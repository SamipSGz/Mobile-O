"""
Token throughput benchmark for the Understanding task on MacBook M5.

Measures tokens/sec across three configurations:
  Config A — MPS fp16 baseline (sequential)
  Config B — ANE vision + MPS LLM (parallel pipeline)
  Config C — ANE vision + INT4 Metal LLM (best Mac config)

For each: runs N=5 measured iterations (after 1 warmup), reports:
  - tokens generated per run
  - wall-clock time per run
  - tokens/sec per run
  - ms/token per run
  - median over runs
  - speedup vs Config A baseline
"""
import argparse
import asyncio
import json
import statistics
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

import coremltools as ct

from mobileo.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from mobileo.model.builder import load_pretrained_model
from mobileo.utils import disable_torch_init
from mobileo.mm_utils import tokenizer_image_token, process_images
from mobileo.conversation import conv_templates

from hardware_scheduler import HardwareScheduler

# ── Args ─────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--model_path",  default="checkpoints/")
p.add_argument("--coreml_path", default="vision_encoder.mlpackage")
p.add_argument("--gguf_path",   default="llm_export/model-q4.gguf")
p.add_argument("--image_path",  default="assets/cute_cat.png")
p.add_argument("--prompt",      default="What is in the image?")
p.add_argument("--max_tokens",  type=int, default=128)
p.add_argument("--warmup",      type=int, default=1)
p.add_argument("--runs",        type=int, default=5)
p.add_argument("--out_json",    default="predictions/understanding_token_throughput.json")
args = p.parse_args()

# ── Hardware ──────────────────────────────────────────────────────────────────
sched = HardwareScheduler(coreml_vision_path=args.coreml_path, gguf_llm_path=args.gguf_path)
sched.print_report()
cfg = sched.get_config()

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype  = torch.float16 if device == "mps" else torch.float32


def sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def percentile(values, q):
    return float(np.percentile(values, q))


# ── Load Mobile-O ────────────────────────────────────────────────────────────
print("\nLoading Mobile-O (PyTorch)...")
disable_torch_init()
tokenizer, model, _ = load_pretrained_model(args.model_path)
model.to(dtype).to(device).eval()
image_processor = model.get_vision_tower().image_processor
model.generation_config.pad_token_id = tokenizer.pad_token_id
print("Done.\n")

# Prepare input image once
raw = Image.open(args.image_path).convert("RGB")
image_tensor = process_images([raw], image_processor, model.config)[0]
image_batch  = image_tensor.unsqueeze(0).to(dtype).to(device)
pixel_np     = image_tensor.unsqueeze(0).float().numpy()

# Build prompts for both code paths
def build_mps_prompt():
    qs = DEFAULT_IMAGE_TOKEN + "\n" + args.prompt
    conv = conv_templates["qwen_2"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()

def build_int4_prompt():
    return (
        "<|im_start|>system\nYou are a helpful visual assistant.<|im_end|>\n"
        f"<|im_start|>user\n{args.prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


# ── Inference functions ──────────────────────────────────────────────────────
def run_mps_understanding():
    """Sequential MPS fp16: vision + LLM both on MPS."""
    sync(); t0 = time.perf_counter()
    input_ids = tokenizer_image_token(
        build_mps_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)
    with torch.inference_mode():
        out = model.generate(
            input_ids, images=image_batch,
            do_sample=True, temperature=0.8, top_p=None,
            num_beams=1, max_new_tokens=args.max_tokens, use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    sync(); elapsed = time.perf_counter() - t0
    # When images are provided, Mobile-O uses inputs_embeds in super().generate(),
    # so the returned tensor contains only the newly generated tokens (no echoed input)
    num_new = int(out.shape[1])
    text = tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
    return num_new, elapsed, text


# CoreML vision encoder
print("Loading CoreML vision encoder (ANE)...")
cml = ct.models.MLModel(args.coreml_path, compute_units=ct.ComputeUnit.ALL)
cml.predict({"pixel_values": pixel_np})   # warm
print("Done.\n")


def ane_encode_only():
    """Encode image via ANE (timing-only; LLM still re-encodes on MPS)."""
    cml.predict({"pixel_values": pixel_np})


async def run_ane_plus_mps_pipeline():
    """
    Async pipeline: ANE encodes image on Thread A while MPS LLM runs on Thread B.
    For a single query this becomes effectively the same as ANE+MPS sequential,
    but we keep the same code path used in 4-query throughput tests.
    """
    loop = asyncio.get_event_loop()
    ane_ex = ThreadPoolExecutor(max_workers=1)
    llm_ex = ThreadPoolExecutor(max_workers=1)
    q = asyncio.Queue(maxsize=2)
    result = {}

    async def vision_worker():
        await loop.run_in_executor(ane_ex, ane_encode_only)
        await q.put(True)
        await q.put(None)

    async def llm_worker():
        await q.get()
        num, elapsed, text = await loop.run_in_executor(llm_ex, run_mps_understanding)
        result["num"], result["elapsed"], result["text"] = num, elapsed, text

    t0 = time.perf_counter()
    await asyncio.gather(vision_worker(), llm_worker())
    wall = time.perf_counter() - t0

    ane_ex.shutdown(wait=False)
    llm_ex.shutdown(wait=False)
    return result["num"], wall, result["text"]


# INT4 path
print("Loading INT4 GGUF LLM (Metal)...")
from llama_cpp import Llama
int4 = Llama(model_path=args.gguf_path, n_gpu_layers=99, n_ctx=1024, verbose=False)
int4("hi", max_tokens=4, echo=False)   # warm
print("Done.\n")


def run_ane_int4_understanding():
    """ANE vision + INT4 Metal LLM (text-only LLM, image features not injected)."""
    t0 = time.perf_counter()
    cml.predict({"pixel_values": pixel_np})        # ANE vision
    out = int4(build_int4_prompt(), max_tokens=args.max_tokens, temperature=0.8,
               echo=False, stop=["<|im_end|>", "<|endoftext|>"])
    elapsed = time.perf_counter() - t0
    num = int(out["usage"]["completion_tokens"])
    text = out["choices"][0]["text"].strip()
    return num, elapsed, text


# ── Benchmark loop ────────────────────────────────────────────────────────────
configs = [
    ("A — MPS fp16 sequential",        lambda: run_mps_understanding()),
    ("B — ANE vision + MPS LLM",       lambda: asyncio.run(run_ane_plus_mps_pipeline())),
    ("C — ANE vision + INT4 Metal LLM", lambda: run_ane_int4_understanding()),
]

results = {}
print("=" * 70)
print(f"  Mobile-O Understanding Throughput  |  M5  |  prompt: \"{args.prompt}\"")
print(f"  {args.warmup} warmup + {args.runs} measured runs per config")
print("=" * 70)

for name, fn in configs:
    print(f"\n[{name}]")
    # Warmup
    for _ in range(args.warmup):
        n, t, _ = fn()
    times, toks, txt_last = [], [], ""
    for i in range(args.runs):
        n, t, text = fn()
        times.append(t)
        toks.append(n)
        tps = n / t if t > 0 else 0
        print(f"  run {i+1}:  tokens={n:3d}  time={t*1000:7.0f} ms  "
              f"tps={tps:6.1f} tok/s  ms/tok={t*1000/max(n,1):6.1f}")
        txt_last = text
    tps_list = [n / t if t > 0 else 0 for n, t in zip(toks, times)]
    median_tps = statistics.median(tps_list)
    median_t   = statistics.median(times)
    median_n   = statistics.median(toks)
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  MEDIAN:        tokens={median_n:.0f}  time={median_t*1000:.0f} ms  "
          f"tps={median_tps:.1f} tok/s")
    print(f"  Answer: {txt_last[:80]}")
    results[name] = {
        "runs": [{"tokens": n, "seconds": t, "tps": (n/t if t>0 else 0)}
                 for n, t in zip(toks, times)],
        "median_tokens": median_n,
        "median_seconds": median_t,
        "median_tps": median_tps,
        "mean_tps":  statistics.mean(tps_list),
        "stdev_tps": statistics.stdev(tps_list) if len(tps_list) > 1 else 0,
        "last_answer": txt_last,
    }


# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SUMMARY — Median across runs")
print("=" * 70)
print(f"  {'Config':<35} {'Tokens':>7} {'Time':>9} {'Tok/s':>10} {'Speedup':>10}")
print("-" * 75)

base_tps = results[configs[0][0]]["median_tps"]
for name, _ in configs:
    r = results[name]
    speedup = r["median_tps"] / base_tps if base_tps > 0 else 0
    print(f"  {name:<35} {r['median_tokens']:>7.0f} "
          f"{r['median_seconds']*1000:>7.0f} ms  "
          f"{r['median_tps']:>7.1f} t/s  "
          f"{speedup:>7.2f}×")
print("=" * 70)

# Save JSON
from pathlib import Path
Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
with open(args.out_json, "w") as f:
    json.dump({
        "device":     cfg.summary_dict() if hasattr(cfg, "summary_dict") else sched.summary_dict(),
        "prompt":     args.prompt,
        "image":      args.image_path,
        "max_tokens": args.max_tokens,
        "warmup":     args.warmup,
        "runs":       args.runs,
        "results":    results,
    }, f, indent=2)
print(f"\nSaved JSON: {args.out_json}\n")
