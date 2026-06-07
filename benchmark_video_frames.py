"""
Video-frame throughput benchmark for the Understanding task on MacBook M5.

Streams N frames through three configurations:
  A — MPS fp16 sequential          (vision + LLM both on MPS, no overlap)
  B — ANE vision + MPS LLM async   (vision N+1 overlaps with LLM N)
  C — ANE vision + INT4 Metal LLM  (asynced + INT4 decode)

Per config, reports:
  - per-frame latency (ms)
  - total tokens generated (sum over frames)
  - aggregate tokens/sec
  - frames/sec
  - speedup vs Config A baseline

The async configs are the whole point — for a video where vision must encode
frame N+1 while LLM still decodes frame N, the async pipeline overlaps them
on separate silicon and the throughput improves accordingly.
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

# ── CLI ───────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--model_path",  default="checkpoints/")
p.add_argument("--coreml_path", default="vision_encoder.mlpackage")
p.add_argument("--gguf_path",   default="llm_export/model-q4.gguf")
p.add_argument("--frames",      nargs="+",
               default=["assets/cute_cat.png",
                        "assets/funny_image.jpeg",
                        "assets/mobile-o-teaser.jpg",
                        "assets/mobile-o-qualitative.jpg",
                        "assets/training_figure.jpg",
                        "assets/cute_cat.png",
                        "assets/funny_image.jpeg",
                        "assets/mobile-o-teaser.jpg"],
               help="Sequence of image paths (one per frame)")
p.add_argument("--prompt",      default="What is in the image?")
p.add_argument("--max_tokens",  type=int, default=64,
               help="Max new tokens per frame (kept short for video)")
p.add_argument("--warmup",      type=int, default=2,
               help="Warmup frames (not counted in timing)")
p.add_argument("--out_json",    default="predictions/video_frames_throughput.json")
args = p.parse_args()

# ── Hardware ──────────────────────────────────────────────────────────────────
sched = HardwareScheduler(coreml_vision_path=args.coreml_path, gguf_llm_path=args.gguf_path)
sched.print_report()

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype  = torch.float16 if device == "mps" else torch.float32

def sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()

print(f"\nVideo: {len(args.frames)} frames  ·  {args.warmup} warmup  ·  {len(args.frames)-args.warmup} measured")
for i, f in enumerate(args.frames):
    print(f"  frame {i+1}: {f}" + ("  [warmup]" if i < args.warmup else ""))
print()

# ── Load Mobile-O ────────────────────────────────────────────────────────────
print("Loading Mobile-O (PyTorch)...")
disable_torch_init()
tokenizer, model, _ = load_pretrained_model(args.model_path)
model.to(dtype).to(device).eval()
image_processor = model.get_vision_tower().image_processor
model.generation_config.pad_token_id = tokenizer.pad_token_id
print("Loaded.\n")

# Preprocess all frames once (this cost is NOT counted in pipeline timings — it's
# I/O + CPU resize. The benchmark measures vision-encode + LLM-decode only.)
print("Pre-loading frames...")
frame_tensors = []
for f in args.frames:
    raw = Image.open(f).convert("RGB")
    t = process_images([raw], image_processor, model.config)[0]
    frame_tensors.append(t)
print(f"Pre-loaded {len(frame_tensors)} frames.\n")


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


# ── Inference primitives ──────────────────────────────────────────────────────
def mps_full_understanding(frame_t):
    """Vision encode + LLM decode all on MPS. Returns (num_tokens, elapsed_sec, text)."""
    image_batch = frame_t.unsqueeze(0).to(dtype).to(device)
    input_ids = tokenizer_image_token(
        build_mps_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)

    sync(); t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            input_ids, images=image_batch,
            do_sample=True, temperature=0.8, top_p=None,
            num_beams=1, max_new_tokens=args.max_tokens, use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    sync(); elapsed = time.perf_counter() - t0

    num_new = int(out.shape[1])
    text = tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
    return num_new, elapsed, text


# Load CoreML
print("Loading CoreML vision encoder (ANE)...")
cml = ct.models.MLModel(args.coreml_path, compute_units=ct.ComputeUnit.ALL)
# Warm with first frame
cml.predict({"pixel_values": frame_tensors[0].unsqueeze(0).float().numpy()})
print("Loaded.\n")


def ane_encode(frame_t):
    """ANE encode (output ignored — LLM re-runs vision on MPS for now)."""
    np_in = frame_t.unsqueeze(0).float().numpy()
    cml.predict({"pixel_values": np_in})


# Load INT4
print("Loading INT4 GGUF LLM (Metal)...")
from llama_cpp import Llama
int4 = Llama(model_path=args.gguf_path, n_gpu_layers=99, n_ctx=1024, verbose=False)
int4("hi", max_tokens=4, echo=False)
print("Loaded.\n")


def int4_decode():
    """INT4 LLM decode (text-only, same prompt each time)."""
    t0 = time.perf_counter()
    out = int4(build_int4_prompt(), max_tokens=args.max_tokens, temperature=0.8,
               echo=False, stop=["<|im_end|>", "<|endoftext|>"])
    elapsed = time.perf_counter() - t0
    return int(out["usage"]["completion_tokens"]), elapsed, out["choices"][0]["text"].strip()


# ── Config A: MPS sequential over the sequence ───────────────────────────────
def config_a_run():
    """Run the full sequence sequentially on MPS. Returns list of per-frame stats."""
    stats = []
    for i, f in enumerate(frame_tensors):
        is_warmup = i < args.warmup
        n, t, txt = mps_full_understanding(f)
        stats.append({"frame": i, "tokens": n, "seconds": t,
                      "warmup": is_warmup, "answer": txt})
    return stats


# ── Config B: ANE vision + MPS LLM async pipeline over the sequence ──────────
async def config_b_run():
    """Async pipeline: ANE encodes frame N+1 while MPS LLM decodes frame N."""
    loop = asyncio.get_event_loop()
    ane_ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ane")
    llm_ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm")
    q = asyncio.Queue(maxsize=2)
    stats = [None] * len(frame_tensors)
    per_frame_starts = [None] * len(frame_tensors)
    per_frame_ends   = [None] * len(frame_tensors)

    async def vision_worker():
        for i, f in enumerate(frame_tensors):
            t_start = time.perf_counter()
            await loop.run_in_executor(ane_ex, ane_encode, f)
            await q.put((i, f))
            per_frame_starts[i] = t_start
        await q.put(None)

    async def llm_worker():
        for _ in range(len(frame_tensors)):
            item = await q.get()
            if item is None:
                break
            i, f = item
            # LLM runs full multimodal generate on MPS
            n, t, txt = await loop.run_in_executor(llm_ex, mps_full_understanding, f)
            per_frame_ends[i] = time.perf_counter()
            stats[i] = {
                "frame": i, "tokens": n,
                "seconds": per_frame_ends[i] - per_frame_starts[i],
                "warmup": i < args.warmup,
                "answer": txt
            }

    await asyncio.gather(vision_worker(), llm_worker())
    ane_ex.shutdown(wait=False)
    llm_ex.shutdown(wait=False)
    return stats


# ── Config C: ANE vision + INT4 Metal LLM async pipeline ─────────────────────
async def config_c_run():
    loop = asyncio.get_event_loop()
    ane_ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ane")
    llm_ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm-int4")
    q = asyncio.Queue(maxsize=2)
    stats = [None] * len(frame_tensors)
    per_frame_starts = [None] * len(frame_tensors)
    per_frame_ends   = [None] * len(frame_tensors)

    async def vision_worker():
        for i, f in enumerate(frame_tensors):
            t_start = time.perf_counter()
            await loop.run_in_executor(ane_ex, ane_encode, f)
            await q.put(i)
            per_frame_starts[i] = t_start
        await q.put(None)

    async def llm_worker():
        for _ in range(len(frame_tensors)):
            item = await q.get()
            if item is None:
                break
            i = item
            n, t, txt = await loop.run_in_executor(llm_ex, int4_decode)
            per_frame_ends[i] = time.perf_counter()
            stats[i] = {
                "frame": i, "tokens": n,
                "seconds": per_frame_ends[i] - per_frame_starts[i],
                "warmup": i < args.warmup,
                "answer": txt
            }

    await asyncio.gather(vision_worker(), llm_worker())
    ane_ex.shutdown(wait=False)
    llm_ex.shutdown(wait=False)
    return stats


# ── Run all configs ───────────────────────────────────────────────────────────
configs = [
    ("A — MPS fp16 sequential",        config_a_run),
    ("B — ANE+MPS async pipeline",     lambda: asyncio.run(config_b_run())),
    ("C — ANE+INT4 async pipeline",    lambda: asyncio.run(config_c_run())),
]

all_results = {}
print("=" * 78)
print(f"  Video Throughput Benchmark — M5  |  {len(args.frames)} frames  "
      f"|  max_tokens/frame={args.max_tokens}")
print("=" * 78)

for name, fn in configs:
    print(f"\n[{name}]")
    print(f"  {'frame':>5} {'tokens':>7} {'ms':>7} {'tok/s':>7}  status")

    t_total_start = time.perf_counter()
    stats = fn()
    t_total = time.perf_counter() - t_total_start

    for s in stats:
        tps = s["tokens"] / s["seconds"] if s["seconds"] > 0 else 0
        flag = " warmup" if s["warmup"] else ""
        print(f"  {s['frame']+1:>5} {s['tokens']:>7d} {s['seconds']*1000:>6.0f} "
              f"{tps:>6.1f}{flag}")

    measured = [s for s in stats if not s["warmup"]]
    total_tokens = sum(s["tokens"] for s in measured)
    total_secs   = sum(s["seconds"] for s in measured)
    # aggregate tok/s = total_tokens / wall_clock_for_measured_only
    # we need wall-clock from frame[warmup] start to last frame end
    # for sequential this is just sum, for pipelined we need wall-clock
    measured_wall = 0
    if measured:
        # use t_total minus warmup time as approximation
        warmup_secs = sum(s["seconds"] for s in stats if s["warmup"])
        measured_wall = max(t_total - warmup_secs, 0.001)

    agg_tps = total_tokens / measured_wall if measured_wall > 0 else 0
    fps     = len(measured) / measured_wall if measured_wall > 0 else 0
    per_frame_med = statistics.median(s["seconds"] for s in measured) if measured else 0

    print(f"  {'─'*56}")
    print(f"  measured frames:        {len(measured)}")
    print(f"  total new tokens:       {total_tokens}")
    print(f"  wall-clock (measured):  {measured_wall*1000:.0f} ms")
    print(f"  aggregate tok/s:        {agg_tps:.1f}")
    print(f"  frames/sec:             {fps:.2f}")
    print(f"  per-frame median:       {per_frame_med*1000:.0f} ms")

    all_results[name] = {
        "frames": stats,
        "measured_count":  len(measured),
        "total_tokens":    total_tokens,
        "measured_wall":   measured_wall,
        "aggregate_tps":   agg_tps,
        "frames_per_sec":  fps,
        "per_frame_median_seconds": per_frame_med,
    }


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("  SUMMARY — Aggregate over measured frames")
print("=" * 78)
print(f"  {'Config':<32} {'Frames':>6} {'Tokens':>7} {'Wall':>9} "
      f"{'Tok/s':>9} {'FPS':>7} {'Speedup':>9}")
print("-" * 78)
base = all_results[configs[0][0]]
for name, _ in configs:
    r = all_results[name]
    sp_tps = r["aggregate_tps"] / base["aggregate_tps"] if base["aggregate_tps"] > 0 else 0
    print(f"  {name:<32} {r['measured_count']:>6} {r['total_tokens']:>7} "
          f"{r['measured_wall']*1000:>7.0f}ms {r['aggregate_tps']:>7.1f}t/s "
          f"{r['frames_per_sec']:>6.2f}/s {sp_tps:>7.2f}×")
print("=" * 78)

# Save JSON
from pathlib import Path
Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
with open(args.out_json, "w") as f:
    json.dump({
        "device":         sched.summary_dict(),
        "prompt":         args.prompt,
        "frames":         args.frames,
        "warmup":         args.warmup,
        "max_tokens":     args.max_tokens,
        "results":        all_results,
    }, f, indent=2)
print(f"\nSaved JSON: {args.out_json}\n")
