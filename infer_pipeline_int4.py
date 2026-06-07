"""
INT4 LLM + ANE Vision parallel pipeline.

Vision Encoder  →  CoreML (.mlpackage)  →  ANE   (~256ms)
LLM decode      →  llama_cpp INT4 GGUF  →  Metal (~93-400ms depending on output length)

Both run on physically separate silicon simultaneously.
This script is self-contained and does NOT modify any existing files.

Usage:
  python infer_pipeline_int4.py \
      --model_path checkpoints/ \
      --gguf_path llm_export/model-q4.gguf \
      --coreml_path vision_encoder.mlpackage \
      --image_path assets/cute_cat.png \
      --repeat 4 --compare
"""
import asyncio
import time
import torch
import numpy as np
import coremltools as ct
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from argparse import ArgumentParser

from mobileo.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from mobileo.model.builder import load_pretrained_model
from mobileo.utils import disable_torch_init
from mobileo.mm_utils import tokenizer_image_token, process_images
from mobileo.conversation import conv_templates


def mps_sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


# ── CLI ───────────────────────────────────────────────────────────────────────
parser = ArgumentParser()
parser.add_argument("--model_path",   type=str, default="checkpoints/")
parser.add_argument("--gguf_path",    type=str, default="llm_export/model-q4.gguf")
parser.add_argument("--coreml_path",  type=str, default="vision_encoder.mlpackage")
parser.add_argument("--image_path",   type=str, default="assets/cute_cat.png")
parser.add_argument("--image_paths",  nargs="+")
parser.add_argument("--prompt",       type=str, default="What is in the image?")
parser.add_argument("--prompts",      nargs="+")
parser.add_argument("--repeat",       type=int, default=4)
parser.add_argument("--max_tokens",   type=int, default=128)
parser.add_argument("--compare",      action="store_true",
                    help="Run all three modes and print speedup table")
args = parser.parse_args()

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype  = torch.float16 if device == "mps" else torch.float32

if args.image_paths:
    image_paths = args.image_paths
    prompts     = args.prompts if args.prompts else [args.prompt] * len(image_paths)
else:
    image_paths = [args.image_path] * args.repeat
    prompts     = [args.prompt]     * args.repeat

N = len(image_paths)
print(f"\nDevice: {device}  |  dtype: {dtype}  |  queries: {N}")
print(f"GGUF:   {args.gguf_path}")
print(f"CoreML: {args.coreml_path}\n")

# ── Load MPS model (used only for image preprocessing + MPS LLM baseline) ────
print("Loading Mobile-O (MPS) model...")
disable_torch_init()
tokenizer, model, _ = load_pretrained_model(args.model_path)
model.to(dtype).to(device)
model.eval()
image_processor = model.get_vision_tower().image_processor
model.generation_config.pad_token_id = tokenizer.pad_token_id

# ── Load CoreML vision encoder (ANE) ─────────────────────────────────────────
print(f"Loading CoreML vision encoder...")
cml_model = ct.models.MLModel(args.coreml_path, compute_units=ct.ComputeUnit.ALL)

# ── Load INT4 LLM via llama_cpp (Metal backend) ───────────────────────────────
print(f"Loading INT4 GGUF LLM (Metal)...")
from llama_cpp import Llama
int4_llm = Llama(
    model_path=args.gguf_path,
    n_gpu_layers=99,          # all layers on Metal
    n_ctx=1024,
    verbose=False,
)
print("All models loaded.\n")


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_llm_prompt(user_prompt: str) -> str:
    """Build Qwen2 chat format prompt (no image tokens — INT4 LLM is text-only)."""
    return (
        "<|im_start|>system\nYou are a helpful visual assistant.<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def build_mps_prompt(user_prompt: str) -> str:
    """Build MPS model prompt (with image token)."""
    qs = DEFAULT_IMAGE_TOKEN + "\n" + user_prompt
    conv = conv_templates["qwen_2"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


# ── Stage A: ANE vision encode ────────────────────────────────────────────────
def encode_image_ane(image_path: str):
    """Preprocess + CoreML vision encoder → returns pixel tensor for MPS reference."""
    raw = Image.open(image_path).convert("RGB")
    tensor = process_images([raw], image_processor, model.config)[0]
    pixel_np = tensor.unsqueeze(0).float().numpy()
    cml_model.predict({"pixel_values": pixel_np})   # warms ANE; output unused by INT4 LLM
    return tensor.unsqueeze(0).to(dtype).to(device)  # for MPS generate() if needed


# ── Stage B: INT4 LLM decode (Metal) ─────────────────────────────────────────
def run_llm_int4(prompt_text: str) -> tuple[str, float]:
    """Run INT4 LLM, returns (answer, elapsed_ms)."""
    t0 = time.perf_counter()
    out = int4_llm(
        prompt_text,
        max_tokens=args.max_tokens,
        temperature=0.8,
        echo=False,
        stop=["<|im_end|>", "<|endoftext|>"],
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return out["choices"][0]["text"].strip(), elapsed_ms


# ── Stage B (baseline): MPS LLM decode ───────────────────────────────────────
def run_llm_mps(image_batch: torch.Tensor, prompt_text: str) -> tuple[str, float]:
    """Run MPS fp16 LLM."""
    input_ids = tokenizer_image_token(
        prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids, images=image_batch,
            do_sample=True, temperature=0.8, top_p=None,
            num_beams=1, max_new_tokens=args.max_tokens,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    mps_sync()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return answer, elapsed_ms


# ── Single-query stage profiler ───────────────────────────────────────────────
def profile_single():
    img_path, prompt = image_paths[0], prompts[0]
    print("=" * 60)
    print("SINGLE-QUERY STAGE PROFILING")
    print("=" * 60)

    # Preprocess
    t0 = time.perf_counter()
    raw = Image.open(img_path).convert("RGB")
    tensor = process_images([raw], image_processor, model.config)[0]
    t_pre = (time.perf_counter() - t0) * 1000

    # ANE vision
    pixel_np = tensor.unsqueeze(0).float().numpy()
    t0 = time.perf_counter()
    cml_model.predict({"pixel_values": pixel_np})
    t_ane = (time.perf_counter() - t0) * 1000

    # MPS vision (for comparison)
    image_batch = tensor.unsqueeze(0).to(dtype).to(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        model.visual(image_batch)
    mps_sync()
    t_mps_vis = (time.perf_counter() - t0) * 1000

    # INT4 LLM
    _, t_int4 = run_llm_int4(build_llm_prompt(prompt))

    # MPS LLM
    _, t_mps_llm = run_llm_mps(image_batch, build_mps_prompt(prompt))

    print(f"  {'Stage':<32} {'MPS':>10}  {'INT4/ANE':>10}")
    print(f"  {'-'*54}")
    print(f"  {'Preprocess (CPU)':32} {'':>10}  {t_pre:>9.1f}ms")
    print(f"  {'Vision encoder':32} {t_mps_vis:>9.1f}ms  {t_ane:>9.1f}ms")
    print(f"  {'LLM decode':32} {t_mps_llm:>9.1f}ms  {t_int4:>9.1f}ms")
    print(f"  {'-'*54}")
    print(f"  {'Sequential total':32} {t_mps_vis+t_mps_llm:>9.1f}ms  {t_ane+t_int4:>9.1f}ms")
    print(f"  {'Ratio (vision:LLM)':32} {'':>10}  {t_ane/max(t_int4,1):.2f} : 1.00")
    print("=" * 60 + "\n")
    return t_ane, t_int4


# ── Sequential MPS baseline ───────────────────────────────────────────────────
def run_sequential_mps(queries):
    results, t0 = [], time.perf_counter()
    for img, prompt in queries:
        raw  = Image.open(img).convert("RGB")
        tensor = process_images([raw], image_processor, model.config)[0]
        image_batch = tensor.unsqueeze(0).to(dtype).to(device)
        ans, _ = run_llm_mps(image_batch, build_mps_prompt(prompt))
        results.append(ans)
    mps_sync()
    return results, time.perf_counter() - t0


# ── Sequential ANE+INT4 (no overlap) ─────────────────────────────────────────
def run_sequential_int4(queries):
    results, t0 = [], time.perf_counter()
    for img, prompt in queries:
        raw = Image.open(img).convert("RGB")
        tensor = process_images([raw], image_processor, model.config)[0]
        cml_model.predict({"pixel_values": tensor.unsqueeze(0).float().numpy()})
        ans, _ = run_llm_int4(build_llm_prompt(prompt))
        results.append(ans)
    return results, time.perf_counter() - t0


# ── Async pipeline: ANE vision overlaps INT4 LLM ─────────────────────────────
async def run_pipeline_int4(queries):
    loop = asyncio.get_event_loop()
    ane_ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ane")
    llm_ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm")
    q: asyncio.Queue = asyncio.Queue(maxsize=2)
    results = [None] * len(queries)

    async def vision_worker():
        for img_path, _ in queries:
            def _ane_encode(p=img_path):
                raw = Image.open(p).convert("RGB")
                tensor = process_images([raw], image_processor, model.config)[0]
                cml_model.predict({"pixel_values": tensor.unsqueeze(0).float().numpy()})
            await loop.run_in_executor(ane_ex, _ane_encode)
            await q.put(True)   # signal: encoded done
        await q.put(None)

    async def llm_worker():
        for i, (_, prompt) in enumerate(queries):
            token = await q.get()
            if token is None:
                break
            prompt_text = build_llm_prompt(prompt)
            ans, _ = await loop.run_in_executor(llm_ex, run_llm_int4, prompt_text)
            results[i] = ans

    t0 = time.perf_counter()
    await asyncio.gather(vision_worker(), llm_worker())
    elapsed = time.perf_counter() - t0

    ane_ex.shutdown(wait=False)
    llm_ex.shutdown(wait=False)
    return results, elapsed


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    queries = list(zip(image_paths, prompts))

    t_ane, t_int4 = profile_single()

    print(f"Pipeline balance: ANE={t_ane:.0f}ms  INT4-LLM={t_int4:.0f}ms  "
          f"ratio={max(t_ane,t_int4)/min(t_ane,t_int4):.2f}× "
          f"({'LLM' if t_int4>t_ane else 'Vision'} is bottleneck)\n")

    if args.compare:
        print("Running MPS-only SEQUENTIAL (original baseline)...")
        _, t_mps = run_sequential_mps(queries)
        print(f"  MPS sequential:       {t_mps*1000:.0f} ms  ({t_mps/N*1000:.0f} ms/query)\n")

        print("Running ANE+INT4 SEQUENTIAL (no overlap)...")
        _, t_int4_seq = run_sequential_int4(queries)
        print(f"  ANE+INT4 sequential:  {t_int4_seq*1000:.0f} ms  ({t_int4_seq/N*1000:.0f} ms/query)\n")

        print("Running ANE+INT4 PIPELINE (async overlap)...")
        results, t_pipe = asyncio.run(run_pipeline_int4(queries))
        print(f"  ANE+INT4 pipeline:    {t_pipe*1000:.0f} ms  ({t_pipe/N*1000:.0f} ms/query)\n")

        print("=" * 60)
        print("SPEEDUP SUMMARY")
        print("=" * 60)
        print(f"  MPS sequential (baseline):   {t_mps*1000:.0f} ms  1.00×")
        print(f"  ANE+INT4 sequential:         {t_int4_seq*1000:.0f} ms  {t_mps/t_int4_seq:.2f}×")
        print(f"  ANE+INT4 pipeline:           {t_pipe*1000:.0f} ms  {t_mps/t_pipe:.2f}×")
        print(f"  Time saved vs baseline:      {(t_mps-t_pipe)*1000:.0f} ms over {N} queries")
        print("=" * 60)

        print("\nAnswers (INT4 pipeline):")
        for i, (ans, (_, p)) in enumerate(zip(results, queries)):
            print(f"  Q{i+1} [{p}]: {ans}")
    else:
        print("Running ANE+INT4 PIPELINE...")
        results, elapsed = asyncio.run(run_pipeline_int4(queries))
        print(f"  Done in {elapsed*1000:.0f} ms  ({elapsed/N*1000:.0f} ms/query)\n")
        for i, (ans, (_, p)) in enumerate(zip(results, queries)):
            print(f"  Q{i+1}: {ans}")


if __name__ == "__main__":
    main()
