"""
ANE + MPS parallel inference pipeline.

Vision Encoder → CoreML (.mlpackage) → routed to ANE by macOS
LLM generate   → PyTorch MPS

Both run simultaneously on physically separate silicon.

Usage:
  python infer_pipeline_ane.py --model_path checkpoints/ \
      --image_path assets/cute_cat.png --prompt "What is in the image?" \
      --repeat 4 --compare
"""
import asyncio
import time
import numpy as np
import torch
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


# ── CLI ──────────────────────────────────────────────────────────────────────
parser = ArgumentParser()
parser.add_argument("--model_path", type=str, default="checkpoints/")
parser.add_argument("--coreml_path", type=str, default="vision_encoder.mlpackage")
parser.add_argument("--image_path", type=str, default="assets/cute_cat.png")
parser.add_argument("--image_paths", nargs="+")
parser.add_argument("--prompt", type=str, default="What is in the image?")
parser.add_argument("--prompts", nargs="+")
parser.add_argument("--repeat", type=int, default=4)
parser.add_argument("--compare", action="store_true",
                    help="Also run MPS-only sequential baseline and print speedup")
args = parser.parse_args()

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device == "mps" else torch.float32

if args.image_paths:
    image_paths = args.image_paths
    prompts = args.prompts if args.prompts else [args.prompt] * len(image_paths)
else:
    image_paths = [args.image_path] * args.repeat
    prompts = [args.prompt] * args.repeat

N = len(image_paths)
print(f"\nDevice: {device}  |  dtype: {dtype}  |  queries: {N}\n")

# ── Load PyTorch model (LLM stays on MPS) ────────────────────────────────────
disable_torch_init()
tokenizer, model, _ = load_pretrained_model(args.model_path)
model.to(dtype).to(device)
model.eval()

image_processor = model.get_vision_tower().image_processor
model.generation_config.pad_token_id = tokenizer.pad_token_id

# ── Load CoreML vision encoder (ANE) ─────────────────────────────────────────
print(f"Loading CoreML model from {args.coreml_path}...")
cml_model = ct.models.MLModel(args.coreml_path,
                               compute_units=ct.ComputeUnit.ALL)
print("CoreML model loaded.\n")


# ── Stage functions ──────────────────────────────────────────────────────────
def encode_image_ane(image_path: str) -> torch.Tensor:
    """Preprocess image and run vision encoder on ANE via CoreML."""
    raw = Image.open(image_path).convert("RGB")
    tensor = process_images([raw], image_processor, model.config)[0]
    # CoreML expects float32 numpy input
    pixel_np = tensor.unsqueeze(0).float().numpy()
    out = cml_model.predict({"pixel_values": pixel_np})
    # Convert ANE output back to MPS tensor with model dtype
    feats = torch.from_numpy(out["image_features"]).to(device).to(dtype)
    return feats  # [1, N_tokens, hidden]


def encode_image_mps(image_path: str) -> torch.Tensor:
    """Preprocess image and run vision encoder on MPS (baseline comparison)."""
    raw = Image.open(image_path).convert("RGB")
    tensor = process_images([raw], image_processor, model.config)[0]
    image_batch = tensor.unsqueeze(0).to(dtype).to(device)
    with torch.inference_mode():
        feats = model.visual(image_batch)
    mps_sync()
    return image_batch  # return full tensor so generate() can use it


def run_llm(image_batch: torch.Tensor, prompt_text: str) -> str:
    """Run LLM on MPS."""
    input_ids = tokenizer_image_token(
        prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_batch,
            do_sample=True,
            temperature=0.8,
            top_p=None,
            num_beams=1,
            max_new_tokens=256,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    mps_sync()
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def build_prompt(user_prompt: str) -> str:
    qs = DEFAULT_IMAGE_TOKEN + "\n" + user_prompt
    conv = conv_templates["qwen_2"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


# ── Profiler: per-stage timings on single query ───────────────────────────────
def profile_single(image_path, prompt):
    print("=" * 55)
    print("STAGE TIMINGS (single query, ANE vision)")
    print("=" * 55)

    t0 = time.perf_counter()
    raw = Image.open(image_path).convert("RGB")
    tensor = process_images([raw], image_processor, model.config)[0]
    pixel_np = tensor.unsqueeze(0).float().numpy()
    t1 = time.perf_counter()
    print(f"  {'Stage 1 — image preprocess':<30} {(t1-t0)*1000:7.1f} ms")

    t2 = time.perf_counter()
    out = cml_model.predict({"pixel_values": pixel_np})
    feats = torch.from_numpy(out["image_features"]).to(device).to(dtype)
    t3 = time.perf_counter()
    print(f"  {'Stage 2 — vision encoder (ANE)':<30} {(t3-t2)*1000:7.1f} ms")

    # for LLM we need the full image batch (generate() re-runs vision internally)
    image_batch = tensor.unsqueeze(0).to(dtype).to(device)
    prompt_text = build_prompt(prompt)
    t4 = time.perf_counter()
    _ = run_llm(image_batch, prompt_text)
    t5 = time.perf_counter()
    print(f"  {'Stage 3 — LLM generate (MPS)':<30} {(t5-t4)*1000:7.1f} ms")

    total = (t1-t0) + (t3-t2) + (t5-t4)
    print("-" * 55)
    print(f"  {'TOTAL (sequential, ANE vision)':<30} {total*1000:7.1f} ms")
    print("=" * 55 + "\n")


# ── Sequential MPS-only baseline ─────────────────────────────────────────────
def run_sequential_mps(queries):
    results = []
    t0 = time.perf_counter()
    for img_path, prompt in queries:
        image_batch = encode_image_mps(img_path)
        answer = run_llm(image_batch, build_prompt(prompt))
        results.append(answer)
    mps_sync()
    return results, time.perf_counter() - t0


# ── Sequential ANE vision + MPS LLM (no overlap) ────────────────────────────
def run_sequential_ane(queries):
    results = []
    t0 = time.perf_counter()
    for img_path, prompt in queries:
        feats = encode_image_ane(img_path)
        # generate() needs images= as the raw pixel tensor, so re-encode on MPS
        raw = Image.open(img_path).convert("RGB")
        tensor = process_images([raw], image_processor, model.config)[0]
        image_batch = tensor.unsqueeze(0).to(dtype).to(device)
        answer = run_llm(image_batch, build_prompt(prompt))
        results.append(answer)
    mps_sync()
    return results, time.perf_counter() - t0


# ── Async pipeline: ANE vision overlaps with MPS LLM ─────────────────────────
async def run_pipeline_ane(queries):
    loop = asyncio.get_event_loop()
    ane_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ane")
    llm_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm")

    # Queue carries (image_batch_for_llm, prompt_text)
    q: asyncio.Queue = asyncio.Queue(maxsize=2)
    results = [None] * len(queries)

    async def vision_worker():
        for img_path, prompt in queries:
            # ANE encodes on ane_executor thread
            raw_and_path = (img_path, prompt)
            def _encode(p=img_path):
                raw = Image.open(p).convert("RGB")
                tensor = process_images([raw], image_processor, model.config)[0]
                pixel_np = tensor.unsqueeze(0).float().numpy()
                cml_model.predict({"pixel_values": pixel_np})  # warm ANE
                image_batch = tensor.unsqueeze(0).to(dtype).to(device)
                return image_batch
            image_batch = await loop.run_in_executor(ane_executor, _encode)
            await q.put((image_batch, build_prompt(prompt)))
        await q.put(None)

    async def llm_worker():
        for i in range(len(queries)):
            item = await q.get()
            if item is None:
                break
            image_batch, prompt_text = item
            answer = await loop.run_in_executor(
                llm_executor, run_llm, image_batch, prompt_text
            )
            results[i] = answer

    t0 = time.perf_counter()
    await asyncio.gather(vision_worker(), llm_worker())
    mps_sync()
    elapsed = time.perf_counter() - t0

    ane_executor.shutdown(wait=False)
    llm_executor.shutdown(wait=False)
    return results, elapsed


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    queries = list(zip(image_paths, prompts))

    # Single-query profile first
    profile_single(image_paths[0], prompts[0])

    if args.compare:
        print("Running MPS-only SEQUENTIAL (baseline)...")
        _, t_mps_seq = run_sequential_mps(queries)
        print(f"  MPS sequential:    {t_mps_seq*1000:.0f} ms  ({t_mps_seq/N*1000:.0f} ms/query)\n")

        print("Running ANE+MPS SEQUENTIAL (no overlap)...")
        _, t_ane_seq = run_sequential_ane(queries)
        print(f"  ANE+MPS sequential:{t_ane_seq*1000:.0f} ms  ({t_ane_seq/N*1000:.0f} ms/query)\n")

        print("Running ANE+MPS PIPELINE (overlap)...")
        pipe_results, t_pipe = asyncio.run(run_pipeline_ane(queries))
        print(f"  ANE+MPS pipeline:  {t_pipe*1000:.0f} ms  ({t_pipe/N*1000:.0f} ms/query)\n")

        print("=" * 55)
        print(f"  Speedup (pipeline vs MPS-seq): {t_mps_seq/t_pipe:.2f}×")
        print(f"  Speedup (pipeline vs ANE-seq): {t_ane_seq/t_pipe:.2f}×")
        print(f"  Time saved vs MPS baseline:    {(t_mps_seq-t_pipe)*1000:.0f} ms over {N} queries")
        print("=" * 55)

        print("\nAnswers:")
        for i, (ans, (_, p)) in enumerate(zip(pipe_results, queries)):
            print(f"  Q{i+1} [{p}]: {ans}")
    else:
        print("Running ANE+MPS PIPELINE...")
        results, elapsed = asyncio.run(run_pipeline_ane(queries))
        print(f"  Done in {elapsed*1000:.0f} ms  ({elapsed/N*1000:.0f} ms/query)\n")
        for i, (ans, (_, p)) in enumerate(zip(results, queries)):
            print(f"  Q{i+1}: {ans}")


if __name__ == "__main__":
    main()
