"""
Async pipelined inference — Vision Encoder and LLM overlap across queries.

While LLM is answering query N, Vision Encoder is already encoding query N+1.
They run on the same MPS device but in separate threads, exploiting the fact
that MPS can queue work from multiple host threads concurrently.

Usage (multiple queries from a file, one image path per line):
  python infer_pipeline.py --model_path checkpoints/ \
      --image_paths assets/cute_cat.png assets/cute_cat.png assets/cute_cat.png \
      --prompts "What is in the image?" "Describe the colors." "How many animals?"

Or benchmark mode (repeats one image N times to measure throughput):
  python infer_pipeline.py --model_path checkpoints/ \
      --image_path assets/cute_cat.png --prompt "What is in the image?" \
      --repeat 4 --compare
"""
import asyncio
import time
import torch
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from argparse import ArgumentParser

from mobileo.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from mobileo.model.builder import load_pretrained_model
from mobileo.utils import disable_torch_init
from mobileo.mm_utils import tokenizer_image_token, process_images
from mobileo.conversation import conv_templates


def sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


# ── CLI ──────────────────────────────────────────────────────────────────────
parser = ArgumentParser()
parser.add_argument("--model_path", type=str, default="checkpoints/mobileo_unified_1.5B")
parser.add_argument("--image_path", type=str, default="assets/cute_cat.png",
                    help="Single image (used with --repeat or --prompts)")
parser.add_argument("--image_paths", nargs="+", help="Multiple image paths")
parser.add_argument("--prompt", type=str, default="What is in the image?")
parser.add_argument("--prompts", nargs="+", help="One prompt per image")
parser.add_argument("--repeat", type=int, default=3,
                    help="Repeat single image N times for benchmarking")
parser.add_argument("--compare", action="store_true",
                    help="Also run sequential baseline and print speedup")
args = parser.parse_args()

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device == "mps" else torch.float32

# ── Build query list ─────────────────────────────────────────────────────────
if args.image_paths:
    image_paths = args.image_paths
    prompts = args.prompts if args.prompts else [args.prompt] * len(image_paths)
else:
    image_paths = [args.image_path] * args.repeat
    prompts = [args.prompt] * args.repeat

assert len(image_paths) == len(prompts), "image_paths and prompts must be same length"
N = len(image_paths)

print(f"\nDevice: {device}  |  dtype: {dtype}  |  queries: {N}\n")

# ── Load model ───────────────────────────────────────────────────────────────
disable_torch_init()
tokenizer, model, _ = load_pretrained_model(args.model_path)
model.to(dtype).to(device)
model.eval()

image_processor = model.get_vision_tower().image_processor
model.generation_config.pad_token_id = tokenizer.pad_token_id


# ── Stage functions ──────────────────────────────────────────────────────────
def encode_image(image_path: str) -> torch.Tensor:
    """Stage 1+2: preprocess + vision encoder. Returns image_features tensor."""
    raw = Image.open(image_path).convert("RGB")
    tensor = process_images([raw], image_processor, model.config)[0]
    image_batch = tensor.unsqueeze(0).to(dtype).to(device)
    with torch.inference_mode():
        features = model.visual(image_batch)  # [1, tokens, hidden]
    sync()
    return image_batch  # keep full tensor so generate() can use it


def run_llm(image_batch: torch.Tensor, prompt_text: str) -> str:
    """Stage 3: LLM generate given pre-encoded image tensor."""
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
    sync()
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def build_prompt(user_prompt: str) -> str:
    qs = DEFAULT_IMAGE_TOKEN + "\n" + user_prompt
    conv = conv_templates["qwen_2"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


# ── Sequential baseline ──────────────────────────────────────────────────────
def run_sequential(queries):
    results = []
    t0 = time.perf_counter()
    for img_path, prompt in queries:
        image_batch = encode_image(img_path)
        answer = run_llm(image_batch, build_prompt(prompt))
        results.append(answer)
    sync()
    elapsed = time.perf_counter() - t0
    return results, elapsed


# ── Async pipelined inference ────────────────────────────────────────────────
async def run_pipeline(queries):
    loop = asyncio.get_event_loop()
    # Two separate single-thread executors so vision and LLM never share a thread
    vision_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vision")
    llm_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm")

    feature_queue: asyncio.Queue = asyncio.Queue(maxsize=2)
    results = [None] * len(queries)

    async def vision_worker():
        for img_path, _ in queries:
            image_batch = await loop.run_in_executor(
                vision_executor, encode_image, img_path
            )
            await feature_queue.put(image_batch)
        await feature_queue.put(None)  # sentinel

    async def llm_worker():
        for i, (_, prompt) in enumerate(queries):
            image_batch = await feature_queue.get()
            if image_batch is None:
                break
            prompt_text = build_prompt(prompt)
            answer = await loop.run_in_executor(
                llm_executor, run_llm, image_batch, prompt_text
            )
            results[i] = answer

    t0 = time.perf_counter()
    await asyncio.gather(vision_worker(), llm_worker())
    sync()
    elapsed = time.perf_counter() - t0

    vision_executor.shutdown(wait=False)
    llm_executor.shutdown(wait=False)
    return results, elapsed


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    queries = list(zip(image_paths, prompts))

    if args.compare:
        print("Running SEQUENTIAL baseline...")
        seq_results, seq_time = run_sequential(queries)
        print(f"  Sequential total: {seq_time * 1000:.0f} ms  ({seq_time / N * 1000:.0f} ms/query)\n")

        print("Running PIPELINE...")
        pipe_results, pipe_time = asyncio.run(run_pipeline(queries))
        print(f"  Pipeline total:   {pipe_time * 1000:.0f} ms  ({pipe_time / N * 1000:.0f} ms/query)\n")

        speedup = seq_time / pipe_time
        saved_ms = (seq_time - pipe_time) * 1000
        print("=" * 50)
        print(f"  Speedup:  {speedup:.2f}×   (saved {saved_ms:.0f} ms over {N} queries)")
        print("=" * 50)

        print("\nAnswers (pipeline):")
        for i, (ans, (img, prompt)) in enumerate(zip(pipe_results, queries)):
            print(f"  Q{i+1} [{prompt}]: {ans}")
    else:
        print("Running PIPELINE...")
        results, elapsed = asyncio.run(run_pipeline(queries))
        print(f"  Done in {elapsed * 1000:.0f} ms  ({elapsed / N * 1000:.0f} ms/query)\n")
        for i, (ans, (img, prompt)) in enumerate(zip(results, queries)):
            print(f"  Q{i+1}: {ans}")


if __name__ == "__main__":
    main()
