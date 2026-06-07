"""
Profiled sequential inference — measures wall time for each stage.
Run: python infer_profiled.py --model_path checkpoints/ --image_path assets/cute_cat.png --prompt "What is in the image?"
"""
import time
import torch
from PIL import Image
from mobileo.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from mobileo.model.builder import load_pretrained_model
from mobileo.utils import disable_torch_init
from mobileo.mm_utils import tokenizer_image_token, process_images
from mobileo.conversation import conv_templates
from argparse import ArgumentParser


def sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def timed(label, fn):
    sync()
    t0 = time.perf_counter()
    result = fn()
    sync()
    t1 = time.perf_counter()
    print(f"  {label:<30} {(t1 - t0) * 1000:7.1f} ms")
    return result, t1 - t0


parser = ArgumentParser()
parser.add_argument("--model_path", type=str, default="checkpoints/mobileo_unified_1.5B")
parser.add_argument("--image_path", type=str, default="assets/cute_cat.png")
parser.add_argument("--prompt", type=str, default="What is in the image?")
args = parser.parse_args()

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device == "mps" else torch.float32

print(f"\nDevice: {device}  |  dtype: {dtype}\n")

# ── Model load (not part of pipeline timing) ────────────────────────────────
disable_torch_init()
tokenizer, model, _ = load_pretrained_model(args.model_path)
model.to(dtype).to(device)
model.eval()

image_processor = model.get_vision_tower().image_processor

# ── Build text inputs (CPU — cheap) ─────────────────────────────────────────
qs = DEFAULT_IMAGE_TOKEN + "\n" + args.prompt
conv = conv_templates["qwen_2"].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt_text = conv.get_prompt()

model.generation_config.pad_token_id = tokenizer.pad_token_id
input_ids = tokenizer_image_token(
    prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
).unsqueeze(0).to(device)

raw_image = Image.open(args.image_path).convert("RGB")

print("=" * 55)
print("STAGE TIMINGS (single query)")
print("=" * 55)

# ── Stage 1: image pre-processing (CPU) ─────────────────────────────────────
image_tensor, t_preproc = timed(
    "Stage 1 — image preprocess",
    lambda: process_images([raw_image], image_processor, model.config)[0],
)

# ── Stage 2: vision encoder + projector (MPS/ANE) ───────────────────────────
with torch.inference_mode():
    image_batch = image_tensor.unsqueeze(0).to(dtype).to(device)
    image_features, t_vision = timed(
        "Stage 2 — vision encoder",
        lambda: model.visual(image_batch),
    )

# ── Stage 3: LLM generate (MPS) ─────────────────────────────────────────────
with torch.inference_mode():
    output_ids, t_llm = timed(
        "Stage 3 — LLM generate",
        lambda: model.generate(
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
        ),
    )

total = t_preproc + t_vision + t_llm
print("-" * 55)
print(f"  {'TOTAL (sequential)':30} {total * 1000:7.1f} ms")
print("=" * 55)
print("\nAnswer:", tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip())
print()
print("Baseline numbers for pipeline design:")
print(f"  Vision encoder fraction : {t_vision / total * 100:.1f}%")
print(f"  LLM generate fraction   : {t_llm    / total * 100:.1f}%")
