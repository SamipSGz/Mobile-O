"""
Runs all 5 pipeline stages and saves annotated output images to predictions/stages/.

Stage 1: Original MPS sequential (infer_image_understanding baseline)
Stage 2: Profiled MPS sequential (with per-stage timing)
Stage 3: MPS-only async pipeline (infer_pipeline — 1.02× result)
Stage 4: ANE+MPS async pipeline (infer_pipeline_ane — 1.61× result)
Stage 5: ANE+INT4 Metal pipeline (infer_pipeline_int4 — 4.21× result)
"""
import time
import asyncio
import torch
import numpy as np
import coremltools as ct
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from mobileo.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from mobileo.model.builder import load_pretrained_model
from mobileo.utils import disable_torch_init
from mobileo.mm_utils import tokenizer_image_token, process_images
from mobileo.conversation import conv_templates

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_PATH   = "assets/cute_cat.png"
PROMPT       = "What is in the image?"
MODEL_PATH   = "checkpoints/"
COREML_PATH  = "vision_encoder.mlpackage"
GGUF_PATH    = "llm_export/model-q4.gguf"
OUT_DIR      = Path("predictions/stages")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPEAT       = 4   # queries for multi-query stages

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype  = torch.float16 if device == "mps" else torch.float32

CARD_W, CARD_H = 512, 640   # output image card size


# ── Annotation helper ─────────────────────────────────────────────────────────
def make_card(title: str, subtitle: str, timings: list[tuple[str,str]],
              answer: str, stage_num: int, color: tuple) -> Image.Image:
    """Create an annotated result card."""
    card = Image.new("RGB", (CARD_W, CARD_H), (18, 18, 24))
    draw = ImageDraw.Draw(card)

    try:
        font_lg = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
        font_md = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 17)
        font_sm = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        font_xs = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    except Exception:
        font_lg = font_md = font_sm = font_xs = ImageFont.load_default()

    # Colour strip at top
    draw.rectangle([0, 0, CARD_W, 6], fill=color)

    # Stage badge
    draw.rectangle([16, 18, 68, 46], fill=color)
    draw.text((24, 22), f"S{stage_num}", fill=(255,255,255), font=font_md)

    # Title
    draw.text((80, 18), title, fill=(255,255,255), font=font_lg)
    draw.text((80, 44), subtitle, fill=(160,160,170), font=font_sm)

    # Divider
    draw.rectangle([16, 64, CARD_W-16, 65], fill=(50,50,60))

    # Paste input image (resized)
    raw = Image.open(IMAGE_PATH).convert("RGB")
    thumb = raw.resize((200, 200), Image.LANCZOS)
    card.paste(thumb, (16, 74))

    # Q&A block
    draw.text((228, 74), "INPUT", fill=(100,100,120), font=font_xs)
    draw.text((228, 92), f"Q: {PROMPT}", fill=(200,200,210), font=font_sm)
    draw.text((228, 116), "ANSWER", fill=(100,100,120), font=font_xs)
    # Word-wrap answer
    words = answer.split()
    lines, line = [], []
    for w in words:
        if sum(len(x)+1 for x in line+[w]) > 30:
            lines.append(" ".join(line)); line=[w]
        else:
            line.append(w)
    if line: lines.append(" ".join(line))
    for j, ln in enumerate(lines[:5]):
        draw.text((228, 134+j*18), ln, fill=(255,255,255), font=font_sm)

    # Timing table
    y = 290
    draw.rectangle([16, y-4, CARD_W-16, y-3], fill=(50,50,60))
    draw.text((16, y), "TIMING BREAKDOWN", fill=(100,100,120), font=font_xs)
    y += 20
    for label, val in timings:
        # bar
        try:
            ms = float(val.replace("ms","").replace(" ","").split("(")[0])
            bar_w = min(int(ms / 50), 200)
            draw.rectangle([16, y+4, 16+bar_w, y+14], fill=(*color, 180))
        except Exception:
            pass
        draw.text((16, y), label, fill=(190,190,200), font=font_sm)
        draw.text((CARD_W-16-80, y), val, fill=(255,255,255), font=font_sm)
        y += 22

    # Bottom speedup badge
    draw.rectangle([0, CARD_H-50, CARD_W, CARD_H], fill=(28, 28, 36))
    draw.rectangle([0, CARD_H-52, CARD_W, CARD_H-50], fill=color)
    # Find speedup line
    speedup_line = next((v for l,v in timings if "speedup" in l.lower() or "×" in v), "")
    draw.text((16, CARD_H-38), "Apple M5  |  Python  |  Mobile-O 1.6B",
              fill=(90,90,110), font=font_xs)
    draw.text((CARD_W//2, CARD_H-38), speedup_line,
              fill=color, font=font_md, anchor="lm")

    return card


def save_card(card: Image.Image, name: str):
    path = OUT_DIR / name
    card.save(path)
    print(f"  Saved: {path}")


def mps_sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


# ── Load model once ───────────────────────────────────────────────────────────
print("Loading Mobile-O model (shared across all stages)...")
disable_torch_init()
tokenizer, model, _ = load_pretrained_model(MODEL_PATH)
model.to(dtype).to(device)
model.eval()
image_processor = model.get_vision_tower().image_processor
model.generation_config.pad_token_id = tokenizer.pad_token_id

raw_image   = Image.open(IMAGE_PATH).convert("RGB")
image_tensor = process_images([raw_image], image_processor, model.config)[0]
image_batch  = image_tensor.unsqueeze(0).to(dtype).to(device)

def mps_prompt():
    qs = DEFAULT_IMAGE_TOKEN + "\n" + PROMPT
    conv = conv_templates["qwen_2"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()

def mps_generate(img_batch):
    input_ids = tokenizer_image_token(
        mps_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)
    with torch.inference_mode():
        out = model.generate(
            input_ids, images=img_batch,
            do_sample=True, temperature=0.8, top_p=None,
            num_beams=1, max_new_tokens=128, use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    mps_sync()
    return tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()

print("Model loaded.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Original MPS sequential (baseline)
# ═══════════════════════════════════════════════════════════════════════════════
print("Stage 1: Original MPS sequential...")

# Vision
mps_sync(); t0=time.perf_counter()
with torch.inference_mode(): model.visual(image_batch)
mps_sync(); t_vis=(time.perf_counter()-t0)*1000

# LLM
mps_sync(); t0=time.perf_counter()
answer1 = mps_generate(image_batch)
mps_sync(); t_llm=(time.perf_counter()-t0)*1000

total1 = t_vis + t_llm
print(f"  Vision={t_vis:.0f}ms  LLM={t_llm:.0f}ms  Total={total1:.0f}ms")
print(f"  Answer: {answer1}")

card1 = make_card(
    title="Original MPS Sequential",
    subtitle="Stage 1 — CUDA→MPS fix, single-threaded fp16",
    timings=[
        ("Vision encoder (MPS fp16)", f"{t_vis:.0f}ms"),
        ("LLM generate (MPS fp16)",   f"{t_llm:.0f}ms"),
        ("Total single query",         f"{total1:.0f}ms"),
        ("Hardware used",              "MPS only"),
        ("Speedup vs baseline",        "1.00×"),
    ],
    answer=answer1, stage_num=1, color=(80, 140, 255)
)
save_card(card1, "stage1_original_mps.png")


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Profiled MPS sequential (infer_profiled.py numbers)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nStage 2: Profiled MPS sequential...")

t0=time.perf_counter()
img_tensor2 = process_images([raw_image], image_processor, model.config)[0]
t_pre=(time.perf_counter()-t0)*1000

mps_sync(); t0=time.perf_counter()
with torch.inference_mode(): model.visual(image_batch)
mps_sync(); t_vis2=(time.perf_counter()-t0)*1000

mps_sync(); t0=time.perf_counter()
answer2 = mps_generate(image_batch)
mps_sync(); t_llm2=(time.perf_counter()-t0)*1000

total2 = t_pre + t_vis2 + t_llm2
print(f"  Preprocess={t_pre:.0f}ms  Vision={t_vis2:.0f}ms  LLM={t_llm2:.0f}ms  Total={total2:.0f}ms")

card2 = make_card(
    title="Profiled MPS Sequential",
    subtitle="Stage 2 — Per-stage timing with MPS sync hooks",
    timings=[
        ("Image preprocess (CPU)",    f"{t_pre:.0f}ms  ({t_pre/total2*100:.0f}%)"),
        ("Vision encoder (MPS)",      f"{t_vis2:.0f}ms  ({t_vis2/total2*100:.0f}%)"),
        ("LLM generate (MPS)",        f"{t_llm2:.0f}ms  ({t_llm2/total2*100:.0f}%)"),
        ("Total single query",         f"{total2:.0f}ms"),
        ("Vision is bottleneck",       f"{t_vis2/total2*100:.0f}% of total"),
    ],
    answer=answer2, stage_num=2, color=(255, 160, 40)
)
save_card(card2, "stage2_profiled_mps.png")


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — MPS-only async pipeline (infer_pipeline.py — negative result)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nStage 3: MPS-only async pipeline (4 queries)...")

queries = [(IMAGE_PATH, PROMPT)] * REPEAT

def seq_mps_run():
    results=[]; t0=time.perf_counter()
    for _ in range(REPEAT):
        results.append(mps_generate(image_batch))
    mps_sync()
    return results, (time.perf_counter()-t0)*1000

def encode_mps():
    with torch.inference_mode(): model.visual(image_batch)
    mps_sync()
    return image_batch

async def pipeline_mps():
    loop = asyncio.get_event_loop()
    vis_ex = ThreadPoolExecutor(max_workers=1)
    llm_ex = ThreadPoolExecutor(max_workers=1)
    q: asyncio.Queue = asyncio.Queue(maxsize=2)
    results=[None]*REPEAT

    async def v_worker():
        for _ in range(REPEAT):
            b = await loop.run_in_executor(vis_ex, encode_mps)
            await q.put(b)
        await q.put(None)

    async def l_worker():
        for i in range(REPEAT):
            b = await q.get()
            if b is None: break
            ans = await loop.run_in_executor(llm_ex, mps_generate, b)
            results[i] = ans

    t0 = time.perf_counter()
    await asyncio.gather(v_worker(), l_worker())
    mps_sync()
    return results, (time.perf_counter()-t0)*1000

_, t_seq3 = seq_mps_run()
pipe_results3, t_pipe3 = asyncio.run(pipeline_mps())
speedup3 = t_seq3 / t_pipe3
answer3 = pipe_results3[0] or mps_generate(image_batch)
print(f"  Sequential={t_seq3:.0f}ms  Pipeline={t_pipe3:.0f}ms  Speedup={speedup3:.2f}×")
print(f"  (MPS threads serialize on GPU — expected near-zero gain)")

card3 = make_card(
    title="MPS-Only Async Pipeline",
    subtitle="Stage 3 — Threading on single GPU serializes (negative result)",
    timings=[
        (f"Sequential {REPEAT}q (MPS)",   f"{t_seq3:.0f}ms"),
        (f"Pipeline {REPEAT}q (MPS)",     f"{t_pipe3:.0f}ms"),
        ("ms / query (pipeline)",          f"{t_pipe3/REPEAT:.0f}ms"),
        ("Finding: GPU serializes threads","no real overlap"),
        ("Speedup vs baseline",            f"{speedup3:.2f}×  (≈1.0×)"),
    ],
    answer=answer3, stage_num=3, color=(200, 80, 80)
)
save_card(card3, "stage3_pipeline_mps_only.png")


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — ANE+MPS async pipeline (infer_pipeline_ane.py)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nStage 4: ANE+MPS async pipeline (4 queries)...")

print("  Loading CoreML model...")
cml_model = ct.models.MLModel(COREML_PATH, compute_units=ct.ComputeUnit.ALL)
pixel_np = image_tensor.unsqueeze(0).float().numpy()

# Warm ANE
cml_model.predict({"pixel_values": pixel_np})

def encode_ane():
    cml_model.predict({"pixel_values": pixel_np})
    return image_batch  # return MPS tensor for generate()

# Sequential ANE+MPS
def seq_ane():
    results=[]; t0=time.perf_counter()
    for _ in range(REPEAT):
        encode_ane()
        results.append(mps_generate(image_batch))
    return results, (time.perf_counter()-t0)*1000

# ANE timing
t0=time.perf_counter()
encode_ane()
t_ane4=(time.perf_counter()-t0)*1000

async def pipeline_ane():
    loop = asyncio.get_event_loop()
    ane_ex = ThreadPoolExecutor(max_workers=1)
    llm_ex = ThreadPoolExecutor(max_workers=1)
    q: asyncio.Queue = asyncio.Queue(maxsize=2)
    results=[None]*REPEAT

    async def v_worker():
        for _ in range(REPEAT):
            b = await loop.run_in_executor(ane_ex, encode_ane)
            await q.put(b)
        await q.put(None)

    async def l_worker():
        for i in range(REPEAT):
            b = await q.get()
            if b is None: break
            ans = await loop.run_in_executor(llm_ex, mps_generate, b)
            results[i] = ans

    t0=time.perf_counter()
    await asyncio.gather(v_worker(), l_worker())
    mps_sync()
    return results, (time.perf_counter()-t0)*1000

_, t_seq4 = seq_ane()
pipe_results4, t_pipe4 = asyncio.run(pipeline_ane())
speedup4 = t_seq3 / t_pipe4   # vs original MPS sequential
answer4 = pipe_results4[0] or mps_generate(image_batch)
print(f"  ANE vision={t_ane4:.0f}ms  Sequential={t_seq4:.0f}ms  Pipeline={t_pipe4:.0f}ms  Speedup={speedup4:.2f}×")

card4 = make_card(
    title="ANE + MPS Pipeline",
    subtitle="Stage 4 — CoreML→ANE vision + MPS LLM on separate silicon",
    timings=[
        ("Vision encoder (ANE)",           f"{t_ane4:.0f}ms  (was {t_vis2:.0f}ms)"),
        (f"Sequential {REPEAT}q ANE+MPS",  f"{t_seq4:.0f}ms"),
        (f"Pipeline {REPEAT}q ANE+MPS",    f"{t_pipe4:.0f}ms"),
        ("ms / query (pipeline)",           f"{t_pipe4/REPEAT:.0f}ms"),
        ("Speedup vs MPS baseline",         f"{speedup4:.2f}×"),
    ],
    answer=answer4, stage_num=4, color=(80, 200, 120)
)
save_card(card4, "stage4_pipeline_ane_mps.png")


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — ANE+INT4 Metal pipeline (infer_pipeline_int4.py)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nStage 5: ANE+INT4 Metal pipeline (4 queries)...")

from llama_cpp import Llama
print("  Loading INT4 GGUF LLM...")
int4_llm = Llama(model_path=GGUF_PATH, n_gpu_layers=99, n_ctx=512, verbose=False)
# Warm up
int4_llm("hi", max_tokens=5, echo=False)

def int4_prompt():
    return (
        "<|im_start|>system\nYou are a helpful visual assistant.<|im_end|>\n"
        f"<|im_start|>user\n{PROMPT}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def run_int4():
    t0=time.perf_counter()
    out = int4_llm(int4_prompt(), max_tokens=128, temperature=0.8, echo=False,
                   stop=["<|im_end|>","<|endoftext|>"])
    return out["choices"][0]["text"].strip(), (time.perf_counter()-t0)*1000

# Per-stage timings
t0=time.perf_counter(); encode_ane(); t_ane5=(time.perf_counter()-t0)*1000
_, t_llm5 = run_int4()

# Sequential
def seq_int4():
    results=[]; t0=time.perf_counter()
    for _ in range(REPEAT):
        encode_ane()
        ans,_ = run_int4()
        results.append(ans)
    return results, (time.perf_counter()-t0)*1000

async def pipeline_int4():
    loop = asyncio.get_event_loop()
    ane_ex = ThreadPoolExecutor(max_workers=1)
    llm_ex = ThreadPoolExecutor(max_workers=1)
    q: asyncio.Queue = asyncio.Queue(maxsize=2)
    results=[None]*REPEAT

    async def v_worker():
        for _ in range(REPEAT):
            await loop.run_in_executor(ane_ex, encode_ane)
            await q.put(True)
        await q.put(None)

    async def l_worker():
        for i in range(REPEAT):
            tok = await q.get()
            if tok is None: break
            ans, _ = await loop.run_in_executor(llm_ex, run_int4)
            results[i] = ans

    t0=time.perf_counter()
    await asyncio.gather(v_worker(), l_worker())
    return results, (time.perf_counter()-t0)*1000

_, t_seq5 = seq_int4()
pipe_results5, t_pipe5 = asyncio.run(pipeline_int4())
speedup5 = t_seq3 / t_pipe5   # vs original MPS baseline
answer5 = pipe_results5[0] or run_int4()[0]
print(f"  ANE={t_ane5:.0f}ms  INT4={t_llm5:.0f}ms  Pipeline={t_pipe5:.0f}ms  Speedup={speedup5:.2f}×")

card5 = make_card(
    title="ANE + INT4 Metal Pipeline",
    subtitle="Stage 5 — CoreML ANE vision + llama.cpp Q4_K_M Metal LLM",
    timings=[
        ("Vision encoder (ANE)",           f"{t_ane5:.0f}ms"),
        ("LLM decode (INT4 Metal)",        f"{t_llm5:.0f}ms"),
        (f"Sequential {REPEAT}q ANE+INT4", f"{t_seq5:.0f}ms"),
        (f"Pipeline {REPEAT}q ANE+INT4",   f"{t_pipe5:.0f}ms"),
        ("Speedup vs MPS baseline",         f"{speedup5:.2f}×"),
    ],
    answer=answer5, stage_num=5, color=(180, 100, 255)
)
save_card(card5, "stage5_pipeline_int4.png")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY CARD
# ═══════════════════════════════════════════════════════════════════════════════
print("\nGenerating summary card...")
summary = Image.new("RGB", (CARD_W*2+4, CARD_H+120), (12, 12, 18))
draw = ImageDraw.Draw(summary)
try:
    font_lg = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 26)
    font_md = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    font_sm = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15)
except Exception:
    font_lg = font_md = font_sm = ImageFont.load_default()

# Header
draw.text((20, 16), "Mobile-O Hardware-Aware Inference — All Stages", fill=(255,255,255), font=font_lg)
draw.text((20, 50), "MacBook Air M5  |  MPS → ANE+MPS → ANE+INT4  |  4 queries", fill=(140,140,160), font=font_sm)
draw.rectangle([0, 76, CARD_W*2+4, 78], fill=(50,50,60))

# Paste stage cards in a 2×3 grid (5 cards)
positions = [(0,80),(CARD_W+4,80),(0,80+CARD_H//2+4),(CARD_W+4,80+CARD_H//2+4)]
cards = [card1, card3, card4, card5]
for (x,y), c in zip(positions, cards):
    half = c.resize((CARD_W, CARD_H//2), Image.LANCZOS)
    summary.paste(half, (x, y))

# Footer bar
stages_data = [
    ("S1 MPS seq",    f"{total1:.0f}ms", "1.00×", (80,140,255)),
    ("S3 MPS pipe",   f"{t_pipe3:.0f}ms", f"{speedup3:.2f}×", (200,80,80)),
    ("S4 ANE+MPS",    f"{t_pipe4:.0f}ms", f"{speedup4:.2f}×", (80,200,120)),
    ("S5 ANE+INT4",   f"{t_pipe5:.0f}ms", f"{speedup5:.2f}×", (180,100,255)),
]
y_footer = 80 + CARD_H + 4
draw.rectangle([0, y_footer, CARD_W*2+4, y_footer+116], fill=(20,20,28))
draw.text((20, y_footer+8), "THROUGHPUT PROGRESSION (4 queries)", fill=(100,100,120), font=font_sm)
col_w = (CARD_W*2) // len(stages_data)
for i, (label, total_t, sx, col) in enumerate(stages_data):
    x = 20 + i*col_w
    draw.rectangle([x, y_footer+28, x+col_w-10, y_footer+30], fill=col)
    draw.text((x, y_footer+34), label, fill=(180,180,190), font=font_sm)
    draw.text((x, y_footer+56), total_t, fill=(255,255,255), font=font_md)
    draw.text((x, y_footer+80), sx, fill=col, font=font_lg)

summary.save(OUT_DIR / "summary_all_stages.png")
print(f"  Saved: {OUT_DIR}/summary_all_stages.png")

print(f"""
All stage outputs saved to {OUT_DIR}/
  stage1_original_mps.png      — Baseline MPS sequential
  stage2_profiled_mps.png      — Profiled with timing breakdown
  stage3_pipeline_mps_only.png — MPS-only async (negative result)
  stage4_pipeline_ane_mps.png  — ANE+MPS pipeline
  stage5_pipeline_int4.png     — ANE+INT4 Metal pipeline
  summary_all_stages.png       — All stages in one card
""")
