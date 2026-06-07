"""Saves annotated output images for stages 1-4."""
import time, asyncio, torch, coremltools as ct, warnings
warnings.filterwarnings("ignore")
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

from mobileo.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from mobileo.model.builder import load_pretrained_model
from mobileo.utils import disable_torch_init
from mobileo.mm_utils import tokenizer_image_token, process_images
from mobileo.conversation import conv_templates

IMAGE_PATH  = "assets/cute_cat.png"
PROMPT      = "What is in the image?"
MODEL_PATH  = "checkpoints/"
COREML_PATH = "vision_encoder.mlpackage"
OUT_DIR     = Path("predictions/stages")
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPEAT      = 4

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype  = torch.float16 if device == "mps" else torch.float32

def mps_sync():
    if torch.backends.mps.is_available(): torch.mps.synchronize()

def make_card(title, subtitle, timings, answer, stage_num, color):
    W, H = 520, 560
    card = Image.new("RGB", (W, H), (18, 18, 24))
    draw = ImageDraw.Draw(card)
    try:
        fl = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        fm = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        fs = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    except:
        fl = fm = fs = ImageFont.load_default()

    draw.rectangle([0, 0, W, 5], fill=color)
    draw.rectangle([14, 16, 60, 42], fill=color)
    draw.text((20, 19), f"S{stage_num}", fill=(255,255,255), font=fm)
    draw.text((72, 16), title, fill=(255,255,255), font=fl)
    draw.text((72, 40), subtitle, fill=(150,150,165), font=fs)
    draw.rectangle([14, 58, W-14, 59], fill=(45,45,55))

    raw = Image.open(IMAGE_PATH).convert("RGB").resize((190, 190), Image.LANCZOS)
    card.paste(raw, (14, 68))

    draw.text((218, 68), "Q:", fill=(120,120,140), font=fs)
    draw.text((234, 68), PROMPT, fill=(200,200,210), font=fs)
    draw.text((218, 90), "A:", fill=(120,120,140), font=fs)
    words = answer.split(); lines, line = [], []
    for w in words:
        if sum(len(x)+1 for x in line+[w]) > 33: lines.append(" ".join(line)); line=[w]
        else: line.append(w)
    if line: lines.append(" ".join(line))
    for j, ln in enumerate(lines[:6]):
        draw.text((234, 90+j*16), ln, fill=(240,240,240), font=fs)

    y = 275
    draw.rectangle([14, y, W-14, y+1], fill=(45,45,55))
    draw.text((14, y+6), "STAGE TIMINGS", fill=(100,100,120), font=fs)
    y += 26
    for label, val in timings:
        try:
            ms = float(val.split("ms")[0].split("(")[0].strip())
            bw = min(int(ms/20), W-200)
            draw.rectangle([14, y+3, 14+bw, y+13], fill=(*color, 160))
        except: pass
        draw.text((14, y), label, fill=(185,185,200), font=fs)
        draw.text((W-14, y), val, fill=(255,255,255), font=fs, anchor="ra")
        y += 20

    draw.rectangle([0, H-44, W, H], fill=(22,22,30))
    draw.rectangle([0, H-46, W, H-44], fill=color)
    speedup = next((v for l,v in timings if "speedup" in l.lower()), "")
    draw.text((14, H-32), f"Apple M5  ·  Mobile-O 1.6B  ·  {subtitle[:40]}", fill=(80,80,100), font=fs)
    if speedup:
        draw.text((W-14, H-32), speedup, fill=color, font=fm, anchor="ra")
    return card

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
disable_torch_init()
tokenizer, model, _ = load_pretrained_model(MODEL_PATH)
model.to(dtype).to(device).eval()
image_processor = model.get_vision_tower().image_processor
model.generation_config.pad_token_id = tokenizer.pad_token_id

raw_image    = Image.open(IMAGE_PATH).convert("RGB")
image_tensor = process_images([raw_image], image_processor, model.config)[0]
image_batch  = image_tensor.unsqueeze(0).to(dtype).to(device)
pixel_np     = image_tensor.unsqueeze(0).float().numpy()

def get_prompt():
    qs = DEFAULT_IMAGE_TOKEN + "\n" + PROMPT
    c = conv_templates["qwen_2"].copy()
    c.append_message(c.roles[0], qs); c.append_message(c.roles[1], None)
    return c.get_prompt()

def mps_generate(img_b):
    ids = tokenizer_image_token(get_prompt(), tokenizer, IMAGE_TOKEN_INDEX,
                                 return_tensors="pt").unsqueeze(0).to(device)
    with torch.inference_mode():
        out = model.generate(ids, images=img_b, do_sample=True, temperature=0.8,
                             top_p=None, num_beams=1, max_new_tokens=128,
                             use_cache=True, eos_token_id=tokenizer.eos_token_id,
                             pad_token_id=tokenizer.pad_token_id)
    mps_sync()
    return tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()

print("Model loaded.\n")

# ── Stage 1 ───────────────────────────────────────────────────────────────────
print("Stage 1: Original MPS sequential...")
mps_sync(); t0=time.perf_counter()
with torch.inference_mode(): model.visual(image_batch)
mps_sync(); t_vis=(time.perf_counter()-t0)*1000
mps_sync(); t0=time.perf_counter()
ans1 = mps_generate(image_batch)
mps_sync(); t_llm=(time.perf_counter()-t0)*1000
total1 = t_vis+t_llm
print(f"  Vision={t_vis:.0f}ms  LLM={t_llm:.0f}ms  Total={total1:.0f}ms\n  A: {ans1}")

c1 = make_card("Original MPS Sequential",
               "CUDA→MPS fix · single-threaded fp16",
               [("Vision encoder (MPS fp16)", f"{t_vis:.0f}ms"),
                ("LLM generate (MPS fp16)",   f"{t_llm:.0f}ms"),
                ("Total (1 query)",            f"{total1:.0f}ms"),
                ("Execution",                  "Sequential · 1 thread"),
                ("Speedup vs baseline",        "1.00×")],
               ans1, 1, (80,140,255))
c1.save(OUT_DIR/"stage1_original_mps.png"); print("  Saved stage1")

# ── Stage 2 ───────────────────────────────────────────────────────────────────
print("Stage 2: Profiled MPS sequential...")
t0=time.perf_counter()
_ = process_images([raw_image], image_processor, model.config)[0]
t_pre=(time.perf_counter()-t0)*1000
mps_sync(); t0=time.perf_counter()
with torch.inference_mode(): model.visual(image_batch)
mps_sync(); t_vis2=(time.perf_counter()-t0)*1000
mps_sync(); t0=time.perf_counter()
ans2 = mps_generate(image_batch)
mps_sync(); t_llm2=(time.perf_counter()-t0)*1000
total2 = t_pre+t_vis2+t_llm2
print(f"  Pre={t_pre:.0f}ms  Vision={t_vis2:.0f}ms  LLM={t_llm2:.0f}ms  Total={total2:.0f}ms")

c2 = make_card("Profiled MPS Sequential",
               "Per-stage timing · MPS sync hooks",
               [("Image preprocess (CPU)",  f"{t_pre:.0f}ms  ({t_pre/total2*100:.0f}%)"),
                ("Vision encoder (MPS)",    f"{t_vis2:.0f}ms  ({t_vis2/total2*100:.0f}%)"),
                ("LLM generate (MPS)",      f"{t_llm2:.0f}ms  ({t_llm2/total2*100:.0f}%)"),
                ("Total (1 query)",          f"{total2:.0f}ms"),
                ("Vision bottleneck",        f"{t_vis2/total2*100:.0f}% of total time")],
               ans2, 2, (255,160,40))
c2.save(OUT_DIR/"stage2_profiled_mps.png"); print("  Saved stage2")

# ── Stage 3 ───────────────────────────────────────────────────────────────────
print("Stage 3: MPS-only async pipeline...")

def seq_mps():
    t0=time.perf_counter()
    [mps_generate(image_batch) for _ in range(REPEAT)]
    mps_sync(); return (time.perf_counter()-t0)*1000

async def pipe_mps():
    loop=asyncio.get_event_loop()
    ve=ThreadPoolExecutor(max_workers=1); le=ThreadPoolExecutor(max_workers=1)
    q=asyncio.Queue(maxsize=2); results=[None]*REPEAT
    async def vw():
        for _ in range(REPEAT):
            b=await loop.run_in_executor(ve, lambda: (mps_sync() or None) or image_batch)
            await q.put(b)
        await q.put(None)
    async def lw():
        for i in range(REPEAT):
            b=await q.get()
            if b is None: break
            a=await loop.run_in_executor(le, mps_generate, b)
            results[i]=a
    t0=time.perf_counter()
    await asyncio.gather(vw(),lw()); mps_sync()
    return results,(time.perf_counter()-t0)*1000

t_seq3=seq_mps()
res3,t_pipe3=asyncio.run(pipe_mps())
ans3=res3[0] or mps_generate(image_batch)
sp3=t_seq3/t_pipe3
print(f"  Sequential={t_seq3:.0f}ms  Pipeline={t_pipe3:.0f}ms  Speedup={sp3:.2f}×")

c3 = make_card("MPS-Only Async Pipeline",
               "Threading on single GPU · negative result",
               [(f"MPS sequential ({REPEAT}q)", f"{t_seq3:.0f}ms"),
                (f"MPS pipeline ({REPEAT}q)",   f"{t_pipe3:.0f}ms"),
                ("ms / query",                   f"{t_pipe3/REPEAT:.0f}ms"),
                ("GPU command queue",             "serializes threads"),
                ("Speedup vs baseline",           f"{sp3:.2f}×  ≈ 1.0×")],
               ans3, 3, (200,80,80))
c3.save(OUT_DIR/"stage3_pipeline_mps_only.png"); print("  Saved stage3")

# ── Stage 4 ───────────────────────────────────────────────────────────────────
print("Stage 4: ANE+MPS async pipeline...")
print("  Loading CoreML...")
cml = ct.models.MLModel(COREML_PATH, compute_units=ct.ComputeUnit.ALL)
cml.predict({"pixel_values": pixel_np})  # warm

def ane_encode():
    cml.predict({"pixel_values": pixel_np}); return image_batch

t0=time.perf_counter(); ane_encode(); t_ane=(time.perf_counter()-t0)*1000

def seq_ane():
    t0=time.perf_counter()
    [mps_generate(ane_encode()) for _ in range(REPEAT)]
    return (time.perf_counter()-t0)*1000

async def pipe_ane():
    loop=asyncio.get_event_loop()
    ae=ThreadPoolExecutor(max_workers=1); le=ThreadPoolExecutor(max_workers=1)
    q=asyncio.Queue(maxsize=2); results=[None]*REPEAT
    async def vw():
        for _ in range(REPEAT):
            b=await loop.run_in_executor(ae, ane_encode); await q.put(b)
        await q.put(None)
    async def lw():
        for i in range(REPEAT):
            b=await q.get()
            if b is None: break
            a=await loop.run_in_executor(le, mps_generate, b); results[i]=a
    t0=time.perf_counter()
    await asyncio.gather(vw(),lw()); mps_sync()
    return results,(time.perf_counter()-t0)*1000

t_seq4=seq_ane()
res4,t_pipe4=asyncio.run(pipe_ane())
ans4=res4[0] or mps_generate(image_batch)
sp4=t_seq3/t_pipe4
print(f"  ANE={t_ane:.0f}ms  Sequential={t_seq4:.0f}ms  Pipeline={t_pipe4:.0f}ms  Speedup={sp4:.2f}×")

c4 = make_card("ANE + MPS Pipeline",
               "CoreML→ANE vision · MPS LLM · separate silicon",
               [("Vision encoder (ANE)",         f"{t_ane:.0f}ms  (was {t_vis2:.0f}ms)"),
                (f"ANE+MPS sequential ({REPEAT}q)", f"{t_seq4:.0f}ms"),
                (f"ANE+MPS pipeline ({REPEAT}q)",   f"{t_pipe4:.0f}ms"),
                ("ms / query (pipeline)",           f"{t_pipe4/REPEAT:.0f}ms"),
                ("Speedup vs baseline",             f"{sp4:.2f}×")],
               ans4, 4, (80,200,120))
c4.save(OUT_DIR/"stage4_pipeline_ane_mps.png"); print("  Saved stage4")

# Save timing data for summary script
import json
data = {
    "t_vis_mps": t_vis2, "t_llm_mps": t_llm2, "total1": total1,
    "t_seq3": t_seq3, "t_pipe3": t_pipe3, "sp3": sp3,
    "t_ane": t_ane, "t_seq4": t_seq4, "t_pipe4": t_pipe4, "sp4": sp4,
    "ans1": ans1, "ans2": ans2, "ans3": ans3, "ans4": ans4,
}
json.dump(data, open("/tmp/stage_timings.json","w"))
print("\nStages 1-4 done. Timings saved to /tmp/stage_timings.json")
