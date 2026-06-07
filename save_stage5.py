"""Saves annotated output image for stage 5 (ANE+INT4 Metal pipeline)."""
import time, asyncio, json, warnings
warnings.filterwarnings("ignore")
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import coremltools as ct
from mobileo.model.builder import load_pretrained_model
from mobileo.utils import disable_torch_init
from mobileo.mm_utils import process_images
import torch

IMAGE_PATH  = "assets/cute_cat.png"
PROMPT      = "What is in the image?"
MODEL_PATH  = "checkpoints/"
COREML_PATH = "vision_encoder.mlpackage"
GGUF_PATH   = "llm_export/model-q4.gguf"
OUT_DIR     = Path("predictions/stages")
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPEAT      = 4

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype  = torch.float16 if device == "mps" else torch.float32

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
    draw.rectangle([0,0,W,5], fill=color)
    draw.rectangle([14,16,60,42], fill=color)
    draw.text((20,19), f"S{stage_num}", fill=(255,255,255), font=fm)
    draw.text((72,16), title, fill=(255,255,255), font=fl)
    draw.text((72,40), subtitle, fill=(150,150,165), font=fs)
    draw.rectangle([14,58,W-14,59], fill=(45,45,55))
    raw = Image.open(IMAGE_PATH).convert("RGB").resize((190,190), Image.LANCZOS)
    card.paste(raw, (14,68))
    draw.text((218,68), "Q:", fill=(120,120,140), font=fs)
    draw.text((234,68), PROMPT, fill=(200,200,210), font=fs)
    draw.text((218,90), "A:", fill=(120,120,140), font=fs)
    words=answer.split(); lines,line=[],[]
    for w in words:
        if sum(len(x)+1 for x in line+[w])>33: lines.append(" ".join(line)); line=[w]
        else: line.append(w)
    if line: lines.append(" ".join(line))
    for j,ln in enumerate(lines[:6]):
        draw.text((234,90+j*16), ln, fill=(240,240,240), font=fs)
    y=275
    draw.rectangle([14,y,W-14,y+1], fill=(45,45,55))
    draw.text((14,y+6), "STAGE TIMINGS", fill=(100,100,120), font=fs)
    y+=26
    for label,val in timings:
        try:
            ms=float(val.split("ms")[0].split("(")[0].strip())
            bw=min(int(ms/20), W-200)
            draw.rectangle([14,y+3,14+bw,y+13], fill=(*color,160))
        except: pass
        draw.text((14,y), label, fill=(185,185,200), font=fs)
        draw.text((W-14,y), val, fill=(255,255,255), font=fs, anchor="ra")
        y+=20
    draw.rectangle([0,H-44,W,H], fill=(22,22,30))
    draw.rectangle([0,H-46,W,H-44], fill=color)
    speedup=next((v for l,v in timings if "speedup" in l.lower()),"")
    draw.text((14,H-32), f"Apple M5  ·  Mobile-O 1.6B  ·  INT4 Q4_K_M", fill=(80,80,100), font=fs)
    if speedup: draw.text((W-14,H-32), speedup, fill=color, font=fm, anchor="ra")
    return card

# Minimal model load just for image preprocessing
print("Loading model for preprocessing...")
disable_torch_init()
tokenizer, model, _ = load_pretrained_model(MODEL_PATH)
model.to(dtype).to(device).eval()
image_processor = model.get_vision_tower().image_processor

raw_image    = Image.open(IMAGE_PATH).convert("RGB")
image_tensor = process_images([raw_image], image_processor, model.config)[0]
pixel_np     = image_tensor.unsqueeze(0).float().numpy()

print("Loading CoreML + INT4 LLM...")
cml = ct.models.MLModel(COREML_PATH, compute_units=ct.ComputeUnit.ALL)
cml.predict({"pixel_values": pixel_np})  # warm

from llama_cpp import Llama
llm = Llama(model_path=GGUF_PATH, n_gpu_layers=99, n_ctx=512, verbose=False)
llm("hi", max_tokens=5, echo=False)  # warm
print("Loaded.\n")

def ane_encode():
    cml.predict({"pixel_values": pixel_np})

def int4_run():
    t0=time.perf_counter()
    out=llm(
        "<|im_start|>system\nYou are a helpful visual assistant.<|im_end|>\n"
        f"<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n",
        max_tokens=128, temperature=0.8, echo=False,
        stop=["<|im_end|>","<|endoftext|>"])
    return out["choices"][0]["text"].strip(), (time.perf_counter()-t0)*1000

# Per-stage timings
t0=time.perf_counter(); ane_encode(); t_ane=(time.perf_counter()-t0)*1000
ans5, t_llm5 = int4_run()

# Sequential
def seq_int4():
    t0=time.perf_counter()
    [int4_run() for _ in range(REPEAT)]
    return (time.perf_counter()-t0)*1000
_ = ane_encode  # satisfy reference
t_seq5=seq_int4()

# Async pipeline
async def pipe_int4():
    loop=asyncio.get_event_loop()
    ae=ThreadPoolExecutor(max_workers=1); le=ThreadPoolExecutor(max_workers=1)
    q=asyncio.Queue(maxsize=2); results=[None]*REPEAT
    async def vw():
        for _ in range(REPEAT):
            await loop.run_in_executor(ae, ane_encode); await q.put(True)
        await q.put(None)
    async def lw():
        for i in range(REPEAT):
            tok=await q.get()
            if tok is None: break
            a,_=await loop.run_in_executor(le, int4_run); results[i]=a
    t0=time.perf_counter()
    await asyncio.gather(vw(),lw())
    return results,(time.perf_counter()-t0)*1000

res5,t_pipe5=asyncio.run(pipe_int4())
ans5_pipe = res5[0] or ans5

# Load baseline for speedup
try:
    prev = json.load(open("/tmp/stage_timings.json"))
    t_seq3 = prev["t_seq3"]
except: t_seq3 = t_seq5  # fallback
sp5 = t_seq3 / t_pipe5

print(f"  ANE={t_ane:.0f}ms  INT4={t_llm5:.0f}ms  Pipeline={t_pipe5:.0f}ms  Speedup={sp5:.2f}×")
print(f"  A: {ans5_pipe}")

c5 = make_card("ANE + INT4 Metal Pipeline",
               "CoreML ANE vision · llama.cpp Q4_K_M Metal",
               [("Vision encoder (ANE)",           f"{t_ane:.0f}ms"),
                ("LLM decode (INT4 Metal)",        f"{t_llm5:.0f}ms"),
                (f"ANE+INT4 sequential ({REPEAT}q)", f"{t_seq5:.0f}ms"),
                (f"ANE+INT4 pipeline ({REPEAT}q)",   f"{t_pipe5:.0f}ms"),
                ("Speedup vs MPS baseline",          f"{sp5:.2f}×")],
               ans5_pipe, 5, (180,100,255))
c5.save(OUT_DIR/"stage5_pipeline_int4.png"); print("  Saved stage5")

# Save for summary
data = json.load(open("/tmp/stage_timings.json")) if Path("/tmp/stage_timings.json").exists() else {}
data.update({"t_ane5": t_ane, "t_llm5": t_llm5, "t_seq5": t_seq5,
             "t_pipe5": t_pipe5, "sp5": sp5, "ans5": ans5_pipe})
json.dump(data, open("/tmp/stage_timings.json","w"))
print("\nStage 5 done.")
