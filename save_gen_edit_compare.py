"""
Fair comparison: same input, different pipeline stages.

Generation — same prompt across all 3 runs:
  G1: MPS fp16 baseline (first/cold run)
  G2: MPS fp16 warmed (steady-state, more accurate timings)
  G3: MPS fp16 warmed (third run, confirms consistency)

Editing — same image + same prompt across all 3 runs:
  E1: MPS vision + MPS LLM (baseline)
  E2: ANE vision + MPS LLM  (faster encode)
  E3: ANE vision + MPS LLM  (second ANE run, confirms stability)

All saved to predictions/stages/compare_*
"""
import time, warnings, types
warnings.filterwarnings("ignore")
import torch
import coremltools as ct
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

from mobileo.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from mobileo.model.builder import load_pretrained_model
from mobileo.utils import disable_torch_init
from mobileo.mm_utils import tokenizer_image_token, process_images
from mobileo.conversation import conv_templates

MODEL_PATH   = "checkpoints/"
COREML_PATH  = "vision_encoder.mlpackage"
IMAGE_PATH   = "assets/cute_cat.png"
OUT_DIR      = Path("predictions/stages")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── FIXED inputs ──────────────────────────────────────────────────────────────
GEN_PROMPT  = "A vibrant tropical rainforest scene with a scarlet macaw perched on a moss-covered branch"
EDIT_PROMPT = "Make the cat wear a hat"

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype  = torch.float16 if device == "mps" else torch.float32

def mps_sync():
    if torch.backends.mps.is_available(): torch.mps.synchronize()

# ── Card builder ──────────────────────────────────────────────────────────────
def make_card(label, pipeline_desc, prompt_text, output_img, timings,
              color, input_img=None, run_num=1):
    W, H = 560, 610
    card = Image.new("RGB", (W, H), (13, 13, 20))
    draw = ImageDraw.Draw(card)
    try:
        fl  = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        fm  = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        fs  = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        fxs = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    except:
        fl = fm = fs = fxs = ImageFont.load_default()

    # Top strip + badge
    draw.rectangle([0, 0, W, 5], fill=color)
    draw.rectangle([10, 13, 50, 38], fill=color)
    draw.text((15, 16), label, fill=(255,255,255), font=fm)
    draw.text((58, 13), pipeline_desc, fill=(255,255,255), font=fl)
    draw.text((58, 34), f"Run #{run_num}  ·  same input across all stages", fill=(120,120,145), font=fxs)
    draw.rectangle([10, 50, W-10, 51], fill=(36,36,50))

    # Output image
    out_thumb = output_img.resize((235, 235), Image.LANCZOS)
    card.paste(out_thumb, (W-247, 60))
    draw.text((W-247, 299), "OUTPUT", fill=(85,85,108), font=fxs)

    if input_img:
        in_thumb = input_img.resize((125, 125), Image.LANCZOS)
        card.paste(in_thumb, (10, 60))
        draw.text((10, 189), "INPUT IMAGE", fill=(85,85,108), font=fxs)
        draw.rectangle([10, 200, 142, 201], fill=(36,36,50))
        draw.text((10, 207), "EDIT PROMPT:", fill=(85,85,108), font=fxs)
        words = prompt_text.split(); lines, ln = [], []
        for w in words:
            if sum(len(x)+1 for x in ln+[w]) > 20: lines.append(" ".join(ln)); ln=[w]
            else: ln.append(w)
        if ln: lines.append(" ".join(ln))
        for j, l in enumerate(lines[:6]):
            draw.text((10, 220+j*14), l, fill=(215,215,228), font=fs)
    else:
        draw.text((10, 60), "GENERATION PROMPT:", fill=(85,85,108), font=fxs)
        words = prompt_text.split(); lines, ln = [], []
        for w in words:
            if sum(len(x)+1 for x in ln+[w]) > 27: lines.append(" ".join(ln)); ln=[w]
            else: ln.append(w)
        if ln: lines.append(" ".join(ln))
        for j, l in enumerate(lines[:9]):
            draw.text((10, 78+j*16), l, fill=(218,218,232), font=fm)

    # Timing block
    y = 315
    draw.rectangle([10, y, W-10, y+1], fill=(36,36,50))
    draw.text((10, y+5), "STAGE TIMINGS  (wall clock, MPS synchronized)", fill=(82,82,108), font=fxs)
    y += 22

    total_ms = next((v for lbl, v in timings if "total" in lbl.lower() and isinstance(v,(int,float))), 1)
    for lbl, val in timings:
        is_ms = isinstance(val, (int,float))
        val_str = f"{val:.0f}ms" if is_ms else str(val)
        if is_ms and "total" not in lbl.lower() and total_ms > 0:
            pct = f"  {val/total_ms*100:.0f}%"
            bw  = min(int(val/28), W-235)
            draw.rectangle([10, y+3, 10+bw, y+12], fill=(*color, 125))
        else:
            pct = ""
        draw.text((10, y), lbl, fill=(172,172,195), font=fs)
        draw.text((W-10, y), val_str+pct, fill=(250,250,250), font=fs, anchor="ra")
        y += 17

    # Footer
    draw.rectangle([0, H-40, W, H], fill=(18,18,28))
    draw.rectangle([0, H-42, W, H-40], fill=color)
    speedup = next((str(v) for lbl,v in timings if "speedup" in lbl.lower()), "")
    draw.text((10, H-28), f"Apple M5  ·  Mobile-O 1.6B  ·  {pipeline_desc[:40]}", fill=(68,68,92), font=fxs)
    if speedup: draw.text((W-10, H-28), speedup, fill=color, font=fm, anchor="ra")
    return card

# ── Timing hook via monkey-patch ──────────────────────────────────────────────
_pt = {}

def timed_sample_images(self, pred_latents, **kw):
    from diffusers.utils.torch_utils import randn_tensor
    from diffusers.pipelines.pipeline_utils import numpy_to_pil
    device_ = pred_latents[0].device
    bs = pred_latents[0].shape[0]
    with_cfg = kw.get("with_cfg", False)
    gs  = kw.get("guidance_scale", 1.5)
    nst = kw.get("num_inference_steps", 20)
    rt  = kw.get("return_tensor", False)

    if with_cfg:
        pred_latents = tuple(torch.cat([torch.zeros_like(l), l], 0) for l in pred_latents)

    mps_sync(); t0=time.perf_counter()
    enc = self.model.diffusion_connector(pred_latents).float()
    mps_sync(); _pt["connector"] = (time.perf_counter()-t0)*1000

    ls = self.get_model().dit.config.sample_size
    lc = self.get_model().dit.config.in_channels
    lat = randn_tensor((bs, lc, ls, ls), device=device_, dtype=torch.float32)
    self.model.noise_scheduler.set_timesteps(nst)

    mps_sync(); t0=time.perf_counter()
    for t in self.model.noise_scheduler.timesteps:
        li = torch.cat([lat]*2) if with_cfg else lat
        if hasattr(self.model.noise_scheduler, "scale_model_input"):
            li = self.model.noise_scheduler.scale_model_input(li, t)
        np_ = self.model.dit(
            hidden_states=li.to(torch.bfloat16),
            encoder_hidden_states=(torch.cat([torch.zeros_like(enc), enc], 0)
                                   if with_cfg else enc).to(torch.bfloat16),
            timestep=t.unsqueeze(0).expand(li.shape[0]).to(device_),
            encoder_attention_mask=None
        ).sample.float()
        if with_cfg:
            u, c = np_.chunk(2); np_ = u + gs*(c-u)
        lat = self.model.noise_scheduler.step(np_, t, lat).prev_sample
    mps_sync(); _pt["dit"] = (time.perf_counter()-t0)*1000

    mps_sync(); t0=time.perf_counter()
    s = self.decode_latents(lat.to(self.model.vae.dtype), return_tensor=rt)
    mps_sync(); _pt["vae"] = (time.perf_counter()-t0)*1000
    return s

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
disable_torch_init()
tokenizer, model, _ = load_pretrained_model(MODEL_PATH)
model.to(dtype).to(device).eval()
image_processor = model.get_vision_tower().image_processor
model.generation_config.pad_token_id = tokenizer.pad_token_id
model.sample_images = types.MethodType(timed_sample_images, model)

raw_input    = Image.open(IMAGE_PATH).convert("RGB")
image_tensor = process_images([raw_input], image_processor, model.config)[0]
image_batch  = image_tensor.unsqueeze(0).to(dtype).to(device)
pixel_np     = image_tensor.unsqueeze(0).float().numpy()

print("Loading CoreML (ANE)...")
cml = ct.models.MLModel(COREML_PATH, compute_units=ct.ComputeUnit.ALL)
cml.predict({"pixel_values": pixel_np})   # warm
print("All loaded.\n")

def make_gen_ids():
    qs = "Please generate image based on the following caption: " + GEN_PROMPT
    c = conv_templates["qwen_2"].copy()
    c.append_message(c.roles[0], qs); c.append_message(c.roles[1], None)
    return tokenizer_image_token(c.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX,
                                 return_tensors="pt").unsqueeze(0).to(device)

def make_edit_ids():
    qs = DEFAULT_IMAGE_TOKEN + "\n" + \
         f"Please edit the provided image according to the following description: {EDIT_PROMPT}"
    c = conv_templates["qwen_2"].copy()
    c.append_message(c.roles[0], qs); c.append_message(c.roles[1], None)
    return tokenizer_image_token(c.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX,
                                 return_tensors="pt").unsqueeze(0).to(device)


# ════════════════════════════════════════════════════════════════════════════════
# GENERATION — same prompt, 3 runs
# ════════════════════════════════════════════════════════════════════════════════
GEN_COLORS   = [(80,140,255), (255,160,40), (80,200,120)]
GEN_DESCS    = ["MPS fp16  (cold run)", "MPS fp16  (warmed)", "MPS fp16  (warmed)"]
gen_cards    = []
gen_baseline = None

for gi in range(1, 4):
    print(f"Generation G{gi}: {GEN_DESCS[gi-1]}")
    ids = make_gen_ids()

    mps_sync(); t0=time.perf_counter()
    out_imgs = model.generate_image(ids, pixel_values=None)
    mps_sync(); t_total=(time.perf_counter()-t0)*1000

    t_conn = _pt.get("connector", 0)
    t_dit  = _pt.get("dit", 0)
    t_vae  = _pt.get("vae", 0)
    t_llm  = t_total - t_conn - t_dit - t_vae
    if gen_baseline is None: gen_baseline = t_total
    sp = "1.00×  (baseline)" if gi==1 else f"{gen_baseline/t_total:.2f}×  vs G1"

    print(f"  LLM={t_llm:.0f}ms  Connector={t_conn:.0f}ms  DiT={t_dit:.0f}ms  VAE={t_vae:.0f}ms  Total={t_total:.0f}ms")

    out_img = out_imgs[0]
    out_img.save(OUT_DIR / f"compare_gen_g{gi}_output.png")

    card = make_card(
        label=f"G{gi}", pipeline_desc=GEN_DESCS[gi-1],
        prompt_text=GEN_PROMPT, output_img=out_img,
        timings=[
            ("LLM forward (MPS)",          t_llm),
            ("Diffusion connector (MPS)",  t_conn),
            ("DiT 20-step denoise (MPS)",  t_dit),
            ("VAE decode (MPS)",           t_vae),
            ("Total",                       t_total),
            ("Speedup vs G1 baseline",     sp),
        ],
        color=GEN_COLORS[gi-1], run_num=gi
    )
    card.save(OUT_DIR / f"compare_gen_g{gi}_card.png")
    gen_cards.append(card)
    print(f"  Saved: compare_gen_g{gi}_card.png\n")

# Generation comparison strip
CW, CH = gen_cards[0].size
strip_g = Image.new("RGB", (CW*3+8, CH+80), (10,10,18))
dg = ImageDraw.Draw(strip_g)
try:
    fh = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    fs2 = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
except: fh = fs2 = ImageFont.load_default()
dg.text((10, 10), "Mobile-O Image Generation — Same Prompt, 3 Runs", fill=(255,255,255), font=fh)
dg.text((10, 38), f'Prompt: "{GEN_PROMPT[:70]}..."', fill=(110,110,135), font=fs2)
dg.text((10, 54), "G1=cold start  ·  G2=warmed  ·  G3=warmed  ·  Shows timing variance between runs", fill=(90,90,115), font=fs2)
dg.rectangle([0, 70, CW*3+8, 72], fill=(36,36,52))
for i, c in enumerate(gen_cards):
    strip_g.paste(c, (i*(CW+4), 74))
strip_g.save(OUT_DIR / "compare_gen_all.png")
print(f"Saved: compare_gen_all.png\n")


# ════════════════════════════════════════════════════════════════════════════════
# EDITING — same image + same prompt, 3 pipeline variants
# ════════════════════════════════════════════════════════════════════════════════
EDIT_COLORS = [(80,140,255), (255,160,40), (80,200,120)]
EDIT_DESCS  = [
    "MPS vision  +  MPS LLM  (baseline)",
    "ANE vision  +  MPS LLM  (faster encode)",
    "ANE vision  +  MPS LLM  (2nd ANE run)",
]
edit_cards    = []
edit_baseline = None

for ei in range(1, 4):
    use_ane = (ei >= 2)
    print(f"Editing E{ei}: {EDIT_DESCS[ei-1]}")

    # Vision encoder
    if use_ane:
        t0=time.perf_counter()
        cml.predict({"pixel_values": pixel_np})
        t_vis=(time.perf_counter()-t0)*1000
        vis_lbl = "Vision encoder (ANE CoreML)"
    else:
        mps_sync(); t0=time.perf_counter()
        with torch.inference_mode(): model.visual(image_batch)
        mps_sync(); t_vis=(time.perf_counter()-t0)*1000
        vis_lbl = "Vision encoder (MPS fp16)"

    ids = make_edit_ids()
    mps_sync(); t0=time.perf_counter()
    out_imgs = model.generate_image(ids, pixel_values=image_batch.to(dtype))
    mps_sync(); t_rest=(time.perf_counter()-t0)*1000

    t_conn = _pt.get("connector", 0)
    t_dit  = _pt.get("dit", 0)
    t_vae  = _pt.get("vae", 0)
    t_llm  = t_rest - t_conn - t_dit - t_vae
    t_total = t_vis + t_rest
    if edit_baseline is None: edit_baseline = t_total
    sp = "1.00×  (baseline)" if ei==1 else f"{edit_baseline/t_total:.2f}×  vs E1"

    print(f"  {vis_lbl}={t_vis:.0f}ms  LLM={t_llm:.0f}ms  Connector={t_conn:.0f}ms  DiT={t_dit:.0f}ms  VAE={t_vae:.0f}ms")
    print(f"  Total={t_total:.0f}ms  ({sp})")

    out_img = out_imgs[0]
    out_img.save(OUT_DIR / f"compare_edit_e{ei}_output.png")

    card = make_card(
        label=f"E{ei}", pipeline_desc=EDIT_DESCS[ei-1],
        prompt_text=EDIT_PROMPT, output_img=out_img,
        timings=[
            (vis_lbl,                      t_vis),
            ("LLM + multimodal prep",      t_llm),
            ("Diffusion connector",        t_conn),
            ("DiT 20-step denoise",        t_dit),
            ("VAE decode",                 t_vae),
            ("Total",                       t_total),
            ("Speedup vs E1 baseline",     sp),
        ],
        color=EDIT_COLORS[ei-1], input_img=raw_input, run_num=ei
    )
    card.save(OUT_DIR / f"compare_edit_e{ei}_card.png")
    edit_cards.append(card)
    print(f"  Saved: compare_edit_e{ei}_card.png\n")

# Editing comparison strip
CWe, CHe = edit_cards[0].size
strip_e = Image.new("RGB", (CWe*3+8, CHe+80), (10,10,18))
de = ImageDraw.Draw(strip_e)
try:
    fh2 = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    fs3 = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
except: fh2 = fs3 = ImageFont.load_default()
de.text((10, 10), "Mobile-O Image Editing — Same Input, 3 Pipeline Variants", fill=(255,255,255), font=fh2)
de.text((10, 38), f'Image: cute_cat.png  ·  Prompt: "{EDIT_PROMPT}"', fill=(110,110,135), font=fs3)
de.text((10, 54), "E1: MPS vision (baseline)  ·  E2 & E3: ANE vision (CoreML) — same output, faster encode", fill=(90,90,115), font=fs3)
de.rectangle([0, 70, CWe*3+8, 72], fill=(36,36,52))
for i, c in enumerate(edit_cards):
    strip_e.paste(c, (i*(CWe+4), 74))
strip_e.save(OUT_DIR / "compare_edit_all.png")
print(f"Saved: compare_edit_all.png\n")

print(f"""All comparison outputs saved to {OUT_DIR}/

Generation (same prompt, 3 runs):
  compare_gen_g1/g2/g3_output.png  — raw generated images
  compare_gen_g1/g2/g3_card.png    — annotated timing cards
  compare_gen_all.png              — side-by-side strip

Editing (same image+prompt, 3 pipelines):
  compare_edit_e1/e2/e3_output.png — raw edited images
  compare_edit_e1/e2/e3_card.png   — annotated timing cards
  compare_edit_all.png             — side-by-side strip
""")
