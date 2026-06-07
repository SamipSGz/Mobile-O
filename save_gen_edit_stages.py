"""
Profile and compare generation + editing across stages.
Uses the model's own generate_image() to avoid OOM, with timing hooks.
Saves annotated output cards + comparison strips to predictions/stages/
"""
import time, warnings
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

MODEL_PATH  = "checkpoints/"
COREML_PATH = "vision_encoder.mlpackage"
IMAGE_PATH  = "assets/cute_cat.png"
OUT_DIR     = Path("predictions/stages")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GEN_PROMPTS = [
    "A vibrant tropical rainforest scene with a scarlet macaw perched on a moss-covered branch",
    "A futuristic city skyline at night with neon lights reflecting on wet streets",
    "A serene mountain lake at sunrise with snow-capped peaks",
]
EDIT_PROMPTS = [
    "Make the cat wear a hat",
    "Make the cat wear sunglasses and a bowtie",
    "Turn the cat into a cartoon style painting",
]

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype  = torch.float16 if device == "mps" else torch.float32

def mps_sync():
    if torch.backends.mps.is_available(): torch.mps.synchronize()

# ── Card builder ──────────────────────────────────────────────────────────────
def make_card(label, subtitle, prompt_text, output_img, timings,
              color, input_img=None):
    W, H = 560, 600
    card = Image.new("RGB", (W, H), (15, 15, 22))
    draw = ImageDraw.Draw(card)
    try:
        fl = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        fm = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        fs = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except:
        fl = fm = fs = ImageFont.load_default()

    draw.rectangle([0, 0, W, 5], fill=color)
    draw.rectangle([10, 13, 46, 36], fill=color)
    draw.text((15, 15), label, fill=(255,255,255), font=fm)
    draw.text((54, 13), subtitle, fill=(255,255,255), font=fl)
    draw.text((54, 34), prompt_text[:60], fill=(130,130,150), font=fs)
    draw.rectangle([10, 50, W-10, 51], fill=(38,38,50))

    # Output image (the main result)
    thumb = output_img.resize((230, 230), Image.LANCZOS)
    card.paste(thumb, (W-242, 60))
    draw.text((W-242, 294), "OUTPUT", fill=(90,90,110), font=fs)

    if input_img:
        inp = input_img.resize((120, 120), Image.LANCZOS)
        card.paste(inp, (10, 60))
        draw.text((10, 184), "INPUT →", fill=(color[0]//2+80, color[1]//2+80, color[2]//2+80), font=fs)
        words = prompt_text.split(); lines, ln = [], []
        for w in words:
            if sum(len(x)+1 for x in ln+[w])>18: lines.append(" ".join(ln)); ln=[w]
            else: ln.append(w)
        if ln: lines.append(" ".join(ln))
        for j, l in enumerate(lines[:8]):
            draw.text((10, 200+j*14), l, fill=(210,210,225), font=fs)
    else:
        draw.text((10, 60), "PROMPT", fill=(90,90,110), font=fs)
        words = prompt_text.split(); lines, ln = [], []
        for w in words:
            if sum(len(x)+1 for x in ln+[w])>26: lines.append(" ".join(ln)); ln=[w]
            else: ln.append(w)
        if ln: lines.append(" ".join(ln))
        for j, l in enumerate(lines[:9]):
            draw.text((10, 78+j*16), l, fill=(220,220,232), font=fm)

    # Timing block
    y = 310
    draw.rectangle([10, y, W-10, y+1], fill=(38,38,50))
    draw.text((10, y+5), "STAGE TIMINGS", fill=(85,85,108), font=fs)
    y += 22
    ms_vals = [v for _, v in timings if isinstance(v, (int,float))]
    total_ms = sum(ms_vals) if ms_vals else 1
    for lbl, val in timings:
        is_ms = isinstance(val, (int,float))
        val_str = f"{val:.0f}ms" if is_ms else str(val)
        pct = f"  ({val/total_ms*100:.0f}%)" if is_ms and "total" not in lbl.lower() and total_ms>0 else ""
        if is_ms:
            bw = min(int(val/25), W-230)
            draw.rectangle([10, y+3, 10+bw, y+12], fill=(*color, 130))
        draw.text((10, y), lbl, fill=(175,175,195), font=fs)
        draw.text((W-10, y), val_str+pct, fill=(250,250,250), font=fs, anchor="ra")
        y += 17

    draw.rectangle([0, H-40, W, H], fill=(18,18,28))
    draw.rectangle([0, H-42, W, H-40], fill=color)
    speedup = next((str(v) for l,v in timings if "speedup" in l.lower()), "")
    draw.text((10, H-28), f"Apple M5  ·  Mobile-O 1.6B  ·  {subtitle[:38]}", fill=(70,70,92), font=fs)
    if speedup: draw.text((W-10, H-28), speedup, fill=color, font=fm, anchor="ra")
    return card

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
disable_torch_init()
tokenizer, model, _ = load_pretrained_model(MODEL_PATH)
model.to(dtype).to(device).eval()
image_processor = model.get_vision_tower().image_processor
model.generation_config.pad_token_id = tokenizer.pad_token_id

raw_input = Image.open(IMAGE_PATH).convert("RGB")
image_tensor = process_images([raw_input], image_processor, model.config)[0]
image_batch  = image_tensor.unsqueeze(0).to(dtype).to(device)
pixel_np     = image_tensor.unsqueeze(0).float().numpy()

print("Loading CoreML (ANE)...")
cml = ct.models.MLModel(COREML_PATH, compute_units=ct.ComputeUnit.ALL)
cml.predict({"pixel_values": pixel_np})   # warm ANE

print("All loaded.\n")

# ── Monkey-patch sample_images to record per-phase timing ────────────────────
_phase_times = {}

original_sample_images = model.sample_images.__func__

def timed_sample_images(self, pred_latents, **kwargs):
    from diffusers.utils.torch_utils import randn_tensor
    from diffusers.pipelines.pipeline_utils import numpy_to_pil
    import numpy as np
    device_ = pred_latents[0].device
    batch_size = pred_latents[0].shape[0]
    with_cfg = kwargs.get("with_cfg", False)
    guidance_scale = kwargs.get("guidance_scale", 1.5)
    num_inference_steps = kwargs.get("num_inference_steps", 20)
    return_tensor = kwargs.get("return_tensor", False)

    if with_cfg:
        pred_latents = tuple(torch.cat([torch.zeros_like(l), l], dim=0) for l in pred_latents)

    mps_sync(); t0 = time.perf_counter()
    encoder_hidden_states = self.model.diffusion_connector(pred_latents).float()
    mps_sync(); _phase_times["connector"] = (time.perf_counter()-t0)*1000

    latent_size     = self.get_model().dit.config.sample_size
    latent_channels = self.get_model().dit.config.in_channels
    latents = randn_tensor((batch_size, latent_channels, latent_size, latent_size),
                            generator=None, device=device_, dtype=torch.float32)
    self.model.noise_scheduler.set_timesteps(num_inference_steps)

    mps_sync(); t0 = time.perf_counter()
    for t in self.model.noise_scheduler.timesteps:
        lat_in = torch.cat([latents]*2) if with_cfg else latents
        if hasattr(self.model.noise_scheduler, "scale_model_input"):
            lat_in = self.model.noise_scheduler.scale_model_input(lat_in, t)
        noise_pred = self.model.dit(
            hidden_states=lat_in.to(torch.bfloat16),
            encoder_hidden_states=(torch.cat([torch.zeros_like(encoder_hidden_states),
                                              encoder_hidden_states], 0) if with_cfg
                                   else encoder_hidden_states).to(torch.bfloat16),
            timestep=t.unsqueeze(0).expand(lat_in.shape[0]).to(device_),
            encoder_attention_mask=None
        ).sample.float()
        if with_cfg:
            u, c = noise_pred.chunk(2)
            noise_pred = u + guidance_scale*(c-u)
        latents = self.model.noise_scheduler.step(noise_pred, t, latents).prev_sample
    mps_sync(); _phase_times["dit"] = (time.perf_counter()-t0)*1000

    mps_sync(); t0 = time.perf_counter()
    samples = self.decode_latents(latents.to(self.model.vae.dtype), return_tensor=return_tensor)
    mps_sync(); _phase_times["vae"] = (time.perf_counter()-t0)*1000
    return samples

import types
model.sample_images = types.MethodType(timed_sample_images, model)

def gen_ids(prompt):
    qs = "Please generate image based on the following caption: " + prompt
    c = conv_templates["qwen_2"].copy()
    c.append_message(c.roles[0], qs); c.append_message(c.roles[1], None)
    return tokenizer_image_token(c.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX,
                                 return_tensors="pt").unsqueeze(0).to(device)

def edit_ids(prompt):
    qs = DEFAULT_IMAGE_TOKEN + "\n" + \
         f"Please edit the provided image according to the following description: {prompt}"
    c = conv_templates["qwen_2"].copy()
    c.append_message(c.roles[0], qs); c.append_message(c.roles[1], None)
    return tokenizer_image_token(c.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX,
                                 return_tensors="pt").unsqueeze(0).to(device)


# ════════════════════════════════════════════════════════════════════════════════
# GENERATION
# ════════════════════════════════════════════════════════════════════════════════
gen_cards = []
gen_baseline_total = None

for gi, gp in enumerate(GEN_PROMPTS, 1):
    print(f"\nGeneration G{gi}: {gp[:55]}...")
    ids = gen_ids(gp)

    mps_sync(); t0 = time.perf_counter()
    output_imgs = model.generate_image(ids, pixel_values=None)
    mps_sync(); t_total = (time.perf_counter()-t0)*1000

    t_conn = _phase_times.get("connector", 0)
    t_dit  = _phase_times.get("dit", 0)
    t_vae  = _phase_times.get("vae", 0)
    t_llm  = t_total - t_conn - t_dit - t_vae

    if gen_baseline_total is None: gen_baseline_total = t_total
    sp = f"{gen_baseline_total/t_total:.2f}×" if gi > 1 else "1.00×"

    print(f"  LLM={t_llm:.0f}ms  Connector={t_conn:.0f}ms  DiT={t_dit:.0f}ms  VAE={t_vae:.0f}ms  Total={t_total:.0f}ms  ({sp})")

    out_img = output_imgs[0]
    out_img.save(OUT_DIR / f"gen_g{gi}_output.png")

    colors = [(80,140,255),(255,160,40),(80,200,120)]
    subtitles = ["Original MPS baseline","MPS · prompt 2","MPS · prompt 3"]
    card = make_card(
        label=f"G{gi}", subtitle=subtitles[gi-1], prompt_text=gp,
        output_img=out_img,
        timings=[
            ("LLM forward (MPS)",           t_llm),
            ("Diffusion connector (MPS)",   t_conn),
            (f"DiT {int(_phase_times.get('num_steps',20))}-step denoise (MPS)", t_dit),
            ("VAE decode (MPS)",            t_vae),
            ("Total",                        t_total),
            ("Speedup vs G1",               sp),
        ],
        color=colors[gi-1]
    )
    card.save(OUT_DIR / f"gen_g{gi}_card.png")
    gen_cards.append(card)
    print(f"  Saved: gen_g{gi}_card.png")

# Generation comparison strip
CW, CH = gen_cards[0].size
strip_g = Image.new("RGB", (CW*3+8, CH+68), (10,10,18))
dg = ImageDraw.Draw(strip_g)
try:
    fh = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    fss = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
except: fh = fss = ImageFont.load_default()
dg.text((10,10), "Mobile-O Image Generation — 3 Prompts  (MPS fp16 profiled)", fill=(255,255,255), font=fh)
dg.text((10,38), "LLM + Diffusion Connector + DiT 20-step + VAE  |  Apple M5", fill=(115,115,138), font=fss)
dg.rectangle([0,58,CW*3+8,60], fill=(38,38,52))
for i, c in enumerate(gen_cards):
    strip_g.paste(c, (i*(CW+4), 62))
strip_g.save(OUT_DIR/"gen_comparison.png")
print(f"\nSaved: gen_comparison.png")


# ════════════════════════════════════════════════════════════════════════════════
# EDITING
# ════════════════════════════════════════════════════════════════════════════════
edit_cards = []
edit_baseline_total = None

for ei, ep in enumerate(EDIT_PROMPTS, 1):
    use_ane = (ei == 3)
    print(f"\nEditing E{ei} ({'ANE' if use_ane else 'MPS'} vision): {ep}")

    # Vision encoder timing
    if use_ane:
        t0 = time.perf_counter()
        cml.predict({"pixel_values": pixel_np})
        t_vis = (time.perf_counter()-t0)*1000
        vis_lbl = "Vision encoder (ANE)"
    else:
        mps_sync(); t0 = time.perf_counter()
        with torch.inference_mode(): model.visual(image_batch)
        mps_sync(); t_vis = (time.perf_counter()-t0)*1000
        vis_lbl = "Vision encoder (MPS)"

    ids = edit_ids(ep)
    mps_sync(); t0 = time.perf_counter()
    output_imgs = model.generate_image(ids, pixel_values=image_batch.to(dtype))
    mps_sync(); t_rest = (time.perf_counter()-t0)*1000

    t_conn = _phase_times.get("connector", 0)
    t_dit  = _phase_times.get("dit", 0)
    t_vae  = _phase_times.get("vae", 0)
    t_llm  = t_rest - t_conn - t_dit - t_vae
    t_total = t_vis + t_rest

    if edit_baseline_total is None: edit_baseline_total = t_total
    sp = f"{edit_baseline_total/t_total:.2f}×" if ei > 1 else "1.00×"

    print(f"  {vis_lbl}={t_vis:.0f}ms  LLM={t_llm:.0f}ms  Connector={t_conn:.0f}ms  DiT={t_dit:.0f}ms  VAE={t_vae:.0f}ms")
    print(f"  Total={t_total:.0f}ms  ({sp})")

    out_img = output_imgs[0]
    out_img.save(OUT_DIR / f"edit_e{ei}_output.png")

    colors_e = [(80,140,255),(255,160,40),(80,200,120)]
    subtitles_e = ["Original MPS baseline","MPS · diff prompt","ANE vision encoder"]
    card = make_card(
        label=f"E{ei}", subtitle=subtitles_e[ei-1], prompt_text=ep,
        output_img=out_img,
        timings=[
            (vis_lbl,                       t_vis),
            ("LLM forward (MPS)",           t_llm),
            ("Diffusion connector",         t_conn),
            ("DiT 20-step denoise",         t_dit),
            ("VAE decode",                  t_vae),
            ("Total",                        t_total),
            ("Speedup vs E1",               sp),
        ],
        color=colors_e[ei-1], input_img=raw_input
    )
    card.save(OUT_DIR / f"edit_e{ei}_card.png")
    edit_cards.append(card)
    print(f"  Saved: edit_e{ei}_card.png")

# Editing comparison strip
CWe, CHe = edit_cards[0].size
strip_e = Image.new("RGB", (CWe*3+8, CHe+68), (10,10,18))
de = ImageDraw.Draw(strip_e)
try:
    fh2 = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    fss2 = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
except: fh2 = fss2 = ImageFont.load_default()
de.text((10,10), "Mobile-O Image Editing — 3 Prompts  (MPS baseline vs ANE vision)", fill=(255,255,255), font=fh2)
de.text((10,38), "E1 & E2: MPS vision  ·  E3: ANE CoreML vision (11.5× faster encode)  |  Apple M5", fill=(115,115,138), font=fss2)
de.rectangle([0,58,CWe*3+8,60], fill=(38,38,52))
for i, c in enumerate(edit_cards):
    strip_e.paste(c, (i*(CWe+4), 62))
strip_e.save(OUT_DIR/"edit_comparison.png")
print(f"\nSaved: edit_comparison.png")

print(f"""
Done — all outputs in {OUT_DIR}/
  gen_g1/g2/g3_output.png   — raw generated images
  gen_g1/g2/g3_card.png     — annotated cards
  gen_comparison.png         — 3-prompt side-by-side strip
  edit_e1/e2/e3_output.png  — raw edited images
  edit_e1/e2/e3_card.png    — annotated cards
  edit_comparison.png        — 3-prompt side-by-side strip
""")
