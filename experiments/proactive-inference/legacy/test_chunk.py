#!/usr/bin/env python3
"""Smoke test: does Qwen2-VL's VIDEO path (consecutive frames -> motion) work?"""
import os, sys, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as R
import torch
from PIL import Image

FFMPEG = str(Path.home() / "miniforge3" / "bin" / "ffmpeg")
TMP = "/tmp/chunk_test"; os.makedirs(TMP, exist_ok=True)

def get_frame(v, t):
    out = os.path.join(TMP, f"f_{t:07.2f}.jpg")
    subprocess.run([FFMPEG,"-y","-ss",f"{t:.2f}","-i",v,"-frames:v","1","-q:v","2",out],
                   capture_output=True, timeout=60)
    return Image.open(out).convert("RGB")

model, processor, tokenizer = R.load_model()
video = R.DEFAULT_VIDEO
VIS = getattr(model, "visual", None) or getattr(getattr(model, "model", None), "visual", None)
print("visual tower at:", "model.visual" if hasattr(model,"visual") else ("model.model.visual" if VIS is not None else "NOT FOUND"))

# single image (for comparison)
f0 = get_frame(video, 20.0)
di = processor.image_processor(images=[f0], return_tensors="pt")
print("IMAGE keys:", list(di.keys()))
print("  image_grid_thw:", di["image_grid_thw"].tolist(), "pixel_values:", tuple(di["pixel_values"].shape))
ve_i = VIS(di["pixel_values"].to("cuda", dtype=model.dtype), grid_thw=di["image_grid_thw"].to("cuda"))
print("  -> visual tokens:", ve_i.shape[0])

# video chunk: 4 consecutive frames ~ 1.6s
chunk = [get_frame(video, 20.0 + j*0.4) for j in range(4)]
try:
    dv = None
    forms = [
        ("image_processor(images=None,videos=)", lambda: processor.image_processor(images=None, videos=[chunk], return_tensors="pt")),
        ("processor(text,videos=)",              lambda: processor(text=["Describe."], videos=[chunk], return_tensors="pt")),
        ("image_processor(chunk-as-images)",     lambda: processor.image_processor(images=chunk, return_tensors="pt")),
    ]
    for name, fn in forms:
        try:
            dv = fn(); print("  video API works via:", name, "| keys:", list(dv.keys())); break
        except Exception as e:
            print("  form failed [" + name + "]:", str(e)[:90])
    assert dv is not None, "no video API form worked"
    print("\nVIDEO keys:", list(dv.keys()))
    gkey = "video_grid_thw" if "video_grid_thw" in dv else "image_grid_thw"
    pkey = "pixel_values_videos" if "pixel_values_videos" in dv else "pixel_values"
    print("  grid:", dv[gkey].tolist(), "pixels:", tuple(dv[pkey].shape))
    ve_v = VIS(dv[pkey].to("cuda", dtype=model.dtype), grid_thw=dv[gkey].to("cuda"))
    print("  -> visual tokens for 4-frame chunk:", ve_v.shape[0])
    # rope index for video
    ntok = ve_v.shape[0]
    vid_tok = getattr(model.config, "video_token_id", model.config.image_token_id)
    dummy = torch.full((1, ntok), vid_tok, dtype=torch.long, device="cuda")
    pos, _ = model.get_rope_index(input_ids=dummy, image_grid_thw=None, video_grid_thw=dv[gkey].to("cuda"),
                                  attention_mask=torch.ones(1, ntok, dtype=torch.long, device="cuda"))
    print("  rope pos shape:", tuple(pos.shape), "temporal max:", int(pos[0].max().item()))
    print("\nVIDEO PATH WORKS ✓")
except Exception as e:
    import traceback; traceback.print_exc()
    print("\nVIDEO PATH FAILED:", e)
