"""
Export the MobileCLIP vision encoder + mm_projector to CoreML (.mlpackage).
CoreML will route conv-heavy layers to ANE automatically.

Usage:
  python export_vision_coreml.py --model_path checkpoints/
"""
import argparse
import torch
import torch.nn as nn
import coremltools as ct
from mobileo.utils import disable_torch_init
from mobileo.model.builder import load_pretrained_model

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="checkpoints/")
parser.add_argument("--out", type=str, default="vision_encoder.mlpackage")
args = parser.parse_args()

# ── Wrapper: makes vision_tower + projector traceable (tensor in, tensor out) ──
class VisionEncoderWrapper(nn.Module):
    def __init__(self, vision_tower, mm_projector, proj_dtype):
        super().__init__()
        self.vision_tower = vision_tower
        self.mm_projector = mm_projector
        self.proj_dtype = proj_dtype

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [B, 3, H, W]  float32 for tracing
        outs = self.vision_tower(pixel_values, return_image_embeddings=True)
        feats = outs["image_embeddings"]          # [B, C, h, w]
        # Use flatten + permute instead of reshape to avoid scalar-conversion issues
        feats = feats.flatten(2).permute(0, 2, 1)            # [B, N, C]
        feats = self.mm_projector(feats.to(self.proj_dtype))  # [B, N, D]
        return feats


# ── Load model on CPU (no MPS needed for export) ─────────────────────────────
print("Loading model...")
disable_torch_init()
tokenizer, model, _ = load_pretrained_model(args.model_path)
model.eval().cpu().float()   # float32 for tracing stability

vision_tower = model.get_model().vision_tower.vision_tower
mm_projector  = model.get_model().mm_projector
proj_dtype    = mm_projector[0].weight.dtype

wrapper = VisionEncoderWrapper(vision_tower, mm_projector, proj_dtype).eval().float()

# ── Dummy input: 1024×1024 matches mobileclip_l_1024 config ─────────────────
image_size = 1024
dummy = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)

print(f"Tracing vision encoder (input {dummy.shape})...")
with torch.no_grad():
    traced = torch.jit.trace(wrapper, dummy)
    # Verify trace output matches eager
    out_eager  = wrapper(dummy)
    out_traced = traced(dummy)
    max_diff = (out_eager - out_traced).abs().max().item()
    print(f"  Trace verification max diff: {max_diff:.6f}  {'OK' if max_diff < 1e-3 else 'WARNING: large diff'}")

# ── Convert to CoreML ─────────────────────────────────────────────────────────
print("Converting to CoreML (compute_units=ALL → ANE preferred)...")
mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(name="pixel_values", shape=dummy.shape, dtype=float)],
    outputs=[ct.TensorType(name="image_features")],
    compute_units=ct.ComputeUnit.ALL,   # lets CoreML schedule to ANE
    minimum_deployment_target=ct.target.macOS13,
)

mlmodel.save(args.out)
print(f"\nSaved: {args.out}")
print("Next: python infer_pipeline_ane.py --model_path checkpoints/ --image_path assets/cute_cat.png --compare")
