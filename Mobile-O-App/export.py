#!/usr/bin/env python3
"""
Export all Mobile-O models for the iOS app.

Exports 5 components from the Mobile-O model:
  1. DiT Transformer    → CoreML FP32  (transformer.mlpackage)
  2. VAE Decoder        → CoreML FP32  (vae-decoder.mlpackage)
  3. Connector          → CoreML FP32  (connector.mlpackage)
  4. Vision Encoder     → CoreML FP16  (vision-encoder.mlpackage)
  5. LLM                → MLX 4-bit    (llm/model.safetensors + config.json)

Usage:
    python export.py                                     # Export all (default HF model)
    python export.py --model ./local-checkpoint           # Export from local path
    python export.py --only dit vae                      # Export specific components
    python export.py --only llm --llm-bits 16
"""

import argparse
import json
import logging
import os
import sys
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import numpy as np
import torch
import coremltools as ct

# Suppress noisy warnings from dependencies
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
logging.getLogger("coremltools").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).parent))

# -- Constants ----------------------------------------------------------

SEQ_LEN_MIN = 77
SEQ_LEN_MAX = 512
SEQ_LEN_DEFAULT = 77

DIT_LATENT_CHANNELS = 32
DIT_LATENT_SIZE = 16
DIT_TEXT_HIDDEN_SIZE = 2304

VAE_IMAGE_SIZE = 512
VAE_LATENT_SCALE = 32

VISION_IMAGE_RES = 1024

MM_PROJECTOR_REMAP = {
    "mm_projector.0.": "multi_modal_projector.linear_0.",
    "mm_projector.2.": "multi_modal_projector.linear_2.",
}

DEFAULT_MODEL = "Amshaker/Mobile-O-0.5B"
ALL_COMPONENTS = ["dit", "vae", "connector", "vision", "llm"]

COMPONENT_OUTPUTS = {
    "dit":       ("DiT",       "transformer.mlpackage"),
    "vae":       ("VAE",       "vae_decoder.mlpackage"),
    "connector": ("Connector", "connector.mlpackage"),
    "vision":    ("Vision",    "vision_encoder.mlpackage"),
    "llm":       ("LLM",       "llm/model.safetensors"),
}

# -- Helpers -------------------------------------------------------------------

def model_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024**2

def trace_and_freeze(module, example_inputs):
    """Trace a module and freeze the resulting graph."""
    module.eval()
    for p in module.parameters():
        p.requires_grad = False
    with torch.no_grad():
        traced = torch.jit.trace(module, example_inputs, strict=False)
    return torch.jit.freeze(traced)

def register_int_converter():
    """
    Monkey-patch coremltools' _cast so aten::Int works on non-scalar tensors.
    The DiT transformer has attention ops where int() is called on dynamic
    tensor dims (e.g. height*width). CoreML's default _cast raises on these.
    We patch it to fall through to a cast op instead of raising.
    """
    if getattr(register_int_converter, "_done", False):
        return
    register_int_converter._done = True

    import coremltools.converters.mil.frontend.torch.ops as _torch_ops
    import numpy as _np

    _original_cast = _torch_ops._cast

    def _patched_cast(context, node, dtype, dtype_name):
        from coremltools.converters.mil.frontend.torch.ops import _get_inputs
        from coremltools.converters.mil import Builder as mb
        inputs = _get_inputs(context, node, expected=1)
        x = inputs[0]
        # Non-scalar, multi-element constant (e.g. height*width from DiT attention)
        # The default path tries dtype(x.val) which fails on nd-arrays.
        # Route through mb.cast instead.
        if x.val is not None and _np.ndim(x.val) > 0 and _np.size(x.val) > 1:
            res = mb.cast(x=x, dtype=dtype_name, name=node.name)
            context.add(res, node.name)
            return
        # Scalar constant — safe to call dtype(x.val)
        if x.val is not None and _np.ndim(x.val) == 0:
            try:
                res = mb.const(val=dtype(x.val), name=node.name)
                context.add(res, node.name)
                return
            except (TypeError, ValueError):
                res = mb.cast(x=x, dtype=dtype_name, name=node.name)
                context.add(res, node.name)
                return
        # dynamic / non-const — original path
        _original_cast(context, node, dtype, dtype_name)

    _torch_ops._cast = _patched_cast


def register_movedim_converter():
    """Register torch.movedim → CoreML transpose (needed by VAE)."""
    if getattr(register_movedim_converter, "_done", False):
        return
    register_movedim_converter._done = True

    from coremltools.converters.mil.frontend.torch.ops import _get_inputs
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op

    @register_torch_op
    def movedim(context, node):
        inputs = _get_inputs(context, node, expected=3)
        x = inputs[0]
        source = inputs[1].val
        destination = inputs[2].val
        rank = x.rank

        source = [int(source)] if isinstance(source, (int, np.integer)) else [int(s) for s in source]
        destination = [int(destination)] if isinstance(destination, (int, np.integer)) else [int(d) for d in destination]
        source = [(s if s >= 0 else rank + s) for s in source]
        destination = [(d if d >= 0 else rank + d) for d in destination]

        perm = list(range(rank))
        for s in sorted(source, reverse=True):
            perm.pop(s)
        for s, d in zip(source, destination):
            perm.insert(d, s)

        context.add(mb.transpose(x=x, perm=perm, name=node.name))

# -- Model wrappers for tracing ------------------------------------------------

class DiTWrapper(torch.nn.Module):
    def __init__(self, dit):
        super().__init__()
        self.dit = dit

    def forward(self, latent, timestep, encoder_hidden_states, encoder_attention_mask):
        return self.dit(
            hidden_states=latent,
            timestep=timestep.float(),
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        ).sample

class VAEDecoderWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latent):
        return self.vae.decode(latent, return_dict=False)[0]

class ConnectorWrapper(torch.nn.Module):
    def __init__(self, connector, num_layers):
        super().__init__()
        self.connector = connector
        self.num_layers = num_layers

    def forward(self, stacked_hidden_states):
        hidden_states_list = [stacked_hidden_states[:, i, :, :] for i in range(self.num_layers)]
        return self.connector(hidden_states_list)

# -- Export functions ----------------------------------------------------------

def export_dit(model, output_dir: Path):
    """Export DiT transformer to CoreML FP32."""
    print("\n--- DiT Transformer (CoreML FP32) ---")
    register_int_converter()
    register_movedim_converter()

    dit = model.model.dit.to(torch.float32).eval().cpu()

    example = (
        torch.randn(1, DIT_LATENT_CHANNELS, DIT_LATENT_SIZE, DIT_LATENT_SIZE),
        torch.tensor([500.0]),
        torch.randn(1, SEQ_LEN_DEFAULT, DIT_TEXT_HIDDEN_SIZE),
        torch.ones(1, SEQ_LEN_DEFAULT),
    )

    traced = trace_and_freeze(DiTWrapper(dit), example)

    seq_range = ct.RangeDim(lower_bound=SEQ_LEN_MIN, upper_bound=SEQ_LEN_MAX, default=SEQ_LEN_DEFAULT)
    inputs = [
        ct.TensorType(name="latent", shape=(1, DIT_LATENT_CHANNELS, DIT_LATENT_SIZE, DIT_LATENT_SIZE)),
        ct.TensorType(name="timestep", shape=(1,)),
        ct.TensorType(name="encoder_hidden_states", shape=(1, seq_range, DIT_TEXT_HIDDEN_SIZE)),
        ct.TensorType(name="encoder_attention_mask", shape=(1, seq_range)),
    ]

    print(f"  Seq len range: {SEQ_LEN_MIN}-{SEQ_LEN_MAX} (default {SEQ_LEN_DEFAULT})")
    mlmodel = ct.convert(
        traced, inputs=inputs,
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.ALL,
    )

    path = output_dir / "transformer.mlpackage"
    mlmodel.save(str(path))
    print(f"  Saved: {path} ({model_size_mb(path):.1f} MB)")

def export_vae(model, output_dir: Path):
    """Export VAE decoder to CoreML FP32."""
    print("\n--- VAE Decoder (CoreML FP32) ---")

    register_movedim_converter()

    vae = model.model.vae.to(torch.float32).eval().cpu()
    h = w = VAE_IMAGE_SIZE // VAE_LATENT_SCALE
    c = vae.config.latent_channels
    example = (torch.randn(1, c, h, w),)

    traced = trace_and_freeze(VAEDecoderWrapper(vae), example)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="latent", shape=example[0].shape)],
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.ALL,
    )

    path = output_dir / "vae_decoder.mlpackage"
    mlmodel.save(str(path))
    print(f"  Saved: {path} ({model_size_mb(path):.1f} MB)")

def export_connector(model, output_dir: Path):
    """Export conditioning connector to CoreML FP32."""
    print("\n--- Connector (CoreML FP32) ---")

    connector = model.model.diffusion_connector.to(torch.float32).eval().cpu()
    num_layers = connector.num_layers
    input_dim = connector.input_dim
    print(f"  Layers: {num_layers}, input_dim: {input_dim}, output_dim: {connector.output_dim}")

    example = (torch.randn(1, num_layers, SEQ_LEN_DEFAULT, input_dim),)
    traced = trace_and_freeze(ConnectorWrapper(connector, num_layers), example)

    seq_range = ct.RangeDim(lower_bound=SEQ_LEN_MIN, upper_bound=SEQ_LEN_MAX, default=SEQ_LEN_DEFAULT)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(
            name="stacked_hidden_states",
            shape=(1, num_layers, seq_range, input_dim),
            dtype=np.float32,
        )],
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS18,
    )

    path = output_dir / "connector.mlpackage"
    mlmodel.save(str(path))
    print(f"  Saved: {path} ({model_size_mb(path):.1f} MB)")

def export_vision(model, output_dir: Path):
    """Export vision encoder to CoreML FP16."""
    print("\n--- Vision Encoder (CoreML FP16) ---")

    vision_tower = model.get_model().get_vision_tower()
    if vision_tower is None:
        raise ValueError("Vision tower not found in model")
    if not vision_tower.is_loaded:
        vision_tower.load_model()

    vision_tower.eval().float()
    dummy = torch.rand(1, 3, VISION_IMAGE_RES, VISION_IMAGE_RES)

    traced = trace_and_freeze(vision_tower, (dummy,))

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="images", shape=dummy.shape)],
        outputs=[ct.TensorType(name="image_features", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.ALL,
    )

    path = output_dir / "vision_encoder.mlpackage"
    mlmodel.save(str(path))
    print(f"  Saved: {path} ({model_size_mb(path):.1f} MB)")

def export_llm(model, output_dir: Path, model_path: Path, bits: int = 4):
    """Export LLM and projector to MLX safetensors."""
    group_size = 64
    print(f"\n--- LLM (MLX {bits}-bit) ---")

    import mlx.core as mx

    llm_model = model.model
    lm_head = model.lm_head
    vocab_size = lm_head.weight.shape[0]

    llm_config = {
        "model_type": "qwen2",
        "hidden_size": llm_model.config.hidden_size,
        "num_hidden_layers": llm_model.config.num_hidden_layers,
        "intermediate_size": llm_model.config.intermediate_size,
        "num_attention_heads": llm_model.config.num_attention_heads,
        "num_key_value_heads": llm_model.config.num_key_value_heads,
        "rms_norm_eps": llm_model.config.rms_norm_eps,
        "vocab_size": vocab_size,
        "max_position_embeddings": llm_model.config.max_position_embeddings,
        "rope_theta": getattr(llm_model.config, "rope_theta", 1000000),
        "tie_word_embeddings": llm_model.config.tie_word_embeddings,
    }
    print(f"  hidden_size={llm_config['hidden_size']}, layers={llm_config['num_hidden_layers']}, vocab={llm_config['vocab_size']}")

    # Extract state dicts (suppress verbose output)
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
        llm_state = llm_model.state_dict()
        head_state = lm_head.state_dict()

    # Filter to LLM + mm_projector weights
    state_dict = {
        k: v for k, v in llm_state.items()
        if k.startswith(("embed_tokens", "layers.", "norm", "mm_projector"))
    }
    state_dict.update({f"lm_head.{k}": v for k, v in head_state.items()})
    print(f"  Extracted {len(state_dict)} tensors")

    # Remap keys to FastVLM format and convert to MLX
    dtype = mx.float32 if bits == 32 else mx.float16
    weights = {}
    for key, value in state_dict.items():
        arr = mx.array(value.cpu().float().numpy()).astype(dtype)
        if key.startswith("mm_projector."):
            new_key = key
            for old, new in MM_PROJECTOR_REMAP.items():
                new_key = new_key.replace(old, new)
            weights[new_key] = arr
        elif key.startswith("lm_head."):
            weights[f"language_model.{key}"] = arr
        else:
            weights[f"language_model.model.{key}"] = arr

    # Detect mm_hidden_size from projected weights
    mm_proj_key = "multi_modal_projector.linear_0.weight"
    if mm_proj_key not in weights:
        raise ValueError("Could not detect mm_hidden_size from mm_projector weights")
    mm_hidden_size = weights[mm_proj_key].shape[1]

    # Quantize 2D weight tensors, skip small layers (norms, biases)
    if bits in (16, 32):
        quantized_weights = weights
    else:
        print(f"  Quantizing {len(weights)} tensors to {bits}-bit...")
        quantized_weights = {}
        min_elements = 512
        for key, value in weights.items():
            if key.endswith(".weight") and value.ndim == 2 and value.size >= min_elements:
                q_weight, scales, biases = mx.quantize(value, group_size=group_size, bits=bits)
                quantized_weights[key] = q_weight
                quantized_weights[key.replace(".weight", ".scales")] = scales
                quantized_weights[key.replace(".weight", ".biases")] = biases
            else:
                quantized_weights[key] = value

    # Save weights
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_file = output_dir / "model.safetensors"
    mx.save_safetensors(str(weights_file), quantized_weights, metadata={"format": "mlx"})

    total_size = sum(w.nbytes for w in quantized_weights.values())
    index_data = {
        "metadata": {"total_size": int(total_size)},
        "weight_map": {k: weights_file.name for k in quantized_weights},
    }
    with open(output_dir / f"{weights_file.name}.index.json", "w") as f:
        json.dump(index_data, f, indent=4)

    # Save config
    config = {
        **llm_config,
        "model_type": "llava_qwen2",
        "mm_hidden_size": mm_hidden_size,
        "mm_vision_tower": "mobileclip_l_1024",
        "image_token_index": 151648,
    }
    if bits not in (16, 32):
        config["quantization"] = {"group_size": group_size, "bits": bits}
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    # Copy tokenizer files from source model
    import shutil
    tokenizer_files = [
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "added_tokens.json", "merges.txt", "vocab.json",
    ]
    copied = 0
    for fname in tokenizer_files:
        src = model_path / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)
            copied += 1

    print(f"  Saved: {weights_file} ({total_size / 1024**3:.2f} GB)")
    print(f"  Config: {output_dir / 'config.json'}")
    print(f"  Tokenizer: copied {copied} files")

# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export all Mobile-O models for the iOS app")
    parser.add_argument("model_path", nargs="?", default=DEFAULT_MODEL, help=f"HuggingFace model ID or local path (default: {DEFAULT_MODEL})")
    parser.add_argument("--output-dir", default="exported_models", help="Output directory (default: exported_models)")
    parser.add_argument("--only", nargs="+", choices=ALL_COMPONENTS, help="Export only specific components")
    parser.add_argument("--llm-bits", type=int, default=4, choices=[4, 8, 16, 32], help="LLM quantization bits (default: 4)")
    args = parser.parse_args()

    raw_path = args.model_path
    local_path = Path(raw_path).expanduser()
    is_local = local_path.exists()
    model_path = str(local_path) if is_local else raw_path

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    components = args.only or ALL_COMPONENTS

    if not is_local and "/" not in raw_path:
        print(f"Error: '{raw_path}' is not a local path or a valid HuggingFace model ID")
        sys.exit(1)

    print(f"Model:      {model_path}" + (" (local)" if is_local else " (HuggingFace Hub)"))
    print(f"Output:     {output_dir}")
    print(f"Components: {', '.join(components)}")

    from mobileo.model import MobileOForInferenceLM
    print("\nLoading model...")
    load_kwargs = {"dtype": torch.float16}
    if is_local:
        load_kwargs["local_files_only"] = True
    model = MobileOForInferenceLM.from_pretrained(model_path, **load_kwargs)

    if "dit" in components:
        export_dit(model, output_dir)
    if "vae" in components:
        export_vae(model, output_dir)
    if "connector" in components:
        export_connector(model, output_dir)
    if "vision" in components:
        export_vision(model, output_dir)
    if "llm" in components:
        if is_local:
            tokenizer_source = local_path
        else:
            from huggingface_hub import snapshot_download
            tokenizer_source = Path(snapshot_download(model_path, allow_patterns=["tokenizer*", "special_tokens*", "added_tokens*", "merges.txt", "vocab.json"]))
        export_llm(model, output_dir / "llm", model_path=tokenizer_source, bits=args.llm_bits)

    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)
    for c in components:
        label, path = COMPONENT_OUTPUTS[c]
        print(f"  {label + ':':<11}{output_dir}/{path}")

if __name__ == "__main__":
    main()