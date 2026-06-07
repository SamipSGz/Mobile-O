"""
Extract the Qwen2 LLM backbone from Mobile-O and save it as a standard
HuggingFace Qwen2 checkpoint, then convert to GGUF + quantize to Q4_K_M.

Run:
  python export_llm_gguf.py --model_path checkpoints/

What this does:
  1. Loads the full Mobile-O model
  2. Extracts only the LLM weights (embed_tokens, 24 decoder layers, norm, lm_head)
  3. Saves them as a plain Qwen2ForCausalLM HF checkpoint in llm_export/
  4. Calls llama-convert-hf-to-gguf (bundled with llama.cpp brew) to make a .gguf
  5. Calls llama-quantize to produce Q4_K_M quantized model

Outputs:
  llm_export/          — HF Qwen2 checkpoint
  llm_export/model.gguf         — fp16 GGUF
  llm_export/model-q4.gguf      — INT4 Q4_K_M GGUF  ← use this in pipeline
"""
import os
import sys
import json
import shutil
import subprocess
import argparse
import torch
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="checkpoints/")
parser.add_argument("--out_dir", type=str, default="llm_export")
parser.add_argument("--quant", type=str, default="Q4_K_M",
                    help="GGUF quantization type (Q4_K_M, Q5_K_M, Q8_0, etc.)")
args = parser.parse_args()

out_dir = Path(args.out_dir)
out_dir.mkdir(exist_ok=True)

# ── Step 1: Load Mobile-O and extract LLM weights ────────────────────────────
print("Loading Mobile-O model...")
import warnings; warnings.filterwarnings("ignore")
from mobileo.utils import disable_torch_init
from mobileo.model.builder import load_pretrained_model

disable_torch_init()
tokenizer, model, _ = load_pretrained_model(args.model_path)
model = model.float().cpu()
base = model.get_model()   # mobileoModel

print("Extracting LLM weights...")
state_dict = {}

# embed_tokens
state_dict["model.embed_tokens.weight"] = base.embed_tokens.weight.detach().clone()

# decoder layers
for i, layer in enumerate(base.layers):
    for k, v in layer.state_dict().items():
        state_dict[f"model.layers.{i}.{k}"] = v.detach().clone()

# final norm
for k, v in base.norm.state_dict().items():
    state_dict[f"model.norm.{k}"] = v.detach().clone()

# lm_head — it's on the top-level model
for k, v in model.lm_head.state_dict().items():
    state_dict[f"lm_head.{k}"] = v.detach().clone()

print(f"  Extracted {len(state_dict)} tensors")

# ── Step 2: Save as standard Qwen2 HF checkpoint ─────────────────────────────
print(f"Saving HF Qwen2 checkpoint to {out_dir}/...")

src_cfg = json.load(open(Path(args.model_path) / "config.json"))
qwen2_cfg = {
    "architectures": ["Qwen2ForCausalLM"],
    "model_type": "qwen2",
    "hidden_size": src_cfg["hidden_size"],
    "intermediate_size": src_cfg["intermediate_size"],
    "num_hidden_layers": src_cfg["num_hidden_layers"],
    "num_attention_heads": src_cfg["num_attention_heads"],
    "num_key_value_heads": src_cfg["num_key_value_heads"],
    "hidden_act": src_cfg["hidden_act"],
    "max_position_embeddings": src_cfg["max_position_embeddings"],
    "rms_norm_eps": src_cfg["rms_norm_eps"],
    "rope_theta": src_cfg.get("rope_theta", src_cfg.get("rope_parameters", {}).get("rope_theta", 1000000.0)),
    "vocab_size": src_cfg["vocab_size"],
    "bos_token_id": src_cfg["bos_token_id"],
    "eos_token_id": src_cfg["eos_token_id"],
    "pad_token_id": src_cfg["pad_token_id"],
    "tie_word_embeddings": src_cfg.get("tie_word_embeddings", False),
    "use_sliding_window": False,
    "attention_dropout": 0.0,
    "torch_dtype": "float16",
    "transformers_version": "4.40.0",
}
json.dump(qwen2_cfg, open(out_dir / "config.json", "w"), indent=2)

# Copy tokenizer files
for fname in ["tokenizer.json", "tokenizer_config.json", "vocab.json",
              "merges.txt", "special_tokens_map.json", "added_tokens.json"]:
    src = Path(args.model_path) / fname
    if src.exists():
        shutil.copy(src, out_dir / fname)

# Save weights as safetensors
from safetensors.torch import save_file
# Save in fp16 to keep file size reasonable
state_dict_fp16 = {k: v.half() for k, v in state_dict.items()}
save_file(state_dict_fp16, out_dir / "model.safetensors")
print(f"  Saved model.safetensors ({(out_dir / 'model.safetensors').stat().st_size / 1e9:.2f} GB)")

# ── Step 3: Verify the extracted weights load correctly ───────────────────────
print("Verifying HF checkpoint loads...")
from transformers import Qwen2ForCausalLM, AutoTokenizer
qwen2 = Qwen2ForCausalLM.from_pretrained(str(out_dir), torch_dtype=torch.float16)
print(f"  Qwen2 loaded: {sum(p.numel() for p in qwen2.parameters())/1e6:.0f}M parameters  OK")
del qwen2

# ── Step 4: Convert to GGUF ──────────────────────────────────────────────────
# Find llama.cpp convert script
convert_candidates = [
    "/opt/homebrew/lib/python3.12/site-packages/llama_cpp/convert_hf_to_gguf.py",
    "/opt/homebrew/share/llama.cpp/convert_hf_to_gguf.py",
    shutil.which("llama-convert-hf-to-gguf"),
]
# Also search brew prefix
brew_prefix = subprocess.check_output(["brew", "--prefix", "llama.cpp"],
                                       text=True).strip()
convert_candidates += [
    f"{brew_prefix}/convert_hf_to_gguf.py",
    f"{brew_prefix}/share/llama.cpp/convert_hf_to_gguf.py",
]
convert_script = None
for c in convert_candidates:
    if c and Path(c).exists():
        convert_script = c
        break

if convert_script is None:
    # Try finding it via find
    result = subprocess.run(
        ["find", str(Path(brew_prefix).parent.parent), "-name", "convert_hf_to_gguf.py", "-maxdepth", "8"],
        capture_output=True, text=True
    )
    found = result.stdout.strip().split("\n")
    for f in found:
        if f:
            convert_script = f
            break

if convert_script is None:
    print("\nERROR: Could not find convert_hf_to_gguf.py from llama.cpp brew install.")
    print("Try: find /opt/homebrew -name 'convert_hf_to_gguf.py'")
    print("Then run manually:")
    print(f"  python <path>/convert_hf_to_gguf.py {out_dir} --outfile {out_dir}/model.gguf --outtype f16")
    sys.exit(1)

print(f"\nConverting to GGUF using: {convert_script}")
gguf_path = out_dir / "model.gguf"
result = subprocess.run(
    [sys.executable, convert_script, str(out_dir),
     "--outfile", str(gguf_path), "--outtype", "f16"],
    capture_output=True, text=True
)
if result.returncode != 0:
    print("STDOUT:", result.stdout[-2000:])
    print("STDERR:", result.stderr[-2000:])
    sys.exit(1)
print(f"  Saved {gguf_path}  ({gguf_path.stat().st_size / 1e9:.2f} GB)")

# ── Step 5: Quantize to Q4_K_M ───────────────────────────────────────────────
q4_path = out_dir / "model-q4.gguf"
print(f"\nQuantizing to {args.quant}...")
result = subprocess.run(
    ["/opt/homebrew/bin/llama-quantize", str(gguf_path), str(q4_path), args.quant],
    capture_output=True, text=True
)
if result.returncode != 0:
    print("STDOUT:", result.stdout[-2000:])
    print("STDERR:", result.stderr[-2000:])
    sys.exit(1)
print(f"  Saved {q4_path}  ({q4_path.stat().st_size / 1e9:.2f} GB)")

print(f"""
Done! Files in {out_dir}/:
  model.safetensors  — HF Qwen2 fp16 checkpoint
  model.gguf         — GGUF fp16
  model-q4.gguf      — GGUF {args.quant} (use this in the pipeline)

Next:
  python infer_pipeline_int4.py --model_path checkpoints/ \\
      --gguf_path {out_dir}/model-q4.gguf \\
      --image_path assets/cute_cat.png --repeat 4 --compare
""")
