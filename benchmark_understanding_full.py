"""
Full understanding-task benchmark across 3 configs on MacBook M5.

Configs:
  A — MPS fp16 sequential          (baseline)
  B — ANE vision + MPS LLM async   (pipelining)
  C — ANE vision + INT4 Metal LLM  (quantized)

For each image × prompt × config:
  - latency (ms)
  - tokens generated
  - tokens/sec
  - decoded answer

Aggregate per config (averaged over all image/prompt pairs):
  - mean/median latency
  - mean/median tokens/sec
  - POPE-style hallucination accuracy (yes/no questions with known answers)
  - cross-config output agreement (vs Config A baseline):
      * exact match
      * Jaccard similarity on words
      * ROUGE-L F1 (longest common subsequence)

Outputs:
  predictions/understanding_full_eval.json  — full per-query results
  Console — paper-ready aggregate tables
"""
import argparse, asyncio, json, statistics, time, warnings, re
warnings.filterwarnings("ignore")
import numpy as np
import torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import coremltools as ct

from mobileo.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from mobileo.model.builder import load_pretrained_model
from mobileo.utils import disable_torch_init
from mobileo.mm_utils import tokenizer_image_token, process_images
from mobileo.conversation import conv_templates
from hardware_scheduler import HardwareScheduler

# ── CLI ──────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--model_path",  default="checkpoints/")
p.add_argument("--coreml_path", default="vision_encoder.mlpackage")
p.add_argument("--gguf_path",   default="llm_export/model-q4.gguf")
p.add_argument("--max_tokens",  type=int, default=64)
p.add_argument("--out_json",    default="predictions/understanding_full_eval.json")
args = p.parse_args()

# ── Test corpus: images + prompts with ground truth ───────────────────────────
# Open-ended descriptions (no ground truth, used for cross-config agreement)
DESCRIBE_PROMPTS = [
    "What is in the image?",
    "Describe the image in one short sentence.",
    "What do you see?",
]

# POPE-style binary questions (have known yes/no ground truth)
# Format: (image_path, question, expected_yes_or_no, "open-ground-truth-keywords")
POPE_QUERIES = [
    # cute_cat.png — has a cat (orange tabby)
    ("assets/cute_cat.png", "Is there a cat in the image?",       "yes", ["cat", "kitten", "feline"]),
    ("assets/cute_cat.png", "Is there a dog in the image?",       "no",  ["dog", "puppy", "canine"]),
    ("assets/cute_cat.png", "Are there whiskers in the image?",   "yes", ["whisker"]),
    ("assets/cute_cat.png", "Is there a banana in the image?",    "no",  ["banana"]),
    ("assets/cute_cat.png", "Is there an animal in the image?",   "yes", ["animal", "cat", "pet"]),
    # funny_image.jpeg
    ("assets/funny_image.jpeg", "Is there a person in the image?", None, None),
    ("assets/funny_image.jpeg", "Is there a car in the image?",    None, None),
    # app_settings.jpg
    ("assets/app_settings.jpg", "Is there text in the image?",     "yes", ["text", "settings", "screen", "interface"]),
    ("assets/app_settings.jpg", "Is there a horse in the image?",  "no",  ["horse"]),
]

# Multi-image describe corpus
DESCRIBE_IMAGES = [
    "assets/cute_cat.png",
    "assets/funny_image.jpeg",
    "assets/app_settings.jpg",
    "assets/mobile-o-teaser.jpg",
    "assets/training_figure.jpg",
]

# ── Hardware ──────────────────────────────────────────────────────────────────
sched = HardwareScheduler(coreml_vision_path=args.coreml_path, gguf_llm_path=args.gguf_path)
sched.print_report()

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype  = torch.float16 if device == "mps" else torch.float32

def sync():
    if torch.backends.mps.is_available(): torch.mps.synchronize()

# ── Model loading ────────────────────────────────────────────────────────────
print("\nLoading Mobile-O (PyTorch)...")
disable_torch_init()
tokenizer, model, _ = load_pretrained_model(args.model_path)
model.to(dtype).to(device).eval()
image_processor = model.get_vision_tower().image_processor
model.generation_config.pad_token_id = tokenizer.pad_token_id
print("Loaded.\n")

# Pre-load all unique images
unique_images = sorted(set([img for img, _, _, _ in POPE_QUERIES] + DESCRIBE_IMAGES))
print(f"Pre-loading {len(unique_images)} images...")
img_tensor_cache = {}
for img_path in unique_images:
    raw = Image.open(img_path).convert("RGB")
    img_tensor_cache[img_path] = process_images([raw], image_processor, model.config)[0]
print("Done.\n")

# CoreML
print("Loading CoreML vision encoder (ANE)...")
cml = ct.models.MLModel(args.coreml_path, compute_units=ct.ComputeUnit.ALL)
cml.predict({"pixel_values": img_tensor_cache[unique_images[0]].unsqueeze(0).float().numpy()})
print("Done.\n")

# INT4
print("Loading INT4 GGUF LLM (Metal)...")
from llama_cpp import Llama
int4 = Llama(model_path=args.gguf_path, n_gpu_layers=99, n_ctx=1024, verbose=False)
int4("hi", max_tokens=4, echo=False)
print("Done.\n")


# ── Prompt builders ──────────────────────────────────────────────────────────
def build_mps_prompt(text):
    qs = DEFAULT_IMAGE_TOKEN + "\n" + text
    conv = conv_templates["qwen_2"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()

def build_int4_prompt(text):
    return (
        "<|im_start|>system\nYou are a helpful visual assistant.<|im_end|>\n"
        f"<|im_start|>user\n{text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

# ── Per-config inference ─────────────────────────────────────────────────────
def run_mps(img_path, prompt_text):
    """Config A: MPS fp16 sequential."""
    tensor = img_tensor_cache[img_path]
    image_batch = tensor.unsqueeze(0).to(dtype).to(device)
    input_ids = tokenizer_image_token(
        build_mps_prompt(prompt_text), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)

    sync(); t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(input_ids, images=image_batch,
            do_sample=True, temperature=0.6, top_p=None,
            num_beams=1, max_new_tokens=args.max_tokens, use_cache=True,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    sync(); elapsed = time.perf_counter() - t0
    n = int(out.shape[1])
    text = tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
    return {"tokens": n, "seconds": elapsed, "answer": text}


async def run_ane_mps_async(img_path, prompt_text):
    """Config B: ANE vision + MPS LLM async pipeline."""
    loop = asyncio.get_event_loop()
    ane_ex = ThreadPoolExecutor(max_workers=1)
    llm_ex = ThreadPoolExecutor(max_workers=1)
    q = asyncio.Queue(maxsize=2)
    result = {}

    tensor = img_tensor_cache[img_path]
    np_in  = tensor.unsqueeze(0).float().numpy()

    async def vision_worker():
        await loop.run_in_executor(ane_ex, lambda: cml.predict({"pixel_values": np_in}))
        await q.put(True)
        await q.put(None)

    async def llm_worker():
        await q.get()
        r = await loop.run_in_executor(llm_ex, run_mps, img_path, prompt_text)
        result.update(r)

    t0 = time.perf_counter()
    await asyncio.gather(vision_worker(), llm_worker())
    result["seconds"] = time.perf_counter() - t0  # wall-clock includes overlap

    ane_ex.shutdown(wait=False); llm_ex.shutdown(wait=False)
    return result


def run_ane_int4(img_path, prompt_text):
    """Config C: ANE vision + INT4 Metal LLM."""
    tensor = img_tensor_cache[img_path]
    np_in  = tensor.unsqueeze(0).float().numpy()

    t0 = time.perf_counter()
    cml.predict({"pixel_values": np_in})
    out = int4(build_int4_prompt(prompt_text), max_tokens=args.max_tokens, temperature=0.6,
               echo=False, stop=["<|im_end|>", "<|endoftext|>"])
    elapsed = time.perf_counter() - t0
    n = int(out["usage"]["completion_tokens"])
    text = out["choices"][0]["text"].strip()
    return {"tokens": n, "seconds": elapsed, "answer": text}


# ── Qualitative scoring helpers ──────────────────────────────────────────────
STOPWORDS = set("a an the is are was were be been being am i it its this that these those of on in at to for with by from has have had does do did did not no yes and or but if then so than as".split())

def words(s):
    return [t.lower() for t in re.findall(r"[a-zA-Z][a-zA-Z']+", s)]

def content_words(s):
    return [w for w in words(s) if w not in STOPWORDS]

def jaccard(a, b):
    if not a and not b: return 1.0
    sa, sb = set(a), set(b)
    if not sa | sb: return 0.0
    return len(sa & sb) / len(sa | sb)

def lcs_len(a, b):
    """Longest common subsequence length (DP)."""
    if not a or not b: return 0
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            if a[i] == b[j]: dp[i+1][j+1] = dp[i][j] + 1
            else:            dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])
    return dp[m][n]

def rouge_l_f1(reference, candidate):
    """ROUGE-L F1 on word sequences."""
    r_words = content_words(reference)
    c_words = content_words(candidate)
    if not r_words or not c_words: return 0.0
    lcs = lcs_len(r_words, c_words)
    if lcs == 0: return 0.0
    p = lcs / len(c_words)
    r = lcs / len(r_words)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def extract_yes_no(text):
    """Heuristic: does the answer say yes or no?"""
    t = text.lower().strip()
    # Look at first 30 chars
    head = t[:30]
    yes_signals = ["yes", "there is", "there are", "i see", "i can see", "it does", "indeed"]
    no_signals  = ["no", "there is no", "there are no", "i don't", "i do not", "no, ", "not"]
    # Strict check for "no" word
    first_word = re.match(r"[^a-zA-Z]*([a-zA-Z]+)", t)
    if first_word:
        fw = first_word.group(1).lower()
        if fw == "yes": return "yes"
        if fw == "no":  return "no"
    for s in no_signals:
        if s in head: return "no"
    for s in yes_signals:
        if s in head: return "yes"
    return "unknown"

def contains_any(text, keywords):
    if not keywords: return None
    t = text.lower()
    return any(k.lower() in t for k in keywords)


# ── Run the benchmark ────────────────────────────────────────────────────────
results = {"A": [], "B": [], "C": []}
print("=" * 80)
print(f"  Running {len(DESCRIBE_IMAGES)*len(DESCRIBE_PROMPTS) + len(POPE_QUERIES)} queries per config")
print("=" * 80)

# Build the full query list
queries = []
for img in DESCRIBE_IMAGES:
    for prompt_text in DESCRIBE_PROMPTS:
        queries.append({"type": "describe", "image": img, "prompt": prompt_text,
                        "expected_yn": None, "expected_keywords": None})
for img, prompt_text, expected_yn, kw in POPE_QUERIES:
    queries.append({"type": "pope", "image": img, "prompt": prompt_text,
                    "expected_yn": expected_yn, "expected_keywords": kw})

# Warmup with one query
print("\nWarming up...")
_ = run_mps(queries[0]["image"], queries[0]["prompt"])
_ = asyncio.run(run_ane_mps_async(queries[0]["image"], queries[0]["prompt"]))
_ = run_ane_int4(queries[0]["image"], queries[0]["prompt"])
print("Warmup done.\n")

for cfg_name, fn in [("A", lambda i, p: run_mps(i, p)),
                     ("B", lambda i, p: asyncio.run(run_ane_mps_async(i, p))),
                     ("C", lambda i, p: run_ane_int4(i, p))]:
    print(f"\n[Config {cfg_name}] running {len(queries)} queries...")
    for q in queries:
        r = fn(q["image"], q["prompt"])
        results[cfg_name].append({
            **q,
            "tokens":  r["tokens"],
            "seconds": r["seconds"],
            "tps":     r["tokens"] / r["seconds"] if r["seconds"] > 0 else 0,
            "answer":  r["answer"],
        })
        # Print compact line
        tag = q["type"][:4]
        print(f"  [{tag}] {q['image'].split('/')[-1]:>25s} | "
              f"{q['prompt'][:32]:<32s} | "
              f"{r['tokens']:>3d} tok | "
              f"{r['seconds']*1000:>5.0f} ms | "
              f"\"{r['answer'][:50]}\"")


# ── Compute aggregate quantitative metrics ───────────────────────────────────
def aggregate(stats_list, key):
    vals = [s[key] for s in stats_list]
    return {
        "mean":   statistics.mean(vals),
        "median": statistics.median(vals),
        "stdev":  statistics.stdev(vals) if len(vals) > 1 else 0,
        "min":    min(vals),
        "max":    max(vals),
    }

quant_summary = {}
for cfg in ["A", "B", "C"]:
    quant_summary[cfg] = {
        "n_queries":         len(results[cfg]),
        "total_tokens":      sum(s["tokens"] for s in results[cfg]),
        "total_seconds":     sum(s["seconds"] for s in results[cfg]),
        "latency_ms":        aggregate(results[cfg], "seconds"),
        "tokens_per_query":  aggregate(results[cfg], "tokens"),
        "tps":               aggregate(results[cfg], "tps"),
    }
    quant_summary[cfg]["aggregate_tps"] = (
        quant_summary[cfg]["total_tokens"] / quant_summary[cfg]["total_seconds"]
        if quant_summary[cfg]["total_seconds"] > 0 else 0
    )


# ── Compute qualitative metrics ──────────────────────────────────────────────
qual_summary = {}

for cfg in ["A", "B", "C"]:
    pope_correct = 0
    pope_total   = 0
    pope_yes_correct = 0
    pope_yes_total   = 0
    pope_no_correct  = 0
    pope_no_total    = 0
    keyword_hits = 0
    keyword_total = 0

    for r in results[cfg]:
        if r["type"] == "pope" and r["expected_yn"] is not None:
            pope_total += 1
            pred = extract_yes_no(r["answer"])
            correct = pred == r["expected_yn"]
            if correct: pope_correct += 1
            if r["expected_yn"] == "yes":
                pope_yes_total += 1
                if correct: pope_yes_correct += 1
            else:
                pope_no_total += 1
                if correct: pope_no_correct += 1

        if r["expected_keywords"]:
            keyword_total += 1
            if contains_any(r["answer"], r["expected_keywords"]):
                keyword_hits += 1

    qual_summary[cfg] = {
        "pope_accuracy":  pope_correct / pope_total if pope_total else 0,
        "pope_yes_recall": pope_yes_correct / pope_yes_total if pope_yes_total else 0,
        "pope_no_recall":  pope_no_correct  / pope_no_total  if pope_no_total  else 0,
        "pope_total":      pope_total,
        "keyword_hit_rate": keyword_hits / keyword_total if keyword_total else 0,
        "keyword_total":   keyword_total,
    }

# Cross-config agreement vs A baseline
agreement = {}
for cfg in ["B", "C"]:
    exact = 0
    jacc_scores = []
    rouge_scores = []
    yn_agreement = 0
    yn_total = 0
    for a_r, c_r in zip(results["A"], results[cfg]):
        # exact match
        if a_r["answer"].strip().lower() == c_r["answer"].strip().lower():
            exact += 1
        # word overlap
        jacc_scores.append(jaccard(content_words(a_r["answer"]), content_words(c_r["answer"])))
        rouge_scores.append(rouge_l_f1(a_r["answer"], c_r["answer"]))
        # yes/no agreement on POPE
        if a_r["type"] == "pope":
            yn_total += 1
            if extract_yes_no(a_r["answer"]) == extract_yes_no(c_r["answer"]):
                yn_agreement += 1
    agreement[f"{cfg}_vs_A"] = {
        "exact_match_rate": exact / len(results["A"]),
        "jaccard_mean":     statistics.mean(jacc_scores),
        "jaccard_median":   statistics.median(jacc_scores),
        "rouge_l_mean":     statistics.mean(rouge_scores),
        "rouge_l_median":   statistics.median(rouge_scores),
        "yn_agreement":     yn_agreement / yn_total if yn_total else 0,
    }


# ── Print paper-ready tables ──────────────────────────────────────────────────
print("\n\n" + "=" * 80)
print("  QUANTITATIVE SUMMARY  (averaged over all queries)")
print("=" * 80)
print(f"  {'Config':<35} {'N':>5} {'Tokens':>7} {'Wall (s)':>10} {'Mean lat':>10} {'Tok/s':>8}")
print("-" * 80)
labels = {"A": "A — MPS fp16 sequential",
          "B": "B — ANE+MPS async pipeline",
          "C": "C — ANE+INT4 Metal pipeline"}
for cfg in ["A", "B", "C"]:
    q = quant_summary[cfg]
    print(f"  {labels[cfg]:<35} {q['n_queries']:>5} "
          f"{q['total_tokens']:>7d} "
          f"{q['total_seconds']:>9.1f}s "
          f"{q['latency_ms']['mean']*1000:>8.0f}ms "
          f"{q['aggregate_tps']:>6.1f}t/s")
print("=" * 80)

print("\n\n" + "=" * 80)
print("  QUALITATIVE — POPE-style hallucination check")
print("=" * 80)
print(f"  {'Config':<35} {'Acc':>7} {'Yes-rec':>9} {'No-rec':>9} {'KW-hit':>9}")
print("-" * 80)
for cfg in ["A", "B", "C"]:
    qq = qual_summary[cfg]
    print(f"  {labels[cfg]:<35} {qq['pope_accuracy']*100:>5.1f}% "
          f"{qq['pope_yes_recall']*100:>7.1f}% "
          f"{qq['pope_no_recall']*100:>7.1f}% "
          f"{qq['keyword_hit_rate']*100:>7.1f}%")
print("=" * 80)
print(f"  POPE total questions per config: {qual_summary['A']['pope_total']}")
print(f"  Keyword-tagged questions per config: {qual_summary['A']['keyword_total']}")

print("\n\n" + "=" * 80)
print("  QUALITATIVE — Cross-config output agreement (vs Config A baseline)")
print("=" * 80)
print(f"  {'Comparison':<25} {'Exact':>8} {'Jaccard':>10} {'ROUGE-L':>10} {'Y/N agree':>11}")
print("-" * 80)
for cmp in ["B_vs_A", "C_vs_A"]:
    a = agreement[cmp]
    print(f"  {cmp:<25} {a['exact_match_rate']*100:>6.1f}% "
          f"{a['jaccard_mean']:>8.3f}  "
          f"{a['rouge_l_mean']:>8.3f}  "
          f"{a['yn_agreement']*100:>9.1f}%")
print("=" * 80)

# ── Save JSON ─────────────────────────────────────────────────────────────────
from pathlib import Path
Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
with open(args.out_json, "w") as f:
    json.dump({
        "device": sched.summary_dict(),
        "max_tokens": args.max_tokens,
        "describe_prompts": DESCRIBE_PROMPTS,
        "describe_images": DESCRIBE_IMAGES,
        "pope_queries": [{"image": i, "prompt": p, "expected_yn": e,
                          "expected_keywords": k}
                         for i, p, e, k in POPE_QUERIES],
        "results": results,
        "quantitative": quant_summary,
        "qualitative":  qual_summary,
        "agreement_vs_A": agreement,
    }, f, indent=2)
print(f"\nSaved: {args.out_json}\n")
