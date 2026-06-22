#!/usr/bin/env python3
"""
proactive-inference/run.py  —  STANDALONE configurable runner (2x2 ablation)
============================================================================
Self-contained: imports NOTHING from sibling exp*.py scripts. It only needs the
environment (torch / transformers / PIL / nltk) plus the method libraries
the in-repo `shared_cache` and `shared_cache_ext` (these ARE the
shared-KV implementation). Everything else — backend, gates, metrics, frame
sampling — is inlined here, so this file works on its own.

Conditions (2x2 of input-gate x output-gate):
  baseline     input gate OFF (admit all)  + output STREAM-ALL (emit every frame)
  input_only   input gate ON  (cosine)     + output STREAM-ALL
  output_only  input gate OFF              + output GATED (AsyncReasoning-style)
  both         input gate ON               + output GATED

Examples:
  python run.py --mode all
  python run.py --mode both
  python run.py --mode baseline,both --frames 30 --delta 0.93 --temp 0.7 --seed 0
"""

import os, sys, glob, json, time, subprocess, argparse
from pathlib import Path

import torch
from PIL import Image

HOME = Path.home()
HERE = Path(__file__).resolve().parent
MOBILE_O = HERE.parent.parent                      # ~/Mobile-O (repo root)
sys.path.insert(0, str(MOBILE_O))                  # vendored shared_cache + shared_cache_ext (no AsyncReasoning code)

FFMPEG    = str(HOME / "miniforge3" / "bin" / "ffmpeg")
FRAME_DIR = str(HERE / "frames")

# ── config ────────────────────────────────────────────────────────────────────
MODEL_NAME        = "Qwen/Qwen2-VL-7B-Instruct"
TOKENS_PER_FRAME  = 1196
TEMP              = 0.7      # sampling temperature (orchestrator + writer)
THINK_TOKS        = 18       # max tokens per orchestrator thought
WRITE_TOKS        = 40       # max tokens per writer caption
DELTA             = 0.93     # input admission threshold (admit if cos < DELTA)
TEMPERATURE_WRITE = 0.7      # (used only by Backend.write_fresh, kept for completeness)
WRITE_MAX_TOKS    = 80

ORCH_HEADER = (
    "<|im_start|>system\nYou are a cooking video analysis assistant.\n<|im_end|>\n"
    "<|im_start|>user\nHere are video frames from a cooking video:\n<|vision_start|>"
)
ORCH_SUFFIX = (
    "<|vision_end|>\nIs something new and significant happening in the latest frame? "
    "Answer Yes or No.\n<|im_end|>\n<|im_start|>assistant\n"
)
WRITE_SUFFIX = (
    "<|vision_end|>\nDescribe what is currently happening in the video in one clear sentence.\n"
    "<|im_end|>\n<|im_start|>assistant\n"
)
THINK_PROMPT = (
    "<|vision_end|>\nIn a few words, note what is happening in the latest frame.\n"
    "<|im_end|>\n<|im_start|>assistant\n"
)

MODES = {
    "baseline":    (False, False),
    "input_only":  (True,  False),
    "output_only": (False, True),
    "both":        (True,  True),
}
DEFAULT_VIDEO = str(HERE.parent / "eval_video.mp4")
DEFAULT_GT    = str(HERE.parent / "eval_video_gt.json")


# ── model ───────────────────────────────────────────────────────────────────--
def load_model():
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    print(f"Loading {MODEL_NAME}...", flush=True)
    t0 = time.time()
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    print(f"Loaded in {time.time()-t0:.1f}s", flush=True)
    return model, processor, processor.tokenizer


# ── shared-KV backend (visual KV computed once into block_P, reused) ──────────
class Backend:
    def __init__(self, model, processor, tokenizer, device):
        from shared_cache.cache_block import CacheBlock
        from shared_cache_ext import RawCacheBlock, MRopeCombinedCacheView, MRopeSharedCacheManager
        self.m, self.proc, self.tok, self.dev = model, processor, tokenizer, device
        self.CacheBlock = CacheBlock
        self.RawCacheBlock = RawCacheBlock
        self.MRopeView = MRopeCombinedCacheView
        self.MRopeCM = MRopeSharedCacheManager
        tk = dict(add_special_tokens=False)
        self.SEP = tokenizer("\n", **tk)["input_ids"][-1]
        self.EOS = tokenizer.eos_token_id
        self.IM_END = 151645
        self.YES = tokenizer("Yes", **tk)["input_ids"][0]
        self.NO  = tokenizer("No",  **tk)["input_ids"][0]
        self.hdr_ids   = tokenizer(ORCH_HEADER, **tk)["input_ids"]
        self.orch_suf  = tokenizer(ORCH_SUFFIX, **tk)["input_ids"]
        self.write_suf = tokenizer(WRITE_SUFFIX, **tk)["input_ids"]
        self.config = model.config
        self._fresh_P()

    def _to2d(self, ids):
        return torch.tensor([ids], dtype=torch.long, device=self.dev)

    def _fresh_P(self):
        self.block_P = self.RawCacheBlock(config=self.config)
        cm = self.MRopeCM([[self.block_P]], write_to=[self.block_P])
        with torch.no_grad():
            self.m(**cm.get_input_kwargs(self._to2d(self.hdr_ids)))

    def encode_frame(self, frame_pil, frame_idx):
        d = self.proc.image_processor(images=[frame_pil], return_tensors="pt")
        pv = d["pixel_values"].to(device=self.dev, dtype=self.m.dtype)
        grid = d["image_grid_thw"].to(device=self.dev)
        with torch.no_grad():
            ve = self.m.visual(pv, grid_thw=grid)
        ntok = ve.shape[0]
        dummy = torch.full((1, ntok), self.config.image_token_id, dtype=torch.long, device=self.dev)
        pos, _ = self.m.get_rope_index(input_ids=dummy, image_grid_thw=grid, video_grid_thw=None,
                                       attention_mask=torch.ones(1, ntok, dtype=torch.long, device=self.dev))
        pos = pos.clone(); mt = int(pos[0].max().item()); pos[0] += frame_idx * (mt + 1)
        return ve, pos, ntok

    def add_frame(self, ve, pos, ntok):
        cur = self.block_P.get_seq_length(0)
        cp = torch.arange(cur, cur + ntok, device=self.dev)
        am = torch.ones(1, cur + ntok, dtype=torch.bool, device=self.dev)
        pkv = self.MRopeView(cache_structure=[[self.block_P]], write_to=[self.block_P],
                             input_mask=None, position_ids=None,
                             override_length=cur + ntok, rotary_cache={})
        with torch.no_grad():
            self.m.model(inputs_embeds=ve.unsqueeze(0), position_ids=pos,
                         attention_mask=am, past_key_values=pkv, cache_position=cp, use_cache=True)

    def write_fresh(self):
        W = self.CacheBlock(config=self.config)
        cm = self.MRopeCM([[self.block_P, W]], write_to=[W])
        with torch.no_grad():
            self.m(**cm.get_input_kwargs(self._to2d(self.write_suf)))
        ids = []; last = self.SEP; nl = 0
        for _ in range(WRITE_MAX_TOKS):
            with torch.no_grad():
                out = self.m(**cm.get_input_kwargs(self._to2d([last])))
                p = torch.softmax(out.logits[0, -1] / TEMPERATURE_WRITE, dim=-1)
                last = int(torch.multinomial(p, 1).item())
            ids.append(last)
            if last in (self.EOS, self.IM_END): break
            nl = nl + 1 if last == self.SEP else 0
            if nl >= 2: break
        return self.tok.decode(ids, skip_special_tokens=True).strip(), len(ids)


# ── frame sampling + novelty ─────────────────────────────────────────────────
def extract_dense(video_path, n):
    os.makedirs(FRAME_DIR, exist_ok=True)
    for f in glob.glob(os.path.join(FRAME_DIR, "*.jpg")):
        os.remove(f)
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
                       capture_output=True, text=True)
    dur = float(r.stdout.strip())
    ts = [dur * (i + 0.5) / n for i in range(n)]
    frames = []
    for i, t in enumerate(ts):
        out = os.path.join(FRAME_DIR, f"f_{i:04d}.jpg")
        subprocess.run([FFMPEG, "-y", "-ss", str(t), "-i", str(video_path),
                        "-frames:v", "1", "-q:v", "2", out], check=True, capture_output=True)
        frames.append(out)
    print(f"Dense sample: {n} frames over {dur:.1f}s (every ~{dur/n:.1f}s)", flush=True)
    return frames, ts, dur


def frame_vec(ve):
    """Mean-pooled, L2-normalized frame embedding for the cosine novelty gate."""
    v = ve.float().mean(dim=0)
    return v / (v.norm() + 1e-8)


# ── prompts + ephemeral generation/gate on block_P ───────────────────────────
def tok(be, s):
    return be.tok(s, add_special_tokens=False)["input_ids"]

def gate_prompt(thoughts, last_caption):
    obs = "\n".join(f"- {t}" for t in thoughts[-6:])
    return ('<|vision_end|>\nObservations so far:\n' + obs +
            '\nAlready told to the user: "' + (last_caption or "nothing yet") + '".\n'
            "Has a NEW and significant event happened that has NOT yet been told to "
            "the user? Answer only Yes or No.\n<|im_end|>\n<|im_start|>assistant\n")

def write_prompt(thoughts, captions):
    obs = "\n".join(f"- {t}" for t in thoughts[-6:])
    said = "; ".join(captions) if captions else "nothing"
    return ('<|vision_end|>\nObservations:\n' + obs +
            "\nAlready said: " + said +
            "\nDescribe the latest NEW event to the user in one short sentence.\n"
            "<|im_end|>\n<|im_start|>assistant\n")

def gen_on_P(be, prompt_ids, max_toks, temp):
    """Generate text reading [block_P, ephemeral]; ephemeral discarded. Returns (text, n)."""
    E = be.CacheBlock(config=be.config)
    cm = be.MRopeCM([[be.block_P, E]], write_to=[E])
    with torch.no_grad():
        be.m(**cm.get_input_kwargs(be._to2d(prompt_ids)))
    ids = []; last = be.SEP; nl = 0
    for _ in range(max_toks):
        with torch.no_grad():
            out = be.m(**cm.get_input_kwargs(be._to2d([last])))
            p = torch.softmax(out.logits[0, -1] / temp, dim=-1)
            last = int(torch.multinomial(p, 1).item())
        ids.append(last)
        if last in (be.EOS, be.IM_END): break
        nl = nl + 1 if last == be.SEP else 0
        if nl >= 1 and len(ids) > 4: break
    return be.tok.decode(ids, skip_special_tokens=True).strip(), len(ids)

def gate_on_P(be, prompt_ids):
    """Ephemeral yes/no gate read at the answer slot. Returns (p_yes, p_no, n_q)."""
    E = be.CacheBlock(config=be.config)
    cm = be.MRopeCM([[be.block_P, E]], write_to=[E])
    with torch.no_grad():
        out = be.m(**cm.get_input_kwargs(be._to2d(prompt_ids)))
    p = torch.softmax(out.logits[0, -1], dim=-1)
    return p[be.YES].item(), p[be.NO].item(), len(prompt_ids)


# ── metrics ───────────────────────────────────────────────────────────────────
STOP = set("the a an is are was were be been being of in on at to and or with into over off it "
           "this that these those what when where which a video shows show showing person people "
           "currently happening clear sentence one frame frames".split())

def content_words(s):
    return set(w.strip(".,;:!?").lower() for w in s.split()
               if w.strip(".,;:!?").lower() not in STOP and len(w.strip(".,;:!?")) > 2)

def timeliness(emissions, gt_events, dur):
    per = []
    ems = sorted(emissions, key=lambda e: e["emit_time"])
    for ev in gt_events:
        ew = content_words(ev["sentence"]); match = None
        for e in ems:
            if len(content_words(e["text"]) & ew) >= 1:
                match = e; break
        if match is None:
            per.append({"event": ev["sentence"], "covered": False, "latency": None})
        else:
            per.append({"event": ev["sentence"], "covered": True,
                        "latency": match["emit_time"] - ev["start"], "emit_time": match["emit_time"]})
    cov = [p for p in per if p["covered"]]
    coverage = len(cov) / len(gt_events) if gt_events else 0.0
    mean_lat = sum(abs(p["latency"]) for p in cov) / len(cov) if cov else None
    return {"coverage": coverage, "mean_abs_latency_s": mean_lat, "per_event": per}

def compute_quality(hyp, references):
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    for pkg in ("punkt", "punkt_tab", "wordnet", "omw-1.4"):
        nltk.download(pkg, quiet=True)
    sm = SmoothingFunction().method1
    h = hyp.lower().split(); refs = [r.lower().split() for r in references]
    bleu = {}
    for n in range(1, 5):
        w = tuple([1.0/n]*n + [0.0]*(4-n))
        bleu[f"bleu{n}"] = sentence_bleu(refs, h, weights=w, smoothing_function=sm) if h else 0.0
    meteor = max(meteor_score([r], h) for r in refs) if h else 0.0
    def lcs(a, b):
        dp = [0]*(len(b)+1)
        for x in a:
            prev = 0
            for j, y in enumerate(b):
                tmp = dp[j+1]; dp[j+1] = prev+1 if x == y else max(dp[j+1], dp[j]); prev = tmp
        return dp[len(b)]
    def rl(hh, r):
        l = lcs(hh, r)
        if not l: return 0.0
        p, rc = l/len(hh), l/len(r); return 2*p*rc/(p+rc)
    rouge = max(rl(h, r) for r in refs) if h else 0.0
    return {**bleu, "meteor": meteor, "rouge_l": rouge}


# ── one condition (input gate x output gate; all conditions stream) ───────────
def run_condition(model, processor, tokenizer, pils, ts, dur, device,
                  input_on, output_gated, delta, max_stream, tag, temp):
    print(f"\n── {tag}  (input_gate={'ON' if input_on else 'OFF'}, "
          f"output={'GATED' if output_gated else 'STREAM-ALL'}) ──", flush=True)
    be = Backend(model, processor, tokenizer, device)
    prototype = None
    admitted, thoughts, captions, emissions, checks = [], [], [], [], []
    think_toks = write_toks = gate_toks = 0
    t0 = time.time()

    for i, pil in enumerate(pils):
        ve, pos, ntok = be.encode_frame(pil, len(admitted))
        vec = frame_vec(ve)
        cos = float(torch.dot(vec, prototype).item()) if prototype is not None else -1.0
        if input_on and prototype is not None and cos >= delta:
            continue                                          # input gate: drop redundant frame
        be.add_frame(ve, pos, ntok)
        admitted.append((i, ts[i])); prototype = vec

        note, nt = gen_on_P(be, tok(be, THINK_PROMPT), THINK_TOKS, temp); think_toks += nt
        if note:
            thoughts.append(note)

        if output_gated:
            p_yes, p_no, nq = gate_on_P(be, tok(be, gate_prompt(thoughts, captions[-1] if captions else None)))
            gate_toks += nq
            fire = (p_yes > p_no) and (len(emissions) < max_stream)
            checks.append({"t": ts[i], "p_yes": p_yes, "p_no": p_no, "fired": fire})
        else:
            fire = len(emissions) < max_stream
            p_yes = p_no = None

        if fire:
            cap, wt = gen_on_P(be, tok(be, write_prompt(thoughts, captions)), WRITE_TOKS, temp); write_toks += wt
            if cap:
                captions.append(cap)
                emissions.append({"t": ts[i], "emit_time": ts[i], "text": cap, "p_yes": p_yes, "p_no": p_no})

    elapsed = time.time() - t0
    seen, uniq = set(), []
    for e in emissions:
        if e["text"] not in seen:
            seen.add(e["text"]); uniq.append(e["text"])
    text = " ".join(uniq)
    print(f"  admitted {len(admitted)}/{len(pils)} · captions {len(emissions)} · "
          f"gen_toks {think_toks+write_toks+gate_toks} · {elapsed:.1f}s", flush=True)
    for e in emissions:
        print(f"    t={e['t']:.1f}s: {e['text']}", flush=True)
    return {
        "tag": tag, "input_on": input_on, "output_gated": output_gated,
        "n_admitted": len(admitted), "n_total": len(pils), "kv_tokens": len(admitted) * TOKENS_PER_FRAME,
        "n_emissions": len(emissions), "emissions": emissions, "thoughts": thoughts, "checks": checks,
        "think_tokens": think_toks, "gate_tokens": gate_toks, "write_tokens": write_toks,
        "total_gen_tokens": think_toks + write_toks + gate_toks, "time_s": elapsed, "text": text,
    }


def main():
    ap = argparse.ArgumentParser(description="Standalone proactive streaming runner (2x2 ablation).")
    ap.add_argument("--mode", default="all",
                    help="comma list of {baseline,input_only,output_only,both} or 'all'")
    ap.add_argument("--video", default=DEFAULT_VIDEO)
    ap.add_argument("--gt", default=DEFAULT_GT)
    ap.add_argument("--frames", type=int, default=30, help="dense frames sampled from the clip")
    ap.add_argument("--delta", type=float, default=DELTA, help="input-gate cosine threshold (admit if cos<delta)")
    ap.add_argument("--max-stream", type=int, default=None, help="cap on captions (default = --frames)")
    ap.add_argument("--temp", type=float, default=TEMP, help="sampling temperature (lower = more stable)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed, applied before each condition")
    ap.add_argument("--out", default=str(HERE / "results.jsonl"))
    args = ap.parse_args()

    modes = list(MODES) if args.mode == "all" else [m.strip() for m in args.mode.split(",")]
    bad = [m for m in modes if m not in MODES]
    if bad:
        ap.error(f"unknown mode(s): {bad}. choose from {list(MODES)} or 'all'")
    max_stream = args.max_stream or args.frames

    gt = json.load(open(args.gt))
    refs = [g["sentence"] for g in gt] + [" ".join(g["sentence"] for g in gt)]
    print("GT events:", flush=True)
    for g in gt:
        print(f"  [{g['start']:.0f}-{g['end']:.0f}s] {g['sentence']}", flush=True)

    device = "cuda"
    model, processor, tokenizer = load_model()
    frames, ts, dur = extract_dense(args.video, args.frames)
    pils = [Image.open(f).convert("RGB") for f in frames]

    res = {}
    for m in modes:
        torch.manual_seed(args.seed)
        inp, out = MODES[m]
        res[m] = run_condition(model, processor, tokenizer, pils, ts, dur, device,
                               inp, out, args.delta, max_stream, m, args.temp)
        res[m]["quality"] = compute_quality(res[m]["text"], refs)
        res[m]["timeliness"] = timeliness(res[m]["emissions"], gt, dur)
        torch.cuda.empty_cache()

    order = [m for m in MODES if m in res]
    def row(label, fn):
        print(f"{label:<18}" + "".join(f"{fn(res[m]):>15}" for m in order), flush=True)
    print("\n" + "=" * (18 + 15 * len(order)), flush=True)
    print(f"{'config (seed=%d)'%args.seed:<18}" + "".join(f"{m:>15}" for m in order), flush=True)
    print("=" * (18 + 15 * len(order)), flush=True)
    row("input gate",    lambda r: "ON" if r["input_on"] else "off")
    row("output gate",   lambda r: "ON" if r["output_gated"] else "off")
    row("frames in mem", lambda r: f"{r['n_admitted']}/{r['n_total']}")
    row("KV tokens",     lambda r: f"{r['kv_tokens']:,}")
    row("captions",      lambda r: str(r["n_emissions"]))
    row("gen tokens",    lambda r: str(r["total_gen_tokens"]))
    row("time (s)",      lambda r: f"{r['time_s']:.1f}")
    row("coverage",      lambda r: f"{r['timeliness']['coverage']*100:.0f}%")
    for mk in ["bleu1", "meteor", "rouge_l"]:
        row(mk, lambda r, k=mk: f"{r['quality'][k]:.4f}")
    print("=" * (18 + 15 * len(order)), flush=True)

    json.dump({"args": vars(args), "dur": dur, "results": res}, open(args.out, "w"))
    print(f"\nSaved {args.out}", flush=True)


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc(); sys.exit(1)
