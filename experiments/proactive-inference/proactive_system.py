#!/usr/bin/env python3
"""
proactive_system.py — unified streaming video understanding: captioning + Video QA.

The ORCHESTRATOR (the model itself) decides everything. There are NO thresholds
set by us. Streaming is forward-only.

INPUT MODES (--input-mode):
  adaptive  RECOMMENDED. For each arriving frame the model gives a keep-probability
            (p_yes). We keep the frame only if p_yes beats the model's OWN running
            average p_yes so far — i.e. "is this more interesting than what I've been
            seeing lately?".  The bar is the model's own behaviour; no constant.
  probe     Keep if p_yes > p_no (absolute self-calibration). Honest but very
            selective — the bare per-frame probe is biased to "no".
  even      Keep every arriving frame up to --cap.                     [baseline]
  novelty   Keep if cosine-to-last < --delta (the old rule-based way).  [reference]

TASK (--mode): caption | vqa | both.  ONE shared-KV memory (block P) is built once
and reused for captions and for every question.

Configuration: defaults live in config.yaml next to this file. CLI flags override
the config file, which overrides the built-in defaults.  Use --config <path> to
point at a different file.

Examples:
  python proactive_system.py                       # uses config.yaml
  python proactive_system.py --mode vqa --input-mode probe \
       --questions "What liquid is poured over the chicken?"
  python proactive_system.py --config experiments/fast.yaml
"""
import os, sys, json, time, subprocess, argparse
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as R
import torch
from PIL import Image

FFMPEG = str(Path.home() / "miniforge3" / "bin" / "ffmpeg")
TMP = "/tmp/psys_frames"; os.makedirs(TMP, exist_ok=True)

# ── video helpers ────────────────────────────────────────────────────────────
def duration(v):
    r = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
                        "-of","default=noprint_wrappers=1:nokey=1", v], capture_output=True, text=True)
    return float(r.stdout.strip())

def get_frame(v, t):
    out = os.path.join(TMP, f"f_{t:08.2f}.jpg")
    subprocess.run([FFMPEG,"-y","-ss",f"{t:.2f}","-i",v,"-frames:v","1","-q:v","2",out],
                   capture_output=True, timeout=60)
    return Image.open(out).convert("RGB")

def get_chunk(v, t, n, cd, dur):
    return [get_frame(v, min(t + j*cd, dur - 0.1)) for j in range(n)]

def encode_chunk(be, frames, chunk_idx):
    """Encode consecutive frames as a VIDEO clip (Qwen2-VL temporal patches -> motion)."""
    d = be.proc.image_processor(images=None, videos=[frames], return_tensors="pt")
    pv = d["pixel_values_videos"].to(device=be.dev, dtype=be.m.dtype)
    grid = d["video_grid_thw"].to(device=be.dev)
    with torch.no_grad():
        ve = be.m.visual(pv, grid_thw=grid)
    ntok = ve.shape[0]
    vid_tok = getattr(be.config, "video_token_id", be.config.image_token_id)
    dummy = torch.full((1, ntok), vid_tok, dtype=torch.long, device=be.dev)
    pos, _ = be.m.get_rope_index(input_ids=dummy, image_grid_thw=None, video_grid_thw=grid,
                                 attention_mask=torch.ones(1, ntok, dtype=torch.long, device=be.dev))
    pos = pos.clone(); mt = int(pos[0].max().item()); pos[0] += chunk_idx * (mt + 1)
    return ve, pos, ntok

# ── orchestrator-driven prompts (the model decides) ──────────────────────────
def keep_prompt(thoughts, q):
    """Memory-aware: model sees a summary of stored content + the new frame, decides keep."""
    notes = ("Things I have already noted:\n- " + "\n- ".join(thoughts[-6:]) + "\n") if thoughts else "Nothing noted yet.\n"
    goal  = (f"(I am trying to answer: '{q}'.)\n" if q else "")
    return ("<|vision_end|>\n" + notes + goal +
            "Compared to what I have already noted, has the scene CHANGED or is a NEW action/step "
            "happening in this latest frame that I have not recorded yet? Answer only Yes or No.\n"
            "<|im_end|>\n<|im_start|>assistant\n")

def vqa_prompt(thoughts, q):
    notes = ("Notes:\n- " + "\n- ".join(thoughts[-8:]) + "\n") if thoughts else ""
    return ("<|vision_end|>\n" + notes + "Question: " + q +
            "\nAnswer the question about the video in one short sentence.\n"
            "<|im_end|>\n<|im_start|>assistant\n")

def keep_probe(scorer, ve, pos, ntok, thoughts, q):
    """Model's keep decision for a candidate frame (memory-aware). Returns (p_yes, p_no)."""
    scorer._fresh_P(); scorer.add_frame(ve, pos, ntok)            # header + this candidate
    p_yes, p_no, _ = R.gate_on_P(scorer, R.tok(scorer, keep_prompt(thoughts, q)))
    return p_yes, p_no

DEFAULT_Q = ["What dish is being prepared?",
             "What liquid is poured over the chicken?",
             "What ingredients are combined for the brine?"]

# ── config ───────────────────────────────────────────────────────────────────
def load_config(path):
    p = Path(path) if path else None
    if not p or not p.exists():
        return {}
    txt = p.read_text()
    if p.suffix in (".yaml", ".yml"):
        import yaml
        return yaml.safe_load(txt) or {}
    return json.loads(txt)

def build_args():
    # 1) pre-parse --config so we can fold the file into argparse defaults
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(HERE / "config.yaml"),
                     help="YAML/JSON config file (default: config.yaml beside this script)")
    known, _ = pre.parse_known_args()
    cfg = load_config(known.config)

    # 2) built-in defaults, then overlay the config file
    defaults = dict(mode="both", input_mode="adaptive", output_gate="on", video=None, gt=None,
                    questions=None, base_interval=2.5, cap=40, chunk=4, chunk_dt=0.4,
                    delta=None, temp=0.7, qa_temp=0.3, seed=0, out=None)
    defaults.update({k.replace("-", "_"): v for k, v in cfg.items() if k != "config"})

    # 3) full parser; CLI flags override the config-file values
    ap = argparse.ArgumentParser(parents=[pre], description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["caption", "vqa", "both"])
    ap.add_argument("--input-mode", choices=["even", "novelty", "probe", "adaptive"])
    ap.add_argument("--output-gate", choices=["on", "off"],
                    help="on = model decides when to speak (proactive); off = emit on every kept frame")
    ap.add_argument("--video"); ap.add_argument("--gt")
    ap.add_argument("--questions", help='pipe-separated, e.g. "Q1|Q2|Q3"')
    ap.add_argument("--base-interval", type=float, help="seconds between arriving frames")
    ap.add_argument("--cap", type=int, help="max frames admitted into memory")
    ap.add_argument("--chunk", type=int, help="frames per motion clip (1 = single still)")
    ap.add_argument("--chunk-dt", type=float, help="seconds between frames within a clip")
    ap.add_argument("--delta", type=float, help="(novelty mode only) cosine threshold")
    ap.add_argument("--temp", type=float, help="caption/think temperature")
    ap.add_argument("--qa-temp", type=float, help="QA temperature")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out", help="output JSON path")
    ap.set_defaults(**defaults)
    args = ap.parse_args()

    # 4) resolve "use the built-in" fallbacks
    if args.video is None: args.video = R.DEFAULT_VIDEO
    if args.gt    is None: args.gt    = R.DEFAULT_GT
    if args.delta is None: args.delta = R.DELTA
    if args.out   is None: args.out   = str(HERE / "results" / "run.json")
    return args

def main():
    args = build_args()
    torch.manual_seed(args.seed)
    os.makedirs(Path(args.out).parent, exist_ok=True)

    questions = (DEFAULT_Q if not args.questions else
                 [q.strip() for q in args.questions.split("|") if q.strip()]) if args.mode in ("vqa","both") else []
    q_for_input = questions[0] if questions else None
    want_cap = args.mode in ("caption","both")

    model, processor, tokenizer = R.load_model()
    main_be = R.Backend(model, processor, tokenizer, "cuda")
    scorer  = R.Backend(model, processor, tokenizer, "cuda") if args.input_mode in ("probe","adaptive") else None
    dur = duration(args.video)

    print(f"\n[STREAMING · orchestrator-controlled] input-mode={args.input_mode} "
          f"output-gate={args.output_gate} base_interval={args.base_interval}s cap={args.cap} chunk={args.chunk}", flush=True)
    print(f"{'t(s)':>6} {'decision (model)':>26}  keep", flush=True); print("-"*46, flush=True)

    t = 0.0; admitted = 0; prototype = None
    thoughts = []; captions = []; emissions = []; sched = []; pyes_hist = []
    t_ing = time.time()
    while t < dur and admitted < args.cap:
        if args.chunk > 1:
            frames = get_chunk(args.video, t, args.chunk, args.chunk_dt, dur)
            ve, pos, ntok = encode_chunk(main_be, frames, admitted)
        else:
            ve, pos, ntok = main_be.encode_frame(get_frame(args.video, t), admitted)

        keep = False; info = ""
        if args.input_mode == "even":
            keep = True; info = "always"
        elif args.input_mode == "novelty":
            vec = R.frame_vec(ve)
            cos = float(torch.dot(vec, prototype).item()) if prototype is not None else -1.0
            keep = (prototype is None) or (cos < args.delta); info = f"cos={cos:.2f}<d{args.delta:.2f}"
            prototype = vec if keep else prototype
        elif args.input_mode in ("probe", "adaptive"):
            p_yes, p_no = keep_probe(scorer, ve, pos, ntok, thoughts, q_for_input)
            if args.input_mode == "probe":
                keep = p_yes > p_no; info = f"yes={p_yes:.2f} no={p_no:.2f}"          # absolute
            else:
                base = sum(pyes_hist)/len(pyes_hist) if pyes_hist else -1.0
                keep = (not pyes_hist) or (p_yes > base)                              # relative to own mean
                info = f"yes={p_yes:.2f} vs avg={max(base,0):.2f}"
            pyes_hist.append(p_yes)

        print(f"{t:6.1f} {info:>26}  {'KEEP' if keep else 'drop'}", flush=True)
        sched.append({"t": round(t,1), "keep": keep, "info": info})

        if keep:
            main_be.add_frame(ve, pos, ntok); admitted += 1
            note, _ = R.gen_on_P(main_be, R.tok(main_be, R.THINK_PROMPT), R.THINK_TOKS, args.temp)
            if note: thoughts.append(note)
            if want_cap:
                last = captions[-1] if captions else None
                if args.output_gate == "on":                              # OUTPUT GATE: model decides when to speak
                    py, pn, _ = R.gate_on_P(main_be, R.tok(main_be, R.gate_prompt(thoughts, last)))
                    emit = py > pn
                else:
                    emit = True                                            # stream-all: emit on every kept frame
                if emit:
                    cap, _ = R.gen_on_P(main_be, R.tok(main_be, R.write_prompt(thoughts, captions)), R.WRITE_TOKS, args.temp)
                    if cap: captions.append(cap); emissions.append({"t": round(t,1), "text": cap})

        t += args.base_interval
    ingest_s = time.time() - t_ing
    print("-"*46, flush=True)
    print(f"[stream] {len(sched)} frames seen · kept {admitted} ({admitted*1196} KV tok) · {ingest_s:.1f}s", flush=True)

    if want_cap:
        print("\n=== CAPTIONS ===", flush=True)
        for e in emissions: print(f"  t={e['t']:.1f}s: {e['text']}", flush=True)

    qa = []
    if questions:
        print("\n=== VIDEO QA (reuses cached block P) ===", flush=True)
        for q in questions:
            ans, _ = R.gen_on_P(main_be, R.tok(main_be, vqa_prompt(thoughts, q)), 60, args.qa_temp)
            qa.append({"q": q, "a": ans}); print(f"  Q: {q}\n  A: {ans}\n", flush=True)

    # ── metrics ──────────────────────────────────────────────────────────────
    metrics = {"seen": len(sched), "kept": admitted, "kv_tokens": admitted * 1196,
               "captions": len(emissions), "ingest_s": round(ingest_s, 1)}
    if want_cap and os.path.exists(args.gt):
        try:
            gt = json.load(open(args.gt))
            refs = [g["sentence"] for g in gt] + [" ".join(g["sentence"] for g in gt)]
            txt = " ".join(dict.fromkeys(e["text"] for e in emissions))
            if txt:
                q = R.compute_quality(txt, refs)
                metrics.update({k: round(float(v), 3) for k, v in q.items()})
            tl = R.timeliness([{"emit_time": e["t"], "text": e["text"]} for e in emissions], gt, dur)
            metrics["coverage"] = round(tl["coverage"], 3)
            metrics["mean_abs_latency_s"] = (round(tl["mean_abs_latency_s"], 2)
                                             if tl["mean_abs_latency_s"] is not None else None)
        except Exception as e:
            print(f"[metrics] skipped: {e}", flush=True)

    print("="*60, flush=True)
    print(f"mode={args.mode} input={args.input_mode} output-gate={args.output_gate}", flush=True)
    for k, v in metrics.items():
        print(f"  {k:>20} : {v}", flush=True)
    json.dump({"args": vars(args), "schedule": sched, "captions": emissions, "qa": qa,
               "metrics": metrics}, open(args.out, "w"))
    print(f"Saved {args.out}", flush=True)

if __name__ == "__main__":
    main()
