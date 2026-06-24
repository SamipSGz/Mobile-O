#!/usr/bin/env python3
"""
run_vqa.py — Video QA + captioning on the shared-KV system.

Encode the video ONCE into block P, then:
  1) (captioning) stream the orchestrator+writer to produce timestamped captions,
  2) (QA) answer any number of questions, each as a CHEAP read over the cached P
     (+ the orchestrator's notes). Multi-question reuses the one encode → the
     shared-KV win: cost ≈ one encode + N small reads.

Usage:
  python run_vqa.py --frames 20 --caption \
      --questions "What dish is being made?|What is poured over the chicken?|What goes into the brine?"
"""
import os, sys, json, time, argparse
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as R
import torch
from PIL import Image

def vqa_prompt(thoughts, q):
    notes = ("Notes about the video:\n- " + "\n- ".join(thoughts[-8:]) + "\n") if thoughts else ""
    return ("<|vision_end|>\n" + notes +
            "Question: " + q + "\nAnswer the question about the video in one short sentence.\n"
            "<|im_end|>\n<|im_start|>assistant\n")

DEFAULT_Q = [
    "What dish is being prepared?",
    "What liquid is poured over the chicken?",
    "What ingredients are combined for the brine?",
    "What does the person do after pouring the brine?",
    "How many bowls are used?",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=R.DEFAULT_VIDEO)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--delta", type=float, default=R.DELTA)
    ap.add_argument("--temp", type=float, default=0.3)        # lower temp = more factual answers
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--caption", action="store_true", help="also produce streaming captions")
    ap.add_argument("--questions", default=None, help="'q1|q2|q3' (defaults to a built-in set)")
    ap.add_argument("--out", default=str(HERE / "results_vqa.json"))
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    questions = DEFAULT_Q if not args.questions else [q.strip() for q in args.questions.split("|") if q.strip()]

    model, processor, tokenizer = R.load_model()
    be = R.Backend(model, processor, tokenizer, "cuda")
    frames, ts, dur = R.extract_dense(args.video, args.frames)
    pils = [Image.open(f).convert("RGB") for f in frames]

    # ── 1) ENCODE VIDEO ONCE into block P (input gate) + build notes/captions ──
    print("\n[ingest] encoding video into shared memory (block P)...", flush=True)
    t_ing = time.time()
    prototype = None; admitted = 0; thoughts = []; captions = []; emissions = []
    for i, pil in enumerate(pils):
        ve, pos, ntok = be.encode_frame(pil, admitted)
        vec = R.frame_vec(ve)
        cos = float(torch.dot(vec, prototype).item()) if prototype is not None else -1.0
        if prototype is not None and cos >= args.delta:
            continue
        be.add_frame(ve, pos, ntok); prototype = vec; admitted += 1
        note, _ = R.gen_on_P(be, R.tok(be, R.THINK_PROMPT), R.THINK_TOKS, 0.7)
        if note: thoughts.append(note)
        if args.caption:
            last = captions[-1] if captions else None
            p_yes, p_no, _ = R.gate_on_P(be, R.tok(be, R.gate_prompt(thoughts, last)))
            if p_yes > p_no:
                cap, _ = R.gen_on_P(be, R.tok(be, R.write_prompt(thoughts, captions)), R.WRITE_TOKS, 0.7)
                if cap: captions.append(cap); emissions.append({"t": ts[i], "text": cap})
    ingest_s = time.time() - t_ing
    print(f"[ingest] done: admitted {admitted}/{len(pils)} frames into block P in {ingest_s:.1f}s "
          f"({admitted*1196} KV tokens cached)", flush=True)

    if args.caption:
        print("\n=== CAPTIONS (streaming) ===", flush=True)
        for e in emissions: print(f"  t={e['t']:.1f}s: {e['text']}", flush=True)

    # ── 2) QA: each question = cheap read over the SAME cached P (+ notes) ──
    print("\n=== VIDEO QA (each answer reuses the one encode) ===", flush=True)
    qa = []
    for q in questions:
        t_q = time.time()
        ans, _ = R.gen_on_P(be, R.tok(be, vqa_prompt(thoughts, q)), 60, args.temp)
        dt = time.time() - t_q
        qa.append({"q": q, "a": ans, "time_s": round(dt, 2)})
        print(f"  Q: {q}\n  A: {ans}   ({dt:.1f}s)\n", flush=True)

    avg_q = sum(x["time_s"] for x in qa) / len(qa) if qa else 0
    print("="*60, flush=True)
    print(f"shared-KV efficiency: 1 encode ({ingest_s:.1f}s) → {len(qa)} questions @ avg {avg_q:.1f}s each.", flush=True)
    print(f"(naive would re-feed all frames per question; here every Q reuses cached block P.)", flush=True)
    json.dump({"args": vars(args), "captions": emissions, "qa": qa,
               "ingest_s": ingest_s, "n_admitted": admitted}, open(args.out, "w"))
    print(f"Saved {args.out}", flush=True)

if __name__ == "__main__":
    main()
