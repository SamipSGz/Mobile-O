#!/usr/bin/env python3
"""
run_qcond.py — proactive input via QUERY-CONDITIONED admission (the "what to keep" axis).

Densely scan M candidate frames (full coverage), score each by relevance to the
question, then KEEP only the top-K (a tight budget). Compare to keeping K
evenly-spaced frames. Tests whether *intelligent selection* beats *even sampling*
when the budget is tight (where proactive input should matter).

Usage:
  python run_qcond.py --question "What liquid is poured over the chicken?" --candidates 24 --budget 5
"""
import os, sys, json, time, argparse
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as R
import torch
from PIL import Image

def rel_prompt(q):
    return ("<|vision_end|>\nQuestion: " + q +
            "\nDoes THIS frame contain information useful for answering the question? "
            "Answer only Yes or No.\n<|im_end|>\n<|im_start|>assistant\n")

def vqa_prompt(q):
    return ("<|vision_end|>\nQuestion: " + q +
            "\nAnswer the question about the video in one short sentence.\n"
            "<|im_end|>\n<|im_start|>assistant\n")

def build_and_answer(model, processor, tokenizer, frames_pos, q, qa_temp):
    """Fresh block P with the given (ve,pos,ntok) frames in time order; answer q."""
    be = R.Backend(model, processor, tokenizer, "cuda")
    for k,(ve,pos,ntok) in enumerate(frames_pos):
        # re-offset temporal band to admission index k
        be.add_frame(ve, pos, ntok)
    ans,_ = R.gen_on_P(be, R.tok(be, vqa_prompt(q)), 60, qa_temp)
    return ans

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=R.DEFAULT_VIDEO)
    ap.add_argument("--question", default="What liquid is poured over the chicken?")
    ap.add_argument("--candidates", type=int, default=24)
    ap.add_argument("--budget", type=int, default=5)
    ap.add_argument("--qa-temp", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "results_qcond.json"))
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    model, processor, tokenizer = R.load_model()
    frames, ts, dur = R.extract_dense(args.video, args.candidates)
    pils = [Image.open(f).convert("RGB") for f in frames]
    print(f"\nQ: {args.question}\ncandidates={args.candidates} budget={args.budget}\n", flush=True)

    # encode all candidates once; score each by relevance (isolated: header+frame)
    scorer = R.Backend(model, processor, tokenizer, "cuda")
    enc = []   # (ve,pos,ntok)
    scores = []
    vecs = []
    for i, pil in enumerate(pils):
        ve, pos, ntok = scorer.encode_frame(pil, 0)     # frame_idx 0 (isolated scoring)
        scorer._fresh_P()                                # reset P = header only
        scorer.add_frame(ve, pos, ntok)                  # P = header + this one frame
        p_yes, p_no, _ = R.gate_on_P(scorer, R.tok(scorer, rel_prompt(args.question)))
        enc.append((ve, pos, ntok)); scores.append(p_yes); vecs.append(R.frame_vec(ve))
        print(f"  t={ts[i]:5.1f}s  relevance={p_yes:.2f}", flush=True)

    order = sorted(range(len(pils)), key=lambda i: -scores[i])
    qcond_idx = sorted(order[:args.budget])                                  # top-K by relevance, time order
    even_idx  = sorted(range(0, len(pils), max(1, len(pils)//args.budget)))[:args.budget]  # K evenly
    print(f"\nQCOND keeps t = {[round(ts[i],1) for i in qcond_idx]}", flush=True)
    print(f"EVEN  keeps t = {[round(ts[i],1) for i in even_idx]}", flush=True)

    # relevance + DIVERSITY (MMR): relevant AND spread out
    def mmr(scores, vecs, K, lam=0.55):
        sel=[]; cand=list(range(len(scores)))
        while len(sel)<K and cand:
            best=None; bv=-1e9
            for i in cand:
                div=max((float(torch.dot(vecs[i],vecs[j]).item()) for j in sel), default=0.0)
                val=lam*scores[i]-(1-lam)*div
                if val>bv: bv=val; best=i
            sel.append(best); cand.remove(best)
        return sorted(sel)
    mmr_idx = mmr(scores, vecs, args.budget)
    print(f"MMR   keeps t = {[round(ts[i],1) for i in mmr_idx]}", flush=True)

    a_q = build_and_answer(model, processor, tokenizer, [enc[i] for i in qcond_idx], args.question, args.qa_temp)
    a_e = build_and_answer(model, processor, tokenizer, [enc[i] for i in even_idx], args.question, args.qa_temp)
    print("\n=== ANSWERS (budget = %d frames) ===" % args.budget, flush=True)
    print(f"  QUERY-CONDITIONED: {a_q}", flush=True)
    a_m = build_and_answer(model, processor, tokenizer, [enc[i] for i in mmr_idx], args.question, args.qa_temp)
    print(f"  EVEN-SPACED:       {a_e}", flush=True)
    print(f"  RELEVANCE+DIVERSE: {a_m}", flush=True)

    json.dump({"q": args.question, "ts": ts, "scores": scores,
               "qcond_idx": qcond_idx, "even_idx": even_idx,
               "ans_qcond": a_q, "ans_even": a_e}, open(args.out, "w"))
    print(f"\nSaved {args.out}", flush=True)

if __name__ == "__main__":
    main()
