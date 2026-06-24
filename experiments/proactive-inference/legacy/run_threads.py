#!/usr/bin/env python3
"""
run_threads.py — REAL single-GPU parallelism via threads + 2 CUDA streams.

The vision encoder produces embeddings and NEVER touches the KV cache; only the
LM thread reads/writes block P/T/W. So we run them concurrently with ZERO
cross-thread cache contention (single-writer/multi-reader stays intact):

  VISION thread (stream s_vis):  encode frame i+1, i+2, ... ahead  (cache-free)
        │  queue of ready (ve,pos) + a CUDA event marking "encode done"
        ▼
  LM thread   (stream s_lm):     input-gate -> add_frame(P) -> think(T) -> gate -> write(W)

A CUDA event makes the LM stream wait for each frame's encode before using it.
--overlap (default) runs the two threads concurrently; --sequential encodes inline
(no second thread) so you can A/B the wall-clock.

NOTE: on one 7B model the heavy LM matmuls saturate the GPU, so the gain is the
overlap of the ViT encode with the LM work — real but modest. True parallel of the
LM itself needs a smaller 2nd model (0.5B writer) / MPS / multi-GPU.

Usage:
  python run_threads.py --frames 16 --overlap
  python run_threads.py --frames 16 --sequential
"""
import os, sys, json, time, threading, queue, argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as R
import torch
from PIL import Image


def process_frame_lm(be, ve, pos, ntok, vec, state, thoughts, captions, emissions,
                     t_i, delta, temp, log):
    """All cache-touching work for one frame — runs ONLY in the LM thread."""
    cos = float(torch.dot(vec, state["prototype"]).item()) if state["prototype"] is not None else -1.0
    if state["prototype"] is not None and cos >= delta:
        return  # input gate: redundant -> drop (frame's KV never enters block P)
    be.add_frame(ve, pos, ntok)                       # WRITE block P (LM thread only)
    state["prototype"] = vec; state["n_admitted"] += 1
    note, _ = R.gen_on_P(be, R.tok(be, R.THINK_PROMPT), R.THINK_TOKS, temp)   # think -> block T
    if note: thoughts.append(note)
    last = captions[-1] if captions else None
    p_yes, p_no, _ = R.gate_on_P(be, R.tok(be, R.gate_prompt(thoughts, last)))  # output gate
    if p_yes > p_no:
        cap, _ = R.gen_on_P(be, R.tok(be, R.write_prompt(thoughts, captions)), R.WRITE_TOKS, temp)
        if cap:
            captions.append(cap); emissions.append({"t": t_i, "emit_time": t_i, "text": cap})
            log(f"  EMIT @t={t_i:.1f}s: {cap[:70]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=R.DEFAULT_VIDEO)
    ap.add_argument("--gt", default=R.DEFAULT_GT)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--delta", type=float, default=R.DELTA)
    ap.add_argument("--temp", type=float, default=R.TEMP)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefetch", type=int, default=3, help="how many frames the vision thread runs ahead")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--overlap", dest="overlap", action="store_true")
    g.add_argument("--sequential", dest="overlap", action="store_false")
    ap.set_defaults(overlap=True)
    ap.add_argument("--out", default=str(HERE / "results_threads.json"))
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    device = "cuda"
    model, processor, tokenizer = R.load_model()
    frames, ts, dur = R.extract_dense(args.video, args.frames)
    pils = [Image.open(f).convert("RGB") for f in frames]
    be = R.Backend(model, processor, tokenizer, device)

    state = {"prototype": None, "n_admitted": 0}
    thoughts, captions, emissions = [], [], []
    def log(m): print(m, flush=True)

    torch.cuda.synchronize(); t0 = time.time()

    if not args.overlap:
        # ── SEQUENTIAL: encode inline then process (no 2nd thread/stream) ──
        log(f"[SEQUENTIAL] {len(pils)} frames")
        for i, pil in enumerate(pils):
            ve, pos, ntok = be.encode_frame(pil, i)
            vec = R.frame_vec(ve)
            process_frame_lm(be, ve, pos, ntok, vec, state, thoughts, captions, emissions,
                             ts[i], args.delta, args.temp, log)
    else:
        # ── OVERLAP: vision thread (s_vis) encodes ahead; LM thread (default) processes ──
        log(f"[OVERLAP] {len(pils)} frames · vision∥LM on 2 streams · prefetch={args.prefetch}")
        s_vis = torch.cuda.Stream()
        q = queue.Queue(maxsize=args.prefetch)

        def vision_worker():
            for i, pil in enumerate(pils):
                with torch.cuda.stream(s_vis):
                    ve, pos, ntok = be.encode_frame(pil, i)   # cache-free; on its own stream
                    ev = torch.cuda.Event(); ev.record(s_vis)
                q.put((i, ts[i], ve, pos, ntok, ev))          # blocks if LM is >prefetch behind
            q.put(None)                                       # sentinel

        vt = threading.Thread(target=vision_worker, daemon=True); vt.start()
        while True:
            item = q.get()
            if item is None: break
            i, t_i, ve, pos, ntok, ev = item
            torch.cuda.current_stream().wait_event(ev)        # LM waits for THIS frame's encode
            vec = R.frame_vec(ve)
            process_frame_lm(be, ve, pos, ntok, vec, state, thoughts, captions, emissions,
                             t_i, args.delta, args.temp, log)
        vt.join()

    torch.cuda.synchronize(); wall = time.time() - t0

    gt = json.load(open(args.gt))
    refs = [g["sentence"] for g in gt] + [" ".join(g["sentence"] for g in gt)]
    text = " ".join(dict.fromkeys(e["text"] for e in emissions))
    qm = R.compute_quality(text, refs); tl = R.timeliness(emissions, gt, dur)
    print("\n" + "=" * 60, flush=True)
    print(f"MODE: {'OVERLAP (threads+2 streams)' if args.overlap else 'SEQUENTIAL'}", flush=True)
    print(f"wall time: {wall:.2f}s · admitted {state['n_admitted']}/{len(pils)} · emissions {len(emissions)}", flush=True)
    print(f"METEOR {qm['meteor']:.3f} · BLEU-1 {qm['bleu1']:.3f} · coverage {tl['coverage']*100:.0f}%", flush=True)
    print("=" * 60, flush=True)
    json.dump({"args": vars(args), "wall_s": wall, "n_admitted": state["n_admitted"],
               "emissions": emissions, "quality": qm, "timeliness": tl}, open(args.out, "w"))


if __name__ == "__main__":
    main()
