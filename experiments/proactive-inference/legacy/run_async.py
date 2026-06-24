#!/usr/bin/env python3
"""
run_async.py — ASYNC version of the proactive orchestrator-writer (full system).

Three concurrent asyncio tasks sharing ONE model + the shared KV cache, talking
ONLY through the cache + two queues (never lockstep):

  INPUT  task: real-time frame arrival -> encode -> input gate -> write block P -> admit_q
  ORCH   task: admit_q -> think (block T) -> output gate -> write_q
  WRITER task: write_q -> generate caption (block W) -> emit

GPU is serialized by an asyncio.Lock + run_in_executor (one op at a time, cache-safe),
but the event loop stays live during GPU ops, so control flow is genuinely concurrent
and the real-time clock / queues keep moving. Under real-time load the INPUT task DROPS
frames (proactiveness under pressure) — impossible in the sequential loop.

NOTE: one GPU + one model => concurrency/decoupling + streaming behavior, NOT parallel
GPU speedup (that needs multi-GPU / MPS / a separate tiny writer model).

Usage:
  python run_async.py --frames 20 --speed 8 --queue 6
"""
import os, sys, json, time, asyncio, argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as R                                   # the standalone runner
import torch
from PIL import Image

T0 = time.time()
def log(tag, msg):
    print(f"[{time.time()-T0:6.2f}s][{tag:<6}] {msg}", flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=R.DEFAULT_VIDEO)
    ap.add_argument("--gt", default=R.DEFAULT_GT)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--delta", type=float, default=R.DELTA)
    ap.add_argument("--temp", type=float, default=R.TEMP)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=8.0,
                    help="real-time factor: frame i arrives at t_i/speed seconds (higher=faster arrival)")
    ap.add_argument("--queue", type=int, default=6, help="admission queue size; overflow => drop under load")
    ap.add_argument("--out", default=str(HERE / "results_async.json"))
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    device = "cuda"
    model, processor, tokenizer = R.load_model()
    frames, ts, dur = R.extract_dense(args.video, args.frames)
    pils = [Image.open(f).convert("RGB") for f in frames]
    be = R.Backend(model, processor, tokenizer, device)

    loop = asyncio.get_event_loop()
    gpu_lock = asyncio.Lock()
    async def gpu(fn):                            # serialize GPU; keep event loop alive (thread)
        async with gpu_lock:
            return await loop.run_in_executor(None, fn)

    # shared state (single process)
    state = {"prototype": None, "n_admitted": 0}
    thoughts, captions, emissions, events = [], [], [], []
    admit_q = asyncio.Queue(maxsize=args.queue)
    write_q = asyncio.Queue()

    # ── INPUT task ────────────────────────────────────────────────────────────
    async def input_task():
        for i, pil in enumerate(pils):
            arrive = ts[i] / args.speed                       # real-time arrival
            wait = arrive - (time.time() - T0_stream[0])
            if wait > 0:
                await asyncio.sleep(wait)
            ve, pos, ntok = await gpu(lambda p=pil, n=state["n_admitted"]: be.encode_frame(p, n))
            vec = R.frame_vec(ve)
            cos = float(torch.dot(vec, state["prototype"]).item()) if state["prototype"] is not None else -1.0
            if state["prototype"] is not None and cos >= args.delta:
                log("INPUT", f"frame {i} (t={ts[i]:.1f}s) redundant cos={cos:.2f} -> drop")
                continue
            await gpu(lambda ve=ve, pos=pos, ntok=ntok: be.add_frame(ve, pos, ntok))
            state["prototype"] = vec; state["n_admitted"] += 1
            ev = {"i": i, "t": ts[i]}
            try:
                admit_q.put_nowait(ev)
                log("INPUT", f"frame {i} (t={ts[i]:.1f}s) ADMITTED -> queue({admit_q.qsize()})")
            except asyncio.QueueFull:
                log("INPUT", f"frame {i} (t={ts[i]:.1f}s) admitted but QUEUE FULL -> DROPPED under load")
        await admit_q.put(None)                               # sentinel

    # ── ORCHESTRATOR task ─────────────────────────────────────────────────────
    async def orch_task():
        while True:
            ev = await admit_q.get()
            if ev is None:
                await write_q.put(None); break
            note, _ = await gpu(lambda: R.gen_on_P(be, R.tok(be, R.THINK_PROMPT), R.THINK_TOKS, args.temp))
            if note:
                thoughts.append(note)
            log("ORCH", f"event t={ev['t']:.1f}s thought=\"{note[:50]}\"")
            last = captions[-1] if captions else None
            p_yes, p_no, _ = await gpu(lambda: R.gate_on_P(be, R.tok(be, R.gate_prompt(thoughts, last))))
            events.append({"t": ev["t"], "p_yes": p_yes, "p_no": p_no, "fired": p_yes > p_no})
            if p_yes > p_no:
                log("ORCH", f"gate p_yes={p_yes:.2f}>p_no={p_no:.2f} -> WRITE @ t={ev['t']:.1f}s")
                await write_q.put({"t": ev["t"]})
            else:
                log("ORCH", f"gate p_yes={p_yes:.2f}<=p_no={p_no:.2f} -> stay silent")

    # ── WRITER task ───────────────────────────────────────────────────────────
    async def writer_task():
        while True:
            req = await write_q.get()
            if req is None:
                break
            cap, _ = await gpu(lambda: R.gen_on_P(be, R.tok(be, R.write_prompt(thoughts, captions)), R.WRITE_TOKS, args.temp))
            if cap:
                captions.append(cap)
                emissions.append({"t": req["t"], "emit_time": req["t"], "text": cap})
                log("WRITER", f"@t={req['t']:.1f}s  EMIT: {cap[:70]}")

    T0_stream = [time.time() - T0]                            # stream clock origin
    log("MAIN", f"starting 3 async tasks · {len(pils)} frames · speed={args.speed}x · queue={args.queue}")
    await asyncio.gather(input_task(), orch_task(), writer_task())

    # ── results ───────────────────────────────────────────────────────────────
    gt = json.load(open(args.gt))
    refs = [g["sentence"] for g in gt] + [" ".join(g["sentence"] for g in gt)]
    text = " ".join(dict.fromkeys(e["text"] for e in emissions))
    q = R.compute_quality(text, refs)
    tl = R.timeliness(emissions, gt, dur)
    print("\n" + "=" * 64, flush=True)
    print(f"ASYNC full-system run · admitted {state['n_admitted']}/{len(pils)} · "
          f"emissions {len(emissions)} · gate-fired {sum(e['fired'] for e in events)}/{len(events)}", flush=True)
    print(f"METEOR {q['meteor']:.3f} · BLEU-1 {q['bleu1']:.3f} · ROUGE-L {q['rouge_l']:.3f} · "
          f"coverage {tl['coverage']*100:.0f}%", flush=True)
    print("=" * 64, flush=True)
    json.dump({"args": vars(args), "emissions": emissions, "events": events,
               "quality": q, "timeliness": tl}, open(args.out, "w"))
    print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
