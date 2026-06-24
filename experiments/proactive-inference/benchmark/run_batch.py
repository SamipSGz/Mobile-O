#!/usr/bin/env python3
"""
run_batch.py — run the 2x2 proactive ablation across many videos and aggregate.

Loads the model ONCE, then loops over the manifest from build_youcook2.py, running
each condition (baseline / input_only / output_only / both) per video via the SAME
code as run.py. Reports the aggregate 2x2 table as mean ± std across videos (this is
what removes the single-clip noise), and writes per-video results to results_batch.jsonl.

Usage:
  python run_batch.py --manifest youcook2/manifest.json --frames 30 --modes all --limit 40
"""
import os, sys, json, argparse, statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import run as R                      # the standalone runner (Backend, gates, metrics, ...)
import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(HERE.parent / "data" / "youcook2" / "manifest.json"))
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--modes", default="all")
    ap.add_argument("--delta", type=float, default=R.DELTA)
    ap.add_argument("--temp", type=float, default=R.TEMP)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(HERE / "results_batch.jsonl"))
    args = ap.parse_args()

    modes = list(R.MODES) if args.modes == "all" else [m.strip() for m in args.modes.split(",")]
    man = json.load(open(args.manifest))
    if args.limit:
        man = man[:args.limit]

    # ---- resume / checkpoint: load prior results, skip already-done videos ----
    per_video = []
    done_ids = set()
    if os.path.exists(args.out):
        try:
            prev = json.load(open(args.out))
            per_video = prev.get("per_video", [])
            done_ids = {v["id"] for v in per_video}
        except Exception:
            per_video, done_ids = [], set()
    remaining = [m for m in man if m["id"] not in done_ids]
    print(f"Batch: {len(man)} total | {len(done_ids)} done | {len(remaining)} remaining x {modes}", flush=True)

    if not remaining:
        print("Nothing remaining — all videos already processed.", flush=True)
    else:
        model, processor, tokenizer = R.load_model()
    device = "cuda"

    for vi, item in enumerate(remaining):
        try:
            gt = json.load(open(item["gt"]))
            refs = [g["sentence"] for g in gt] + [" ".join(g["sentence"] for g in gt)]
            frames, ts, dur = R.extract_dense(item["video"], args.frames)
            pils = [Image.open(f).convert("RGB") for f in frames]
        except Exception as e:
            print(f"  [skip {item['id']}] prep failed: {e}", flush=True)
            continue

        vres = {}
        for m in modes:
            torch.manual_seed(args.seed)
            inp, outg = R.MODES[m]
            try:
                r = R.run_condition(model, processor, tokenizer, pils, ts, dur, device,
                                    inp, outg, args.delta, args.frames, m, args.temp)
                q = R.compute_quality(r["text"], refs)
                tl = R.timeliness(r["emissions"], gt, dur)
                vres[m] = {"n_admitted": r["n_admitted"], "n_total": r["n_total"],
                           "n_emissions": r["n_emissions"], "kv_tokens": r["kv_tokens"],
                           "time_s": r["time_s"], "bleu1": q["bleu1"], "meteor": q["meteor"],
                           "rouge_l": q["rouge_l"], "coverage": tl["coverage"],
                           "latency": tl["mean_abs_latency_s"]}
            except Exception as e:
                print(f"  [{item['id']} {m}] FAILED: {e}", flush=True)
            torch.cuda.empty_cache()

        per_video.append({"id": item["id"], "n_events": item.get("n_events"), "res": vres})
        print(f"[{vi+1}/{len(remaining)} this chunk | {len(per_video)} total] {item['id']} done", flush=True)
        json.dump({"args": vars(args), "per_video": per_video}, open(args.out, "w"))   # checkpoint after each video

    # ── aggregate (mean ± std across videos) ──────────────────────────────────
    def agg(metric, m):
        vals = [v["res"][m][metric] for v in per_video
                if m in v["res"] and v["res"][m].get(metric) is not None]
        if not vals:
            return None
        mu = statistics.mean(vals)
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        return mu, sd, len(vals)

    print("\n" + "=" * (16 + 17 * len(modes)), flush=True)
    print(f"AGGREGATE over {len(per_video)} videos  (mean ± std)", flush=True)
    print("=" * (16 + 17 * len(modes)), flush=True)
    print(f"{'':<16}" + "".join(f"{m:>17}" for m in modes), flush=True)
    print("-" * (16 + 17 * len(modes)), flush=True)
    for metric in ["meteor", "bleu1", "rouge_l", "coverage", "latency",
                   "n_emissions", "n_admitted", "kv_tokens", "time_s"]:
        row = f"{metric:<16}"
        for m in modes:
            a = agg(metric, m)
            row += f"{'n/a':>17}" if a is None else f"{('%.3f±%.3f' % (a[0], a[1])):>17}"
        print(row, flush=True)
    print("=" * (16 + 17 * len(modes)), flush=True)
    print(f"\nSaved {args.out}", flush=True)


if __name__ == "__main__":
    main()
