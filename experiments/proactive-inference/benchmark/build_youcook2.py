#!/usr/bin/env python3
"""
build_youcook2.py — pick N YouCook2 val videos, download them, emit per-video GT.

Uses the existing annotation list (junk/youcook2_captions.json: flat list of segments
with youtube_id / segment[start,end] / sentence). Groups by video, downloads each via
yt-dlp, and writes a GT json per video in the eval_video_gt.json format
([{sentence,start,end}, ...]). Skips videos that fail to download (dead/region-locked).

Output:
  youcook2/videos/<id>.mp4
  youcook2/gt/<id>.json
  youcook2/manifest.json   (list of {id, video, gt, n_events})

Usage:
  python build_youcook2.py --n 40 --res 360
"""
import json, collections, subprocess, argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAP  = Path.home() / "Mobile-O" / "experiments" / "junk" / "youcook2_captions.json"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="number of videos to fetch")
    ap.add_argument("--res", default="360", help="max video height (keep small)")
    ap.add_argument("--min-seg", type=int, default=4)
    ap.add_argument("--max-seg", type=int, default=12)
    ap.add_argument("--outdir", default=str(HERE.parent / "data" / "youcook2"))
    args = ap.parse_args()

    out = Path(args.outdir)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    (out / "gt").mkdir(parents=True, exist_ok=True)

    data = json.load(open(CAP))
    g = collections.defaultdict(list)
    for s in data:
        g[s["youtube_id"]].append(s)
    cands = sorted([(vid, segs) for vid, segs in g.items()
                    if args.min_seg <= len(segs) <= args.max_seg], key=lambda x: x[0])
    print(f"{len(cands)} candidate videos ({args.min_seg}-{args.max_seg} segments); fetching up to {args.n}", flush=True)

    manifest = []
    tried = 0
    for vid, segs in cands:
        if len(manifest) >= args.n:
            break
        tried += 1
        vp = out / "videos" / f"{vid}.mp4"
        if not vp.exists():
            url = f"https://www.youtube.com/watch?v={vid}"
            try:
                r = subprocess.run(
                    ["yt-dlp", "--no-warnings", "-q", "--no-playlist",
                     "--socket-timeout", "30", "--retries", "2", "--no-part",
                     "-f", f"best[height<={args.res}][ext=mp4]/best[height<={args.res}]/best",
                     "-o", str(vp), url],
                    capture_output=True, text=True, timeout=300)
            except subprocess.TimeoutExpired:
                print(f"  SKIP {vid}: download timeout", flush=True)
                vp.unlink(missing_ok=True); continue
            if r.returncode != 0 or not vp.exists():
                print(f"  SKIP {vid}: {(r.stderr or '').strip().splitlines()[-1][:100] if r.stderr else 'failed'}", flush=True)
                continue
        segs2 = sorted(segs, key=lambda s: s["segment"][0])
        gt = [{"sentence": s["sentence"], "start": float(s["segment"][0]), "end": float(s["segment"][1])}
              for s in segs2]
        gp = out / "gt" / f"{vid}.json"
        json.dump(gt, open(gp, "w"))
        manifest.append({"id": vid, "video": str(vp), "gt": str(gp), "n_events": len(gt)})
        print(f"  OK {vid}: {len(gt)} events  ({len(manifest)}/{args.n})", flush=True)

    json.dump(manifest, open(out / "manifest.json", "w"), indent=2)
    print(f"\n{len(manifest)} videos ready (tried {tried}) -> {out / 'manifest.json'}", flush=True)

if __name__ == "__main__":
    main()
