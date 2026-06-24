#!/usr/bin/env python3
"""
run_adaptive_input.py — DEMO of idea #1: adaptive sampling rate (proactive input).

Pulls frames ON DEMAND from the video (ffmpeg seek). After each frame the model
runs a tiny probe -> p_yes("is something actively happening?") -> sets the next
gap Δt (high activity => small Δt = look soon; calm => big Δt = skip ahead).
Logs the timeline so you can SEE where it looks densely vs skips.

Usage: python run_adaptive_input.py --video eval_video.mp4
"""
import os, sys, subprocess, argparse, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as R
import torch
from PIL import Image

FFMPEG = str(Path.home() / "miniforge3" / "bin" / "ffmpeg")
TMP = "/tmp/adapt_frames"; os.makedirs(TMP, exist_ok=True)

RATE_PROBE = ("<|vision_end|>\nIs something actively happening or changing right now in the "
              "latest frame — an action in progress (someone doing something)? Answer only Yes or No.\n"
              "<|im_end|>\n<|im_start|>assistant\n")

def duration(video):
    r = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
                        "-of","default=noprint_wrappers=1:nokey=1", video], capture_output=True, text=True)
    return float(r.stdout.strip())

def get_frame(video, t):
    out = os.path.join(TMP, f"f_{t:07.2f}.jpg")
    subprocess.run([FFMPEG,"-y","-ss",f"{t:.2f}","-i",video,"-frames:v","1","-q:v","2",out],
                   capture_output=True, timeout=60)
    return Image.open(out).convert("RGB")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=R.DEFAULT_VIDEO)
    ap.add_argument("--dt-min", type=float, default=0.5)
    ap.add_argument("--dt-max", type=float, default=9.0)
    ap.add_argument("--cap", type=int, default=40, help="max frames to pull")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    model, processor, tokenizer = R.load_model()
    be = R.Backend(model, processor, tokenizer, "cuda")
    dur = duration(args.video)
    print(f"\nvideo duration {dur:.1f}s · Δt in [{args.dt_min},{args.dt_max}]s · model chooses each Δt\n", flush=True)
    print(f"{'t (s)':>7} {'cos→prev':>9} {'p_yes':>6} {'Δt next':>8}   activity", flush=True)
    print("-"*64, flush=True)

    t = 0.0; admitted = 0; prev = None; rows = []
    while t < dur and admitted < args.cap:
        frame = get_frame(args.video, t)
        ve, pos, ntok = be.encode_frame(frame, admitted)
        vec = R.frame_vec(ve)
        cos = float(torch.dot(vec, prev).item()) if prev is not None else float("nan")
        be.add_frame(ve, pos, ntok); admitted += 1; prev = vec
        p_yes, p_no, _ = R.gate_on_P(be, R.tok(be, RATE_PROBE))
        # high activity (p_yes↑) -> small Δt ; calm -> big Δt
        dt = args.dt_max - (args.dt_max - args.dt_min) * p_yes
        bar = "#" * max(1, int(p_yes * 24))
        print(f"{t:7.1f} {cos:9.3f} {p_yes:6.2f} {dt:8.1f}   {bar}", flush=True)
        rows.append({"t": round(t,2), "cos_prev": None if prev is None else round(cos,3),
                     "p_yes": round(p_yes,3), "dt_next": round(dt,2)})
        t += dt

    fixed_n = int(dur / (sum(r["dt_next"] for r in rows)/len(rows)))
    print("-"*64, flush=True)
    print(f"\nADAPTIVE used {admitted} frames over {dur:.0f}s.", flush=True)
    print(f"Δt ranged {min(r['dt_next'] for r in rows):.1f}s (busy) .. {max(r['dt_next'] for r in rows):.1f}s (calm).", flush=True)
    print("Dense clusters = where the model decided to look closely; big gaps = it skipped ahead.", flush=True)
    import json; json.dump(rows, open(HERE/"adaptive_timeline.json","w"))

if __name__ == "__main__":
    main()
