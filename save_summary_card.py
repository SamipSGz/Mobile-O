"""Assembles all 5 stage cards into one summary image."""
import json
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

OUT_DIR = Path("predictions/stages")
data    = json.load(open("/tmp/stage_timings.json"))

COLORS = {
    1: (80,140,255), 2: (255,160,40),
    3: (200,80,80),  4: (80,200,120), 5: (180,100,255)
}
LABELS = {
    1: ("S1 Original MPS",  f"{data['total1']:.0f}ms",   "1.00×"),
    2: ("S2 Profiled",      f"{data['total1']:.0f}ms",   "1.00×"),
    3: ("S3 MPS Pipeline",  f"{data['t_pipe3']:.0f}ms",  f"{data['sp3']:.2f}×"),
    4: ("S4 ANE+MPS",       f"{data['t_pipe4']:.0f}ms",  f"{data['sp4']:.2f}×"),
    5: ("S5 ANE+INT4",      f"{data['t_pipe5']:.0f}ms",  f"{data['sp5']:.2f}×"),
}

# Load the 5 cards
cards = [Image.open(OUT_DIR / f"stage{i}_{['original_mps','profiled_mps','pipeline_mps_only','pipeline_ane_mps','pipeline_int4'][i-1]}.png") for i in range(1,6)]
W, H = cards[0].size

# Layout: 3 on top row, 2 centered on bottom row
PAD = 6
HEADER = 100
FOOTER = 120
total_w = W*3 + PAD*2
total_h = HEADER + H*2 + PAD + FOOTER
summary = Image.new("RGB", (total_w, total_h), (10,10,16))
draw = ImageDraw.Draw(summary)

try:
    f_xl = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
    f_lg = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    f_md = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 17)
    f_sm = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
except:
    f_xl = f_lg = f_md = f_sm = ImageFont.load_default()

# Header
draw.text((PAD, 12), "Mobile-O: Hardware-Aware Inference — All 5 Stages", fill=(255,255,255), font=f_xl)
draw.text((PAD, 48), "MacBook Air M5  ·  MPS → Profiled → Async → ANE+MPS → ANE+INT4  ·  Mobile-O 1.6B", fill=(130,130,150), font=f_sm)
draw.rectangle([0, 82, total_w, 84], fill=(40,40,50))

# Top row: stages 1, 2, 3
for i, (idx, card) in enumerate(zip([1,2,3], cards[:3])):
    x = i*(W+PAD)
    summary.paste(card, (x, HEADER))

# Bottom row: stages 4, 5 — centred
offset = (total_w - W*2 - PAD) // 2
for i, (idx, card) in enumerate(zip([4,5], cards[3:])):
    x = offset + i*(W+PAD)
    summary.paste(card, (x, HEADER + H + PAD))

# Footer progression bar
fy = HEADER + H*2 + PAD + 8
draw.text((PAD, fy), "THROUGHPUT PROGRESSION  (4-query batch, Apple M5)", fill=(100,100,120), font=f_sm)
fy += 22
col_w = total_w // 5
for i in range(1, 6):
    label, total_t, speedup = LABELS[i]
    col = COLORS[i]
    x = (i-1)*col_w + PAD
    # bar proportional to speedup (inverse of time)
    try:
        sp = float(speedup.replace("×",""))
        bar_h = int(30 * min(sp/5, 1.0))
    except: bar_h = 4
    draw.rectangle([x, fy+30-bar_h, x+col_w-PAD*2, fy+30], fill=col)
    draw.text((x, fy+36), label,   fill=(180,180,195), font=f_sm)
    draw.text((x, fy+54), total_t, fill=(220,220,230), font=f_md)
    draw.text((x, fy+74), speedup, fill=col,           font=f_lg)

out_path = OUT_DIR / "summary_all_stages.png"
summary.save(out_path)
print(f"Saved: {out_path}  ({summary.size[0]}×{summary.size[1]}px)")
