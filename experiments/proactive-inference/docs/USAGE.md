# Usage

## Running

```bash
source ~/AsyncReasoning/.venv/bin/activate     # transformers 4.51.0 (required)
cd proactive-inference
python proactive_system.py                     # uses config.yaml
```

Configuration precedence (highest wins):

```
CLI flag   >   config.yaml   >   built-in default
```

So `config.yaml` sets your normal settings, and you override per-run on the command
line. Point at a different file with `--config path/to/file.yaml` (`.yaml`, `.yml`, or
`.json` are accepted).

## Config keys

Every key in `config.yaml` has a matching CLI flag (`input_mode` ↔ `--input-mode`).

| key | flag | default | meaning |
|-----|------|---------|---------|
| `mode` | `--mode` | `both` | `caption` \| `vqa` \| `both` |
| `input_mode` | `--input-mode` | `adaptive` | frame-keep policy (see below) |
| `video` | `--video` | demo video | input video path (`null` = built-in) |
| `gt` | `--gt` | demo GT | caption reference for METEOR (`null` = built-in) |
| `questions` | `--questions` | demo Qs | pipe-separated, e.g. `"Q1\|Q2\|Q3"` |
| `base_interval` | `--base-interval` | `2.5` | seconds between arriving frames |
| `cap` | `--cap` | `40` | max frames admitted into memory |
| `chunk` | `--chunk` | `4` | frames per motion clip (`1` = single still) |
| `chunk_dt` | `--chunk-dt` | `0.4` | seconds between frames within a clip |
| `temp` | `--temp` | `0.7` | caption / think temperature |
| `qa_temp` | `--qa-temp` | `0.3` | QA temperature |
| `delta` | `--delta` | backend | cosine threshold (**novelty mode only**) |
| `seed` | `--seed` | `0` | RNG seed |
| `out` | `--out` | `results/run.json` | output JSON path |

## Input modes

| mode | rule | threshold from us? | use it for |
|------|------|:--:|------|
| **adaptive** | keep if `p_yes` > model's own running-mean `p_yes` | **no** | **default / recommended** |
| **probe** | keep if `p_yes > p_no` | no | maximally selective, threshold-free baseline |
| **even** | keep every frame | n/a | upper-bound baseline |
| **novelty** | keep if cosine-to-last `< delta` | yes (`delta`) | old rule-based reference only |

## Examples

```bash
# Default: adaptive keep, captions + QA, demo video
python proactive_system.py

# Video QA only, custom questions, absolute (probe) keep
python proactive_system.py --mode vqa --input-mode probe \
    --questions "What dish is being prepared?|What is added to the bowl?"

# Your own video, single-still input, write somewhere specific
python proactive_system.py --video /path/clip.mp4 --chunk 1 \
    --out results/clip.json

# Use an alternate config profile
python proactive_system.py --config experiments/fast.yaml
```

## Output

A JSON file at `out` containing: the resolved `args`, the per-frame `schedule`
(keep/drop decisions), emitted `captions`, `qa` answers, frames `kept`, `ingest_s`,
and `cap_meteor`. Everything in `results/` is regeneratable — safe to delete.

## Benchmark

```bash
python benchmark/build_youcook2.py --n 40 --res 360   # build dataset (downloads videos)
bash scripts/sweep.sh                                  # resumable debug-partition sweep
```

The benchmark needs the YouCook2 annotation list (not committed); `build_youcook2.py`
expects it and downloads the videos it references.
