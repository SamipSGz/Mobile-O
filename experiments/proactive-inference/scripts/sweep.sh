#!/bin/bash
# Auto-resubmitting YouCook2 benchmark sweep on the debug partition.
# Resumable: run_batch.py checkpoints after every video, so re-running this
# continues from where the last debug job stopped.
#
# Build the dataset first:  python benchmark/build_youcook2.py --n 40 --res 360
set -e
cd "$(dirname "$0")/.."
source ~/AsyncReasoning/.venv/bin/activate

MAN=data/youcook2/manifest.json
OUT=results/youcook2_sweep.json
SR="srun --partition=debug --nodes=1 --ntasks=1 --gpus-per-task=1 --time=01:30:00 -A a168"

if [ ! -f "$MAN" ]; then
  echo "[sweep] manifest $MAN not found — run benchmark/build_youcook2.py first." >&2
  exit 1
fi

mkdir -p results
$SR python benchmark/run_batch.py --manifest "$MAN" --out "$OUT" --modes all "$@"
