#!/bin/bash
# Single-video proactive run on a CSCS debug GPU.
# Any extra args are passed straight to proactive_system.py, e.g.
#   bash scripts/run.sh --mode vqa --input-mode probe
set -e
cd "$(dirname "$0")/.."
source ~/AsyncReasoning/.venv/bin/activate
srun --partition=debug --nodes=1 --ntasks=1 --gpus-per-task=1 --time=00:40:00 -A a168 \
     python proactive_system.py "$@"
