# Proactive Streaming Video Understanding

A **training-free** streaming system on Qwen2-VL-7B that does live **captioning** and
**Video QA** over one shared-KV memory. The model (the *orchestrator*) decides
everything — which frames to remember and when to speak — with **no thresholds set
by us**.

## Quickstart

```bash
source ~/AsyncReasoning/.venv/bin/activate      # transformers 4.51.0
cd proactive-inference

python proactive_system.py                      # uses config.yaml (adaptive, both)
```

On the CSCS cluster, launch through SLURM:

```bash
bash scripts/run.sh                             # single-video run on a debug GPU
```

## How it works (one paragraph)

Frames arrive one motion-clip at a time. For each clip the orchestrator reports a
keep-probability `p_yes`; in the default **adaptive** mode we keep the clip only if
`p_yes` beats the model's *own running average* — "is this more interesting than what
I've seen lately?". Kept clips go into a single shared-KV block (block **P**) once.
Both the live captioner and every QA question read from that same memory, so video is
encoded only once. A separate yes/no **output gate** decides when a caption is worth
emitting. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Layout

```
proactive-inference/
├── proactive_system.py   # main entry — streaming caption + Video QA
├── run.py                # shared-KV backend (Backend, gates, metrics)
├── config.yaml           # all settings (CLI flags override)
├── README.md
├── docs/
│   ├── ARCHITECTURE.md   # how the shared-KV cache and the two gates work
│   └── USAGE.md          # config keys, CLI reference, input modes
├── scripts/
│   ├── run.sh            # single-video SLURM launcher
│   └── sweep.sh          # auto-resubmitting YouCook2 benchmark sweep
├── benchmark/
│   ├── build_youcook2.py # download N YouCook2 videos + build manifest/GT
│   └── run_batch.py      # batch evaluator over the manifest
├── data/youcook2/        # manifest + videos (built by benchmark/)
├── results/              # all run outputs land here (safe to delete/regenerate)
└── legacy/               # superseded single-purpose experiment scripts
```

## Configuration

Everything is driven by `config.yaml`. Override any key on the command line, e.g.

```bash
python proactive_system.py --mode vqa --input-mode probe \
    --questions "What liquid is poured over the chicken?"
python proactive_system.py --config experiments/fast.yaml
```

Full key reference: [docs/USAGE.md](docs/USAGE.md).

## Benchmark (YouCook2)

The benchmark needs the YouCook2 annotation list and downloaded videos. The
annotation source is not committed, so rebuild the dataset first:

```bash
python benchmark/build_youcook2.py --n 40 --res 360   # downloads + builds manifest/GT
bash scripts/sweep.sh                                  # resumable debug-partition sweep
```

Outputs are written under `results/` and are fully regeneratable.
