# Architecture

A training-free, proactive, streaming video understanding system on **Qwen2-VL-7B**.
The model itself — the *orchestrator* — makes every decision. We set **no thresholds**.

## 1. Shared-KV memory

Video is expensive to encode, so we encode each kept clip **once** and store its
key/value tensors in a persistent cache block, **block P**. Everything downstream —
the running captioner and every QA question — attends to that same block instead of
re-encoding the video.

This reuses the shared-cache machinery vendored in `~/Mobile-O/shared_cache`. Cache
blocks:

| block | holds |
|-------|-------|
| **P** | visual key/values of the kept clips (the video memory) |
| **T** | the orchestrator's running thoughts/notes |
| **W** | emitted captions |
| **E** | ephemeral probes (yes/no questions), discarded after reading |

Because attention under RoPE depends only on the *relative* position `i_k − i_q`, a
cached clip can be relocated in the sequence by rotating the query — so we append new
clips to block P without recomputing the old ones.

## 2. Motion clips, not stills

Each "frame" admitted is actually a short **clip** of `chunk` frames (default 4) fed
through Qwen2-VL's video path (`pixel_values_videos` + `video_grid_thw`). Temporal
patches give the model *motion* — it can tell stirring from a still bowl — which makes
the keep decision far more reliable than single thumbnails.

## 3. The two gates — the model decides everything

### Input gate (which clips to remember)

For every arriving clip the orchestrator answers a memory-aware yes/no probe
("compared to what I've noted, is something new happening here?"). Reading the answer
logits gives `p_yes` / `p_no`. The keep rule depends on `--input-mode`:

- **adaptive (default, recommended)** — keep if `p_yes` is above the model's *own
  running mean* of `p_yes` so far. The bar is the model's behaviour, not a constant.
  This removes the bias of the bare probe (which skews to "no") and tracks the video:
  dull stretches sit at the average and drop, changes spike above it and are kept.
- **probe** — keep if `p_yes > p_no` (absolute). Honest and threshold-free but very
  selective, since most frames look redundant in absolute terms.
- **even** / **novelty** — baselines (keep all / cosine rule) for comparison only.

### Output gate (when to speak)

After a clip is stored, a second ephemeral yes/no probe asks whether a *new* caption
is warranted. We emit only if `p_yes > p_no`. Same self-calibrating mechanism as the
input gate — no threshold.

Both gates use the identical `p_yes`-vs-baseline pattern, so the system is
**fully orchestrator-controlled on both the input and output sides**.

## 4. Why adaptive beats raw `p_yes > p_no`

The output gate's "should I speak?" question is naturally balanced and reads against
rich accumulated context, so `p_yes > p_no` works well there. The input gate asks
about a *bare candidate clip* with no comparison anchor, so the model defaults to
"no" and absolute `p_yes > p_no` keeps almost nothing (1 frame in our test). Comparing
each clip to the model's *own running average* restores useful recall without us
injecting any number.

Observed on the demo clip (4-frame motion chunks):

| mode | static threshold? | kept | capMETEOR | "What dish?" |
|------|:--:|:--:|:--:|------|
| τ=0.32 (hand-tuned) | yes | 11 | 0.219 | fried chicken ✓ |
| probe (`p_yes>p_no`) | no | 1 | 0.037 | wrong ✗ |
| **adaptive (vs own mean)** | **no** | 9 | 0.133 | **fried chicken ✓** |

Adaptive is the only variant that is both fully model-decided and gives usable
captioning/QA quality, so it is the default.

## 5. Flow

```
for each arriving clip:
    encode clip -> KV          (Qwen2-VL video path, motion)
    INPUT GATE  (model)        keep? p_yes vs running-mean p_yes
    if keep:
        append KV to block P
        update thoughts (T)
        OUTPUT GATE (model)    speak? p_yes vs p_no  -> maybe emit caption (W)
# after the stream:
for each question:
    answer from block P        (no re-encoding)
```
