# PuLID/LoRA/FaceDetailer Causal Benchmark

Codex-approved isolation harness. Answers **causal** questions:
- Foundation mı, LoRA mı, PuLID mı, FaceDetailer mı bozulma yaratıyor?
- Rebalance kalibrasyonu FaceFusion swap'ından iyi mi?

**Not a production decision maker.** This is directional evidence + must be paired with
human eval. Plastic-skin metric is a weak heuristic; FP8 vs BF16 decision cannot rest on it alone.

## Pipelines

| id | Uses LoRA | Uses PuLID | Uses FaceDetailer |
|----|-----------|------------|-------------------|
| `p0_flux_only` | ❌ | ❌ | ❌ |
| `p1_lora_only` | @ 1.0 | ❌ | ❌ |
| `p2_pulid_only` | ❌ | weight 0.9 | ❌ |
| `p3_lora_pulid_fd` | @ 0.7 | weight 0.85 | denoise 0.28 |

## Prerequisites

1. Custom candidate image built + pushed (see `../Dockerfile.pulid.{a,b}`)
2. Isolated RunPod endpoint created (NEW — do NOT touch prod `etu96zf31linwc`)
3. Network Volume `ugcinf-lora` (id `cec0y73w93`) attached at `/runpod-volume`
4. Alara canonical face reference at a known path

## Setup

```bash
export RUNPOD_ENDPOINT_ID_CANDIDATE=<new_endpoint_id>
export RUNPOD_GPU_USD_PER_HOUR=0.44  # set per your endpoint tier

pip install requests python-dotenv insightface onnxruntime opencv-python-headless
```

## Run 24-image benchmark

```bash
python run_ab.py \
  --face /Users/doni/AI_Influencer_Studio/character/references/alara_canonical.png \
  --out ./ab_output_candidate_a \
  --concurrency 2
```

Outputs 24 PNGs + `manifest.json` with real `executionTime`/cost per job.

## Smoke test single job first

```bash
# Fastest sanity check — one image, PuLID pipeline (exercises face load)
python run_ab.py \
  --face .../alara_canonical.png \
  --out ./smoke_out \
  --only-scene s1_sfw --only-pipeline p2_pulid_only --only-seed 42
```

## Score

```bash
python score.py \
  --face .../alara_canonical.png \
  --out ./ab_output_candidate_a
```

Writes `scores.json`. Pipeline summary reports:
- `detection_rate` (SEPARATE — failures do NOT inflate other means)
- `arcface_mean/min/max` (identity similarity)
- `plastic_mean/max` (weak proxy)
- `symmetry_norm_mean` (face-width normalized)

## Interpreting

- **`p0_flux_only` low arcface**: expected — no identity signal.
- **`p1_lora_only` low arcface**: LoRA v2 identity capture broken → dataset audit needed.
- **`p2_pulid_only` low arcface**: PuLID misconfigured or bad reference.
- **`p3_lora_pulid_fd` < `p1_lora_only`**: PuLID/FD hurt more than helped in this config.
- **Any pipeline with high `plastic_mean`**: could be FP8, could be sampler/LoRA — do NOT auto-conclude BF16 needed.

Codex-flagged limitations (respect):
1. plastic-skin metric IS lighting-sensitive
2. hands/teeth/clothing NOT scored — human review still gates publish
3. n=3 seeds per (scene, pipeline) — 6 detected per pipeline is small sample
