# PuLID/LoRA/FaceDetailer Causal Benchmark

Codex-approved 5-way isolation harness. Answers **causal** questions:
- Foundation mı, LoRA mı, PuLID mı, FaceDetailer mı bozulma yaratıyor?
- FaceDetailer katkısı ölçülebilir mi? (P3→P4 delta)

**Not a production decision maker.** Triage layer + human eval gates publish.

## Pipelines (5)

| id | LoRA | PuLID | FaceDetailer |
|----|------|-------|--------------|
| `p0_flux_only` | ❌ | ❌ | ❌ |
| `p1_lora_only` | @ 1.0 | ❌ | ❌ |
| `p2_pulid_only` | ❌ | weight 0.9 | ❌ |
| `p3_lora_pulid` | @ 0.7 | weight 0.85 | ❌ |
| `p4_lora_pulid_fd` | @ 0.7 | weight 0.85 | denoise 0.28 |

Codex causal reads:
- **P0 arcface** = baseline drift (no identity signal)
- **P1 arcface** = LoRA v2 alone quality
- **P2 arcface** = PuLID alone quality
- **P3 → P2/P1 delta** = LoRA+PuLID interaction (helps or hurts?)
- **P4 → P3 delta** = FaceDetailer causal contribution ← this is the KEY question

## Prerequisites

1. Custom candidate A image built + pushed (`Dockerfile.pulid.a`, see workflow)
2. Isolated RunPod endpoint (NEW — do NOT touch prod `etu96zf31linwc`)
3. Network Volume `ugcinf-lora` (id `cec0y73w93`) attached at `/runpod-volume`
4. Alara canonical face reference PNG on local disk

## Env vars

```bash
export RUNPOD_ENDPOINT_ID_CANDIDATE=<new_endpoint_id>
export RUNPOD_GPU_USD_PER_HOUR=0.684  # required, no default; check your endpoint tier
# Approximate 24 GB tiers: L4/A5000/3090 ~$0.684, 4090 Pro ~$1.116
```

```bash
pip install requests python-dotenv insightface onnxruntime opencv-python-headless
```

## Sequential smoke — REQUIRED before full benchmark

Run one job per pipeline first (5 jobs, ~$0.10-0.30). If any pipeline fails at
this stage, do NOT proceed to full benchmark.

```bash
FACE=/Users/doni/AI_Influencer_Studio/character/references/alara_canonical.png
OUT=./smoke_out

python run_ab.py --face "$FACE" --out "$OUT" --only-pipeline p0_flux_only  --only-scene s1_sfw --only-seed 42
python run_ab.py --face "$FACE" --out "$OUT" --only-pipeline p1_lora_only  --only-scene s1_sfw --only-seed 42
python run_ab.py --face "$FACE" --out "$OUT" --only-pipeline p2_pulid_only --only-scene s1_sfw --only-seed 42
python run_ab.py --face "$FACE" --out "$OUT" --only-pipeline p3_lora_pulid --only-scene s1_sfw --only-seed 42
python run_ab.py --face "$FACE" --out "$OUT" --only-pipeline p4_lora_pulid_fd --only-scene s1_sfw --only-seed 42
```

All 5 must succeed before moving on.

## Full 30-image benchmark

```bash
python run_ab.py \
  --face "$FACE" \
  --out ./ab_output_candidate_a \
  --concurrency 2
```

## Score

```bash
python score.py --face "$FACE" --out ./ab_output_candidate_a
```

Writes `scores.json`. Pipeline summary reports:
- `detection_rate` (SEPARATE — failures do NOT inflate other means)
- `arcface_mean/min/max` (identity similarity vs canonical reference)
- `plastic_mean/max` (weak proxy — NOT a reliable FP8/BF16 signal)
- `symmetry_norm_mean` (face-width normalized)

## Cost reporting caveats (Codex-flagged)

- `execution_only_cost_est_usd` = `executionTime × GPU_USD_PER_HOUR` only
- **Does NOT include cold-start or idle-timeout billing**
- Cross-check actual cost via RunPod Billing dashboard delta
- Single endpoint may span multiple GPU tiers → single-rate estimate is directional
