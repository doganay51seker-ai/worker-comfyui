# PuLID/LoRA/FaceDetailer Causal Benchmark

Codex-approved 5-way isolation harness. Answers **causal** questions:
- Foundation mı, LoRA mı, PuLID mı, FaceDetailer mı bozulma yaratıyor?
- FaceDetailer katkısı ölçülebilir mi? (P3→P4 delta)

**Not a production decision maker.** Triage layer; human eval gates publish.

## Pipelines (5) — matched parameters for causal isolation

All pipelines share identical LoRA/PuLID parameter values so only the presence
of a component differs between them.

| id | LoRA | PuLID | FaceDetailer |
|----|------|-------|--------------|
| `p0_flux_only` | ❌ | ❌ | ❌ |
| `p1_lora_only` | strength_model 0.7 / strength_clip 0.7 | ❌ | ❌ |
| `p2_pulid_only` | ❌ | weight 0.85, start_at 0, end_at 0.7 | ❌ |
| `p3_lora_pulid` | 0.7 / 0.7 | 0.85 / 0 / 0.7 | ❌ |
| `p4_lora_pulid_fd` | 0.7 / 0.7 | 0.85 / 0 / 0.7 | denoise 0.28, feather 24 |

Codex causal reads:
- P1 → P0 delta = **LoRA alone** contribution
- P2 → P0 delta = **PuLID alone** contribution
- P3 → P1 delta = **PuLID added on top of LoRA**
- P4 → P3 delta = **FaceDetailer added** ← THE key question

## Face reference (verified exists)

`/Users/doni/AI_Influencer_Studio/character/references/character_b_primary.png`
(1024×1536 RGBA, current canonical Alara face).

Workflow `LoadImage` nodes and RunPod payload use the basename `character_b_primary.png`.

## Prerequisites

1. Candidate A image built + pushed to GHCR (see `../.github/workflows/pulid-candidates.yml`)
2. GHCR package visibility set to **public** (default is private on first publish)
   OR RunPod endpoint given ghcr.io registry credentials
3. NEW isolated RunPod endpoint (do NOT touch prod `etu96zf31linwc`)
4. Network Volume `ugcinf-lora` (id `cec0y73w93`) attached at `/runpod-volume`
5. RunPod endpoint config: **active workers = 0, max workers = 1** (cost control)

## Env vars

```bash
export RUNPOD_ENDPOINT_ID_CANDIDATE=<new_endpoint_id>
export RUNPOD_GPU_USD_PER_HOUR=0.684  # required; no default. Check endpoint tier.
# L4/A5000/3090 ~$0.684, 4090 Pro ~$1.116
```

```bash
pip install requests python-dotenv insightface onnxruntime opencv-python-headless
```

## Sequential smoke — 1 seed × 1 scene × 5 pipelines (~5 jobs)

One command runs all 5 pipelines sequentially at concurrency=1. If any pipeline
fails at this stage, do NOT proceed to full benchmark.

```bash
FACE=/Users/doni/AI_Influencer_Studio/character/references/character_b_primary.png

python run_ab.py \
  --face "$FACE" \
  --out ./smoke_out \
  --only-scene s1_sfw \
  --only-seed 42 \
  --concurrency 1
```

All 5 must succeed. Check `smoke_out/manifest.json` — `ok_count` must equal 5.

## Full 30-image benchmark

Only if smoke passed:

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
- Cross-check via RunPod Billing dashboard delta
- Multi-GPU endpoints render single-rate estimate directional only

## GHCR image visibility (default: private)

GHCR publishes new packages as **private by default**. After first successful build:

- **A** (simplest for research): Make package public — GitHub → your profile →
  Packages → `ugc-pulid-candidate-a` → Package settings → Change visibility → Public
- **B** (keep private): Add ghcr credentials to the RunPod endpoint —
  Console → Endpoint → Registry Auth → registry `ghcr.io`,
  user `<github-username>`, token = GitHub PAT with `read:packages`

**A** is fine because the LoRA lives on the RunPod volume; nothing private is
inside the image itself.
