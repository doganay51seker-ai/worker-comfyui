#!/usr/bin/env python3
"""
Causal 5-way isolation benchmark (Codex-approved).

Pipelines (2 sahne × 5 pipeline × 3 seed = 30 imaj):
  p0_flux_only            — baseline (no LoRA, no PuLID)
  p1_lora_only            — LoRA v2 alone
  p2_pulid_only           — PuLID alone (no LoRA)
  p3_lora_pulid           — LoRA + PuLID (NO FaceDetailer) ← Codex: needed to isolate FD effect
  p4_lora_pulid_fd        — LoRA + PuLID + conservative FaceDetailer

P3→P4 delta = causal FaceDetailer contribution.

Cost reporting (Codex-flagged):
  Reports 'execution_only_cost_est' — does NOT include cold-start / idle timeout
  billing. Real cost measured from RunPod Billing dashboard delta.
  GPU $/hr must be set explicitly via env; NO default (24 GB tiers vary
  $0.68-$1.12/hr depending on GPU type).

Job status handling: COMPLETED / FAILED / CANCELLED / TIMED_OUT all terminal.

Requires: RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID_CANDIDATE, RUNPOD_GPU_USD_PER_HOUR.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

RUNPOD_API_KEY = os.environ["RUNPOD_API_KEY"]
ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID_CANDIDATE")
if not ENDPOINT_ID:
    sys.exit("Set RUNPOD_ENDPOINT_ID_CANDIDATE (NEW endpoint, not prod etu96zf31linwc)")

# NO DEFAULT — must be set explicitly (Codex: $0.44 assumption wrong for L4/A5000/4090)
_gpu_rate = os.environ.get("RUNPOD_GPU_USD_PER_HOUR")
if not _gpu_rate:
    sys.exit(
        "Set RUNPOD_GPU_USD_PER_HOUR explicitly. Approximate 24 GB tiers:\n"
        "  L4/A5000/3090: ~0.684\n"
        "  4090 Pro:      ~1.116\n"
        "Check RunPod Console for your endpoint's actual rate."
    )
GPU_USD_PER_HOUR = float(_gpu_rate)
POLL_TIMEOUT_S = int(os.environ.get("RUNPOD_POLL_TIMEOUT_S", "1800"))

WF_DIR = Path(__file__).parent / "workflows"

SCENES = {
    "s1_sfw": (
        "karlisa woman, cool platinum blonde hair, freckles, blue eyes, round soft face, "
        "medium shot, sitting on window ledge, hands relaxed on knees, "
        "cream ribbed tank top, "
        "warm morning window light, Cihangir apartment interior, natural iPhone raw photo, "
        "sharp anatomy, five fingers each hand, thumb opposing four"
    ),
    "s2_topless": (
        "karlisa woman, cool platinum blonde hair, freckles, blue eyes, round soft face, "
        "medium shot, standing at bathroom sink, one hand loosely covering breast, "
        "topless natural anatomy, "
        "warm morning light through frosted window, "
        "small tiled bathroom, natural iPhone raw photo, "
        "sharp anatomy, symmetric breasts natural size, five fingers each hand"
    ),
}

PIPELINES = [
    "p0_flux_only",
    "p1_lora_only",
    "p2_pulid_only",
    "p3_lora_pulid",       # Codex-restored: needed to isolate FaceDetailer causal effect
    "p4_lora_pulid_fd",
]
SEEDS = [42, 137, 2718]

TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}

RUN_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"
STATUS_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status"


def load_workflow(pipeline: str) -> dict:
    return json.loads((WF_DIR / f"{pipeline}.json").read_text())


def patch(wf: dict, prompt: str, seed: int, face_name: str) -> dict:
    for k, node in wf.items():
        cls = node["class_type"]
        if cls == "CLIPTextEncode" and "PROMPT_POSITIVE_PLACEHOLDER" in node["inputs"].get("text", ""):
            wf[k]["inputs"]["text"] = prompt
        if cls == "KSampler":
            wf[k]["inputs"]["seed"] = seed
        if cls == "FaceDetailer":
            wf[k]["inputs"]["seed"] = seed
        if cls == "LoadImage":
            wf[k]["inputs"]["image"] = face_name
    return wf


def workflow_needs_face(pipeline: str) -> bool:
    return "pulid" in pipeline


def submit(wf: dict, face_path: Path | None) -> str:
    payload: dict = {"input": {"workflow": wf}}
    if face_path is not None:
        payload["input"]["images"] = [
            {"name": face_path.name, "image": base64.b64encode(face_path.read_bytes()).decode("ascii")}
        ]
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}
    r = requests.post(RUN_URL, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["id"]


def poll(job_id: str, timeout: int = POLL_TIMEOUT_S) -> dict:
    """Wait for terminal status. Handles COMPLETED/FAILED/CANCELLED/TIMED_OUT."""
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        r = requests.get(f"{STATUS_URL}/{job_id}", headers=headers, timeout=30)
        r.raise_for_status()
        d = r.json()
        status = d.get("status")
        if status in TERMINAL_STATUSES:
            if status == "COMPLETED":
                return d
            raise RuntimeError(f"job {job_id} ended {status}: {d}")
        time.sleep(4)
    raise TimeoutError(f"{job_id} client timeout after {timeout}s (last status={d.get('status')})")


def save_image(data: dict, out_path: Path) -> None:
    output = data.get("output", {})
    images = output.get("images") or output.get("message", {}).get("images", [])
    if not images:
        raise RuntimeError(f"no images in output: {data}")
    b64 = images[0].get("image") or images[0].get("data")
    out_path.write_bytes(base64.b64decode(b64))


def one_job(scene: str, pipeline: str, seed: int, face_path: Path, out_dir: Path) -> dict:
    started = time.time()
    job_id: str | None = None
    wf = load_workflow(pipeline)
    face_arg = face_path if workflow_needs_face(pipeline) else None
    wf = patch(wf, SCENES[scene], seed, face_path.name)
    out_file = out_dir / f"{scene}__{pipeline}__seed{seed}.png"
    try:
        job_id = submit(wf, face_arg)
        print(
            f"[submitted] {scene}/{pipeline}/s{seed} job_id={job_id}",
            flush=True,
        )
        data = poll(job_id)
        save_image(data, out_file)
        exec_ms = int(data.get("executionTime", 0))
        delay_ms = int(data.get("delayTime", 0))
        # Execution-only cost estimate — does NOT include cold-start / idle billing
        cost_usd = round((exec_ms / 1000.0 / 3600.0) * GPU_USD_PER_HOUR, 5)
        return {
            "scene": scene, "pipeline": pipeline, "seed": seed,
            "status": "ok",
            "wall_s": round(time.time() - started, 1),
            "execution_ms": exec_ms,
            "delay_ms": delay_ms,
            "execution_only_cost_est_usd": cost_usd,
            "output": str(out_file),
            "job_id": job_id,
        }
    except Exception as e:
        return {
            "scene": scene, "pipeline": pipeline, "seed": seed,
            "status": "error", "error": str(e),
            "wall_s": round(time.time() - started, 1),
            "job_id": job_id,
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--face", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("./ab_output"))
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--only-scene", default=None)
    ap.add_argument("--only-pipeline", default=None)
    ap.add_argument("--only-seed", type=int, default=None)
    args = ap.parse_args()

    if not args.face.exists():
        sys.exit(f"face not found: {args.face}")
    args.out.mkdir(parents=True, exist_ok=True)

    scenes = [args.only_scene] if args.only_scene else list(SCENES.keys())
    pipelines = [args.only_pipeline] if args.only_pipeline else PIPELINES
    seeds = [args.only_seed] if args.only_seed else SEEDS

    jobs = [(s, p, seed) for s in scenes for p in pipelines for seed in seeds]
    print(f"[ab] endpoint={ENDPOINT_ID} gpu_usd_per_hour={GPU_USD_PER_HOUR}")
    print(f"[ab] total_jobs={len(jobs)} concurrency={args.concurrency}")
    print(f"[ab] scenes={scenes} pipelines={pipelines} seeds={seeds}")

    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(one_job, s, p, seed, args.face, args.out): (s, p, seed) for s, p, seed in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            icon = "OK" if r["status"] == "ok" else "ERR"
            print(f"[{i}/{len(jobs)}] {icon} {r['scene']}/{r['pipeline']}/s{r['seed']} wall={r['wall_s']}s "
                  f"exec={r.get('execution_ms', '?')}ms exec_cost=${r.get('execution_only_cost_est_usd', '?')}")
            results.append(r)

    ok_rs = [r for r in results if r["status"] == "ok"]
    total_exec_cost = round(sum(r["execution_only_cost_est_usd"] for r in ok_rs), 4)
    total_exec_s = round(sum(r["execution_ms"] for r in ok_rs) / 1000.0, 1)

    manifest = {
        "endpoint": ENDPOINT_ID,
        "gpu_usd_per_hour": GPU_USD_PER_HOUR,
        "total_jobs": len(jobs),
        "ok_count": len(ok_rs),
        "total_execution_s": total_exec_s,
        "total_execution_only_cost_est_usd": total_exec_cost,
        "cost_note": (
            "Execution-only estimate. Actual RunPod bill also includes cold-start "
            "and idle-timeout minutes. Cross-check with Billing dashboard delta."
        ),
        "results": results,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[done] ok={len(ok_rs)}/{len(jobs)} total_exec={total_exec_s}s exec_cost=${total_exec_cost}")
    print(f"[note] cold-start + idle NOT included — cross-check RunPod Billing dashboard")

    # Non-zero exit if any job failed — prevents accidentally advancing to full
    # benchmark after a smoke run had errors.
    if len(ok_rs) != len(jobs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
