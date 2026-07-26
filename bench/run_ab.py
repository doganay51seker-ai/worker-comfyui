#!/usr/bin/env python3
"""
Causal A/B benchmark — Codex-approved 4-way isolation.

Pipelines (2 sahne × 4 pipeline × 3 seed = 24 imaj):
  p0_flux_only            — baseline (no LoRA, no PuLID)
  p1_lora_only            — LoRA v2 alone
  p2_pulid_only           — PuLID alone (no LoRA)
  p3_lora_pulid_fd        — LoRA + PuLID + conservative FaceDetailer

Reports real GPU-second cost from RunPod status executionTime, not per-image guesses.
Requires: RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID_CANDIDATE (must be NEW isolated endpoint).
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

# .env expected at ~/Desktop/ugcinf/.env (2 levels up from worker-comfyui/bench/)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

RUNPOD_API_KEY = os.environ["RUNPOD_API_KEY"]
ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID_CANDIDATE")
if not ENDPOINT_ID:
    sys.exit("Set RUNPOD_ENDPOINT_ID_CANDIDATE (NEW endpoint, not prod etu96zf31linwc)")

# GPU $/hr — set per real endpoint tier (Codex flag: guess is unreliable)
GPU_USD_PER_HOUR = float(os.environ.get("RUNPOD_GPU_USD_PER_HOUR", "0.44"))

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

PIPELINES = ["p0_flux_only", "p1_lora_only", "p2_pulid_only", "p3_lora_pulid_fd"]
SEEDS = [42, 137, 2718]

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


def poll(job_id: str, timeout: int = 480) -> dict:
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        r = requests.get(f"{STATUS_URL}/{job_id}", headers=headers, timeout=30)
        r.raise_for_status()
        d = r.json()
        status = d.get("status")
        if status == "COMPLETED":
            return d
        if status == "FAILED":
            raise RuntimeError(f"failed: {d}")
        time.sleep(4)
    raise TimeoutError(f"{job_id} timeout after {timeout}s")


def save_image(data: dict, out_path: Path) -> None:
    output = data.get("output", {})
    images = output.get("images") or output.get("message", {}).get("images", [])
    if not images:
        raise RuntimeError(f"no images in output: {data}")
    b64 = images[0].get("image") or images[0].get("data")
    out_path.write_bytes(base64.b64decode(b64))


def one_job(scene: str, pipeline: str, seed: int, face_path: Path, out_dir: Path) -> dict:
    started = time.time()
    wf = load_workflow(pipeline)
    face_arg = face_path if workflow_needs_face(pipeline) else None
    wf = patch(wf, SCENES[scene], seed, face_path.name)
    out_file = out_dir / f"{scene}__{pipeline}__seed{seed}.png"
    try:
        job_id = submit(wf, face_arg)
        data = poll(job_id)
        save_image(data, out_file)
        exec_ms = int(data.get("executionTime", 0))
        delay_ms = int(data.get("delayTime", 0))
        cost_usd = round((exec_ms / 1000.0 / 3600.0) * GPU_USD_PER_HOUR, 5)
        return {
            "scene": scene, "pipeline": pipeline, "seed": seed,
            "status": "ok",
            "wall_s": round(time.time() - started, 1),
            "execution_ms": exec_ms,
            "delay_ms": delay_ms,
            "cost_usd_est": cost_usd,
            "output": str(out_file),
            "job_id": job_id,
        }
    except Exception as e:
        return {
            "scene": scene, "pipeline": pipeline, "seed": seed,
            "status": "error", "error": str(e),
            "wall_s": round(time.time() - started, 1),
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
                  f"exec={r.get('execution_ms', '?')}ms cost=${r.get('cost_usd_est', '?')}")
            results.append(r)

    ok_rs = [r for r in results if r["status"] == "ok"]
    total_cost = round(sum(r["cost_usd_est"] for r in ok_rs), 4)
    total_exec_s = round(sum(r["execution_ms"] for r in ok_rs) / 1000.0, 1)

    manifest = {
        "endpoint": ENDPOINT_ID,
        "gpu_usd_per_hour": GPU_USD_PER_HOUR,
        "total_jobs": len(jobs),
        "ok_count": len(ok_rs),
        "total_execution_s": total_exec_s,
        "total_cost_usd_est": total_cost,
        "results": results,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[done] ok={len(ok_rs)}/{len(jobs)} total_exec={total_exec_s}s cost=${total_cost}")


if __name__ == "__main__":
    main()
