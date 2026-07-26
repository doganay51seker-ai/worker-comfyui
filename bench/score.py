#!/usr/bin/env python3
"""
Score benchmark output — Codex-approved metrics with acknowledged limitations.

Metrics (ALL per-image, no missing-value averaging):
  arcface_similarity  — vs canonical reference, largest face only
  plastic_skin_ratio  — face-crop luminance variance heuristic (weak, NOT precision proxy)
  face_symmetry_norm  — landmark mirror distance / face_width
  face_detected       — boolean, tracked separately (does NOT inflate other averages)

Explicitly out of scope (per Codex): hands, teeth, prompt fidelity, clothing.
Human review still required — this is triage, not final grading.

Deps:
    pip install insightface onnxruntime opencv-python-headless numpy pillow
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np


def largest_face(faces: list) -> Any:
    """Pick face with largest bbox area (not faces[0])."""
    if not faces:
        return None

    def area(f):
        x1, y1, x2, y2 = f.bbox
        return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))

    return max(faces, key=area)


def load_reference_embedding(app, ref_path: Path) -> tuple[np.ndarray, float]:
    import cv2
    img = cv2.imread(str(ref_path))
    if img is None:
        raise RuntimeError(f"cannot read reference {ref_path}")
    faces = app.get(img)
    f = largest_face(faces)
    if f is None:
        raise RuntimeError(f"no face in reference {ref_path}")
    return f.normed_embedding, float(f.det_score)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def plastic_skin_ratio(cv_img, face_bbox) -> float:
    """
    Fraction of 16x16 face-crop patches with luminance variance < 25.
    KNOWN LIMITATION: sensitive to lighting/blur/exposure — NOT a reliable
    FP8-vs-BF16 proxy. Use only as directional signal alongside human eval.
    """
    import cv2
    x1, y1, x2, y2 = [int(v) for v in face_bbox]
    face = cv_img[max(0, y1):y2, max(0, x1):x2]
    if face.size == 0:
        return 0.0
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    patch = 16
    h, w = gray.shape
    low_var = total = 0
    for y in range(0, h - patch, patch):
        for x in range(0, w - patch, patch):
            block = gray[y:y + patch, x:x + patch]
            total += 1
            if float(np.var(block)) < 25.0:
                low_var += 1
    return low_var / total if total else 0.0


def face_symmetry_normalized(landmarks_2d, face_width: float) -> float | None:
    """Landmark mirror distance / face_width — scale-invariant."""
    if landmarks_2d is None or face_width <= 0:
        return None
    lm = np.array(landmarks_2d)
    midx = float(np.mean(lm[:, 0]))
    left = lm[lm[:, 0] < midx]
    right = lm[lm[:, 0] >= midx]
    if len(left) == 0 or len(right) == 0:
        return None
    mirrored = np.copy(left)
    mirrored[:, 0] = 2 * midx - mirrored[:, 0]
    dists = [float(np.min(np.linalg.norm(right - m, axis=1))) for m in mirrored]
    return float(np.mean(dists)) / face_width


def score_image(app, ref_emb: np.ndarray, img_path: Path) -> dict[str, Any]:
    import cv2
    img = cv2.imread(str(img_path))
    if img is None:
        return {"face_detected": False, "error": "cv2 read failed"}
    faces = app.get(img)
    face = largest_face(faces)
    if face is None:
        return {"face_detected": False, "faces_found": len(faces)}
    x1, y1, x2, y2 = [float(v) for v in face.bbox]
    face_w = max(1.0, x2 - x1)
    return {
        "face_detected": True,
        "faces_found": len(faces),
        "arcface_similarity": round(cosine(face.normed_embedding, ref_emb), 4),
        "plastic_skin_ratio": round(plastic_skin_ratio(img, face.bbox), 4),
        "face_symmetry_norm": round(face_symmetry_normalized(
            getattr(face, "landmark_2d_106", None), face_w
        ), 4) if hasattr(face, "landmark_2d_106") else None,
        "face_bbox": [int(v) for v in face.bbox],
        "face_width_px": round(face_w, 1),
        "det_score": round(float(face.det_score), 4),
    }


def summarize(records: list[dict]) -> dict:
    """Group by pipeline; report detection rate SEPARATELY so failures don't inflate means."""
    by_pipeline: dict[str, list[dict]] = {}
    for r in records:
        by_pipeline.setdefault(r["pipeline"], []).append(r)

    summary = {}
    for pipeline, rs in by_pipeline.items():
        total = len(rs)
        detected = [r for r in rs if r.get("score", {}).get("face_detected")]
        det_rate = len(detected) / total if total else 0.0
        if not detected:
            summary[pipeline] = {
                "total": total, "detected": 0, "detection_rate": 0.0,
                "note": "no faces detected — cannot compare",
            }
            continue
        sims = [r["score"]["arcface_similarity"] for r in detected]
        plastics = [r["score"]["plastic_skin_ratio"] for r in detected]
        syms = [r["score"]["face_symmetry_norm"] for r in detected if r["score"]["face_symmetry_norm"] is not None]
        summary[pipeline] = {
            "total": total,
            "detected": len(detected),
            "detection_rate": round(det_rate, 3),
            "arcface_mean": round(statistics.mean(sims), 4),
            "arcface_min": round(min(sims), 4),
            "arcface_max": round(max(sims), 4),
            "plastic_mean": round(statistics.mean(plastics), 4),
            "plastic_max": round(max(plastics), 4),
            "symmetry_norm_mean": round(statistics.mean(syms), 4) if syms else None,
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--face", type=Path, required=True, help="canonical reference face")
    ap.add_argument("--out", type=Path, default=Path("./ab_output"))
    args = ap.parse_args()

    import insightface

    print("[score] loading InsightFace antelopev2...")
    app = insightface.app.FaceAnalysis(name="antelopev2", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    print(f"[score] embedding reference: {args.face}")
    ref_emb, ref_det = load_reference_embedding(app, args.face)
    print(f"[score] reference det_score={ref_det}")

    manifest_path = args.out / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"no manifest at {manifest_path} — run run_ab.py first")
    manifest = json.loads(manifest_path.read_text())

    scored = []
    for r in manifest["results"]:
        if r["status"] != "ok":
            scored.append({**r, "score": {"face_detected": False, "error": r.get("error", "run failed")}})
            continue
        img_path = Path(r["output"])
        if not img_path.exists():
            scored.append({**r, "score": {"face_detected": False, "error": "missing file"}})
            continue
        sc = score_image(app, ref_emb, img_path)
        scored.append({**r, "score": sc})
        if sc.get("face_detected"):
            print(f"  {img_path.name}: arcface={sc['arcface_similarity']} plastic={sc['plastic_skin_ratio']} sym={sc.get('face_symmetry_norm')}")
        else:
            print(f"  {img_path.name}: NO FACE ({sc.get('faces_found', 0)} candidates)")

    summary = summarize(scored)
    out = args.out / "scores.json"
    out.write_text(json.dumps({
        "endpoint": manifest.get("endpoint"),
        "total_cost_usd_est": manifest.get("total_cost_usd_est"),
        "summary": summary,
        "records": scored,
    }, indent=2))
    print(f"\n[done] wrote {out}")

    print("\n=== pipeline summary (detection rate SEPARATE; no averaging of failures) ===")
    for pipeline, s in summary.items():
        print(f"  {pipeline}: {s}")
    print("\nCaveats (Codex-flagged):")
    print("  - plastic_skin_ratio is a weak heuristic — NOT a reliable FP8/BF16 proxy")
    print("  - hands/teeth/prompt fidelity NOT scored — human eval still required")
    print("  - n=3 seeds per (scene, pipeline) is small — treat as directional, not conclusive")


if __name__ == "__main__":
    main()
