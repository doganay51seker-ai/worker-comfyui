#!/usr/bin/env python3
"""Fetch private runtime models from RunPod's S3-compatible storage.

The public worker image intentionally does not contain the character LoRA.
At cold start, the model is downloaded into ComfyUI's local model directory,
verified by SHA-256, and atomically installed before the worker starts.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import boto3


EXPECTED_SHA256 = os.environ.get(
    "RUNTIME_LORA_SHA256",
    "ae18bc60182c55dd6e99ee7883269d5a9badda6256f0641c2792dd764049dbd0",
)
DESTINATION = Path(
    os.environ.get(
        "RUNTIME_LORA_DEST",
        "/comfyui/models/loras/alara_karlisa_v2.safetensors",
    )
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def main() -> None:
    if DESTINATION.exists() and sha256_file(DESTINATION) == EXPECTED_SHA256:
        print(f"runtime-models: LoRA already verified at {DESTINATION}")
        return

    endpoint_url = required_env("RUNPOD_S3_ENDPOINT_URL")
    bucket = required_env("RUNPOD_S3_BUCKET")
    key = os.environ.get(
        "RUNPOD_S3_LORA_KEY",
        "models/loras/alara_karlisa_v2.safetensors",
    )
    access_key = required_env("RUNPOD_S3_ACCESS_KEY")
    secret_key = required_env("RUNPOD_S3_SECRET")
    region = os.environ.get("RUNPOD_S3_REGION", "EU-CZ-1")

    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    partial = DESTINATION.with_name(f"{DESTINATION.name}.part")
    partial.unlink(missing_ok=True)

    print(
        "runtime-models: downloading private LoRA "
        f"from bucket={bucket} key={key}"
    )
    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    client.download_file(bucket, key, str(partial))

    actual_sha256 = sha256_file(partial)
    if actual_sha256 != EXPECTED_SHA256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            "runtime LoRA checksum mismatch: "
            f"expected={EXPECTED_SHA256} actual={actual_sha256}"
        )

    partial.replace(DESTINATION)
    print(
        f"runtime-models: verified LoRA at {DESTINATION} "
        f"({DESTINATION.stat().st_size} bytes)"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"runtime-models: FAIL: {exc}", file=sys.stderr)
        raise
