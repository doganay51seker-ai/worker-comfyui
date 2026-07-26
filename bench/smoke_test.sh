#!/usr/bin/env bash
# Runtime smoke test — file/mount presence checks.
# ComfyUI-level import smoke test lives in Dockerfile (uses upstream's
# `python main.py --quick-test-for-ci --cpu` pattern, which correctly
# exercises the node registration graph).
#
# Runtime usage: docker run --rm --entrypoint smoke-test.sh <image>
set -euo pipefail

echo "[smoke] === Python + torch presence ==="
python -c "import sys, torch; print(f'python={sys.version.split()[0]} torch={torch.__version__} cuda_available={torch.cuda.is_available()}')"

echo "[smoke] === pip check (dependency conflicts) ==="
uv pip check

echo "[smoke] === Model files present + size ==="
declare -A REQUIRED=(
  ["/comfyui/models/pulid/pulid_flux_v0.9.1.safetensors"]=1142099520
  ["/comfyui/models/clip/EVA02_CLIP_L_336_psz14_s6B.pt"]=856461210
  ["/comfyui/models/ultralytics/bbox/face_yolov8m.pt"]=52026019
)
for path in "${!REQUIRED[@]}"; do
  if [ ! -f "$path" ]; then
    echo "[smoke] FAIL: missing $path"; exit 1
  fi
  expected=${REQUIRED[$path]}
  actual=$(stat -c%s "$path")
  if [ "$actual" != "$expected" ]; then
    echo "[smoke] FAIL: size mismatch $path expected=$expected got=$actual"; exit 1
  fi
  echo "[smoke]   ok $path ($actual bytes)"
done

echo "[smoke] === antelopev2 pack (5 ONNX files) ==="
antelope_dir="/comfyui/models/insightface/models/antelopev2"
if [ ! -d "$antelope_dir" ]; then
  echo "[smoke] FAIL: missing $antelope_dir"; exit 1
fi
for f in 1k3d68.onnx 2d106det.onnx genderage.onnx glintr100.onnx scrfd_10g_bnkps.onnx; do
  if [ ! -f "$antelope_dir/$f" ]; then
    echo "[smoke] FAIL: missing antelopev2/$f"; exit 1
  fi
done
echo "[smoke]   ok antelopev2"

echo "[smoke] === Custom node directories present ==="
for d in ComfyUI-Impact-Pack ComfyUI-Impact-Subpack ComfyUI_PuLID_Flux_ll ComfyUI-PuLID-Flux; do
  if [ -d "/comfyui/custom_nodes/$d" ]; then
    echo "[smoke]   ok custom_nodes/$d"
  fi
done

echo "[smoke] === LoRA mount check (runtime-only via network volume) ==="
lora="/runpod-volume/models/loras/alara_karlisa_v2.safetensors"
if [ -f "$lora" ]; then
  echo "[smoke]   ok LoRA present at runtime: $lora"
else
  echo "[smoke]   note: LoRA not present (mount RunPod Network Volume 'ugcinf-lora' → /runpod-volume)"
fi

echo "[smoke] === ALL PASS ==="
