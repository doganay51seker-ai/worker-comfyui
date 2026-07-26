#!/usr/bin/env bash
# Smoke test — run at Docker build time (COPY + RUN) to fail fast on:
# - custom node import errors
# - missing model files
# - broken Python deps
#
# Also usable at runtime: `docker run --rm --entrypoint smoke-test.sh <image>`
set -euo pipefail

echo "[smoke] === Python + torch presence ==="
python -c "import sys, torch; print(f'python={sys.version.split()[0]} torch={torch.__version__} cuda_available={torch.cuda.is_available()}')"

echo "[smoke] === pip check (dependency conflicts) ==="
uv pip check

echo "[smoke] === Model files present ==="
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

echo "[smoke] === antelopev2 pack ==="
antelope_dir="/comfyui/models/insightface/models/antelopev2"
if [ ! -d "$antelope_dir" ]; then
  echo "[smoke] FAIL: missing $antelope_dir"; exit 1
fi
for f in 1k3d68.onnx 2d106det.onnx genderage.onnx glintr100.onnx scrfd_10g_bnkps.onnx; do
  if [ ! -f "$antelope_dir/$f" ]; then
    echo "[smoke] FAIL: missing antelopev2/$f"; exit 1
  fi
done
echo "[smoke]   ok antelopev2 (5 ONNX files)"

echo "[smoke] === Custom node directories present ==="
for d in ComfyUI-Impact-Pack ComfyUI-Impact-Subpack ComfyUI_PuLID_Flux_ll ComfyUI-PuLID-Flux; do
  if [ -d "/comfyui/custom_nodes/$d" ]; then
    echo "[smoke]   ok custom_nodes/$d"
  fi
done

echo "[smoke] === Custom node Python import (surfaces node registration errors) ==="
python - <<'PY'
import importlib.util
import os
import sys
import traceback

CUSTOM_NODES = "/comfyui/custom_nodes"
sys.path.insert(0, "/comfyui")

errors = []
loaded_modules = []

# Attempt to import each custom node __init__.py directly
for name in sorted(os.listdir(CUSTOM_NODES)):
    init_path = os.path.join(CUSTOM_NODES, name, "__init__.py")
    if not os.path.isfile(init_path):
        continue
    try:
        spec = importlib.util.spec_from_file_location(f"custom_nodes.{name}", init_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded_modules.append(name)
    except Exception as e:
        errors.append((name, repr(e), traceback.format_exc()))

print(f"[smoke] loaded custom_nodes: {loaded_modules}")

if errors:
    print(f"[smoke] FAIL: {len(errors)} custom_node import errors")
    for name, err, tb in errors:
        print(f"\n--- {name} ---\n{err}\n{tb}")
    sys.exit(1)
PY

echo "[smoke] === InsightFace antelope pack loadable on CPU ==="
python -c "
import insightface
app = insightface.app.FaceAnalysis(name='antelopev2', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
print('[smoke]   ok insightface antelopev2 loaded')
"

echo "[smoke] === LoRA mount check (WARNING only — mounted at runtime via network volume) ==="
lora="/runpod-volume/models/loras/alara_karlisa_v2.safetensors"
if [ -f "$lora" ]; then
  echo "[smoke]   ok LoRA present at runtime: $lora"
else
  echo "[smoke]   note: LoRA not present at build time — must be mounted via RunPod Network Volume 'ugcinf-lora' at runtime"
fi

echo "[smoke] === ALL PASS ==="
