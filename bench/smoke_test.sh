#!/usr/bin/env bash
# Build-time + runtime smoke test — CODEX v3 fail-fast spec.
#
# Non-zero exit conditions:
#   1. dill or piexif cannot be imported by /opt/venv/bin/python
#   2. ComfyUI --quick-test-for-ci output contains "IMPORT FAILED"
#      OR "Cannot import" for any required custom_node
#   3. PuLID module cannot be imported at all
#   4. PulidFluxInsightFaceLoader not present in NODE_CLASS_MAPPINGS
#   5. ComfyUI-Impact-Pack or ComfyUI-Impact-Subpack not loaded (grep INFO line)
#
# All uv/python calls target /opt/venv (base image runtime venv). comfy-cli's
# workspace venv (/comfyui/.venv) is intentionally ignored.
set -euo pipefail

PY_BIN=/opt/venv/bin/python

echo "[smoke] === venv sanity ==="
"$PY_BIN" -c "import sys, torch; print(f'python={sys.version.split()[0]} venv={sys.prefix} torch={torch.__version__}')"

echo "[smoke] === uv pip check (/opt/venv only) ==="
uv pip check --python "$PY_BIN"

echo "[smoke] === Required deps import ==="
"$PY_BIN" -c "import dill, piexif; print('  ok dill+piexif')"

echo "[smoke] === ONNX Runtime CPU/GPU distribution hygiene ==="
# uv pip check does NOT flag onnxruntime + onnxruntime-gpu coexistence because
# they share the same importable module name. We check distribution metadata
# explicitly and require CUDAExecutionProvider capability.
"$PY_BIN" - <<'PYORT'
import sys
import importlib.metadata as md

fail = []

# onnxruntime-gpu present and pinned to 1.19.2
try:
    gpu_ver = md.version("onnxruntime-gpu")
    if gpu_ver != "1.19.2":
        fail.append(f"onnxruntime-gpu version {gpu_ver} != 1.19.2")
    else:
        print(f"  ok onnxruntime-gpu=={gpu_ver}")
except md.PackageNotFoundError:
    fail.append("onnxruntime-gpu distribution not installed")

# CPU onnxruntime must NOT coexist (they share the same top-level module)
try:
    cpu_ver = md.version("onnxruntime")
    fail.append(f"CPU onnxruntime=={cpu_ver} is installed alongside onnxruntime-gpu (module conflict)")
except md.PackageNotFoundError:
    print("  ok CPU onnxruntime distribution absent")

# Import + provider capability
try:
    import onnxruntime
    providers = onnxruntime.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        fail.append(f"CUDAExecutionProvider missing from providers: {providers}")
    else:
        print(f"  ok CUDAExecutionProvider present ({providers})")
except Exception as e:
    fail.append(f"onnxruntime import failed: {e!r}")

if fail:
    for f in fail:
        print(f"  FAIL: {f}")
    sys.exit(1)
PYORT

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
[ -d "$antelope_dir" ] || { echo "[smoke] FAIL: missing $antelope_dir"; exit 1; }
for f in 1k3d68.onnx 2d106det.onnx genderage.onnx glintr100.onnx scrfd_10g_bnkps.onnx; do
  [ -f "$antelope_dir/$f" ] || { echo "[smoke] FAIL: missing antelopev2/$f"; exit 1; }
done
echo "[smoke]   ok antelopev2"

echo "[smoke] === PuLID module direct import + INSIGHT loader in mappings ==="
# ComfyUI's own custom-node loader is stateful; here we import the module
# in isolation and inspect its published NODE_CLASS_MAPPINGS. This does NOT
# require ComfyUI's init flow.
"$PY_BIN" - <<'PYCHK'
import importlib.util, os, sys, pathlib
sys.path.insert(0, "/comfyui")

# Try both known repo paths; the image contains exactly ONE
candidates = [
    "/comfyui/custom_nodes/ComfyUI_PuLID_Flux_ll",   # candidate A (lldacing)
    "/comfyui/custom_nodes/ComfyUI-PuLID-Flux",       # candidate B (balazik)
]
node_dir = next((c for c in candidates if os.path.isdir(c)), None)
if not node_dir:
    print("FAIL: no PuLID custom_node dir found")
    sys.exit(1)

# Import the package __init__.py under a synthetic name
pkg_name = "pulid_pkg_smoke"
spec = importlib.util.spec_from_file_location(
    pkg_name,
    pathlib.Path(node_dir, "__init__.py"),
    submodule_search_locations=[node_dir],
)
mod = importlib.util.module_from_spec(spec)
sys.modules[pkg_name] = mod
spec.loader.exec_module(mod)

ncm = getattr(mod, "NODE_CLASS_MAPPINGS", None)
if not ncm:
    print("FAIL: PuLID module has no NODE_CLASS_MAPPINGS")
    sys.exit(1)

if "PulidFluxInsightFaceLoader" not in ncm:
    print("FAIL: PulidFluxInsightFaceLoader not in NODE_CLASS_MAPPINGS")
    print("      registered:", sorted(ncm.keys()))
    sys.exit(1)

print("  ok PuLID imported +", len(ncm), "nodes registered")
print("  ok PulidFluxInsightFaceLoader is present")
if "PulidFluxFaceNetLoader" in ncm:
    print("  info FaceNet loader also present (facenet_pytorch installed)")
else:
    print("  info FaceNet loader intentionally skipped (facenet_pytorch not installed)")
PYCHK

echo "[smoke] === ComfyUI --quick-test-for-ci (fail-fast on IMPORT FAILED) ==="
QT_LOG=/tmp/quick_test.log
cd /comfyui && timeout 300 "$PY_BIN" main.py --quick-test-for-ci --cpu 2>&1 | tee "$QT_LOG"

# Any IMPORT FAILED line is fatal
if grep -qE "IMPORT FAILED" "$QT_LOG"; then
  echo "[smoke] FAIL: quick-test reported IMPORT FAILED:"
  grep -E "IMPORT FAILED" "$QT_LOG"
  exit 1
fi

# Any "Cannot import ... custom_nodes/<REQUIRED>" is fatal
REQUIRED_NODES=("ComfyUI-Impact-Pack" "ComfyUI-Impact-Subpack")
# PuLID dir name differs per candidate
if [ -d "/comfyui/custom_nodes/ComfyUI_PuLID_Flux_ll" ]; then
  REQUIRED_NODES+=("ComfyUI_PuLID_Flux_ll")
elif [ -d "/comfyui/custom_nodes/ComfyUI-PuLID-Flux" ]; then
  REQUIRED_NODES+=("ComfyUI-PuLID-Flux")
fi

FAIL=0
for node in "${REQUIRED_NODES[@]}"; do
  if grep -qE "Cannot import /comfyui/custom_nodes/${node}\b" "$QT_LOG"; then
    echo "[smoke] FAIL: quick-test could not import $node"
    grep -E "Cannot import /comfyui/custom_nodes/${node}" "$QT_LOG"
    FAIL=1
  fi
done
if [ $FAIL -ne 0 ]; then exit 1; fi

# Positive-signal check: each required node must show a Loading line
for node in "${REQUIRED_NODES[@]}"; do
  # Impact loaders print "### Loading: ComfyUI-Impact-Pack" / Subpack.
  # PuLID lldacing prints nothing custom; falling back to no-Cannot-import above is enough.
  case "$node" in
    ComfyUI-Impact-Pack|ComfyUI-Impact-Subpack)
      if ! grep -qE "### Loading: ${node}" "$QT_LOG"; then
        echo "[smoke] FAIL: no Loading line for $node in quick-test output"
        exit 1
      fi
      echo "[smoke]   ok Loading line for $node"
      ;;
    *)
      echo "[smoke]   ok no Cannot-import for $node"
      ;;
  esac
done

echo "[smoke] === LoRA mount check (runtime-only via network volume) ==="
lora="/runpod-volume/models/loras/alara_karlisa_v2.safetensors"
if [ -f "$lora" ]; then
  echo "[smoke]   ok LoRA present at runtime: $lora"
else
  echo "[smoke]   note: LoRA not present (mount RunPod Network Volume 'ugcinf-lora' → /runpod-volume)"
fi

echo "[smoke] === ALL PASS ==="
