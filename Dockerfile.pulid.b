# ============================================================================
# CANDIDATE B: balazik/ComfyUI-PuLID-Flux (original FLUX port, alpha/prototype)
# Same base + FaceDetailer + models — only PuLID node source differs from A
# Used for A/B comparison against Candidate A (lldacing fork)
# NOTE: Alara LoRA mounted via RunPod Network Volume `ugcinf-lora` (cec0y73w93)
# ============================================================================

FROM runpod/worker-comfyui:5.8.6-flux1-dev-fp8

ARG PULID_NODE_SHA=a80912fc3435c358607bf4b43a58dbcbebdb09ff
ARG IMPACT_PACK_SHA=429d0159ad429e64d2b3916e6e7be9c22d025c3c
ARG IMPACT_SUBPACK_SHA=50c7b71a6a224734cc9b21963c6d1926816a97f1
ARG HF_PULID_REV=492b1451255dc9d9bc3c857259690b5f8b998d4a
ARG HF_EVACLIP_REV=11afd202f2ae80869d6cef18b1ec775e79bd8d12
ARG HF_ADETAILER_REV=53cc19de382014514d9d4038601d261a7faa9b7b
ARG INSIGHTFACE_TAG=v0.7

ARG SIZE_PULID=1142099520
ARG SIZE_EVACLIP=856461210
ARG SIZE_YOLO_BBOX=52026019

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake libgl1 libglib2.0-0 wget unzip ca-certificates \
    python3.12-dev && \
    rm -rf /var/lib/apt/lists/*

# facenet-pytorch removed — incompatible with base torch 2.12 (see .a for details)
RUN uv pip install \
    insightface==0.7.3 \
    onnxruntime-gpu==1.19.2 \
    facexlib==0.3.0 \
    ftfy==6.2.3 \
    einops==0.8.0 \
    timm==1.0.11 \
    ultralytics==8.3.162

# PuLID FLUX node (balazik original — pinned to master branch commit)
RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/balazik/ComfyUI-PuLID-Flux.git && \
    cd ComfyUI-PuLID-Flux && \
    git checkout ${PULID_NODE_SHA} && \
    if [ -f requirements.txt ]; then \
      grep -viE '^(torch|torchvision|torchaudio|xformers)([<>=~! ]|$)' requirements.txt > /tmp/req.txt; \
      uv pip install -r /tmp/req.txt; \
    fi

RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git && \
    cd ComfyUI-Impact-Pack && git checkout ${IMPACT_PACK_SHA} && \
    if [ -f requirements.txt ]; then \
      grep -viE '^(torch|torchvision|torchaudio|xformers|git\+)' requirements.txt > /tmp/ip.txt; \
      uv pip install -r /tmp/ip.txt; \
    fi && \
    cd /comfyui/custom_nodes && \
    git clone https://github.com/ltdrdata/ComfyUI-Impact-Subpack.git && \
    cd ComfyUI-Impact-Subpack && git checkout ${IMPACT_SUBPACK_SHA} && \
    if [ -f requirements.txt ]; then \
      grep -viE '^(torch|torchvision|torchaudio|xformers)([<>=~! ]|$)' requirements.txt > /tmp/is.txt; \
      uv pip install -r /tmp/is.txt; \
    fi

RUN uv pip check

RUN mkdir -p /comfyui/models/pulid && \
    wget -q --tries=3 -O /comfyui/models/pulid/pulid_flux_v0.9.1.safetensors \
      "https://huggingface.co/guozinan/PuLID/resolve/${HF_PULID_REV}/pulid_flux_v0.9.1.safetensors" && \
    ACTUAL_SIZE=$(stat -c%s /comfyui/models/pulid/pulid_flux_v0.9.1.safetensors) && \
    if [ "$ACTUAL_SIZE" != "$SIZE_PULID" ]; then \
      echo "FAIL: pulid size mismatch expected=$SIZE_PULID got=$ACTUAL_SIZE"; exit 1; \
    fi

RUN mkdir -p /comfyui/models/clip && \
    wget -q --tries=3 -O /comfyui/models/clip/EVA02_CLIP_L_336_psz14_s6B.pt \
      "https://huggingface.co/QuanSun/EVA-CLIP/resolve/${HF_EVACLIP_REV}/EVA02_CLIP_L_336_psz14_s6B.pt" && \
    ACTUAL_SIZE=$(stat -c%s /comfyui/models/clip/EVA02_CLIP_L_336_psz14_s6B.pt) && \
    if [ "$ACTUAL_SIZE" != "$SIZE_EVACLIP" ]; then \
      echo "FAIL: eva-clip size mismatch expected=$SIZE_EVACLIP got=$ACTUAL_SIZE"; exit 1; \
    fi

RUN mkdir -p /comfyui/models/insightface/models && \
    cd /comfyui/models/insightface/models && \
    wget -q --tries=3 \
      "https://github.com/deepinsight/insightface/releases/download/${INSIGHTFACE_TAG}/antelopev2.zip" && \
    unzip -q antelopev2.zip && rm antelopev2.zip

RUN mkdir -p /comfyui/models/ultralytics/bbox && \
    wget -q --tries=3 -O /comfyui/models/ultralytics/bbox/face_yolov8m.pt \
      "https://huggingface.co/Bingsu/adetailer/resolve/${HF_ADETAILER_REV}/face_yolov8m.pt" && \
    ACTUAL_SIZE=$(stat -c%s /comfyui/models/ultralytics/bbox/face_yolov8m.pt) && \
    if [ "$ACTUAL_SIZE" != "$SIZE_YOLO_BBOX" ]; then \
      echo "FAIL: yolo bbox size mismatch expected=$SIZE_YOLO_BBOX got=$ACTUAL_SIZE"; exit 1; \
    fi

ENV COMFY_MANAGER_MODE=offline

# Load insightface antelope pack — FAIL the build if it can't initialize
RUN python -c "import insightface; app=insightface.app.FaceAnalysis(name='antelopev2', root='/comfyui/models/insightface', providers=['CPUExecutionProvider']); app.prepare(ctx_id=0, det_size=(640,640))"

# NOTE (Codex): balazik EVA loader does NOT use baked /comfyui/models/clip/... —
# it downloads via HF cache on first inference. Baked EVA-CLIP file is unused in this
# candidate. Only build B if candidate A fails; expect a first-inference download.

# Build-time smoke test 1/2: boot ComfyUI on CPU, catches broken imports
RUN cd /comfyui && timeout 300 python main.py --quick-test-for-ci --cpu

# Build-time smoke test 2/2: actually EXECUTE runtime check script
COPY bench/smoke_test.sh /usr/local/bin/smoke-test.sh
RUN chmod +x /usr/local/bin/smoke-test.sh && /usr/local/bin/smoke-test.sh

LABEL org.opencontainers.image.title="ugc-pulid-candidate-b" \
      org.opencontainers.image.description="PuLID FLUX (balazik original) + Impact Pack FaceDetailer + Alara LoRA support (LoRA mounted via network volume)" \
      org.opencontainers.image.source="https://github.com/doganay51seker-ai/worker-comfyui"
