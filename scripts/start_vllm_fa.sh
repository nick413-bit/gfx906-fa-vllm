#!/bin/bash
# gfx906-fa-vllm — "короткий" профиль (≤ 32K токенов в запросе).
# CUDA Graphs включены, без speculative decoding.
# Производительность (MiniMax-M2.7-AWQ-4bit, 8× MI50 TP=8):
#   1K ≈ 27 tok/s   8K ≈ 15 tok/s   32K ≈ 7.7 tok/s
#
# Для длинных контекстов (RAG / code / analysis ≥ 32K) — используйте
# start_vllm_fa_ngram.sh (ngram speculative decoding, +29% на 100K+).
#
# Usage:
#   MODEL=/path/to/model bash start_vllm_fa.sh
# Override args via env:
#   TP, PORT, MAX_MODEL_LEN, MAX_NUM_SEQS, GPU_UTIL, IMAGE, MODEL_MOUNT,
#   EXTRA_MOUNT, CONTAINER_NAME

set -euo pipefail

: "${IMAGE:=nickoptimal/gfx906-fa-vllm:mvp}"
: "${CONTAINER_NAME:=gfx906-fa-vllm}"
: "${MODEL:=/models/cyankiwi/MiniMax-M2.7-AWQ-4bit}"
: "${SERVED_MODEL_NAME:=minimax}"
: "${PORT:=40044}"
: "${TP:=8}"
: "${MAX_MODEL_LEN:=65536}"
: "${MAX_NUM_SEQS:=4}"
: "${MAX_NUM_BATCHED_TOKENS:=4096}"
: "${GPU_UTIL:=0.78}"

# P2P отключён по умолчанию — на нашей топологии (Microsemi PCIe switches)
# прирост 3–6% не оправдывает risk-а стабильности. Включается явно:
# P2P_DIS=0 HIP_PEER=1 bash start_vllm_fa.sh
: "${P2P_DIS:=1}"
: "${HIP_PEER:=0}"

# Backend-specific knobs (см. extension/gfx906_fa_backend.py):
: "${LEGACY:=1}"        # inline fp16→Q8 quantize (стабильный путь)
: "${FUSED:=1}"         # HIP gather kernel (vs Python fancy-index)
: "${DIRECT_PAGED:=auto}"  # adaptive path (direct-paged FA для long ctx)
: "${GATHER_V:=2}"      # paged-block-coalesced gather

: "${MODEL_MOUNT:=/var/lib/gpustack/cache/huggingface}"
: "${EXTRA_MOUNT:=}"

docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

docker run -d --name "$CONTAINER_NAME" \
    --network host --ipc host --shm-size 16g \
    --device /dev/kfd --device /dev/dri \
    --group-add "${KFD_GROUP:-44}" --group-add "${RENDER_GROUP:-993}" \
    ${HOST_GROUPS:+--group-add $HOST_GROUPS} \
    --cap-add CAP_SYS_PTRACE \
    --security-opt seccomp=unconfined --security-opt label=disable \
    -v "$MODEL_MOUNT":/models \
    ${EXTRA_MOUNT:+-v $EXTRA_MOUNT} \
    -e OMP_NUM_THREADS=4 \
    -e VLLM_USE_V1=1 \
    -e VLLM_USE_TRITON_AWQ=1 \
    -e VLLM_USE_TRITON_FLASH_ATTN=1 \
    -e FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
    -e FLASH_ATTENTION_TRITON_AMD_AUTOTUNE=0 \
    -e TORCH_BLAS_PREFER_HIPBLASLT=0 \
    -e NCCL_P2P_DISABLE="$P2P_DIS" \
    -e HIP_ENABLE_PEER_ACCESS="$HIP_PEER" \
    -e GFX906_FA_DIRECT_PAGED="$DIRECT_PAGED" \
    -e GFX906_FA_LEGACY="$LEGACY" \
    -e GFX906_FA_FUSED="$FUSED" \
    -e GFX906_FA_GATHER_V="$GATHER_V" \
    -e NCCL_IB_DISABLE=1 \
    -e NCCL_DEBUG=WARN \
    -e RCCL_MSCCL_ENABLE=0 \
    -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
    "$IMAGE" \
    vllm serve "$MODEL" \
        --host 0.0.0.0 --port "$PORT" \
        --served-model-name "$SERVED_MODEL_NAME" \
        --trust-remote-code \
        --dtype float16 \
        --tensor-parallel-size "$TP" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
        --disable-custom-all-reduce \
        --attention-backend CUSTOM

echo "Started $CONTAINER_NAME (IMAGE=$IMAGE MODEL=$MODEL PORT=$PORT TP=$TP)"
echo "Logs:     docker logs -f $CONTAINER_NAME"
echo "Health:   curl http://localhost:$PORT/v1/models"
