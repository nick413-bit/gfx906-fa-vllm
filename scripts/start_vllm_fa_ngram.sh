#!/bin/bash
# gfx906-fa-vllm — "долгий" профиль (≥ 32K, RAG / code / document analysis).
# EAGER mode + ngram speculative decoding (K=5, lookup=3..4).
# Производительность (MiniMax-M2.7-AWQ-4bit, 8× MI50 TP=8, повторяющийся текст):
#   32K ≈ 7.4 tok/s   65K ≈ 4.0 tok/s   100K ≈ 3.9 tok/s   130K ≈ 3.0 tok/s
#   (+29% к baseline на 100K+; на коротких ctx прирост меньше)
#
# Usage:
#   MODEL=/path/to/model bash start_vllm_fa_ngram.sh
# Override args via env (см. start_vllm_fa.sh) + ngram:
#   NGRAM_K=5 NGRAM_MIN=3 NGRAM_MAX=4 bash start_vllm_fa_ngram.sh

set -euo pipefail

: "${IMAGE:=nickoptimal/gfx906-fa-vllm:mvp-0.1}"
: "${CONTAINER_NAME:=gfx906-fa-vllm}"
: "${MODEL:=/models/cyankiwi/MiniMax-M2.7-AWQ-4bit}"
: "${SERVED_MODEL_NAME:=minimax}"
: "${PORT:=40044}"
: "${TP:=8}"
: "${MAX_MODEL_LEN:=131072}"
: "${MAX_NUM_SEQS:=2}"
: "${MAX_NUM_BATCHED_TOKENS:=8192}"  # увеличено для speculative draft slots
: "${GPU_UTIL:=0.88}"

: "${P2P_DIS:=1}"
: "${HIP_PEER:=0}"

: "${LEGACY:=1}"
: "${FUSED:=1}"
: "${DIRECT_PAGED:=auto}"
: "${GATHER_V:=2}"

# ngram speculative decoding
: "${NGRAM_K:=5}"
: "${NGRAM_MIN:=3}"
: "${NGRAM_MAX:=4}"

SPEC_CONFIG="{\"method\":\"ngram\",\"num_speculative_tokens\":$NGRAM_K,\"prompt_lookup_min\":$NGRAM_MIN,\"prompt_lookup_max\":$NGRAM_MAX}"

: "${MODEL_MOUNT:=/var/lib/vllm/models}"
: "${EXTRA_MOUNT:=}"

docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

docker run -d --name "$CONTAINER_NAME" \
    --network host --ipc host --shm-size 16g \
    --device /dev/kfd --device /dev/dri \
    --group-add video --group-add render \
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
        --attention-backend CUSTOM \
        --enforce-eager \
        --speculative-config "$SPEC_CONFIG"

echo "Started $CONTAINER_NAME (ngram K=$NGRAM_K, MAX_MODEL_LEN=$MAX_MODEL_LEN)"
echo "Logs:     docker logs -f $CONTAINER_NAME"
echo "Health:   curl http://localhost:$PORT/v1/models"
