# gfx906-fa-vllm — кастомный FlashAttention-style attention backend
# для AMD MI50/MI60 (gfx906), работает поверх vLLM.
#
# Базовый образ: mixa3607/vllm-gfx906:0.19.1-rocm-7.2.1-aiinfos
# (vLLM 0.19.1-dev + ROCm 7.2.1 + PyTorch 2.9 для gfx906).
# Спасибо @mixa3607 за maintained image.
#
# Multi-stage build: (1) компиляция HIP extension, (2) рантайм-образ.
#
# Patches applied to base image:
#   - fa_utils.py: except ImportError → except Exception
#     (Triton AMD driver init fails in worker subprocesses before
#     libamdhip64.so is loaded by PyTorch; catch RuntimeError gracefully)
#   - mla_attention.py: same patch for the ROCm fallback import block
#   These patches are safe with --attention-backend CUSTOM because our
#   backend handles attention directly without flash_attn_varlen_func.

ARG BASE_IMAGE=docker.io/mixa3607/vllm-gfx906:0.19.1-rocm-7.2.1-aiinfos

# ---------------------------------------------------------------------------
# STAGE 1: build HIP extension (.so)
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS builder

ENV PYTORCH_ROCM_ARCH=gfx906 \
    HIP_ARCHS=gfx906 \
    AMDGPU_TARGETS=gfx906 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /src
COPY extension/ ./extension/
COPY kernel/    ./kernel/
WORKDIR /src/extension

RUN pip install --no-cache-dir -e . --no-deps 2>&1 | tail -20

# ---------------------------------------------------------------------------
# STAGE 2: runtime
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="gfx906-fa-vllm" \
      org.opencontainers.image.description="FlashAttention-style custom attention backend for vLLM on AMD MI50/MI60 (gfx906)" \
      org.opencontainers.image.source="https://github.com/nick413/gfx906-fa-vllm" \
      org.opencontainers.image.licenses="Apache-2.0"

COPY --from=builder /src/extension /opt/gfx906_fa
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Patch vLLM: catch RuntimeError (Triton AMD init) in flash_attn imports.
# In worker subprocesses (spawn), libamdhip64.so is not yet loaded by
# PyTorch when flash_attn is imported at module level — Triton's
# hipGetProcAddress then fails with RuntimeError (not ImportError).
# For CUSTOM backend we don't use flash_attn_varlen_func, so this is safe.
RUN FA=/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/fa_utils.py && \
    MLA=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/mla_attention.py && \
    sed -i 's/except ImportError:/except Exception:/g' "$FA" && \
    sed -i 's/except ImportError:/except Exception:/g' "$MLA"

# Extension path + gfx906 FA knobs (overridable via -e at docker run)
ENV PYTHONPATH=/opt/gfx906_fa \
    GFX906_FA_LEGACY=1 \
    GFX906_FA_FUSED=1 \
    GFX906_FA_DIRECT_PAGED=auto \
    GFX906_FA_GATHER_V=2

# Recommended vLLM flags (overridable)
ENV VLLM_USE_TRITON_AWQ=1 \
    VLLM_USE_V1=1 \
    TORCH_BLAS_PREFER_HIPBLASLT=0 \
    NCCL_P2P_DISABLE=1 \
    HIP_ENABLE_PEER_ACCESS=0 \
    NCCL_IB_DISABLE=1 \
    RCCL_MSCCL_ENABLE=0 \
    PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
    OMP_NUM_THREADS=4

# entrypoint.sh: pip install -e /opt/gfx906_fa then exec "$@"
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
