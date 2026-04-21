# gfx906-fa-vllm

[![Docker](https://img.shields.io/badge/docker-nickoptimal%2Fgfx906--fa--vllm-blue?logo=docker)](https://hub.docker.com/r/nickoptimal/gfx906-fa-vllm)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![ROCm](https://img.shields.io/badge/ROCm-7.2.1-red?logo=amd)](https://rocm.docs.amd.com/)
[![Upstream](https://img.shields.io/badge/upstream-mixa3607%2FML--gfx906-lightgrey?logo=github)](https://github.com/mixa3607/ML-gfx906)

**FlashAttention-style custom attention backend for vLLM on AMD MI50 / MI60 / Radeon VII (gfx906).**

This is a **downstream fork** of [`mixa3607/ML-gfx906`](https://github.com/mixa3607/ML-gfx906) — the
community-maintained ROCm / PyTorch / vLLM stack for deprecated gfx906 GPUs.
We reuse the upstream Docker image (`mixa3607/vllm-gfx906:0.19.1-rocm-7.2.1-aiinfos`) as the base
layer and ship **replacement attention kernels** plus a vLLM plugin that registers them as
`--attention-backend CUSTOM`.

On 8× MI50 with `MiniMax-M2.7-AWQ-4bit` and contexts up to 130K tokens, the custom kernels deliver
**+20…40 %** throughput over the stock `TRITON_ATTN` backend, and remain usable at 100K+ ctx where
Triton-generated code degrades sharply on Vega 20.

---

## Relationship to the upstream `mixa3607/ML-gfx906`

| Layer                              | Upstream (`mixa3607/ML-gfx906`)                        | This fork (`gfx906-fa-vllm`)                                       |
| :--------------------------------- | :----------------------------------------------------- | :----------------------------------------------------------------- |
| ROCm 7.2.1 patched for gfx906      | ✅ provides `rocm-gfx906:7.2.1-complete`                | — reused as-is                                                     |
| PyTorch 2.9 for gfx906             | ✅ provides `pytorch-gfx906:v2.9.0-rocm-7.2.1`          | — reused as-is                                                     |
| vLLM 0.19.1 for gfx906             | ✅ provides `vllm-gfx906:0.19.1-rocm-7.2.1-aiinfos`     | — used as Docker **base image**                                    |
| Attention backend                  | Default `TRITON_ATTN`                                  | 🔁 **replaced** by custom HIP kernels (this repo)                  |
| `vllm.general_plugins` entry point | —                                                      | ➕ `gfx906_fa = gfx906_fa_backend:register`                         |
| vLLM worker-spawn patches          | —                                                      | ➕ `sed` patches on `fa_utils.py` / `mla_attention.py` (see below) |

Everything below the attention layer (ROCm driver, hipBLAS, Tensile, RCCL, PyTorch, vLLM runtime)
comes from the upstream image unchanged. We do not ship our own ROCm / PyTorch builds and we do
not compete with upstream — we plug into it.

If you only need vLLM on gfx906 **without** the custom FA kernels, use the upstream image
directly. If you hit throughput issues on long contexts (≥ 32K) or want `CUSTOM` backend — use
this repo.

---

## Why a custom backend

Stock vLLM on ROCm routes attention through `TRITON_ATTN`. Its auto-generated kernels on gfx906
yield mediocre throughput at long contexts because of (a) sub-optimal occupancy for Vega 20 (wavefront = 64,
VGPR = 256), and (b) the absence of a paged-KV fast path tuned for this arch. `gfx906-fa-vllm`
addresses both:

- **Paged-KV attention directly in the kernel** (Level 3c direct-paged) — no intermediate
  `gather → forward` pipeline for decode.
- **Q8_0 quantization for K** in the KV cache — saves HBM bandwidth on decode.
- **Occupancy-aware launch bounds** tuned for gfx906 (`__launch_bounds__(128, 4)`, VGPR = 128 sweet spot).
- **Adaptive path selection** (short `Sq` → tile kernel, long `Sq` → direct-paged kernel).

## Performance

Benchmarked with `tests/bench_vllm.py` on `MiniMax-M2.7-AWQ-4bit`, 8× MI50, TP = 8, BS = 1:

### Short-context profile (`start_vllm_fa.sh`, CUDA Graphs ON)

|  prompt ctx | TTFT (ms) | ITL (ms) | **TG (tok/s)** |
| ----------: | --------: | -------: | -------------: |
|          1K |        89 |       36 |       **27.7** |
|          8K |       263 |       64 |       **15.6** |
|         32K |       360 |      130 |        **7.7** |

### Long-context profile (`start_vllm_fa_ngram.sh`, EAGER + ngram speculative decoding, K = 5)

|  prompt ctx | TTFT (ms) | ITL (ms) | **TG (tok/s)** | Δ vs `TRITON_ATTN` |
| ----------: | --------: | -------: | -------------: | -----------------: |
|         32K |       638 |      135 |        **7.4** |                +6% |
|         65K |       647 |      225 |        **4.0** |                  — |
|        100K |      1711 |      256 |        **3.9** |           **+32%** |
|        130K |      1246 |      332 |        **3.0** |           **+29%** |

Ngram speculative decoding is effective on repetitive text (RAG, code, document summarization):
acceptance rate 100 %, mean accepted length 4 / 5.

---

## Quick start

### 1. Pre-built Docker image

```bash
docker pull nickoptimal/gfx906-fa-vllm:mvp

docker run -d --name vllm \
    --network host --ipc host --shm-size 16g \
    --device /dev/kfd --device /dev/dri \
    --group-add 44 --group-add 993 \
    --cap-add CAP_SYS_PTRACE --security-opt seccomp=unconfined \
    -v /path/to/your/models:/models \
    -e NCCL_P2P_DISABLE=1 -e HIP_ENABLE_PEER_ACCESS=0 \
    nickoptimal/gfx906-fa-vllm:mvp \
    vllm serve /models/your-awq-model \
        --tensor-parallel-size 8 \
        --attention-backend CUSTOM \
        --disable-custom-all-reduce \
        --max-model-len 65536 \
        --gpu-memory-utilization 0.78
```

Or use the ready-made launch scripts:

```bash
git clone https://github.com/nick413-bit/gfx906-fa-vllm && cd gfx906-fa-vllm
MODEL=/models/MiniMax-M2.7-AWQ-4bit bash scripts/start_vllm_fa.sh
```

### 2. Long contexts (≥ 32K, RAG / code)

```bash
MODEL=/models/MiniMax-M2.7-AWQ-4bit bash scripts/start_vllm_fa_ngram.sh
```

### 3. Build from source

```bash
docker build -f docker/Dockerfile -t gfx906-fa-vllm:dev .

# Or build the extension in-place inside the upstream container:
docker run -it --rm --device /dev/kfd --device /dev/dri \
    -v $PWD:/src \
    docker.io/mixa3607/vllm-gfx906:0.19.1-rocm-7.2.1-aiinfos bash
# inside:
cd /src/extension && pip install -e . --no-deps
```

---

## Architecture

```
vLLM  →  --attention-backend CUSTOM  →  Gfx906FAImpl (extension/gfx906_fa_backend.py)
                                        │
                                        ├─ do_kv_cache_update:  fp16 key/value → paged KV cache (fp16)
                                        │
                                        └─ forward:  ─┬─ adaptive path select (extension/gfx906_fa_paged.py):
                                                      │    • seq_q ≤ 32 → direct-paged FA (paged kernel)
                                                      │    • seq_q >  32 → gather + flat FA (tile kernel)
                                                      │
                                                      ├─ gfx906_fa.reshape_and_cache_q8 (inline fp16 → Q8 on the fly)
                                                      ├─ gfx906_fa.gather_paged_kv_q8   (when gather is needed)
                                                      └─ gfx906_fa.forward              (core FA kernel)

HIP kernels (kernel/):
  fattn-q8.cuh         — flat FlashAttention kernel (Q8_0 K, fp16 V)
  fattn-q8-paged.cuh   — direct-paged variant (no gather)
  gfx906-common.cuh    — wavefront utilities, shuffle, reductions
```

### Plugin registration

The extension registers itself as a vLLM general plugin via `setup.py`:

```python
entry_points={
    "vllm.general_plugins": [
        "gfx906_fa = gfx906_fa_backend:register",
    ],
}
```

`vllm.plugins.load_general_plugins()` is called in every process (main, engine core, TP workers),
so `AttentionBackendEnum.CUSTOM` is registered **everywhere** — no `PYTHONPATH` hacks or manual
imports in user code.

### Base-image patches applied in `docker/Dockerfile`

Two tiny patches are applied to the upstream image:

```bash
sed -i 's/except ImportError:/except Exception:/g' \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/fa_utils.py
sed -i 's/except ImportError:/except Exception:/g' \
    /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/mla_attention.py
```

Rationale: in Python `spawn` multiprocessing, vLLM workers import `flash_attn` at module load
time, **before** `libamdhip64.so` is fully linked by PyTorch. Triton's AMD driver init then
fails with a `RuntimeError` (not `ImportError`), which the stock `except ImportError:` blocks
do not catch, and the worker crashes with `Triton cache error`. Since we drive attention
through `--attention-backend CUSTOM` (which does not use `flash_attn_varlen_func`), broadening
the exception to `except Exception:` is safe.

### Kernel design notes

1. **Q8_0 K quantization** — the side-buffer design (Level 0) is temporarily disabled because of
   synchronization issues with vLLM warm-up passes; the stable path is inline quantize inside
   `forward`. See `BASELINE.md §14` in the internal development history.
2. **Paged KV cache format** is bit-compatible with vLLM stock layout
   `(num_blocks, 2, block_size, num_kv_heads, head_dim)` — no allocator changes.
3. **Launch bounds** are tuned for gfx906: `__launch_bounds__(128, 4)` — 2 wavefronts per block
   (gfx906 wavefront = 64). More wavefronts push us past VGPR = 128 and cause spills.

---

## Compatibility

| Component     | Version                                                            |
| :------------ | :----------------------------------------------------------------- |
| GPU           | AMD MI50 / MI60 / Radeon VII (gfx906)                              |
| ROCm          | 7.2.1 (from `mixa3607/rocm-gfx906:7.2.1-complete`)                 |
| PyTorch       | 2.9                                                                |
| vLLM          | 0.19.1 (from `mixa3607/vllm-gfx906:0.19.1-rocm-7.2.1-aiinfos`)     |
| Models tested | `MiniMax-M2.7-AWQ-4bit`                                            |

### Verified topologies

- 8× MI50 on AMD EPYC 7002, PCIe Gen4 x16, Microsemi switches — **OK**
- TP = 8, `--disable-custom-all-reduce`, P2P OFF — default configuration

Not tested on MI100 / MI200 (gfx908 / gfx90a). The kernels are gfx906-specific and would need
adaptation for other architectures.

---

## Environment variables

| Env                      | Default | Purpose                                                                                         |
| :----------------------- | :------ | :---------------------------------------------------------------------------------------------- |
| `GFX906_FA_LEGACY`       | `1`     | `1` — inline fp16 → Q8 (stable path); `0` — Q8 side-buffer (experimental)                       |
| `GFX906_FA_FUSED`        | `1`     | `1` — HIP gather kernel; `0` — Python fancy-index                                               |
| `GFX906_FA_DIRECT_PAGED` | `auto`  | `auto` / `1` / `0` — direct-paged FA path                                                       |
| `GFX906_FA_GATHER_V`     | `2`     | `1` — per-token gather; `2` — paged-block-coalesced gather                                      |
| `NCCL_P2P_DISABLE`       | `1`     | P2P on our topology gave only +3…6 %, not worth the complexity — disabled by default            |
| `HIP_ENABLE_PEER_ACCESS` | `0`     | Paired with `NCCL_P2P_DISABLE`                                                                  |

---

## Limitations and roadmap

- [ ] **Q8 V quantization** — currently V stays fp16 in KV cache. Biggest remaining headroom for
      decode at 100K+ (+20…30 % expected).
- [ ] **EAGLE draft model** — +2…3× on generic prompts, requires training a drafter.
- [ ] **Persistent kernel** — remove per-layer launch overhead over 62 layers. High risk.
- [ ] **Q8 K side-buffer** — fix synchronization with vLLM warm-up passes (re-enable Level 0).

---

## Tests

```bash
# Unit tests (inside the container):
pytest tests/test_smoke.py                    # basic sanity
pytest tests/test_backend.py                  # vLLM AttentionImpl integration
pytest tests/test_backend_vs_legacy.py        # Q8 side-buffer reproducer
pytest tests/test_gather.py                   # gather paged KV → Q8
pytest tests/test_fa_mixed_batch.py           # mixed batch (prefill + decode)
pytest tests/test_direct_paged_bitexact.py    # direct-paged FA vs flat path

# End-to-end benchmark (server must be running):
python tests/bench_vllm.py --url http://localhost:40044/v1 --model minimax \
    --lengths 1000 8000 32000 --max-new-tokens 32 --label baseline
```

---

## License

Apache-2.0 — see [LICENSE](LICENSE).

The core FlashAttention-Q8_0 math is derived from ideas in
[llama.cpp](https://github.com/ggerganov/llama.cpp) (MIT), adapted to the vLLM paged-KV layout
and re-tuned for gfx906 occupancy.

---

## Credits

- [**@mixa3607**](https://github.com/mixa3607) / [**ML-gfx906**](https://github.com/mixa3607/ML-gfx906) —
  upstream project this repo forks from. Ships the ROCm 7.2.1, PyTorch 2.9 and vLLM 0.19.1
  images for gfx906 (`mixa3607/vllm-gfx906:0.19.1-rocm-7.2.1-aiinfos` is our Docker base).
- [**llama.cpp**](https://github.com/ggerganov/llama.cpp) — original FlashAttention Q8_0 kernel
  design.
- [**vLLM**](https://github.com/vllm-project/vllm) — paged-KV cache layout, `AttentionImpl` API
  and plugin system (`vllm.general_plugins`) that makes `CUSTOM` backend registration clean.

## Authors

- **Nick** — [nick413@gmail.com](mailto:nick413@gmail.com)
