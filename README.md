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

## Two serving profiles

This repo ships **two pre-configured launch profiles**. They are not different
builds — it is the same image and the same kernels, just different vLLM
runtime settings. The trade-off is fundamental:

### Profile A — short context, high throughput

Script: [`scripts/start_vllm_fa.sh`](scripts/start_vllm_fa.sh)

- **CUDA Graphs ON** (no `--enforce-eager`)
- Attention routed through our custom kernels (`--attention-backend CUSTOM`)
- No speculative decoding

**When to use:** chat / Q&A / short prompts, typical context ≤ 32K tokens.

**What you get:** the best decode speed on MI50 for short contexts (27.7 tok/s
at 1K ctx on MiniMax-M2.7-AWQ-4bit, 8× MI50 TP = 8). CUDA Graphs eliminate
per-layer launch overhead, which matters a lot when each decode step is cheap.

**What to expect at long context:** throughput degrades roughly linearly with
KV-cache size — at 32K it is already ~3.5× slower than at 1K (7.7 vs 27.7 tok/s).
This is fundamental to attention complexity on Vega 20 HBM bandwidth, not a
bug we can fix without speculative decoding.

### Profile B — long context, stable throughput via n-gram speculative decoding

Script: [`scripts/start_vllm_fa_ngram.sh`](scripts/start_vllm_fa_ngram.sh)

- **`--enforce-eager`** (CUDA Graphs OFF — required for speculative decoding today)
- Same custom attention backend
- **N-gram speculative decoding** (`--speculative-config` with `method: ngram`,
  `num_speculative_tokens: 5`, `prompt_lookup_max: 4`)

**When to use:** RAG, long document summarization, code generation, agents,
anything with ≥ 32K context or repetitive / retrievable text.

**How it works:** the draft model is a trivial n-gram lookup over the current
prompt + generated text — no separate draft model, no extra VRAM, zero setup
cost. For each decode step we propose up to 5 tokens by finding matching
n-grams in the context; the main model then verifies all 5 in a single forward
pass. When the context is genuinely repetitive (RAG quotes, code, re-mentions
of named entities) acceptance rate is close to 100 % and we effectively run
~4× faster per token on long prompts.

**Why `--enforce-eager`:** with speculative decoding, decode batches have
variable shape (1 to `num_speculative_tokens + 1` tokens per step), so vLLM's
CUDA Graphs path cannot capture a single graph and currently falls back to
eager anyway. Explicitly disabling graphs avoids a startup hang we observed
during graph capture.

### Which profile should I pick?

| Your workload                                 | Profile | Why                                                                  |
| :-------------------------------------------- | :------ | :------------------------------------------------------------------- |
| Chat, ≤ 8K context                            | **A**   | CUDA Graphs win by a mile on short ctx                               |
| 8–32K short-answer Q&A                        | **A**   | Still faster; n-gram gain not big enough yet                         |
| RAG with cited passages                       | **B**   | Heavy quoting from context → n-gram acceptance ≈ 100 %               |
| Code generation / completion                  | **B**   | Repetitive tokens (indent, symbols, imports) → high acceptance       |
| Long-document summarization / rewrite         | **B**   | Output reuses source phrasing                                        |
| 100K+ context of any kind                     | **B**   | At this size eager + ngram beats CUDA Graphs without spec decoding   |
| Creative / open-ended generation, low ctx     | **A**   | Low acceptance on open-ended text — no benefit from n-gram           |

If you are unsure, run both on your actual traffic and measure. Switching is
just re-launching the container with a different script.

## Performance

Benchmarked with `tests/bench_vllm.py` on `MiniMax-M2.7-AWQ-4bit`, 8× MI50, TP = 8, BS = 1:

### Profile A — `start_vllm_fa.sh` (CUDA Graphs ON, no spec decoding)

|  prompt ctx | TTFT (ms) | ITL (ms) | **TG (tok/s)** |
| ----------: | --------: | -------: | -------------: |
|          1K |        89 |       36 |       **27.7** |
|          8K |       263 |       64 |       **15.6** |
|         32K |       360 |      130 |        **7.7** |

Observation: TG drops from 27.7 → 7.7 tok/s going 1K → 32K (~3.6×). This is
why Profile B exists.

### Profile B — `start_vllm_fa_ngram.sh` (EAGER + n-gram spec decoding, K = 5)

|  prompt ctx | TTFT (ms) | ITL (ms) | **TG (tok/s)** | Δ vs stock `TRITON_ATTN` |
| ----------: | --------: | -------: | -------------: | -----------------------: |
|         32K |       638 |      135 |        **7.4** |                      +6% |
|         65K |       647 |      225 |        **4.0** |                        — |
|        100K |      1711 |      256 |        **3.9** |                 **+32%** |
|        130K |      1246 |      332 |        **3.0** |                 **+29%** |

Observation: TG is essentially **flat** from 32K to 130K (7.4 → 3.0 tok/s is
only ~2.5× over a 4× context increase), and at 100K+ it beats stock
`TRITON_ATTN` by ~30 %. On the benchmark prompts n-gram acceptance rate is
100 %, mean accepted length 4 / 5.

Profile A at 130K would either OOM or collapse into single-digit tok/s
without spec decoding — Profile B is the only practical way to serve that
range on MI50 today.

---

## Quick start

```bash
docker pull nickoptimal/gfx906-fa-vllm:latest
git clone https://github.com/nick413-bit/gfx906-fa-vllm && cd gfx906-fa-vllm
```

### Profile A — short context (chat, Q&A, ≤ 32K)

```bash
MODEL=/models/MiniMax-M2.7-AWQ-4bit bash scripts/start_vllm_fa.sh
```

### Profile B — long context (RAG, code, ≥ 32K) — n-gram speculative decoding

```bash
MODEL=/models/MiniMax-M2.7-AWQ-4bit bash scripts/start_vllm_fa_ngram.sh
```

### Or run the container by hand

```bash
docker run -d --name vllm \
    --network host --ipc host --shm-size 16g \
    --device /dev/kfd --device /dev/dri \
    --group-add 44 --group-add 993 \
    --cap-add CAP_SYS_PTRACE --security-opt seccomp=unconfined \
    -v /path/to/your/models:/models \
    -e NCCL_P2P_DISABLE=1 -e HIP_ENABLE_PEER_ACCESS=0 \
    nickoptimal/gfx906-fa-vllm:latest \
    vllm serve /models/your-awq-model \
        --tensor-parallel-size 8 \
        --attention-backend CUSTOM \
        --disable-custom-all-reduce \
        --max-model-len 65536 \
        --gpu-memory-utilization 0.78
# Profile B: add on the same line:
#   --enforce-eager \
#   --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}'
```

### Worked example — MiniMax-M2.7-AWQ-4bit on 8× MI50 32 GB (Profile A)

Verified launch command (model loads in ~83 s; KV cache ≈ 130 k tokens at
`max_model_len=32768`, giving ~4× headroom over the 16 concurrent slots).
Tool-calling and reasoning are enabled, so the model is usable out of the box
from Roo Code, Cline, OpenAI-compatible clients, etc.

```bash
docker run -d --name vllm-minimax \
    --network host --ipc host --shm-size 16g \
    --device /dev/kfd --device /dev/dri \
    --group-add 44 --group-add 993 \
    --cap-add CAP_SYS_PTRACE --security-opt seccomp=unconfined \
    -v /path/to/models:/models \
    -e NCCL_P2P_DISABLE=1 -e HIP_ENABLE_PEER_ACCESS=0 \
    nickoptimal/gfx906-fa-vllm:latest \
    vllm serve /models/MiniMax-M2.7-AWQ-4bit \
        --served-model-name minimax-m2.7-awq-4bit \
        --trust-remote-code \
        --dtype float16 \
        --attention-backend CUSTOM \
        --disable-custom-all-reduce \
        --tensor-parallel-size 8 \
        --gpu-memory-utilization 0.90 \
        --max-model-len 32768 \
        --max-num-seqs 16 \
        --max-num-batched-tokens 8192 \
        --enable-chunked-prefill \
        --enable-auto-tool-choice \
        --tool-call-parser minimax_m2 \
        --reasoning-parser minimax_m2_append_think
```

Why these values (Profile A, max throughput on 8× MI50):

| Flag | Value | Reason |
| --- | --- | --- |
| `--tensor-parallel-size` | `8` | One rank per MI50; MoE weights split evenly. |
| `--gpu-memory-utilization` | `0.90` | Safe on Profile A (short ctx, no spec-decoding overhead). |
| `--max-model-len` | `32768` | Good balance for agentic clients (Roo, Cline). Use `65536` if you need more; then drop `--max-num-seqs` to `8`. |
| `--max-num-seqs` | `16` | 2 concurrent slots per GPU — healthy for MI50 decode throughput. |
| `--max-num-batched-tokens` | `8192` | Fills MFU on prefill without stalling decode when chunked prefill is on. |
| `--enable-chunked-prefill` | — | Large prompts are sliced and interleaved with ongoing decode → stable ITL. |
| `--tool-call-parser` | `minimax_m2` | Required for OpenAI-style `"tool_choice": "auto"` requests. |
| `--reasoning-parser` | `minimax_m2_append_think` | Strips / exposes `<think>…</think>` reasoning blocks correctly. |
| `--attention-backend` | `CUSTOM` | Activates the gfx906 FA backend from this repo. |
| `--disable-custom-all-reduce` | — | Must be set when P2P is disabled (default on most G292-Z20 / gfx906 nodes). |

Do **not** add `--enforce-eager` or `--speculative-config` on Profile A — they
hurt short-context throughput; those belong to Profile B only.

### Build from source

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
   `forward`.
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
