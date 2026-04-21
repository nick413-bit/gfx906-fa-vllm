# Architecture

## Position in the stack

`gfx906-fa-vllm` is a thin layer on top of the upstream
[`mixa3607/ML-gfx906`](https://github.com/mixa3607/ML-gfx906) stack. It does **not**
replace ROCm, PyTorch or vLLM — it plugs a custom attention backend into the
vLLM that ships inside `docker.io/mixa3607/vllm-gfx906:0.19.1-rocm-7.2.1-aiinfos`:

```
┌──────────────────────────────────────────────────────────────────────┐
│                          gfx906-fa-vllm                              │  ← this repo
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │ extension/gfx906_fa_backend.py  (Gfx906FABackend)             │   │
│  │ extension/gfx906_fa_paged.py    (path selection)              │   │
│  │ kernel/fattn-q8*.cuh            (HIP FlashAttention kernels)  │   │
│  │ setup.py entry_point  vllm.general_plugins = …:register       │   │
│  └───────────────────────────────────────────────────────────────┘   │
├──────────────────────────────────────────────────────────────────────┤
│                  mixa3607/vllm-gfx906:0.19.1 image                   │  ← upstream
│  vLLM 0.19.1  ·  PyTorch 2.9  ·  Triton (AMD fork)                   │
│  ROCm 7.2.1 patched for gfx906  ·  hipBLAS / Tensile / RCCL          │
├──────────────────────────────────────────────────────────────────────┤
│                       AMD MI50 / MI60 / Radeon VII                   │  ← hardware
│                       gfx906  ·  Vega 20  ·  HBM2                    │
└──────────────────────────────────────────────────────────────────────┘
```

## Data path inside the backend

```
vLLM  →  --attention-backend CUSTOM  →  Gfx906FAImpl
                                          │
                                          ├─ do_kv_cache_update
                                          │     fp16 key/value  →  paged KV cache (fp16)
                                          │
                                          └─ forward
                                                │
                                                ├── adaptive path (gfx906_fa_paged.py):
                                                │     Sq ≤ 32  →  direct-paged FA kernel
                                                │     Sq >  32  →  gather + flat FA kernel
                                                │
                                                ├── gfx906_fa.reshape_and_cache_q8
                                                │     inline fp16 → Q8 on the fly
                                                ├── gfx906_fa.gather_paged_kv_q8
                                                │     when gather path is taken
                                                └── gfx906_fa.forward
                                                      core FlashAttention HIP kernel
```

## HIP kernels

| File                               | Role                                                  |
| :--------------------------------- | :---------------------------------------------------- |
| `kernel/fattn-q8.cuh`              | Flat FlashAttention kernel (Q8_0 K, fp16 V)           |
| `kernel/fattn-q8-paged.cuh`        | Direct-paged variant (no gather step)                 |
| `kernel/gfx906-common.cuh`         | Wavefront utilities, shuffle, reductions              |

Launch config: `__launch_bounds__(128, 4)` — 2 wavefronts per block on gfx906
(wavefront = 64). Larger blocks push us past VGPR = 128 and cause register spills
into scratch, which tanks occupancy.

## KV-cache format

We keep the vLLM stock paged layout unchanged so the vLLM block allocator keeps
working as-is:

```
(num_blocks, 2, block_size, num_kv_heads, head_dim)
         ^  ^       ^              ^          ^
         │  │       │              │          └─ 128
         │  │       │              └─ depends on model (8 for MiniMax-M2.7)
         │  │       └─ 16 by default
         │  └─ 0 = keys, 1 = values
         └─ allocated by vLLM
```

The Q8-K path writes quantized keys into a **separate side-buffer** the same shape,
but the side-buffer is currently disabled (`GFX906_FA_LEGACY=1`) pending a
synchronization fix with vLLM's warm-up passes. In legacy mode the kernel quantizes
fp16 → Q8 inline during `forward`, which is slower than using a ready-made Q8
cache but stable.

## Plugin registration

`setup.py` exposes a vLLM general plugin:

```python
entry_points={
    "vllm.general_plugins": [
        "gfx906_fa = gfx906_fa_backend:register",
    ],
}
```

`vllm.plugins.load_general_plugins()` runs in **every** process — the main process,
the engine core process, and every TP worker spawned by `multiproc_executor`. This
is crucial: `AttentionBackendEnum.CUSTOM` is a process-local registration
(`_ATTN_OVERRIDES` is a module-level dict), so it must be re-registered in each
worker after `spawn`. The plugin mechanism handles that automatically; no
`PYTHONPATH` hacks or manual imports on the user side.

## Base-image patches

Two `sed` patches are applied inside `docker/Dockerfile`:

```bash
sed -i 's/except ImportError:/except Exception:/g' \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/fa_utils.py
sed -i 's/except ImportError:/except Exception:/g' \
    /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/mla_attention.py
```

Reason: under Python `spawn`, vLLM workers deserialize and import `flash_attn`
at module load time, before `libamdhip64.so` is fully linked by PyTorch.
Triton's AMD driver init (`hipGetProcAddress`) then raises `RuntimeError`, not
`ImportError`, so the upstream `except ImportError:` block lets it propagate
and crashes the worker with `Triton cache error`. Since `CUSTOM` does not use
`flash_attn_varlen_func`, catching `Exception` is safe and restores clean
worker startup.

## Path-selection heuristic

`gfx906_fa_paged.forward_paged()` picks the kernel based on query length:

| `Sq` (query tokens) | Path                                  | Why                                                             |
| ------------------: | :------------------------------------ | :-------------------------------------------------------------- |
|                   1 | direct-paged FA, mask elided          | Pure decode, causal mask trivial                                |
|             2 … 32  | direct-paged FA                       | Speculative decode / short mixed batch — avoid gather overhead  |
|             > 32    | gather → flat FA                      | Prefill — amortize gather with a large tile                     |

The cutoff of 32 is empirical on MI50 with MiniMax-M2.7 (8 KV-heads, 128 head-dim).

## Environment knobs

See the main [README](../README.md#environment-variables) for the full list of
`GFX906_FA_*` environment variables and their defaults.
