# Roadmap & call for contributors

This document tracks concrete research-backed optimizations we want to integrate
into the gfx906-fa-vllm attention backend and MoE path, with published papers
as the source of each idea. All of them are expected to bring **measurable**
decode throughput gains on the `v_dot2_f32_f16` / `v_dot4_i32_i8` instruction
family that gfx906 actually ships with (no MFMA, no tensor cores).

We explicitly welcome help. If you have access to an MI50 / MI60 / Radeon VII
rig (or any other `gfx906` box), even a single card, and want to try one of
these items — please open a draft PR or an issue and we will help scope it.
Each task below is sized so it can be done by one person in a few days to a
few weeks, not months.

**How to claim a task:** open an issue titled `[ROADMAP] <item>` with a short
plan and expected measurement methodology. We will update this file with your
GitHub handle next to the item.

---

## Priority 1 — directly applicable, high ROI

### 1.1 Asynchronous softmax with unified-max value (FlashDecoding++ §3.1)

**Paper:**
[FlashDecoding++: Faster LLM Inference on GPUs (MLSys 2024)](https://proceedings.mlsys.org/paper_files/paper/2024/file/5321b1dabcd2be188d796c21b733e8c7-Paper-Conference.pdf)
· [arXiv 2311.01282](https://arxiv.org/abs/2311.01282)
· follow-up [FlashDecoding++Next (IEEE TC, 2025)](https://www.computer.org/csdl/journal/tc/2025/10/11062854/281HFjayFLa)

**Idea.** The standard attention softmax needs two passes per tile (find the
global max, then `exp` + normalize). FlashDecoding++ predicts the max ahead of
time using a unified value and lets threads run `exp / sum` **in parallel**
without a global sync, correcting at the very end. The trick is purely
algorithmic — it does not need tensor cores and maps well onto our
wave-level reductions in `fattn-q8-paged.cuh`.

**Why it's relevant to gfx906.** The paper explicitly targets both
tensor-core-equipped and tensor-core-free GPUs and reports improvements on
AMD hardware. Our current softmax is a classic two-pass implementation — same
baseline the paper measures against.

**Expected gain.** +15 … 25 % decode throughput at ≥ 32K context.

**Scope.** Patch `kernel/fattn-q8-paged.cuh` and the flat kernel in
`kernel/fattn-q8.cuh` (the unified-max estimate must match). ~2 weeks for one
contributor with HIP experience.

**Status.** Open. Issue: _TBD_.

---

### 1.2 Flat GEMM with double buffering (FlashDecoding++ §3.2)

**Paper:** same as above.

**Idea.** The decode step is "flat-long" (M=1, K=d_head, N=seq_len). Its HBM
latency is almost fully exposed. Double buffering loads the next `K`-tile into
LDS while the current one is being computed, hiding ~80 % of the HBM cost.
The technique is implementation-level and completely architecture-agnostic —
`v_dot2_f32_f16` pipelines it the same way `mma.sync` does.

**Expected gain.** +5 … 10 % on top of 1.1 (they are additive — 1.1 hides
softmax sync, 1.2 hides HBM).

**Scope.** 3–5 days. Mostly affects the tile kernel inner loop.

**Status.** Open.

---

### 1.3 PARD — Parallel Draft Model for speculative decoding

**Paper:**
[Accelerating Generative LLMs with Parallel Draft (AMD Developer, 2024)](https://www.amd.com/en/developer/resources/technical-articles/accelerating-generative-llms-interface-with-parallel-draft-model-pard.html)
· companion article on [AMD ROCm blog](https://rocm.blogs.amd.com/)

**Idea.** Replace the current n-gram speculative decoding (Profile B) with
a trained draft model that predicts 4–8 tokens ahead in parallel. N-gram only
helps on repetitive text (RAG, code completion) — it gives ~0 % on
free-form generation. PARD works on every prompt.

**Why it's relevant.** Published by AMD, explicitly tested on MI250X / MI300X,
does **not** use MFMA (the speed-up comes from parallelizing the draft run, not
from kernel-level acceleration).

**Expected gain.** +40 … 70 % TG on non-repetitive generation, where our
current n-gram profile gives 0 %. Probably stacks with 1.1–1.2.

**Scope.** ~1 week. The bulk is wiring the draft model into vLLM's
`--speculative-config`; vLLM already accepts generic drafter plug-ins.
Finding / training a small drafter for MiniMax-M2.7 is the harder part — a
Qwen2-0.5B-Instruct drafter is a plausible starting point.

**Status.** Open. Drafter training is the blocker.

---

### 1.4 Softmax FTZ threshold (from `noflash-attention`)

**Code:**
[github.com/Lowkey-Loki-SN/noflash-attention](https://github.com/Lowkey-Loki-SN/noflash-attention) ·
discussion on [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1s614i8/built_a_simple_pytorch_flashattention_alternative/)

**Idea.** FP16 denormals near zero occasionally produce NaN in the online
softmax on `gfx906`. `noflash-attention` flushes values below a small
threshold to zero, which is a one-line change but fixes silent stalls when
the attention scores are extreme (long-context, tool-calling prompts with
big system preambles).

**Expected gain.** Stability only (no speed). But cheap insurance — relevant
since our current stack already does `bf16 → fp16` cast (see vLLM log:
`Casting torch.bfloat16 to torch.float16`).

**Scope.** 1 day. Add a compile-time constant `FTZ_EPS = 1e-20f` in both
attention kernels.

**Status.** Open, trivial, unclaimed.

---

## Priority 2 — partially applicable, need adaptation

### 2.1 Q8 quantization for V in the KV cache (BitDecoding-inspired)

**Paper:**
[BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache (arXiv 2503.18773, 2025)](https://arxiv.org/abs/2503.18773)

**Idea.** The paper targets Hopper/Blackwell tensor cores, but the **KV
layout** — how INT4/INT8 K and V are packed so that on-the-fly dequant
overlaps MatMul — is architecture-independent. We already do this for K;
extending it to V is the single largest remaining decode-time win.

**Expected gain.** −25 … 30 % HBM bandwidth on decode → +15 … 20 % TG.

**Scope.** ~2 weeks. Changes split across
`kernel/fattn-q8-paged.cuh` (attention reads V) and
`extension/gfx906_fa_backend.py` (KV-cache-update path).

**Status.** Open. Has been on our internal roadmap for a while; paper
provides a clean reference.

---

### 2.2 In-register INT4 dequant + wave shuffle for MoE GEMV

**Papers:**
[tinygemm: Fast CUDA Kernels for Quantized LLMs (Meta AI, 2024)](https://www.reddit.com/r/MachineLearning/comments/1lzzi5p/p_tinygemm_fast_cuda_kernels_for_quantized_llms/)
· [LiquidGEMM: Hardware-Efficient W4A8 GEMM Kernel (SC'25, arXiv 2509.01229)](https://arxiv.org/abs/2509.01229)

**Idea.** Unpack INT4 weights **in-register** using `v_and_b32` /
`v_lshrrev_b32`, distribute quant scales across lanes via `__shfl_xor_sync`
(HIP equivalent works on 64-lane wavefronts), then feed into
`v_dot2_f32_f16`. The papers are NVIDIA-focused but the instruction semantics
are identical on gfx906 — we already use
`v_dot2_f32_f16` inside Q8-K dequant, it is just not applied to MoE.

**Expected gain.** +10 … 20 % on the MoE GEMM path. Relevant specifically
for `MiniMax-M2.7-AWQ-4bit` and other W4 MoE models.

**Scope.** 1–2 weeks. Requires a tuned `fused_moe` configuration
(`E=256,N=192,device_name=AMD_GFX906,dtype=int4_w4a16.json`) — see 2.3.

**Status.** Open.

---

### 2.3 Tuned `fused_moe` config for MiniMax-M2.7 on gfx906

**Upstream reference:**
[vllm benchmark_moe.py](https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py)
· [smcleod.net — Qwen3.5 + 2×RTX3090 MoE tuning](https://smcleod.net/) reports
+10 … 30 % from a full sweep on that config.

**Idea.** The stock vLLM config that ships for gfx906 covers only `N=1024`.
MiniMax-M2.7 has `N=192` (hidden shard per rank at TP=8), and the runtime
currently falls back to the **default** MoE config — visible in every vLLM
start-up log as `Using default MoE config. Performance might be sub-optimal!`.

A full tune sweeps `BLOCK_SIZE_M / N / K`, `GROUP_SIZE_M`, `num_warps ∈ {1,2,4}`
(note: `num_warps=8` causes VGPR OOM on gfx906), `num_stages`, `waves_per_eu`,
`kpack`, plus ROCm-specific `matrix_instr_nonkdim=0` (no effect on gfx906 but
required to sit in the JSON). Our internal runs saw **+12 % end-to-end at
BS=64** from this alone.

**Expected gain.** +10 … 15 % throughput on MoE decode, across all contexts.

**Scope.** 4–30 h of GPU time for the sweep itself; half a day of
engineering to integrate the resulting JSON into the Docker image.

**Caveat.** On the first attempt, a hand-picked config triggered
`Memory access fault by GPU` on all 8 cards at `profile_run`. Always use the
tuner output, not hand-tuned values, and validate with
`pytest tests/test_smoke.py` before committing.

**Status.** Open. Tuner script (`benchmark_moe.py`) exists on our internal
tuning host but was not published in this repo yet; we will upload it as a
separate PR once it runs cleanly.

---

### 2.4 Learned 4-bit format (`any4`)

**Paper / code:**
[any4 (Meta AI, 2025)](https://github.com/facebookresearch/any4) · tinygemm
integrates LUT dequant on SIMD cores.

**Idea.** Replace fixed W4 groupwise quant with a **learned LUT** per group.
+5 % quality retention at the same bit budget. Works on SIMD-only
architectures because the dequant is a table lookup + `v_dot2`, no matrix
hardware required.

**Expected gain.** Quality, not speed — allows tighter quant (W3 territory)
without accuracy cliff, which *indirectly* speeds up decode (smaller weights →
less HBM traffic).

**Scope.** Research-level integration: needs quantization re-run on
MiniMax-M2.7, new AWQ-replacement loader in vLLM, custom dequant kernel.

**Status.** Long-term, not claimed.

---

## Priority 3 — watch list (not actionable yet)

### 3.1 EAGLE / EAGLE-2 / EAGLE-3 speculative heads
[arXiv 2401.10774](https://arxiv.org/abs/2401.10774) · supersedes n-gram and
in many cases matches PARD. Blocker is training cost and the fact that the
head weights would have to be shipped per model.

### 3.2 FlashDecoding++Next (extended paper)
[IEEE TC 2025](https://www.computer.org/csdl/journal/tc/2025/10/11062854/281HFjayFLa)
— extends FlashDecoding++ with buffer reusing and better memory management.
We'd absorb this once 1.1–1.2 are landed and measured.

### 3.3 Categorical foundations of CuTe layouts
[Colfax Research, 2025](https://research.colfax-intl.com/categorical-foundations-for-cute-layouts/)
· [arXiv preprint](https://arxiv.org/abs/2505.xxxxx)

**Verdict: not applicable.** CuTe / CUTLASS is NVIDIA-only and assumes tensor
cores. Included here so that contributors don't spend weeks on it — the
gains this paper enables live in a world we physically do not have on gfx906.

---

## Expected cumulative impact

If items 1.1, 1.2, 1.3, 2.1 and 2.3 all land cleanly:

| Workload                      | Now (Profile A/B) | After roadmap   |
| ----------------------------- | ----------------- | --------------- |
| Decode @ 1K ctx (chat)        | 27.7 tok/s        | ~40 tok/s       |
| Decode @ 32K ctx              | 7.4 tok/s         | ~13 tok/s       |
| Decode @ 100K ctx (Profile B) | 3.9 tok/s         | ~8 tok/s        |

These are estimates; they assume the gains are mostly multiplicative, which
is true when each item tackles a different part of the decode pipeline
(softmax sync vs HBM hiding vs draft model vs KV bandwidth vs MoE tiling).

---

## How to contribute a benchmark

We standardize on the same workload used in the README performance tables:

- Model: `MiniMax-M2.7-AWQ-4bit` (or any MoE W4 for 2.2 / 2.3).
- Hardware: 8× MI50 32 GB with TP=8, P2P off, `--disable-custom-all-reduce`.
  Single-card or 4× results are welcome but please label them clearly.
- Script: `tests/bench_vllm.py --lengths 1000 8000 32000 100000 --max-new-tokens 32`.

Post numbers in the issue tracking the feature, as
`(ctx, TTFT ms, ITL ms, TG tok/s, Δ vs baseline %)`. That makes it trivial to
tell whether a proposed optimization actually moved the needle on gfx906 or
just on the paper's target hardware.
