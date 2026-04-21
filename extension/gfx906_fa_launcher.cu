// gfx906_fa_launcher.cu — host launcher для flash_attn_tile_q8<>
//
// Запускает __global__ kernel flash_attn_tile_q8<DKQ, DV, ncols1, ncols2, use_logit_softcap>
// на паре torch::Tensor Q/K/V + metadata → Output.
//
// MVP ограничения (будут расширены позже):
//   - DKQ = DV = 128 (MiniMax M2.7)
//   - use_logit_softcap = false
//   - Без mask, sinks, KV_max (paged KV — TODO: vLLM block table)
//   - K — block_q8_0 pre-quantized, V — fp16 native
//   - Q — fp32 contiguous
//   - Output — BSHD layout (host API делает transpose → BHSD)
//
// Prefill dispatcher (t6b): cols_per_block выбирается по Sq как в llama.cpp:
//   Sq >  32 → 64, >16 → 32, >8 → 16, >4 → 8, >2 → 4, else 2.
//
// Раскладка tensors (как ggml):
//   Q: [batch, heads_q, seq_q, head_dim]        float32, contiguous
//   K: [batch, heads_kv, seq_kv, head_dim/QK8_0] block_q8_0 (34 bytes per block)
//   V: [batch, heads_kv, seq_kv, head_dim]       float16, contiguous
//   O: [batch, heads_q, seq_q, head_dim]         float32 (output)

// КРИТИЧНО: torch cpp_extension форсит -D__HIP_NO_HALF_OPERATORS__=1 и
// -D__HIP_NO_HALF_CONVERSIONS__=1 в cmdline. Эти defines ломают fattn-q8.cuh
// (там `half2 z[N] = {{0.0f, 0.0f}}`, `h2 *= h2`, implicit float→half).
// Снимаем ДО включения любых ROCm-заголовков.
#ifdef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_OPERATORS__
#endif
#ifdef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_CONVERSIONS__
#endif
#ifdef __HIP_NO_HALF2_OPERATORS__
#undef __HIP_NO_HALF2_OPERATORS__
#endif

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// ВАЖНО: shim ставит все defines (GGML_USE_HIP, GGML_HIP_GFX906, WARP_SIZE=64, FLASH_ATTN_AVAILABLE)
// ДО включения fattn-q8.cuh
#include "ggml_shim.cuh"
#include "fattn-q8.cuh"
#include "fattn-q8-paged.cuh"

#include <cstdio>
#include <cstdint>
#include <type_traits>

// ============================================================================
// Entry point из C++/pybind11: C-linkage удобнее для диагностики
// ============================================================================
extern "C" hipError_t gfx906_fa_launch(
    const float *      Q_fp32,
    const void  *      K_q8,
    const __half *     V_f16,
    float *            O_fp32,
    float2 *           O_meta,
    const int *        KV_max_d,
    // Optional mask [batch, Sq, seq_kv_padded] fp16. Additive bias to scores
    // (0 → keep, -inf / -65504 → mask out). Если nullptr — без mask.
    const __half *     MASK_f16,
    int32_t            mask_seq_kv_padded,   // stride по Sq в элементах half
    // Level 3a: inline causal mask. int32[batch], q_abs_offset[b] = seq_len[b] - n_q[b].
    // Mutually exclusive with MASK_f16 (either mask or q_abs_offset, not both).
    const int32_t *    Q_ABS_OFFSET_d,
    int                batch,
    int                heads_q,
    int                heads_kv,
    int                seq_q,
    int                seq_kv,
    int                head_dim,
    float              scale,
    hipStream_t        stream
) {
    if (head_dim != 128) {
        fprintf(stderr, "[gfx906_fa] Unsupported head_dim=%d (only 128 in MVP)\n", head_dim);
        return hipErrorInvalidValue;
    }
    if (heads_q % heads_kv != 0) {
        fprintf(stderr, "[gfx906_fa] heads_q=%d must be divisible by heads_kv=%d\n", heads_q, heads_kv);
        return hipErrorInvalidValue;
    }

    constexpr int DKQ = 128;
    constexpr int DV  = 128;
    constexpr bool use_logit_softcap = false;

    // nb* computed in BYTES (ggml convention)
    const int32_t nb00 = sizeof(float);
    const int32_t nb01 = nb00 * head_dim;
    const int32_t nb02 = nb01 * seq_q;
    const int32_t nb03 = nb02 * heads_q;

    // K is block_q8_0: 34 bytes per 32-elem block
    const int32_t nb10 = sizeof(block_q8_0);                 // per block
    const int32_t nb11 = nb10 * (head_dim / QK8_0);          // per K-token row
    const int32_t nb12 = nb11 * seq_kv;                      // per head
    const int64_t nb13 = (int64_t) nb12 * heads_kv;          // per batch

    // V is fp16
    const int32_t nb20 = sizeof(__half);
    const int32_t nb21 = nb20 * head_dim;
    const int32_t nb22 = nb21 * seq_kv;
    const int64_t nb23 = (int64_t) nb22 * heads_kv;

    // mask layout: [batch, Sq, mask_seq_kv_padded] fp16 (one plane per batch).
    // ne31 — Sq; ne32 — 1; ne33 — batch; nb31 — stride по Sq в байтах;
    // nb32 — stride по «heads» (не используем → 0); nb33 — stride по sequence.
    const int32_t ne31 = MASK_f16 ? seq_q : 0;
    const int32_t ne32 = MASK_f16 ? 1     : 0;
    const int32_t ne33 = MASK_f16 ? batch : 1;          // %ne33 → не 0
    const int32_t nb31 = MASK_f16 ? (int32_t)(mask_seq_kv_padded * sizeof(__half)) : 0;
    const int32_t nb32 = 0;
    const int64_t nb33 = MASK_f16 ? (int64_t)seq_q * nb31 : 0;

    // ne shapes
    const int32_t ne00 = head_dim;
    // ВАЖНО: kernel использует только ne01.z (где ggml хранит оригинальный divisor,
    // см. init_fastdiv_values в common.cuh: uint3 = (mp, L, d)).
    // Проверено: в fattn-q8.cuh нет вызовов fastdiv/fastmodulo с ne01,
    // так что .x/.y можно оставить нулями, но .z ДОЛЖНО быть = seq_q.
    const uint3   ne01 = make_uint3(0u, 0u, (unsigned) seq_q);
    const int32_t ne02 = heads_q;
    const int32_t ne03 = batch;

    const int32_t ne10 = head_dim;
    const int32_t ne11 = seq_kv;
    const int32_t ne12 = heads_kv;
    const int32_t ne13 = batch;

    // ------------------------------------------------------------------
    // Kernel dispatch: выбор cols_per_block (ncols1) в зависимости от Sq
    // ------------------------------------------------------------------
    //
    // Оригинал llama.cpp (launch_fattn_tile_q8_switch_ncols1, fattn-q8.cuh:846):
    //   Sq >  32  → ncols1 = 64
    //   Sq >  16  → ncols1 = 32
    //   Sq >   8  → ncols1 = 16
    //   Sq >   4  → ncols1 =  8
    //   Sq >   2  → ncols1 =  4
    //   Sq <= 2   → ncols1 =  2
    //
    // Для DKQ=DV=128 таблица (см. fattn-q8.cuh:55-60):
    //   ncols=2  → nthreads=256
    //   ncols=4  → nthreads=128   ← обратите внимание
    //   ncols=8  → nthreads=256
    //   ncols=16 → nthreads=256
    //   ncols=32 → nthreads=256
    //   ncols=64 → nthreads=256
    //
    // NC2 = 1 всегда (mask=nullptr → GQA-packing отключён).
    //
    // Lambda-макрос для DRY — все инстанциации идентичны кроме NC1.

    constexpr int NC2 = 1;
    const int ntiles_z = (heads_q + NC2 - 1) / NC2;

    auto launch = [&](auto NC1_tag, int nthreads) {
        constexpr int NC1 = decltype(NC1_tag)::value;
        dim3 grid(
            /*x=*/ (seq_q + NC1 - 1) / NC1,
            /*y=*/ 1,                          // KV-split (stream-k) disabled
            /*z=*/ batch * ntiles_z
        );
        dim3 block(32 /* warp_size */, nthreads / 32 /* nwarps */, 1);

        flash_attn_tile_q8<DKQ, DV, NC1, NC2, use_logit_softcap><<<grid, block, 0, stream>>>(
            (const char *) Q_fp32,
            (const char *) K_q8,
            (const char *) V_f16,
            /*mask=*/   (const char *) MASK_f16,
            /*sinks=*/  (const char *) nullptr,
            /*KV_max=*/ KV_max_d,
            /*q_abs_offset=*/ Q_ABS_OFFSET_d,
            O_fp32,
            O_meta,
            scale,
            /*max_bias=*/ 0.0f,
            /*m0=*/ 1.0f, /*m1=*/ 1.0f,
            /*n_head_log2=*/ 0u,
            /*logit_softcap=*/ 0.0f,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
            ne31, ne32, ne33,
                  nb31, nb32, nb33
        );
    };

    // std::integral_constant trick — передаём compile-time int в lambda
    using T2  = std::integral_constant<int,  2>;
    using T4  = std::integral_constant<int,  4>;
    using T8  = std::integral_constant<int,  8>;
    using T16 = std::integral_constant<int, 16>;
    using T32 = std::integral_constant<int, 32>;
    using T64 = std::integral_constant<int, 64>;

    if      (seq_q > 32) launch(T64{}, 256);
    else if (seq_q > 16) launch(T32{}, 256);
    else if (seq_q >  8) launch(T16{}, 256);
    else if (seq_q >  4) launch(T8{},  256);
    else if (seq_q >  2) launch(T4{},  128);
    else                 launch(T2{},  256);

    return hipGetLastError();
}

// ============================================================================
// Level 3c: Direct-paged FlashAttention entry point.
//
// No gather. K/V read directly from paged KV cache via block_table.
// Layout (vLLM compat, block_size=16):
//   K_paged: [num_blocks, 16, Hkv, (D/32)*34]  uint8
//   V_paged: [num_blocks, 16, Hkv,  D        ]  fp16
//   block_table: [num_seqs, max_blocks_per_seq]  int32
//
// Strides пересчитываются host'ом в bytes (как в gather path).
// ============================================================================
extern "C" hipError_t gfx906_fa_launch_paged(
    const float *      Q_fp32,
    const void  *      K_paged,           // uint8, [num_blocks, bs, Hkv, bpr]
    const __half *     V_paged,           // fp16,  [num_blocks, bs, Hkv, D]
    const int32_t *    block_table,       // [num_seqs, max_blocks_per_seq]
    const int32_t *    kv_max_d,          // [B, grid_x] already expanded by caller
    float *            O_fp32,            // [B, Sq_pad, Hq, D]  (BSHD)
    float2 *           O_meta,            // [B, Sq_pad, Hq, 2]
    const __half *     MASK_f16,
    int32_t            mask_seq_kv_padded,
    const int32_t *    Q_ABS_OFFSET_d,
    int                batch,
    int                heads_q,
    int                heads_kv,
    int                seq_q,
    int                max_seq_kv,        // max(seq_lens) — для ne11
    int                head_dim,
    int                block_size,
    int                max_blocks_per_seq,
    int64_t            k_block_stride,
    int64_t            k_token_stride,
    int64_t            k_head_stride,
    int64_t            v_block_stride,
    int64_t            v_token_stride,
    int64_t            v_head_stride,
    float              scale,
    hipStream_t        stream
) {
    if (head_dim != 128) {
        fprintf(stderr, "[gfx906_fa_paged] Unsupported head_dim=%d (only 128)\n", head_dim);
        return hipErrorInvalidValue;
    }
    if (heads_q % heads_kv != 0) {
        fprintf(stderr, "[gfx906_fa_paged] heads_q=%d not divisible by heads_kv=%d\n",
                heads_q, heads_kv);
        return hipErrorInvalidValue;
    }
    if (block_size != 16) {
        fprintf(stderr, "[gfx906_fa_paged] Only block_size=16 supported, got %d\n", block_size);
        return hipErrorInvalidValue;
    }

    constexpr int DKQ = 128;
    constexpr int DV  = 128;
    constexpr bool use_logit_softcap = false;

    // Q layout (contiguous, same as non-paged).
    const int32_t nb00 = sizeof(float);
    const int32_t nb01 = nb00 * head_dim;
    const int32_t nb02 = nb01 * seq_q;
    const int32_t nb03 = nb02 * heads_q;

    const int32_t ne31 = MASK_f16 ? seq_q : 0;
    const int32_t ne32 = MASK_f16 ? 1     : 0;
    const int32_t ne33 = MASK_f16 ? batch : 1;
    const int32_t nb31 = MASK_f16 ? (int32_t)(mask_seq_kv_padded * sizeof(__half)) : 0;
    const int32_t nb32 = 0;
    const int64_t nb33 = MASK_f16 ? (int64_t)seq_q * nb31 : 0;

    const int32_t ne00 = head_dim;
    const uint3   ne01 = make_uint3(0u, 0u, (unsigned) seq_q);
    const int32_t ne02 = heads_q;
    const int32_t ne03 = batch;

    const int32_t ne10 = head_dim;
    const int32_t ne11 = max_seq_kv;
    const int32_t ne12 = heads_kv;
    const int32_t ne13 = batch;

    constexpr int NC2 = 1;
    const int ntiles_z = (heads_q + NC2 - 1) / NC2;

    auto launch = [&](auto NC1_tag, int nthreads) {
        constexpr int NC1 = decltype(NC1_tag)::value;
        const int grid_x = (seq_q + NC1 - 1) / NC1;

        dim3 grid(
            /*x=*/ grid_x,
            /*y=*/ 1,
            /*z=*/ batch * ntiles_z
        );
        dim3 block(32, nthreads / 32, 1);

        flash_attn_tile_q8_paged<DKQ, DV, NC1, NC2, use_logit_softcap><<<grid, block, 0, stream>>>(
            (const char *) Q_fp32,
            (const char *) K_paged,
            (const char *) V_paged,
            /*mask=*/   (const char *) MASK_f16,
            /*sinks=*/  (const char *) nullptr,
            /*KV_max=*/ kv_max_d,
            Q_ABS_OFFSET_d,
            block_table,
            O_fp32,
            O_meta,
            scale,
            /*max_bias=*/ 0.0f,
            /*m0=*/ 1.0f, /*m1=*/ 1.0f,
            /*n_head_log2=*/ 0u,
            /*logit_softcap=*/ 0.0f,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
            ne31, ne32, ne33,
                  nb31, nb32, nb33,
            max_blocks_per_seq,
            k_block_stride, k_token_stride, k_head_stride,
            v_block_stride, v_token_stride, v_head_stride
        );
    };

    using T2  = std::integral_constant<int,  2>;
    using T4  = std::integral_constant<int,  4>;
    using T8  = std::integral_constant<int,  8>;
    using T16 = std::integral_constant<int, 16>;
    using T32 = std::integral_constant<int, 32>;
    using T64 = std::integral_constant<int, 64>;

    // Paged-specific: для NC1=64 (DKQ=DV=128, ncols=64) увеличиваем
    // nthreads 256→512 (nwarps=8→16), cpw=8→4. VGPR=256→128, spill=198→117.
    // Выбор 512 (не 1024) обоснован gfx906 VGPR budget: при 1024 threads
    // compiler ограничивает VGPR до 64 → spill возвращается к 199.
    // Значение синхронизировано с ggml_cuda_fattn_tile_q8_paged_get_nthreads.
    if      (seq_q > 32) launch(T64{}, 512);
    else if (seq_q > 16) launch(T32{}, 256);
    else if (seq_q >  8) launch(T16{}, 256);
    else if (seq_q >  4) launch(T8{},  256);
    else if (seq_q >  2) launch(T4{},  128);
    else                 launch(T2{},  256);

    return hipGetLastError();
}
