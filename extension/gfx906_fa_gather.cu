// gfx906_fa_gather.cu — fused HIP gather для paged KV → contiguous BHSD.
//
// Level 1 оптимизация: заменяет fancy-indexing путь _gather_kv_q8 в Python,
// который делал 2 тура через HBM:
//   1) key_cache_q8[block_table]  → temp [B, n_blocks, bs, Hkv, bytes]
//   2) permute + contiguous       → [B, Hkv, Sk, bytes]
//
// Здесь делаем то же самое за ОДИН проход: каждый workgroup обрабатывает один
// (seq_idx, kv_head, token_pos) триплет; читает block_table[seq_idx, token_pos/bs],
// затем копирует K_q8 row и V_fp16 row в contiguous output BHSD.
//
// Дополнительно V-row за seq_lens[seq_idx] обнуляется inline (это требование
// kernel'а: «хвост» V не должен вносить вклад в softmax). K — мусор в хвосте
// безразличен, потому что FA kernel отсекает по KV_max.
//
// ---------------------------------------------------------------------------
// Параметризация:
//   Block(64, 1, 1) — 1 wavefront на 1 (seq, head, tok).
//   Grid(num_seqs, Hkv, max_seqlen_k) — big, но каждый workgroup лёгкий
//     (копия D*34/32 байт для K + D*2 байт для V, чтение block_table = 1 int).
//
// Оптимизации для gfx906 (LDS 64KB/CU, 64-wide waves):
//   - byte copy через unsigned int (4 байта за load/store) → coalesced HBM.
//   - V копируем как __half4 (8 байт за тред) — 128-бит HBM burst.
//   - block_table / seq_lens читаем ONE thread per workgroup + broadcast через
//     shared mem.
// ---------------------------------------------------------------------------

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
#include <cstdint>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Fused gather K_q8 + V_fp16 → contiguous BHSD.
//
// Shape contracts:
//   key_cache_q8  [num_blocks, block_size, Hkv, bytes_per_row]  uint8
//   value_cache   [num_blocks, block_size, Hkv, D]              fp16
//   block_table   [num_seqs, max_num_blocks]                    int32
//   seq_lens      [num_seqs]                                    int32
//   k_out         [num_seqs, Hkv, Sk, bytes_per_row]            uint8
//   v_out         [num_seqs, Hkv, Sk, D]                        fp16
//
// Где Sk = max_seqlen_k (округление до кратного 32 делает host).
// ---------------------------------------------------------------------------
extern "C" __global__ void gather_paged_kv_q8_kernel(
    const uint8_t * __restrict__ key_cache_q8,
    const __half  * __restrict__ value_cache,
    const int32_t * __restrict__ block_table,
    const int32_t * __restrict__ seq_lens,
    uint8_t       * __restrict__ k_out,
    __half        * __restrict__ v_out,
    int num_seqs,
    int num_kv_heads,
    int Sk,                       // max_seqlen_k (кратно 32)
    int D,                        // head_size
    int bytes_per_row,            // (D/32) * 34
    int block_size,
    int max_blocks_per_seq,
    int64_t cache_block_stride,   // block_size * Hkv * bytes_per_row
    int64_t cache_token_stride,   // Hkv * bytes_per_row
    int64_t cache_head_stride_q8, // bytes_per_row (K)
    int64_t v_cache_block_stride, // block_size * Hkv * D
    int64_t v_cache_token_stride, // Hkv * D
    int64_t v_cache_head_stride   // D
) {
    const int seq_idx  = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tok_pos  = blockIdx.z;
    if (seq_idx >= num_seqs || head_idx >= num_kv_heads || tok_pos >= Sk) return;

    const int lane = threadIdx.x;   // 0..63

    // Читаем seq_len ОДИН раз на workgroup (lane 0), broadcast через __shfl.
    int seq_len = 0;
    if (lane == 0) seq_len = seq_lens[seq_idx];
    seq_len = __shfl(seq_len, 0, 64);

    // Тоже для block_table[seq, tok_pos / block_size].
    const int block_tab_idx = tok_pos / block_size;
    const int block_offset  = tok_pos % block_size;

    int64_t v_dst_base =
        ((int64_t)seq_idx * num_kv_heads + head_idx) * (int64_t)Sk * D
        + (int64_t)tok_pos * D;
    int64_t k_dst_base =
        ((int64_t)seq_idx * num_kv_heads + head_idx) * (int64_t)Sk * bytes_per_row
        + (int64_t)tok_pos * bytes_per_row;

    // Hot-case: out-of-range → обнулить V (K не трогаем, kernel отсекает).
    if (tok_pos >= seq_len) {
        __half * vdst = v_out + v_dst_base;
        for (int i = lane; i < D; i += 64) {
            vdst[i] = __float2half(0.0f);
        }
        return;
    }

    // Валидный токен → читаем block_table[seq, block_tab_idx].
    int phys_block = 0;
    if (lane == 0) {
        phys_block = block_table[seq_idx * max_blocks_per_seq + block_tab_idx];
    }
    phys_block = __shfl(phys_block, 0, 64);

    // ---------- K copy (uint8, bytes_per_row) ----------
    const uint8_t * k_src =
        key_cache_q8
        + (int64_t)phys_block   * cache_block_stride
        + (int64_t)block_offset * cache_token_stride
        + (int64_t)head_idx     * cache_head_stride_q8;
    uint8_t * k_dst = k_out + k_dst_base;

    // bytes_per_row = (D/32)*34. Для D=128 это 136 байт. Копируем uint32_t-ами
    // где это возможно, хвост — побайтно.
    const int n_u32 = bytes_per_row >> 2;          // ← целых 4-байтных chunks
    const int tail_start = n_u32 << 2;             // хвост в байтах
    const uint32_t * k_src_u32 = reinterpret_cast<const uint32_t *>(k_src);
    uint32_t       * k_dst_u32 = reinterpret_cast<uint32_t       *>(k_dst);
    for (int i = lane; i < n_u32; i += 64) {
        k_dst_u32[i] = k_src_u32[i];
    }
    // tail (0..3 байт). Для D%32==0 и 34*(D/32) → bytes_per_row % 4 ∈ {0, 2}:
    // (D/32)*34 mod 4 = (D/32)*2 mod 4 → D=64 → 68 → 0; D=128→136→0. Ok.
    // Но на всякий случай обрабатываем хвост через первого lane'а.
    if (lane == 0) {
        for (int i = tail_start; i < bytes_per_row; ++i) {
            k_dst[i] = k_src[i];
        }
    }

    // ---------- V copy (fp16 × D) ----------
    const __half * v_src =
        value_cache
        + (int64_t)phys_block   * v_cache_block_stride
        + (int64_t)block_offset * v_cache_token_stride
        + (int64_t)head_idx     * v_cache_head_stride;
    __half * vdst = v_out + v_dst_base;

    // Копируем через uint2 (8 байт = 4 × fp16) когда D выровнен по 4.
    // Для D=128: 32 итерации по 4 fp16 → 2 burst per lane на 64 threads.
    const int n_u2 = D >> 2;                       // D/4 chunks по 8 байт
    const uint2 * v_src_u2 = reinterpret_cast<const uint2 *>(v_src);
    uint2       * v_dst_u2 = reinterpret_cast<uint2       *>(vdst);
    for (int i = lane; i < n_u2; i += 64) {
        v_dst_u2[i] = v_src_u2[i];
    }
    // tail: D % 4. Для D=128 и D=64 это 0 → можно не обрабатывать, но оставим.
    const int v_tail = D & 3;
    if (v_tail != 0 && lane == 0) {
        for (int i = D - v_tail; i < D; ++i) vdst[i] = v_src[i];
    }
}

// ---------------------------------------------------------------------------
// V2: Paged-block-coalesced gather (1 workgroup = 1 paged block, 16 tokens).
//
// Мотивация (rocprof_decode.py на 60K, batch=4):
//   Текущий per-token kernel: grid(4, 8, 61440) = ~2M workgroups, 64 threads.
//   Эффективная HBM BW ~330 GB/s из ~1 TB/s пика MI50 (33% utilization).
//   Основная причина — launch overhead от 2M wavefronts + мелкие transfers
//   по 392 bytes/wg без полного burst-fill.
//
// V2 подход:
//   * 1 workgroup обслуживает ОДИН (seq, head, paged-block) тройник
//   * block_size=16 токенов сразу → 16×(136+256)=6272 bytes per wg
//   * 128 threads/wg → ~49 bytes/thread = 2-3 uint4 each
//   * block_table читается 1 раз за весь блок (вместо 1 на каждый токен)
//
// Layout контракт — идентичен V1 (чтобы безопасно переключаться через env):
//   src:  [num_blocks, block_size, Hkv, bytes_per_row | D]
//   dst:  [num_seqs, Hkv, Sk, bytes_per_row | D]
//   Sk кратно 32 (округляет host).
//
// bytes_per_row = (D/32)*34 для q8_0 (D=128 → 136 bytes). Не кратно 16,
// так что для K делаем 8×uint4 (128 bytes) + 1×uint2 (8 bytes) tail.
// Для V (D*2 bytes, D%4==0) полностью кратно 16 → чистые uint4 loads.
// ---------------------------------------------------------------------------
extern "C" __global__ void gather_paged_kv_q8_kernel_v2(
    const uint8_t * __restrict__ key_cache_q8,
    const __half  * __restrict__ value_cache,
    const int32_t * __restrict__ block_table,
    const int32_t * __restrict__ seq_lens,
    uint8_t       * __restrict__ k_out,
    __half        * __restrict__ v_out,
    int num_seqs,
    int num_kv_heads,
    int Sk,
    int D,
    int bytes_per_row,
    int block_size,
    int max_blocks_per_seq,
    int64_t cache_block_stride,
    int64_t cache_token_stride,
    int64_t cache_head_stride_q8,
    int64_t v_cache_block_stride,
    int64_t v_cache_token_stride,
    int64_t v_cache_head_stride
) {
    const int seq_idx        = blockIdx.x;
    const int head_idx       = blockIdx.y;
    const int paged_block    = blockIdx.z;              // 0..ceil(Sk/block_size)-1
    const int block_start_tok = paged_block * block_size;
    if (seq_idx >= num_seqs || head_idx >= num_kv_heads || block_start_tok >= Sk) return;

    const int tid = threadIdx.x;
    const int nth = blockDim.x;   // 128

    // ---------- Прочитать phys_block + seq_len ОДИН раз на wg ----------
    __shared__ int s_phys_block;
    __shared__ int s_seq_len;
    if (tid == 0) {
        s_seq_len = seq_lens[seq_idx];
        const int block_tab_idx = block_start_tok / block_size;
        s_phys_block = (block_tab_idx < max_blocks_per_seq)
            ? block_table[seq_idx * max_blocks_per_seq + block_tab_idx]
            : -1;
    }
    __syncthreads();
    const int phys_block = s_phys_block;
    const int seq_len    = s_seq_len;

    // Базовые указатели источника (для валидного phys_block).
    const uint8_t * k_src_base_bh = (phys_block >= 0)
        ? key_cache_q8
          + (int64_t)phys_block * cache_block_stride
          + (int64_t)head_idx   * cache_head_stride_q8
        : nullptr;
    const __half  * v_src_base_bh = (phys_block >= 0)
        ? value_cache
          + (int64_t)phys_block   * v_cache_block_stride
          + (int64_t)head_idx     * v_cache_head_stride
        : nullptr;

    // Базовые offset'ы dst: [seq, head, tok, 0]. Fixed per wg.
    const int64_t dst_K_sh_base =
        ((int64_t)seq_idx * num_kv_heads + head_idx) * (int64_t)Sk * bytes_per_row;
    const int64_t dst_V_sh_base =
        ((int64_t)seq_idx * num_kv_heads + head_idx) * (int64_t)Sk * D;

    // Предрасчёт uint4-/uint2-границ для K.
    // bytes_per_row может быть не кратен 16. Для D=128: 136 = 8*16 + 8.
    const int k_n_u4  = bytes_per_row >> 4;           // 8 при D=128
    const int k_tail  = bytes_per_row & 15;           // 8 при D=128
    const int k_tail_u2 = k_tail >> 3;                // 1 (если есть 8 байт)
    const int k_tail_byte = k_tail & 7;               // оставшиеся 0..7 байт (обычно 0)

    // V layout: D*2 bytes per token. D=128 → 256 bytes = 16 × uint4. Чистый vectorised путь.
    const int v_n_u4 = (D * (int)sizeof(__half)) >> 4;  // 16 при D=128

    // Если ни один токен paged-блока не валиден — просто обнуляем V, K не трогаем.
    const bool full_oob = (block_start_tok >= seq_len) || (phys_block < 0);

    // ---------- FLAT ITERATION: вся работа WG распределена равномерно ----------
    //
    // Главное отличие от первой версии V2: нет внутреннего `for t` цикла. Вместо
    // этого все uint4-chunks для K, V по ВСЕМ 16 токенам blok'a распределены
    // между 128 threads одним глобальным range-for. Это даёт:
    //   * равномерную загрузку всех threads (раньше 8 из 128 делали работу)
    //   * consecutive threads обращаются к consecutive адресам (coalesced HBM)
    //   * меньше divergence на token-boundary
    //
    // Размеры работы per wg:
    //   V: block_size × v_n_u4 = 16 × 16 = 256 uint4  (4096 bytes)
    //   K: block_size × k_n_u4 = 16 × 8  = 128 uint4  (2048 bytes)
    //   K tail: block_size × k_tail_u2 = 16 × 1 = 16 uint2 (128 bytes)
    //
    // На 128 threads это 2 uint4/thread для V и 1 uint4/thread для K — отличная
    // утилизация и одновременно сhort программа.

    const int v_total_u4 = block_size * v_n_u4;       // 256
    const int k_total_u4 = block_size * k_n_u4;       // 128
    const int k_total_u2 = block_size * k_tail_u2;    // 16

    // --------- V pass: copy-or-zero per index ----------
    // Раскладка idx -> (t, c): consecutive threads в wave попадают на соседние
    // uint4-chunks ВНУТРИ одного токена (так как v_n_u4=16 ≥ warpsize тоже 16
    // хотя на gfx906 лучше 64; здесь wave=64 покрывает 4 токена что ок).
    for (int idx = tid; idx < v_total_u4; idx += nth) {
        const int t = idx / v_n_u4;
        const int c = idx - t * v_n_u4;
        const int tok_global = block_start_tok + t;
        if (tok_global >= Sk) continue;

        const bool tok_valid = !full_oob && (tok_global < seq_len);
        uint4 val;
        if (tok_valid) {
            const __half * v_src_tok = v_src_base_bh + (int64_t)t * v_cache_token_stride;
            val = reinterpret_cast<const uint4 *>(v_src_tok)[c];
        } else {
            val = make_uint4(0u, 0u, 0u, 0u);
        }
        __half * v_dst_tok = v_out + dst_V_sh_base + (int64_t)tok_global * D;
        reinterpret_cast<uint4 *>(v_dst_tok)[c] = val;
    }

    // --------- K pass (uint4 body): copy только для tok_valid ----------
    // K tail (8 байт из 136) обрабатывается отдельно внизу.
    if (!full_oob) {
        for (int idx = tid; idx < k_total_u4; idx += nth) {
            const int t = idx / k_n_u4;
            const int c = idx - t * k_n_u4;
            const int tok_global = block_start_tok + t;
            if (tok_global >= Sk) continue;
            if (tok_global >= seq_len) continue;  // out-of-range — не трогаем K

            const uint8_t * k_src_tok = k_src_base_bh + (int64_t)t * cache_token_stride;
            uint8_t       * k_dst_tok = k_out + dst_K_sh_base + (int64_t)tok_global * bytes_per_row;
            reinterpret_cast<uint4 *>(k_dst_tok)[c] =
                reinterpret_cast<const uint4 *>(k_src_tok)[c];
        }

        // --------- K tail uint2 (8 байт) — один на токен при D=128 ----------
        if (k_tail_u2 > 0) {
            for (int idx = tid; idx < k_total_u2; idx += nth) {
                const int t = idx / k_tail_u2;
                const int c = idx - t * k_tail_u2;
                const int tok_global = block_start_tok + t;
                if (tok_global >= Sk) continue;
                if (tok_global >= seq_len) continue;

                const uint8_t * k_src_tail = k_src_base_bh
                    + (int64_t)t * cache_token_stride + (int64_t)(k_n_u4 << 4);
                uint8_t * k_dst_tail = k_out + dst_K_sh_base
                    + (int64_t)tok_global * bytes_per_row + (int64_t)(k_n_u4 << 4);
                reinterpret_cast<uint2 *>(k_dst_tail)[c] =
                    reinterpret_cast<const uint2 *>(k_src_tail)[c];
            }
        }

        // --------- K byte-tail (0..7 байт, при D=128 их 0) — cold path ----------
        if (k_tail_byte > 0) {
            for (int idx = tid; idx < block_size * k_tail_byte; idx += nth) {
                const int t = idx / k_tail_byte;
                const int c = idx - t * k_tail_byte;
                const int tok_global = block_start_tok + t;
                if (tok_global >= Sk) continue;
                if (tok_global >= seq_len) continue;
                const int base = (k_n_u4 << 4) + (k_tail_u2 << 3);
                const uint8_t * k_src_tok = k_src_base_bh + (int64_t)t * cache_token_stride;
                uint8_t * k_dst_tok = k_out + dst_K_sh_base + (int64_t)tok_global * bytes_per_row;
                k_dst_tok[base + c] = k_src_tok[base + c];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------
extern "C" hipError_t launch_gather_paged_kv_q8(
    const uint8_t * key_cache_q8,
    const __half  * value_cache,
    const int32_t * block_table,
    const int32_t * seq_lens,
    uint8_t       * k_out,
    __half        * v_out,
    int num_seqs,
    int num_kv_heads,
    int Sk,
    int D,
    int bytes_per_row,
    int block_size,
    int max_blocks_per_seq,
    int64_t cache_block_stride,
    int64_t cache_token_stride,
    int64_t cache_head_stride_q8,
    int64_t v_cache_block_stride,
    int64_t v_cache_token_stride,
    int64_t v_cache_head_stride,
    hipStream_t stream
) {
    if (num_seqs == 0 || num_kv_heads == 0 || Sk == 0) return hipSuccess;
    if (D % 32 != 0) return hipErrorInvalidValue;

    // Level 3c-step-A: GFX906_FA_GATHER_V=2 включает paged-block-coalesced kernel.
    // По умолчанию — V2 (newer, faster). Старый per-token kernel остаётся как
    // safety fallback через GFX906_FA_GATHER_V=1.
    // env var читается один раз (thread-safe: amort over all calls).
    static int cached_version = -1;
    if (cached_version < 0) {
        const char * env = getenv("GFX906_FA_GATHER_V");
        cached_version = (env && env[0] == '1') ? 1 : 2;
    }

    if (cached_version == 1) {
        dim3 block(64, 1, 1);
        dim3 grid(num_seqs, num_kv_heads, Sk);
        gather_paged_kv_q8_kernel<<<grid, block, 0, stream>>>(
            key_cache_q8, value_cache,
            block_table, seq_lens,
            k_out, v_out,
            num_seqs, num_kv_heads, Sk, D, bytes_per_row, block_size,
            max_blocks_per_seq,
            cache_block_stride, cache_token_stride, cache_head_stride_q8,
            v_cache_block_stride, v_cache_token_stride, v_cache_head_stride
        );
    } else {
        // V2: 1 wg = 1 paged block, 128 threads, grid уменьшен в block_size раз.
        const int n_paged_blocks = (Sk + block_size - 1) / block_size;
        dim3 block(128, 1, 1);
        dim3 grid(num_seqs, num_kv_heads, n_paged_blocks);
        gather_paged_kv_q8_kernel_v2<<<grid, block, 0, stream>>>(
            key_cache_q8, value_cache,
            block_table, seq_lens,
            k_out, v_out,
            num_seqs, num_kv_heads, Sk, D, bytes_per_row, block_size,
            max_blocks_per_seq,
            cache_block_stride, cache_token_stride, cache_head_stride_q8,
            v_cache_block_stride, v_cache_token_stride, v_cache_head_stride
        );
    }
    return hipGetLastError();
}
