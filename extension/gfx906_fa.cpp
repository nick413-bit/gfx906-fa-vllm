// gfx906_fa.cpp — pybind11 extension module для gfx906 FlashAttention Q8_0
//
// Python-доступный API:
//   gfx906_fa_forward(q, k_q8, v_fp16, scale) -> out
//     q:      fp32  [B, Hq, Sq, D]    (contiguous)
//     k_q8:   uint8 [B, Hkv, Skv, D*34/32] — pre-quantized block_q8_0 flat bytes
//     v_fp16: fp16  [B, Hkv, Skv, D]  (contiguous)
//     out:    fp32  [B, Hq, Sq, D]    (BHSD-compatible output)
//
// Внутри kernel работает в BSHD layout для output; API делает transpose.
// Квантизация K (из fp16 в block_q8_0) делается отдельной утилитой.

#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Launcher объявлен в gfx906_fa_launcher.cu
extern "C" hipError_t gfx906_fa_launch(
    const float *  Q_fp32,
    const void  *  K_q8,
    const __half * V_f16,
    float *        O_fp32,
    float2 *       O_meta,
    const int *    KV_max_d,
    const __half * MASK_f16,
    int32_t        mask_seq_kv_padded,
    // Level 3a: если non-null, kernel использует inline causal вместо mask
    // (mask_ptr тогда ДОЛЖЕН быть nullptr). q_abs_offset[b] = seq_len_total[b] - n_q[b].
    const int32_t * Q_ABS_OFFSET_d,
    int batch, int heads_q, int heads_kv,
    int seq_q, int seq_kv, int head_dim,
    float scale,
    hipStream_t stream
);

// Device-side quantize launchers (gfx906_fa_quant.cu)
extern "C" hipError_t launch_quantize_q8_0_dense(
    const __half * x,
    uint8_t      * y,
    int N,
    int D,
    hipStream_t stream
);

extern "C" hipError_t launch_reshape_and_cache_q8(
    const __half  * key,
    const int64_t * slot_mapping,
    uint8_t       * k_cache_q8,
    int num_tokens,
    int num_kv_heads,
    int head_size,
    int block_size,
    int64_t key_token_stride,
    int64_t key_head_stride,
    int64_t cache_block_stride,
    int64_t cache_token_stride,
    int64_t cache_head_stride,
    hipStream_t stream
);

// Level 3c: direct-paged FA launcher (no gather).
extern "C" hipError_t gfx906_fa_launch_paged(
    const float *   Q_fp32,
    const void  *   K_paged,
    const __half *  V_paged,
    const int32_t * block_table,
    const int32_t * kv_max_d,
    float *         O_fp32,
    float2 *        O_meta,
    const __half *  MASK_f16,
    int32_t         mask_seq_kv_padded,
    const int32_t * Q_ABS_OFFSET_d,
    int             batch,
    int             heads_q,
    int             heads_kv,
    int             seq_q,
    int             max_seq_kv,
    int             head_dim,
    int             block_size,
    int             max_blocks_per_seq,
    int64_t         k_block_stride,
    int64_t         k_token_stride,
    int64_t         k_head_stride,
    int64_t         v_block_stride,
    int64_t         v_token_stride,
    int64_t         v_head_stride,
    float           scale,
    hipStream_t     stream
);

// Level 1: fused paged gather K_q8 + V_fp16 → contiguous BHSD.
// Заменяет Python-путь _gather_kv_q8 (fancy-indexing + permute).
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
);

#define TORCH_CHECK_CUDA(x) TORCH_CHECK((x).device().is_cuda(), #x " must be CUDA/HIP")
#define TORCH_CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

torch::Tensor gfx906_fa_forward(
    torch::Tensor q,       // fp32 [B, Hq, Sq, D]
    torch::Tensor k_q8,    // uint8 [B, Hkv, Skv, D*34/32]
    torch::Tensor v_fp16,  // fp16 [B, Hkv, Skv, D]
    double scale,          // softmax scale, типично 1/sqrt(D)
    c10::optional<torch::Tensor> kv_max = c10::nullopt,  // int32 [B] или [B, Sq_tiles]
    c10::optional<torch::Tensor> mask  = c10::nullopt,    // fp16 [B, Sq, Skv_pad]
    // Level 3a inline causal mask. Mutually exclusive with `mask`.
    //   q_abs_offset: int32 [B] — абсолютная позиция 0-го query-токена в sequence.
    //   При non-nullopt kernel сам выкидывает k > (q_abs_offset[b] + col_Q_0 + j).
    // Избавляет от материальной [B, Sq_pad, Sk_pad] fp16 маски, критично для
    // prefill chunks >16K (mask был бы 100+ MB на GPU и приводил к OOM).
    c10::optional<torch::Tensor> q_abs_offset = c10::nullopt
) {
    TORCH_CHECK_CUDA(q);
    TORCH_CHECK_CUDA(k_q8);
    TORCH_CHECK_CUDA(v_fp16);
    TORCH_CHECK_CONTIG(q);
    TORCH_CHECK_CONTIG(k_q8);
    TORCH_CHECK_CONTIG(v_fp16);

    TORCH_CHECK(q.dtype() == torch::kFloat32,  "q must be fp32");
    TORCH_CHECK(k_q8.dtype() == torch::kUInt8, "k_q8 must be uint8 (block_q8_0 bytes)");
    TORCH_CHECK(v_fp16.dtype() == torch::kFloat16, "v_fp16 must be fp16");

    TORCH_CHECK(q.dim() == 4,      "q must be 4D [B, Hq, Sq, D]");
    TORCH_CHECK(v_fp16.dim() == 4, "v_fp16 must be 4D [B, Hkv, Skv, D]");

    const int batch     = q.size(0);
    const int heads_q   = q.size(1);
    const int seq_q     = q.size(2);
    const int head_dim  = q.size(3);

    const int heads_kv  = v_fp16.size(1);
    const int seq_kv    = v_fp16.size(2);

    TORCH_CHECK(v_fp16.size(0) == batch,    "v batch mismatch");
    TORCH_CHECK(v_fp16.size(3) == head_dim, "v head_dim mismatch");

    // K — layout [B, Hkv, Skv, (D/QK8_0) * sizeof(block_q8_0)] уложенный uint8
    constexpr int QK8_0_SZ = 32;
    constexpr int BLOCK_SZ = 34;  // sizeof(block_q8_0) = fp16 scale + 32 int8
    TORCH_CHECK(head_dim % QK8_0_SZ == 0, "head_dim must be multiple of 32");
    const int expected_k_bytes = (head_dim / QK8_0_SZ) * BLOCK_SZ;
    TORCH_CHECK(k_q8.size(0) == batch && k_q8.size(1) == heads_kv && k_q8.size(2) == seq_kv,
                "k_q8 first 3 dims must match [B, Hkv, Skv]");
    TORCH_CHECK(k_q8.size(3) == expected_k_bytes, "k_q8 last dim = (D/32)*34 bytes");

    // Output.
    // KERNEL пишет в layout [B, Sq, Hq, D] (BSHD):
    //   j_dst_unrolled = ((seq*Sq + sq_idx)*Hq + head)*gridY + split
    // Мы поэтому аллоцируем o_bshd как BSHD, передаём его в kernel,
    // а для API возвращаем BHSD через .transpose(1,2).contiguous().
    // Это совпадает с MiniMax / vLLM layout (они уже BSHD internally на SDPA уровне).
    auto opts_f32 = q.options().dtype(torch::kFloat32);
    torch::Tensor o_bshd = torch::empty({batch, seq_q, heads_q, head_dim}, opts_f32);

    // meta buffer (для stream-k, gridDim.y; сейчас gridY=1 → не используется,
    // но kernel может писать при прохождении dead branches — выделяем).
    torch::Tensor o_meta = torch::empty({batch, seq_q, heads_q, 2}, opts_f32);

    auto stream = c10::hip::getCurrentHIPStream().stream();

    // kv_max: опциональный int32 tensor[B] для per-sequence cutoff.
    // Kernel ждёт layout [B, gridDim.x] где gridDim.x = ceil(seq_q/ncols1).
    // ncols1 выбирается launcher'ом динамически из seq_q:
    //   seq_q > 32 → 64 ; > 16 → 32 ; > 8 → 16 ; > 4 → 8 ; > 2 → 4 ; else 2
    // Launcher сам создаёт буфер нужного размера и реплицирует kv_max[b]
    // на все gridDim.x тайлов.
    const int * kv_max_ptr = nullptr;
    torch::Tensor kv_max_expanded;
    if (kv_max.has_value()) {
        auto kvm = kv_max.value();
        TORCH_CHECK_CUDA(kvm);
        TORCH_CHECK(kvm.dtype() == torch::kInt32, "kv_max must be int32");
        TORCH_CHECK(kvm.dim() == 1, "kv_max must be 1D [B]");
        TORCH_CHECK(kvm.size(0) == batch, "kv_max[0] must equal batch");

        // Compute ncols1 same way as launcher dispatcher (keep in sync!)
        int ncols1;
        if      (seq_q > 32) ncols1 = 64;
        else if (seq_q > 16) ncols1 = 32;
        else if (seq_q >  8) ncols1 = 16;
        else if (seq_q >  4) ncols1 =  8;
        else if (seq_q >  2) ncols1 =  4;
        else                 ncols1 =  2;
        const int grid_x = (seq_q + ncols1 - 1) / ncols1;

        // Expand [B] → [B, grid_x] (contiguous), каждая sequence получает
        // одинаковое значение для всех Q-tiles (без causal mask в MVP).
        kv_max_expanded = kvm.unsqueeze(1).expand({batch, grid_x}).contiguous();
        kv_max_ptr = kv_max_expanded.data_ptr<int32_t>();
    }

    const __half * mask_ptr = nullptr;
    int32_t mask_seq_kv_padded = 0;
    torch::Tensor mask_contig;
    if (mask.has_value()) {
        auto m = mask.value();
        TORCH_CHECK_CUDA(m);
        TORCH_CHECK(m.dtype() == torch::kFloat16, "mask must be fp16");
        TORCH_CHECK(m.dim() == 3, "mask must be 3D [B, Sq, Skv_pad]");
        TORCH_CHECK(m.size(0) == batch, "mask.size(0) must equal batch");
        TORCH_CHECK(m.size(1) == seq_q, "mask.size(1) must equal seq_q");
        TORCH_CHECK(m.size(2) >= seq_kv, "mask.size(2) must be >= seq_kv");
        mask_contig = m.contiguous();
        mask_ptr = reinterpret_cast<const __half *>(mask_contig.data_ptr<at::Half>());
        mask_seq_kv_padded = (int32_t) mask_contig.size(2);
    }

    const int32_t * q_abs_offset_ptr = nullptr;
    torch::Tensor q_abs_offset_contig;
    if (q_abs_offset.has_value()) {
        TORCH_CHECK(!mask.has_value(),
            "q_abs_offset and mask are mutually exclusive (inline causal replaces mask)");
        auto qo = q_abs_offset.value();
        TORCH_CHECK_CUDA(qo);
        TORCH_CHECK(qo.dtype() == torch::kInt32, "q_abs_offset must be int32");
        TORCH_CHECK(qo.dim() == 1 && qo.size(0) == batch,
            "q_abs_offset must be 1D [B]");
        q_abs_offset_contig = qo.contiguous();
        q_abs_offset_ptr = q_abs_offset_contig.data_ptr<int32_t>();
    }

    hipError_t err = gfx906_fa_launch(
        q.data_ptr<float>(),
        k_q8.data_ptr<uint8_t>(),
        reinterpret_cast<const __half *>(v_fp16.data_ptr<at::Half>()),
        o_bshd.data_ptr<float>(),
        reinterpret_cast<float2 *>(o_meta.data_ptr<float>()),
        kv_max_ptr,
        mask_ptr,
        mask_seq_kv_padded,
        q_abs_offset_ptr,
        batch, heads_q, heads_kv, seq_q, seq_kv, head_dim,
        (float) scale,
        stream
    );

    TORCH_CHECK(err == hipSuccess, "gfx906_fa_launch failed: ", hipGetErrorString(err));

    // Transpose [B, Sq, Hq, D] → [B, Hq, Sq, D] для API совместимости с torch SDPA.
    // .contiguous() делает единоразовую копию (~1% от времени attention на больших Sq).
    // Для интеграции с vLLM стоит экспонировать и BSHD native путь — чтобы избежать копии.
    return o_bshd.transpose(1, 2).contiguous();
}

// ============================================================================
// Q8_0 quantization utility: fp16 K-tensor → block_q8_0 uint8 tensor
//
// Device-side (gfx906_fa_quant.cu). Работает с N-мерным contiguous тензором
// чей последний dim = D (кратно 32). Выход: тот же layout с заменой D → D/32*34.
// ============================================================================
torch::Tensor quantize_q8_0(torch::Tensor k_fp16) {
    TORCH_CHECK_CUDA(k_fp16);
    TORCH_CHECK_CONTIG(k_fp16);
    TORCH_CHECK(k_fp16.dtype() == torch::kFloat16, "k_fp16 must be fp16");
    TORCH_CHECK(k_fp16.dim() >= 1, "k_fp16 must have at least 1 dim");

    const int D = k_fp16.size(-1);
    TORCH_CHECK(D % 32 == 0, "last dim must be multiple of 32");
    const int blocks_per_row = D / 32;
    const int bytes_per_row  = blocks_per_row * 34;

    // N = произведение всех измерений кроме последнего.
    int64_t N = 1;
    for (int i = 0; i < k_fp16.dim() - 1; ++i) {
        N *= k_fp16.size(i);
    }

    // out shape: те же dims + (bytes_per_row) вместо D.
    std::vector<int64_t> out_shape;
    for (int i = 0; i < k_fp16.dim() - 1; ++i) {
        out_shape.push_back(k_fp16.size(i));
    }
    out_shape.push_back(bytes_per_row);
    auto out = torch::empty(out_shape, k_fp16.options().dtype(torch::kUInt8));

    auto stream = c10::hip::getCurrentHIPStream().stream();
    hipError_t err = launch_quantize_q8_0_dense(
        reinterpret_cast<const __half*>(k_fp16.data_ptr<at::Half>()),
        out.data_ptr<uint8_t>(),
        (int)N,
        D,
        stream
    );
    TORCH_CHECK(err == hipSuccess,
                "launch_quantize_q8_0_dense failed: ", hipGetErrorString(err));
    return out;
}

// ============================================================================
// reshape_and_cache_q8: квантует новые K-токены и пишет их в paged Q8 cache
// по slot_mapping. Вызывается из do_kv_cache_update в backend.
//
// Аргументы:
//   key           [num_tokens, Hkv, D]                                 fp16
//   slot_mapping  [num_tokens]                                         int64
//   k_cache_q8    [num_blocks, block_size, Hkv, (D/32)*34]             uint8
//
// slot_mapping[i] < 0 → токен пропускается (padding).
// ============================================================================
void reshape_and_cache_q8(
    torch::Tensor key,            // [num_tokens, Hkv, D] fp16
    torch::Tensor slot_mapping,   // [num_tokens]         int64
    torch::Tensor k_cache_q8      // [num_blocks, block_size, Hkv, (D/32)*34] uint8
) {
    TORCH_CHECK_CUDA(key);
    TORCH_CHECK_CUDA(slot_mapping);
    TORCH_CHECK_CUDA(k_cache_q8);
    TORCH_CHECK(key.dtype() == torch::kFloat16, "key must be fp16");
    TORCH_CHECK(slot_mapping.dtype() == torch::kInt64, "slot_mapping must be int64");
    TORCH_CHECK(k_cache_q8.dtype() == torch::kUInt8, "k_cache_q8 must be uint8");

    TORCH_CHECK(key.dim() == 3, "key must be [num_tokens, Hkv, D]");
    TORCH_CHECK(k_cache_q8.dim() == 4, "k_cache_q8 must be [B, bs, Hkv, bytes]");

    const int num_tokens   = key.size(0);
    const int num_kv_heads = key.size(1);
    const int head_size    = key.size(2);
    TORCH_CHECK(head_size % 32 == 0, "head_size must be multiple of 32");
    const int bytes_per_row = (head_size / 32) * 34;

    const int block_size = k_cache_q8.size(1);
    TORCH_CHECK(k_cache_q8.size(2) == num_kv_heads,
                "k_cache_q8 Hkv mismatch: got ", k_cache_q8.size(2),
                " vs key Hkv=", num_kv_heads);
    TORCH_CHECK(k_cache_q8.size(3) == bytes_per_row,
                "k_cache_q8 last dim mismatch: got ", k_cache_q8.size(3),
                " expected ", bytes_per_row);

    TORCH_CHECK(slot_mapping.size(0) == num_tokens,
                "slot_mapping length mismatch");

    // Strides (в элементах для key, в байтах для cache).
    const int64_t key_token_stride = key.stride(0);  // обычно Hkv*D
    const int64_t key_head_stride  = key.stride(1);  // обычно D

    const int64_t cache_block_stride = (int64_t)block_size * num_kv_heads * bytes_per_row;
    const int64_t cache_token_stride = (int64_t)num_kv_heads * bytes_per_row;
    const int64_t cache_head_stride  = bytes_per_row;

    auto stream = c10::hip::getCurrentHIPStream().stream();
    hipError_t err = launch_reshape_and_cache_q8(
        reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
        slot_mapping.data_ptr<int64_t>(),
        k_cache_q8.data_ptr<uint8_t>(),
        num_tokens, num_kv_heads, head_size, block_size,
        key_token_stride, key_head_stride,
        cache_block_stride, cache_token_stride, cache_head_stride,
        stream
    );
    TORCH_CHECK(err == hipSuccess,
                "launch_reshape_and_cache_q8 failed: ", hipGetErrorString(err));
}

// ============================================================================
// Level 1: fused gather K_q8 + V_fp16 → contiguous BHSD.
//
// Вход:
//   key_cache_q8  [num_blocks, block_size, Hkv, bytes_per_row]  uint8
//   value_cache   [num_blocks, block_size, Hkv, D]              fp16
//   block_table   [num_seqs, max_num_blocks]                    int32
//   seq_lens      [num_seqs]                                    int32
//   Sk            — max_seqlen_k (padded до кратного 32 на host)
//
// Выход (ОДНИМ вызовом, без лишних temp buffers):
//   K_out = uint8 [num_seqs, Hkv, Sk, bytes_per_row]
//   V_out = fp16  [num_seqs, Hkv, Sk, D]
//
// Tail-токены в V_out обнуляются, в K_out оставляются мусором (kernel отсекает
// через KV_max). Это идентично семантике _gather_kv_q8 в Python.
// ============================================================================
std::vector<torch::Tensor> gather_paged_kv_q8(
    torch::Tensor key_cache_q8,  // uint8 [num_blocks, block_size, Hkv, bytes_per_row]
    torch::Tensor value_cache,   // fp16  [num_blocks, block_size, Hkv, D]
    torch::Tensor block_table,   // int32 [num_seqs, max_num_blocks]
    torch::Tensor seq_lens,      // int32 [num_seqs]
    int64_t Sk,
    c10::optional<torch::Tensor> k_out_opt = c10::nullopt,  // uint8 [B,Hkv,Sk,bytes] — grow-buffer
    c10::optional<torch::Tensor> v_out_opt = c10::nullopt   // fp16  [B,Hkv,Sk,D]
) {
    TORCH_CHECK_CUDA(key_cache_q8);
    TORCH_CHECK_CUDA(value_cache);
    TORCH_CHECK_CUDA(block_table);
    TORCH_CHECK_CUDA(seq_lens);

    TORCH_CHECK(key_cache_q8.dtype() == torch::kUInt8, "key_cache_q8 must be uint8");
    TORCH_CHECK(value_cache.dtype()  == torch::kFloat16, "value_cache must be fp16");
    TORCH_CHECK(seq_lens.dtype()     == torch::kInt32,  "seq_lens must be int32");

    TORCH_CHECK(key_cache_q8.dim() == 4, "key_cache_q8 must be 4D");
    TORCH_CHECK(value_cache.dim()  == 4, "value_cache must be 4D");

    const int num_blocks        = key_cache_q8.size(0);
    const int block_size        = key_cache_q8.size(1);
    const int num_kv_heads      = key_cache_q8.size(2);
    const int bytes_per_row     = key_cache_q8.size(3);
    const int D                 = value_cache.size(3);

    TORCH_CHECK(value_cache.size(0) == num_blocks, "value_cache blocks mismatch");
    TORCH_CHECK(value_cache.size(1) == block_size, "value_cache block_size mismatch");
    TORCH_CHECK(value_cache.size(2) == num_kv_heads, "value_cache Hkv mismatch");
    TORCH_CHECK(D % 32 == 0, "D must be multiple of 32");
    TORCH_CHECK(bytes_per_row == (D / 32) * 34,
                "bytes_per_row must equal (D/32)*34, got ", bytes_per_row,
                " vs expected ", (D / 32) * 34);

    // block_table может быть int32 или int64 — приведём к int32 на стороне вызова
    // (vLLM уже int32). Для безопасности делаем contiguous + проверяем.
    TORCH_CHECK(block_table.dtype() == torch::kInt32,
                "block_table must be int32 (caller should .to(torch.int32))");
    TORCH_CHECK(block_table.is_contiguous(), "block_table must be contiguous");
    TORCH_CHECK(seq_lens.is_contiguous(),   "seq_lens must be contiguous");

    const int num_seqs           = block_table.size(0);
    const int max_blocks_per_seq = block_table.size(1);
    TORCH_CHECK(seq_lens.size(0) == num_seqs, "seq_lens vs block_table batch mismatch");

    // Sk-padding: требование FA kernel — кратно 32 (Sk_pad в Python-обёртке).
    TORCH_CHECK(Sk > 0 && (Sk % 32) == 0, "Sk must be positive multiple of 32, got ", Sk);

    // Outputs — contiguous BHSD.
    // Если caller передал pre-allocated buffer ТОЧНО нужного shape/dtype/contig —
    // используем без копии. Иначе аллоцируем новый (поведение как раньше).
    //
    // Зачем точный size: slice [B', :, Sk', :] от большего buffer [B_max, Hkv, Sk_max, D]
    // остаётся contiguous только когда B' == B_max И Sk' == Sk_max, т.е. на практике
    // всегда false. Вызывающий код сам делает grow-strategy в attention backend.
    auto use_or_alloc = [&](const c10::optional<torch::Tensor>& buf,
                            at::TensorOptions opts,
                            int64_t dim3) -> torch::Tensor {
        if (buf.has_value()) {
            const auto & t = buf.value();
            if (t.dim() == 4
                && t.size(0) == num_seqs
                && t.size(1) == num_kv_heads
                && t.size(2) == Sk
                && t.size(3) == dim3
                && t.dtype() == opts.dtype()
                && t.device() == opts.device()
                && t.is_contiguous()) {
                return t;
            }
        }
        return torch::empty({(int64_t)num_seqs, (int64_t)num_kv_heads, Sk, dim3}, opts);
    };

    auto k_out = use_or_alloc(k_out_opt, key_cache_q8.options(), (int64_t)bytes_per_row);
    auto v_out = use_or_alloc(v_out_opt, value_cache.options(),  (int64_t)D);

    // Strides.
    const int64_t cache_block_stride    = (int64_t)block_size * num_kv_heads * bytes_per_row;
    const int64_t cache_token_stride    = (int64_t)num_kv_heads * bytes_per_row;
    const int64_t cache_head_stride_q8  = (int64_t)bytes_per_row;

    const int64_t v_cache_block_stride  = (int64_t)block_size * num_kv_heads * D;
    const int64_t v_cache_token_stride  = (int64_t)num_kv_heads * D;
    const int64_t v_cache_head_stride   = (int64_t)D;

    auto stream = c10::hip::getCurrentHIPStream().stream();
    hipError_t err = launch_gather_paged_kv_q8(
        key_cache_q8.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(value_cache.data_ptr<at::Half>()),
        block_table.data_ptr<int32_t>(),
        seq_lens.data_ptr<int32_t>(),
        k_out.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(v_out.data_ptr<at::Half>()),
        num_seqs, num_kv_heads, (int)Sk, D, bytes_per_row, block_size,
        max_blocks_per_seq,
        cache_block_stride, cache_token_stride, cache_head_stride_q8,
        v_cache_block_stride, v_cache_token_stride, v_cache_head_stride,
        stream
    );
    TORCH_CHECK(err == hipSuccess,
                "launch_gather_paged_kv_q8 failed: ", hipGetErrorString(err));

    return {k_out, v_out};
}

// ============================================================================
// Level 3c: Direct-paged FlashAttention.
//
// НЕТ gather step: kernel читает K/V directly from paged cache через
// block_table indirection. Уменьшает HBM трафик в 2x (нет промежуточной копии),
// ожидаемый выигрыш +15-25% на длинных контекстах.
//
// Layout (vLLM-compatible, block_size=16):
//   q              : fp32  [B, Hq, Sq_pad, D]
//   key_cache_q8   : uint8 [num_blocks, 16, Hkv, (D/32)*34]
//   value_cache    : fp16  [num_blocks, 16, Hkv, D]
//   block_table    : int32 [num_seqs, max_blocks_per_seq]
//   seq_lens       : int32 [num_seqs]
//
// Выход: [B, Sq_pad, Hq, D] fp32 в BSHD (тот же формат что у forward).
// ============================================================================
torch::Tensor gfx906_fa_forward_paged_direct(
    torch::Tensor q,               // fp32  [B, Hq, Sq_pad, D]
    torch::Tensor key_cache_q8,    // uint8 [num_blocks, 16, Hkv, (D/32)*34]
    torch::Tensor value_cache,     // fp16  [num_blocks, 16, Hkv, D]
    torch::Tensor block_table,     // int32 [num_seqs, max_blocks_per_seq]
    torch::Tensor seq_lens,        // int32 [num_seqs]
    double scale,
    c10::optional<torch::Tensor> mask         = c10::nullopt,
    c10::optional<torch::Tensor> q_abs_offset = c10::nullopt
) {
    TORCH_CHECK_CUDA(q);
    TORCH_CHECK_CUDA(key_cache_q8);
    TORCH_CHECK_CUDA(value_cache);
    TORCH_CHECK_CUDA(block_table);
    TORCH_CHECK_CUDA(seq_lens);
    TORCH_CHECK_CONTIG(q);

    TORCH_CHECK(q.dtype() == torch::kFloat32, "q must be fp32");
    TORCH_CHECK(key_cache_q8.dtype() == torch::kUInt8, "key_cache_q8 must be uint8");
    TORCH_CHECK(value_cache.dtype() == torch::kFloat16, "value_cache must be fp16");
    TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must be int32");
    TORCH_CHECK(seq_lens.dtype() == torch::kInt32, "seq_lens must be int32");

    TORCH_CHECK(q.dim() == 4, "q must be 4D [B, Hq, Sq, D]");
    TORCH_CHECK(key_cache_q8.dim() == 4,
                "key_cache_q8 must be [num_blocks, block_size, Hkv, bytes_per_row]");
    TORCH_CHECK(value_cache.dim() == 4,
                "value_cache must be [num_blocks, block_size, Hkv, D]");

    const int batch    = q.size(0);
    const int heads_q  = q.size(1);
    const int seq_q    = q.size(2);
    const int head_dim = q.size(3);

    const int num_blocks        = key_cache_q8.size(0);
    const int block_size        = key_cache_q8.size(1);
    const int heads_kv          = key_cache_q8.size(2);
    const int bytes_per_row     = key_cache_q8.size(3);
    const int D_val             = value_cache.size(3);

    TORCH_CHECK(value_cache.size(0) == num_blocks, "value_cache blocks mismatch");
    TORCH_CHECK(value_cache.size(1) == block_size, "value_cache block_size mismatch");
    TORCH_CHECK(value_cache.size(2) == heads_kv, "value_cache Hkv mismatch");
    TORCH_CHECK(D_val == head_dim, "value_cache D mismatch q D");
    TORCH_CHECK(bytes_per_row == (head_dim/32)*34,
                "bytes_per_row must equal (D/32)*34");
    TORCH_CHECK(block_size == 16, "Only block_size=16 supported in v1, got ", block_size);

    TORCH_CHECK(block_table.dim() == 2, "block_table must be [num_seqs, max_blocks]");
    TORCH_CHECK(block_table.size(0) == batch, "block_table[0] must equal batch");
    TORCH_CHECK(seq_lens.dim() == 1 && seq_lens.size(0) == batch,
                "seq_lens must be [batch]");

    auto block_table_c = block_table.is_contiguous() ? block_table : block_table.contiguous();
    auto seq_lens_c    = seq_lens.is_contiguous() ? seq_lens : seq_lens.contiguous();

    const int max_blocks_per_seq = block_table_c.size(1);

    // Strides (bytes) for paged layout.
    const int64_t k_token_stride = (int64_t) heads_kv * bytes_per_row;
    const int64_t k_head_stride  = (int64_t) bytes_per_row;
    const int64_t k_block_stride = (int64_t) block_size * k_token_stride;

    const int64_t v_token_stride = (int64_t) heads_kv * head_dim * sizeof(__half);
    const int64_t v_head_stride  = (int64_t) head_dim * sizeof(__half);
    const int64_t v_block_stride = (int64_t) block_size * v_token_stride;

    // max_seq_kv используется kernel'ом ТОЛЬКО как fallback для ne11 когда
    // KV_max==nullptr. Мы всегда передаём kv_max_d, поэтому ne11 фактически
    // не читается. Ставим верхнюю оценку max_blocks_per_seq*block_size —
    // корректно и без D2H sync overhead.
    const int max_seq_kv = max_blocks_per_seq * block_size;

    // Output allocation — тот же BSHD как в forward.
    auto opts_f32 = q.options().dtype(torch::kFloat32);
    torch::Tensor o_bshd = torch::empty({batch, seq_q, heads_q, head_dim}, opts_f32);
    torch::Tensor o_meta = torch::empty({batch, seq_q, heads_q, 2}, opts_f32);

    // KV_max expansion [B] → [B, grid_x] (тот же приём что в forward).
    int ncols1;
    if      (seq_q > 32) ncols1 = 64;
    else if (seq_q > 16) ncols1 = 32;
    else if (seq_q >  8) ncols1 = 16;
    else if (seq_q >  4) ncols1 =  8;
    else if (seq_q >  2) ncols1 =  4;
    else                 ncols1 =  2;
    const int grid_x = (seq_q + ncols1 - 1) / ncols1;

    auto kv_max_expanded = seq_lens_c.unsqueeze(1).expand({batch, grid_x}).contiguous();

    // Optional mask.
    const __half * mask_ptr = nullptr;
    int32_t mask_seq_kv_padded = 0;
    torch::Tensor mask_contig;
    if (mask.has_value()) {
        auto m = mask.value();
        TORCH_CHECK_CUDA(m);
        TORCH_CHECK(m.dtype() == torch::kFloat16, "mask must be fp16");
        TORCH_CHECK(m.dim() == 3, "mask must be 3D");
        mask_contig = m.contiguous();
        mask_ptr = reinterpret_cast<const __half *>(mask_contig.data_ptr<at::Half>());
        mask_seq_kv_padded = (int32_t) mask_contig.size(2);
    }

    // Optional q_abs_offset (inline causal).
    const int32_t * q_abs_offset_ptr = nullptr;
    torch::Tensor q_abs_offset_c;
    if (q_abs_offset.has_value()) {
        TORCH_CHECK(!mask.has_value(),
            "q_abs_offset and mask are mutually exclusive");
        auto qo = q_abs_offset.value();
        TORCH_CHECK_CUDA(qo);
        TORCH_CHECK(qo.dtype() == torch::kInt32, "q_abs_offset must be int32");
        TORCH_CHECK(qo.dim() == 1 && qo.size(0) == batch,
            "q_abs_offset must be [batch]");
        q_abs_offset_c = qo.contiguous();
        q_abs_offset_ptr = q_abs_offset_c.data_ptr<int32_t>();
    }

    auto stream = c10::hip::getCurrentHIPStream().stream();
    hipError_t err = gfx906_fa_launch_paged(
        q.data_ptr<float>(),
        key_cache_q8.data_ptr<uint8_t>(),
        reinterpret_cast<const __half *>(value_cache.data_ptr<at::Half>()),
        block_table_c.data_ptr<int32_t>(),
        kv_max_expanded.data_ptr<int32_t>(),
        o_bshd.data_ptr<float>(),
        reinterpret_cast<float2 *>(o_meta.data_ptr<float>()),
        mask_ptr,
        mask_seq_kv_padded,
        q_abs_offset_ptr,
        batch, heads_q, heads_kv, seq_q, max_seq_kv, head_dim,
        block_size, max_blocks_per_seq,
        k_block_stride, k_token_stride, k_head_stride,
        v_block_stride, v_token_stride, v_head_stride,
        (float) scale,
        stream
    );

    TORCH_CHECK(err == hipSuccess, "gfx906_fa_launch_paged failed: ", hipGetErrorString(err));

    return o_bshd.transpose(1, 2).contiguous();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "gfx906 FlashAttention Q8_0 KV — ported from iacopPBK/llama.cpp-gfx906";
    m.def("forward",       &gfx906_fa_forward, "FA forward (Q fp32, K q8_0, V fp16)",
          py::arg("q"), py::arg("k_q8"), py::arg("v_fp16"), py::arg("scale"),
          py::arg("kv_max")        = c10::nullopt,
          py::arg("mask")          = c10::nullopt,
          py::arg("q_abs_offset")  = c10::nullopt);
    m.def("quantize_q8_0", &quantize_q8_0,
          "Quantize fp16 tensor (last dim D) → block_q8_0 uint8 (device-side)",
          py::arg("k_fp16"));
    m.def("reshape_and_cache_q8", &reshape_and_cache_q8,
          "Quantize new K tokens (fp16) and scatter into paged Q8 KV-cache "
          "by slot_mapping. In-place write to k_cache_q8.",
          py::arg("key"), py::arg("slot_mapping"), py::arg("k_cache_q8"));
    m.def("gather_paged_kv_q8", &gather_paged_kv_q8,
          "Fused gather: paged K_q8 + paged V_fp16 → contiguous BHSD outputs. "
          "Returns [k_out, v_out]. V tail zeroed per seq_lens; K tail unmasked. "
          "k_out/v_out: optional pre-allocated buffers (exact shape match) to "
          "avoid peak VRAM spikes on large Sk.",
          py::arg("key_cache_q8"), py::arg("value_cache"),
          py::arg("block_table"), py::arg("seq_lens"), py::arg("Sk"),
          py::arg("k_out") = c10::nullopt,
          py::arg("v_out") = c10::nullopt);
    m.def("forward_paged_direct", &gfx906_fa_forward_paged_direct,
          "Level 3c: Direct-paged FA (no gather). Reads K/V from paged cache "
          "via block_table indirection. Output: fp32 [B, Hq, Sq, D].",
          py::arg("q"), py::arg("key_cache_q8"), py::arg("value_cache"),
          py::arg("block_table"), py::arg("seq_lens"), py::arg("scale"),
          py::arg("mask")         = c10::nullopt,
          py::arg("q_abs_offset") = c10::nullopt);
}
