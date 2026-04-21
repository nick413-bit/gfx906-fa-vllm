"""
Intergrated test: полный pipeline gather_paged_kv_q8 → forward (FA kernel) на mixed-batch.

Сравниваем fullpath:
  1. gather_paged_kv_q8(Sk_pad) → forward → output
  2. _gather_kv_q8_py(max_sk) → quantize_q8_0 → forward → output
  3. SDPA reference (eager math)

Если (1) ≠ (3) но (2) ≈ (3) → баг в связке gather_kernel + attention_kernel (Sk_pad vs max_sk).
Если (1) ≈ (2) ≠ (3) → оба наших пути численно кривые vs SDPA.
"""

import os, sys, torch
import os
sys.path.insert(0, "/gfx906_fa")
import gfx906_fa


def dequantize_q8(q8, D):
    shape = q8.shape
    pre = shape[:-1]
    bpr = D // 32
    q8 = q8.reshape(-1, bpr, 34)
    N = q8.shape[0]
    scale = q8[:, :, :2].contiguous().view(torch.float16).float()[..., 0:1]
    q = q8[:, :, 2:].to(torch.int8).float()
    out = (q * scale).reshape(N, D)
    return out.reshape(*pre, D)


def main():
    torch.manual_seed(42)
    dev = torch.device("cuda:0")

    # vLLM-like params from fwd debug log: Hq=6, Hkv=1 per TP-slice
    Hq = int(os.environ.get("HQ", "6"))
    Hkv = int(os.environ.get("HKV", "1"))
    D = 128
    block_size = 16
    num_blocks = 512
    scale = 1.0 / (D ** 0.5)

    # Sq_max=50, Sk_max=50 (prefill) или Sq=1 (decode)
    sq_from_env = int(os.environ.get("SQ_ALL", "0"))  # если >0 → Sk=Sq (prefill)
    if sq_from_env > 0:
        seq_lens_list = [sq_from_env]
    else:
        seq_lens_list = [47, 63, 128, 200]
    num_seqs = len(seq_lens_list)
    max_sk = max(seq_lens_list)
    Sk_pad = ((max_sk + 31) // 32) * 32
    print(f"num_seqs={num_seqs}, seq_lens={seq_lens_list}, max_sk={max_sk}, Sk_pad={Sk_pad}")

    # Allocate caches
    bytes_per_row = (D // 32) * 34
    k_cache_q8 = torch.zeros(num_blocks, block_size, Hkv, bytes_per_row,
                             dtype=torch.uint8, device=dev)
    v_cache = torch.zeros(num_blocks, block_size, Hkv, D, dtype=torch.float16, device=dev)

    max_blocks_per_seq = (max_sk + block_size - 1) // block_size
    block_table = torch.zeros(num_seqs, max_blocks_per_seq, dtype=torch.int32, device=dev)

    next_block = 0
    k_orig = []  # [B][sl, Hkv, D] fp16 исходник
    v_orig = []
    for s, sl in enumerate(seq_lens_list):
        nb = max_blocks_per_seq
        block_ids = torch.arange(next_block, next_block + nb, dtype=torch.int32, device=dev)
        block_table[s, :] = block_ids
        next_block += nb

        k_s = (torch.randn(sl, Hkv, D, dtype=torch.float16, device=dev) * 0.3)
        v_s = (torch.randn(sl, Hkv, D, dtype=torch.float16, device=dev) * 0.3)
        k_orig.append(k_s)
        v_orig.append(v_s)

        slots = torch.empty(sl, dtype=torch.int64, device=dev)
        for i in range(sl):
            slots[i] = int(block_ids[i // block_size].item()) * block_size + (i % block_size)
        gfx906_fa.reshape_and_cache_q8(k_s.contiguous(), slots, k_cache_q8)
        # V запись
        for i in range(sl):
            slot = slots[i].item()
            bi = slot // block_size
            bo = slot % block_size
            v_cache[bi, bo] = v_s[i]

    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=dev)

    # Query: prefill-step — несколько токенов на sequence (chunked prefill)
    Sq = int(os.environ.get("SQ", "1"))
    # Для prefill-сценария Sq=Sk_i (т.е. каждая seq префилит все свои токены).
    # В этом случае query per seq должен быть [Hq, sl, D], НЕ одинаковый Sq.
    # Но наш forward API требует [B, Hq, Sq, D] с одинаковым Sq. Поэтому
    # берём Sq=max_sk и кладём query только в последние sl токенов каждой seq.
    if sq_from_env > 0:
        # Честный prefill: Sq = sl (one seq)
        Sq = seq_lens_list[0]
    query = torch.randn(num_seqs, Hq, Sq, D, dtype=torch.float32, device=dev) * 0.5
    print(f"SQ={Sq}, Hq={Hq}, Hkv={Hkv}")

    # q_abs_offset для causal (Sq > 1): будем считать что query это "последние Sq токенов" seq
    if Sq > 1:
        q_abs_offset = (seq_lens - Sq).to(torch.int32).contiguous()
    else:
        q_abs_offset = None

    # --- Path 1: gather_paged_kv_q8 (FUSED=1) ---
    K_kn, V_kn = gfx906_fa.gather_paged_kv_q8(
        k_cache_q8, v_cache, block_table, seq_lens, Sk_pad)
    out1 = gfx906_fa.forward(query, K_kn, V_kn, scale,
                             kv_max=seq_lens.to(torch.int32),
                             q_abs_offset=q_abs_offset)
    print(f"Path 1 (FUSED kernel gather, Sk_pad={Sk_pad}): out shape={tuple(out1.shape)}")
    # out: [B, Hq, Sq, D]

    # --- Path 2: _gather_kv_q8_py (FUSED=0) ---
    # Используем точно тот же код что в gfx906_fa_paged.py
    max_blocks_needed = (max_sk + block_size - 1) // block_size
    bt = block_table[:, :max_blocks_needed].to(torch.long)
    kg = k_cache_q8[bt].view(num_seqs, -1, Hkv, bytes_per_row)[:, :max_sk].contiguous()
    vg = v_cache[bt].view(num_seqs, -1, Hkv, D)[:, :max_sk].contiguous()
    # V mask
    pos = torch.arange(max_sk, device=dev, dtype=seq_lens.dtype)
    mask = pos.unsqueeze(0) < seq_lens.unsqueeze(1)
    vg = vg * mask.view(num_seqs, max_sk, 1, 1).to(vg.dtype)
    K_py = kg.permute(0, 2, 1, 3).contiguous()  # [B, Hkv, max_sk, bytes]
    V_py = vg.permute(0, 2, 1, 3).contiguous()  # [B, Hkv, max_sk, D]
    out2 = gfx906_fa.forward(query, K_py, V_py, scale,
                             kv_max=seq_lens.to(torch.int32),
                             q_abs_offset=q_abs_offset)
    # out: [B, Hq, Sq, D]
    print(f"Path 2 (Python gather, max_sk={max_sk}): out shape={tuple(out2.shape)}")

    # --- Path 3: SDPA reference ---
    # K_recon (dequantize из K_py для честности — наш FA тоже работает с q8)
    K_deq = dequantize_q8(K_py, D)  # [B, Hkv, max_sk, D]
    rep = Hq // Hkv
    outs3 = []
    for s in range(num_seqs):
        sl = seq_lens_list[s]
        q_s = query[s, :, :, :]  # [Hq, Sq, D]
        k_s = K_deq[s, :, :sl, :]  # [Hkv, sl, D]
        v_s = V_py[s, :, :sl, :].float()  # [Hkv, sl, D]
        k_rep = k_s.repeat_interleave(rep, dim=0)  # [Hq, sl, D]
        v_rep = v_s.repeat_interleave(rep, dim=0)
        # Causal mask: q_i (abs_pos = sl - Sq + i) can see k_j for j <= abs_pos
        if Sq > 1:
            q_abs_pos = torch.arange(sl - Sq, sl, device=dev).view(Sq, 1)  # [Sq,1]
            k_pos = torch.arange(sl, device=dev).view(1, sl)  # [1, sl]
            attn_mask = (q_abs_pos >= k_pos).unsqueeze(0)  # [1, Sq, sl]
        else:
            attn_mask = None
        ref = torch.nn.functional.scaled_dot_product_attention(
            q_s.unsqueeze(0), k_rep.unsqueeze(0), v_rep.unsqueeze(0),
            attn_mask=attn_mask, scale=scale,
        ).squeeze(0)  # [Hq, Sq, D]
        outs3.append(ref)  # [Hq, Sq, D]
    out3 = torch.stack(outs3, dim=0)  # [B, Hq, Sq, D]
    print(f"Path 3 (SDPA): out shape={tuple(out3.shape)}")

    # Compare paths per seq
    print("\n-- Comparisons per sequence --")
    for s in range(num_seqs):
        # out1/out2: [B, Hq, Sq, D] — возможно кернел возвращает [B, Sq, Hq, D]? Проверим форму
        o1 = out1[s].reshape(-1).float()
        o2 = out2[s].reshape(-1).float()
        o3 = out3[s].reshape(-1).float()
        d12 = (o1 - o2).abs().max().item()
        d13 = (o1 - o3).abs().max().item()
        d23 = (o2 - o3).abs().max().item()
        print(f"seq {s} (sl={seq_lens_list[s]}): "
              f"|P1-P2|={d12:.4e}, |P1-SDPA|={d13:.4e}, |P2-SDPA|={d23:.4e}")


if __name__ == "__main__":
    main()
