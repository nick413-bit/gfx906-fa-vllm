"""Тест paged KV cache path для gfx906_fa.

Шаги:
  1. Генерируем vLLM-style paged kv_cache.
  2. Отдельно имеем «contiguous» K, V для reference SDPA.
  3. Вызываем forward_paged(...) → out_flat.
  4. Сравниваем с торч SDPA (fp16, repeated K/V для GQA).
  5. Проходим кейсы: pure decode (Sq=1 per seq), prefill (Sq>1 per seq),
     mixed batch, разные Hq/Hkv.
"""
import math
import sys
import time

import torch

try:
    import gfx906_fa
    from gfx906_fa_paged import forward_paged
except ImportError as e:
    print(f"FAIL import: {e}", file=sys.stderr); sys.exit(1)


dev = 'cuda'
torch.manual_seed(0)


def make_paged_kv(num_seqs, seq_lens, Hkv, D, block_size=16, dtype=torch.float16):
    """Создаёт vLLM-style paged kv_cache + block_table + 'reference' K/V в BHSD.

    Returns:
      kv_cache:    [num_blocks, 2, block_size, Hkv, D]
      block_table: [num_seqs, max_blocks_per_seq]  int32
      K_ref:       [num_seqs, Hkv, max_seqlen_k, D]  fp16 (уже замаскирован нулями за seq_lens)
      V_ref:       same
    """
    max_seqlen = int(seq_lens.max().item())
    max_blocks = (max_seqlen + block_size - 1) // block_size

    # Суммарное число блоков (вкл. немного запас на паддинг)
    total_blocks_needed = num_seqs * max_blocks
    num_blocks = total_blocks_needed + 4  # запас

    kv_cache = torch.zeros((num_blocks, 2, block_size, Hkv, D), dtype=dtype, device=dev)

    # Block table: уникальные блоки для каждой последовательности, последовательно
    block_table = torch.zeros((num_seqs, max_blocks), dtype=torch.int32, device=dev)

    K_ref = torch.zeros((num_seqs, Hkv, max_seqlen, D), dtype=dtype, device=dev)
    V_ref = torch.zeros((num_seqs, Hkv, max_seqlen, D), dtype=dtype, device=dev)

    next_block = 0
    for s in range(num_seqs):
        n = int(seq_lens[s].item())
        nb = (n + block_size - 1) // block_size
        for b in range(nb):
            block_table[s, b] = next_block
            # Заполняем блок случайными данными
            kb = (torch.randn((block_size, Hkv, D), dtype=dtype, device=dev) * 0.1)
            vb = (torch.randn((block_size, Hkv, D), dtype=dtype, device=dev) * 0.1)
            kv_cache[next_block, 0] = kb
            kv_cache[next_block, 1] = vb
            # Ref: [Hkv, seq, D]
            start = b * block_size
            end = min(start + block_size, n)
            valid = end - start
            if valid > 0:
                K_ref[s, :, start:end, :] = kb[:valid].permute(1, 0, 2)
                V_ref[s, :, start:end, :] = vb[:valid].permute(1, 0, 2)
            next_block += 1

    return kv_cache, block_table, K_ref, V_ref


def run_paged_test(name, seq_lens_q, seq_lens_kv, Hq, Hkv, D=128, block_size=16):
    """seq_lens_q/kv — list of int для каждой последовательности."""
    num_seqs = len(seq_lens_q)
    assert len(seq_lens_kv) == num_seqs

    seq_lens_q_t  = torch.tensor(seq_lens_q, dtype=torch.int32, device=dev)
    seq_lens_kv_t = torch.tensor(seq_lens_kv, dtype=torch.int32, device=dev)

    max_seqlen_q  = int(seq_lens_q_t.max().item())
    max_seqlen_k  = int(seq_lens_kv_t.max().item())
    num_tokens    = int(seq_lens_q_t.sum().item())

    cu_seqlens_q = torch.zeros(num_seqs + 1, dtype=torch.int32, device=dev)
    cu_seqlens_q[1:] = torch.cumsum(seq_lens_q_t, dim=0)

    # Build Q flat
    query = torch.randn((num_tokens, Hq, D), dtype=torch.float32, device=dev)

    # Build paged KV + ref
    kv_cache, block_table, K_ref, V_ref = make_paged_kv(
        num_seqs, seq_lens_kv_t, Hkv, D, block_size=block_size
    )
    key_cache, value_cache = kv_cache.unbind(1)  # [num_blocks, block_size, Hkv, D]

    scale = 1.0 / math.sqrt(D)

    # ---------- Вызов gfx906_fa paged ----------
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out_flat = forward_paged(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        seq_lens=seq_lens_kv_t,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        scale=scale,
    )
    torch.cuda.synchronize()
    t_paged = (time.perf_counter() - t0) * 1000

    # ---------- Reference per-seq SDPA ----------
    # Для каждой последовательности отдельно: [1, Hq, Sq_i, D] vs [1, Hkv_repeated, Sk_i, D]
    gqa = Hq // Hkv
    out_ref_flat = torch.empty_like(out_flat)
    for s in range(num_seqs):
        Sq_s = int(seq_lens_q[s])
        Sk_s = int(seq_lens_kv[s])
        if Sq_s == 0:
            continue
        q_s = query[cu_seqlens_q[s]:cu_seqlens_q[s] + Sq_s].permute(1, 0, 2).unsqueeze(0)  # [1, Hq, Sq, D]
        k_s = K_ref[s:s+1, :, :Sk_s, :]
        v_s = V_ref[s:s+1, :, :Sk_s, :]
        if gqa > 1:
            k_s = k_s.repeat_interleave(gqa, dim=1)
            v_s = v_s.repeat_interleave(gqa, dim=1)
        # Kernel теперь применяет CAUSAL mask (vLLM всегда causal=True).
        # В reference нужен такой же causal: q_i видит k_j где k_j <= (Sk-Sq)+i.
        # torch SDPA is_causal=True работает только если Sq==Sk. Для decode/mixed
        # Sq<Sk — строим явный attn_mask.
        attn_mask = None
        if Sq_s > 0 and Sk_s >= Sq_s:
            # bool mask [Sq, Sk]: True = keep
            kv_offset = Sk_s - Sq_s
            q_pos = torch.arange(Sq_s, device=dev).unsqueeze(1)      # [Sq,1]
            k_pos = torch.arange(Sk_s, device=dev).unsqueeze(0)      # [1,Sk]
            keep  = k_pos <= (q_pos + kv_offset)                     # [Sq,Sk]
            attn_mask = keep  # torch SDPA примет bool
        ref_s = torch.nn.functional.scaled_dot_product_attention(
            q_s.to(torch.float16), k_s, v_s, scale=scale,
            attn_mask=attn_mask,
        ).to(torch.float32)  # [1, Hq, Sq, D]
        out_ref_flat[cu_seqlens_q[s]:cu_seqlens_q[s] + Sq_s] = ref_s[0].permute(1, 0, 2).reshape(Sq_s, Hq * D)

    mse = torch.mean((out_flat - out_ref_flat) ** 2).item()
    max_err = torch.max(torch.abs(out_flat - out_ref_flat)).item()
    ref_mag = torch.mean(torch.abs(out_ref_flat)).item()
    rel_mse = mse / (ref_mag ** 2 + 1e-9)

    ok = rel_mse < 1e-2 and max_err < 2.0

    print(f"{name:<26} num_seqs={num_seqs:<2} Hq={Hq:<3} Hkv={Hkv:<2} "
          f"Sq={seq_lens_q} Sk={seq_lens_kv} "
          f"mse={mse:.6f} max_err={max_err:.5f} rel_mse={rel_mse:.5f} "
          f"t={t_paged:.2f}ms {'OK' if ok else 'FAIL'}")
    return ok


def main():
    cases = [
        # name, Sq_list, Sk_list, Hq, Hkv
        ("decode_single",        [1],            [128],           8,  4),
        ("decode_single_1k",     [1],            [1024],          8,  4),
        ("decode_batch3",        [1, 1, 1],      [128, 256, 512], 8,  4),
        ("decode_batch3_1k",     [1, 1, 1],      [1024, 512, 2048], 8, 4),
        ("prefill_single",       [128],          [128],           8,  4),
        ("prefill_single_512",   [512],          [512],           8,  4),
        ("prefill_batch2",       [64, 32],       [64, 32],        8,  4),
        ("mixed_batch",          [1, 1, 64],     [1024, 2048, 64], 8, 4),
        # MiniMax-style GQA
        ("minimax_decode",       [1],            [1024],         48,  8),
        ("minimax_decode_8k",    [1],            [8192],         48,  8),
        ("minimax_mixed",        [1, 1],         [8192, 2048],   48,  8),
    ]
    n_ok, n_fail = 0, 0
    for c in cases:
        ok = run_paged_test(*c)
        n_ok += int(ok); n_fail += int(not ok)
    print(f"\nTotal: {n_ok} OK, {n_fail} FAIL")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
