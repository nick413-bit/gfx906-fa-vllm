"""test_gather.py — верификация fused gather kernel gfx906_fa.gather_paged_kv_q8.

Сверяем с эталонной Python-реализацией (аналог _gather_kv_q8 в gfx906_fa_paged.py):
  1) Корректное чтение paged K_q8 через block_table для валидных позиций (tok_pos < seq_len).
  2) V обнуляется для tok_pos >= seq_len (tail-masking).
  3) K tail может быть мусором — не сверяем (kernel в FA отсекает через KV_max).
"""
import torch
import gfx906_fa


dev = torch.device("cuda")


def _ref_gather(key_cache_q8, value_cache, block_table, seq_lens, Sk):
    """Python-эталон fused gather — повторяет семантику kernel'а точно.

    К разнице с _gather_kv_q8 в gfx906_fa_paged.py: здесь K не маскируем tail,
    чтобы совпадало с kernel (маска применяется только к V).
    """
    num_blocks, block_size, Hkv, bytes_per_row = key_cache_q8.shape
    _, _, _, D = value_cache.shape
    num_seqs = block_table.shape[0]
    max_blocks_needed = (Sk + block_size - 1) // block_size

    bt = block_table[:, :max_blocks_needed].to(torch.long)
    k_g = key_cache_q8[bt].reshape(num_seqs, -1, Hkv, bytes_per_row)[:, :Sk]
    v_g = value_cache[bt].reshape(num_seqs, -1, Hkv, D)[:, :Sk]

    # V tail masking.
    positions = torch.arange(Sk, device=dev, dtype=seq_lens.dtype)
    mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)   # [B, Sk]
    mask_v = mask.view(num_seqs, Sk, 1, 1).to(v_g.dtype)
    v_g = v_g * mask_v

    k_bhsd = k_g.permute(0, 2, 1, 3).contiguous()
    v_bhsd = v_g.permute(0, 2, 1, 3).contiguous()
    return k_bhsd, v_bhsd


def test_gather_basic():
    print("[test_gather_basic]")
    # Маленький пример — легко отладить.
    num_blocks  = 8
    block_size  = 16
    Hkv         = 2
    D           = 64
    bytes_per_row = (D // 32) * 34
    num_seqs    = 3
    max_blocks  = 4
    Sk          = 64  # кратно 32

    torch.manual_seed(0)
    key_cache_q8 = torch.randint(0, 256, (num_blocks, block_size, Hkv, bytes_per_row),
                                 dtype=torch.uint8, device=dev)
    value_cache  = torch.randn((num_blocks, block_size, Hkv, D),
                               dtype=torch.float16, device=dev)

    # Случайные block_table — каждая seq получает разные block_ids.
    block_table = torch.randint(0, num_blocks, (num_seqs, max_blocks),
                                dtype=torch.int32, device=dev)

    # Разнообразные seq_lens: пустая, короткая, полная.
    seq_lens = torch.tensor([0, 30, Sk], dtype=torch.int32, device=dev)

    k_out, v_out = gfx906_fa.gather_paged_kv_q8(
        key_cache_q8, value_cache, block_table, seq_lens, Sk
    )
    k_ref, v_ref = _ref_gather(key_cache_q8, value_cache, block_table, seq_lens, Sk)

    # V: точное совпадение везде (и valid, и tail=0).
    assert v_out.shape == v_ref.shape, f"V shape {v_out.shape} vs {v_ref.shape}"
    v_diff = (v_out.float() - v_ref.float()).abs().max().item()
    assert v_diff == 0.0, f"V mismatch: max diff {v_diff}"
    print(f"  V OK  (shape={tuple(v_out.shape)}, max_diff={v_diff})")

    # K: совпадение только для валидных позиций (tok_pos < seq_len).
    assert k_out.shape == k_ref.shape, f"K shape {k_out.shape} vs {k_ref.shape}"
    for s in range(num_seqs):
        sl = int(seq_lens[s].item())
        if sl == 0:
            continue
        ks = k_out[s, :, :sl, :]
        kr = k_ref[s, :, :sl, :]
        diff = (ks.int() - kr.int()).abs().max().item()
        assert diff == 0, f"K[seq={s}, :sl={sl}] mismatch: max byte diff {diff}"
    print(f"  K OK  (valid positions per seq match ref)")


def test_gather_random_stress():
    print("[test_gather_random_stress]")
    # Более крупный тест — ближе к prod-размерам одного MI50 attention layer.
    num_blocks    = 64
    block_size    = 16
    Hkv           = 4
    D             = 128
    bytes_per_row = (D // 32) * 34
    num_seqs      = 8
    max_blocks    = 8
    Sk            = 128

    torch.manual_seed(42)
    key_cache_q8 = torch.randint(0, 256, (num_blocks, block_size, Hkv, bytes_per_row),
                                 dtype=torch.uint8, device=dev)
    value_cache  = torch.randn((num_blocks, block_size, Hkv, D),
                               dtype=torch.float16, device=dev)
    block_table = torch.randint(0, num_blocks, (num_seqs, max_blocks),
                                dtype=torch.int32, device=dev)
    seq_lens = torch.randint(1, Sk + 1, (num_seqs,),
                             dtype=torch.int32, device=dev)

    k_out, v_out = gfx906_fa.gather_paged_kv_q8(
        key_cache_q8, value_cache, block_table, seq_lens, Sk
    )
    k_ref, v_ref = _ref_gather(key_cache_q8, value_cache, block_table, seq_lens, Sk)

    v_diff = (v_out.float() - v_ref.float()).abs().max().item()
    assert v_diff == 0.0, f"V mismatch: max diff {v_diff}"

    for s in range(num_seqs):
        sl = int(seq_lens[s].item())
        if sl == 0:
            continue
        ks = k_out[s, :, :sl, :]
        kr = k_ref[s, :, :sl, :]
        diff = (ks.int() - kr.int()).abs().max().item()
        assert diff == 0, f"K[seq={s}, :sl={sl}] mismatch: {diff}"

    print(f"  OK  (num_seqs={num_seqs} Hkv={Hkv} D={D} Sk={Sk})")


def test_gather_matches_old_path():
    """Сверка fused kernel с старым _gather_kv_q8 (torch fancy-indexing)."""
    print("[test_gather_matches_old_path]")
    from gfx906_fa_paged import _gather_kv_q8

    num_blocks    = 32
    block_size    = 16
    Hkv           = 2
    D             = 128
    bytes_per_row = (D // 32) * 34
    num_seqs      = 4
    max_blocks    = 4
    max_seqlen_k  = 48
    Sk_pad        = ((max_seqlen_k + 31) // 32) * 32  # = 64

    torch.manual_seed(123)
    key_cache_q8 = torch.randint(0, 256, (num_blocks, block_size, Hkv, bytes_per_row),
                                 dtype=torch.uint8, device=dev)
    value_cache  = torch.randn((num_blocks, block_size, Hkv, D),
                               dtype=torch.float16, device=dev)
    block_table = torch.randint(0, num_blocks, (num_seqs, max_blocks),
                                dtype=torch.int32, device=dev)
    seq_lens = torch.tensor([10, 32, 48, 5], dtype=torch.int32, device=dev)

    # Fused: Sk = Sk_pad
    k_fused, v_fused = gfx906_fa.gather_paged_kv_q8(
        key_cache_q8, value_cache, block_table, seq_lens, Sk_pad
    )

    # Old torch path: возвращает [B,Hkv,max_seqlen_k,...] без padding
    k_old, v_old = _gather_kv_q8(
        key_cache_q8, value_cache, block_table, seq_lens, max_seqlen_k
    )

    # Сравниваем только первые max_seqlen_k позиций (остальное в fused — padding).
    # V: валидные позиции должны совпадать. Tail обоих — 0.
    v_fused_trim = v_fused[:, :, :max_seqlen_k, :]
    v_diff = (v_fused_trim.float() - v_old.float()).abs().max().item()
    assert v_diff == 0.0, f"V old-vs-fused mismatch: {v_diff}"

    # K: сравниваем только для валидных (< seq_len) позиций.
    for s in range(num_seqs):
        sl = int(seq_lens[s].item())
        kf = k_fused[s, :, :sl, :]
        ko = k_old[s, :, :sl, :]
        diff = (kf.int() - ko.int()).abs().max().item()
        assert diff == 0, f"K[s={s}, :sl={sl}] old-vs-fused: {diff}"

    print(f"  OK  (fused kernel matches torch fancy-indexing path)")


if __name__ == "__main__":
    test_gather_basic()
    test_gather_random_stress()
    test_gather_matches_old_path()
    print("\n[test_gather] ALL PASSED")
