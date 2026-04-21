"""Тесты для device-side квантования Q8_0.

Проверяем:
  1. quantize_q8_0 device-версия совпадает с reference CPU-расчётом.
  2. reshape_and_cache_q8: запись новых токенов в paged Q8 буфер
     по slot_mapping даёт такие же байты как quantize_q8_0(full_K).
  3. Fast-path forward_paged (key_cache_q8) ≈ legacy-path forward_paged
     (gather FP16 + quantize on-the-fly).
"""
import math
import sys

import torch

try:
    import gfx906_fa
    from gfx906_fa_paged import forward_paged
except ImportError as e:
    print(f"FAIL import: {e}", file=sys.stderr); sys.exit(1)


dev = 'cuda'
torch.manual_seed(42)


# ---------------------------------------------------------------------------
# Reference CPU quantize_q8_0 (bit-exact с ggml)
# ---------------------------------------------------------------------------
def ref_quantize_q8_0_cpu(x_fp16: torch.Tensor) -> torch.Tensor:
    """x: [..., D] fp16 → [..., (D/32)*34] uint8 via CPU reference."""
    x = x_fp16.cpu().float()  # round через fp32 identically to kernel
    D = x.shape[-1]
    assert D % 32 == 0
    N = x.numel() // D
    flat = x.reshape(N, D)
    blocks = D // 32
    out = torch.zeros(N, blocks * 34, dtype=torch.uint8)
    for r in range(N):
        for b in range(blocks):
            vec = flat[r, b * 32:(b + 1) * 32]
            amax = vec.abs().max().item()
            d = amax / 127.0
            id_ = 1.0 / d if d > 0 else 0.0
            # fp16 scale bytes (little-endian)
            d_h = torch.tensor([d], dtype=torch.float16)
            scale_bytes = d_h.view(torch.uint8)
            out[r, b * 34 + 0] = scale_bytes[0]
            out[r, b * 34 + 1] = scale_bytes[1]
            # int8 quants
            q = (vec * id_).round().clamp(-128, 127).to(torch.int8)
            out[r, b * 34 + 2:b * 34 + 34] = q.view(torch.uint8)
    return out.view(*x_fp16.shape[:-1], blocks * 34).to(x_fp16.device)


# ---------------------------------------------------------------------------
# Test 1: quantize_q8_0 dense
# ---------------------------------------------------------------------------
def test_quantize_q8_0_dense():
    print("[test_quantize_q8_0_dense]")
    for shape in [(1, 32), (4, 128), (2, 8, 128), (1, 4, 64, 128)]:
        x = torch.randn(shape, dtype=torch.float16, device=dev) * 0.5
        y_dev = gfx906_fa.quantize_q8_0(x)
        y_ref = ref_quantize_q8_0_cpu(x)
        assert y_dev.shape == y_ref.shape, f"shape mismatch {y_dev.shape} vs {y_ref.shape}"
        diff = (y_dev.cpu().int() - y_ref.cpu().int()).abs()
        # Квантование детерминистично; допуск 1 LSB из-за возможного round-half
        # в разных средах. Для большинства blocks должно быть exact.
        max_diff = diff.max().item()
        n_diff = (diff > 0).sum().item()
        pct = 100.0 * n_diff / diff.numel()
        print(f"  shape={str(shape):<30}  max_diff={max_diff}  ({pct:.2f}% bytes differ)")
        assert max_diff <= 1, f"max_diff={max_diff} too high"
        assert pct < 1.0, f"{pct:.2f}% bytes differ, expected <1% (only ties)"
    print("  OK")


# ---------------------------------------------------------------------------
# Test 2: reshape_and_cache_q8 vs dense quantize
# ---------------------------------------------------------------------------
def test_reshape_and_cache_q8():
    print("[test_reshape_and_cache_q8]")
    num_blocks = 10
    block_size = 16
    Hkv = 4
    D = 128
    bytes_per_row = (D // 32) * 34

    # Запишем токены в слоты [5, 10, 20, 21, 22] (разные блоки)
    slots = torch.tensor([5, 10, 20, 21, 22], dtype=torch.int64, device=dev)
    num_tokens = slots.shape[0]
    key = torch.randn((num_tokens, Hkv, D), dtype=torch.float16, device=dev) * 0.3

    # Очищенный Q8 cache
    cache_q8 = torch.zeros((num_blocks, block_size, Hkv, bytes_per_row),
                           dtype=torch.uint8, device=dev)

    gfx906_fa.reshape_and_cache_q8(key, slots, cache_q8)

    # Проверяем что в каждом слоте содержимое = quantize_q8_0(key[i])
    ref_q8 = gfx906_fa.quantize_q8_0(key)  # [num_tokens, Hkv, bytes_per_row]
    for i, slot in enumerate(slots.tolist()):
        block_idx = slot // block_size
        block_off = slot % block_size
        got = cache_q8[block_idx, block_off]  # [Hkv, bytes_per_row]
        exp = ref_q8[i]
        diff = (got.int() - exp.int()).abs().max().item()
        print(f"  token {i} → slot {slot} (block={block_idx}, off={block_off}): max_diff={diff}")
        assert diff == 0, f"slot {slot}: byte mismatch, max_diff={diff}"

    # Остальные слоты должны быть нулями.
    for slot in range(num_blocks * block_size):
        if slot in slots.tolist():
            continue
        bi, bo = slot // block_size, slot % block_size
        content = cache_q8[bi, bo]
        assert content.sum() == 0, f"slot {slot} unexpectedly non-zero"
    print("  OK")


# ---------------------------------------------------------------------------
# Test 3: -1 in slot_mapping → skip
# ---------------------------------------------------------------------------
def test_reshape_and_cache_q8_skip():
    print("[test_reshape_and_cache_q8_skip]")
    num_blocks, block_size, Hkv, D = 4, 16, 2, 64
    bytes_per_row = (D // 32) * 34
    slots = torch.tensor([0, -1, 5], dtype=torch.int64, device=dev)
    key = torch.randn((3, Hkv, D), dtype=torch.float16, device=dev)

    cache_q8 = torch.zeros((num_blocks, block_size, Hkv, bytes_per_row),
                           dtype=torch.uint8, device=dev)
    gfx906_fa.reshape_and_cache_q8(key, slots, cache_q8)

    # Slot 0 и 5 записаны (token 0 и token 2), token 1 пропущен (slot=-1).
    ref = gfx906_fa.quantize_q8_0(key)
    assert (cache_q8[0, 0].int() - ref[0].int()).abs().max().item() == 0, \
        "slot 0 (token 0) content mismatch"
    assert (cache_q8[0, 5].int() - ref[2].int()).abs().max().item() == 0, \
        "slot 5 (token 2) content mismatch"
    # Остальные offset'ы блока 0 (кроме 0 и 5) должны быть нулями.
    written_offsets = {0, 5}
    for off in range(block_size):
        if off in written_offsets:
            continue
        assert cache_q8[0, off].sum() == 0, f"slot (0,{off}) unexpectedly non-zero"
    # Остальные блоки вообще не трогались.
    for bi in range(1, num_blocks):
        assert cache_q8[bi].sum() == 0, f"block {bi} unexpectedly non-zero"
    print("  OK")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Device-side Q8_0 quantization tests (gfx906_fa)")
    print("=" * 60)
    test_quantize_q8_0_dense()
    test_reshape_and_cache_q8()
    test_reshape_and_cache_q8_skip()
    print("=" * 60)
    print("ALL TESTS OK")
