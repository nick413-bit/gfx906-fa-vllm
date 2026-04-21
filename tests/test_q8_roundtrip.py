"""
Изолированный reproducer: Q8 side-buffer write→read roundtrip.

Имитирует то, что делает vLLM backend:
 1) reshape_and_cache_q8(new_key[num_tokens, Hkv, D], slot_mapping, k_cache_q8)
 2) gather_paged_kv_q8(k_cache_q8, value_cache_dummy, block_table, seq_lens, Sk) → K_q8
 3) dequantize(K_q8)                                        → K_recon
 4) reference: quantize_q8_0(key_gathered_by_slot) → reshape → dequantize → K_ref

Если shape/layout/strides правильны, K_recon == K_ref (bit-exact).
"""

import os
import sys
import torch

sys.path.insert(0, "/gfx906_fa")
import gfx906_fa  # type: ignore


def dequantize_q8_0(q8: torch.Tensor, D: int) -> torch.Tensor:
    """
    q8: uint8 [..., (D/32)*34]
    returns fp32 [..., D]
    """
    shape = q8.shape
    prefix = shape[:-1]
    blocks_per_row = D // 32
    q8 = q8.reshape(-1, blocks_per_row, 34)
    N = q8.shape[0]
    scale_bytes = q8[:, :, :2].contiguous().view(torch.float16).float()  # [N, B, 1]
    scale = scale_bytes[..., 0:1]  # [N, B, 1]
    q = q8[:, :, 2:].to(torch.int8).float()  # [N, B, 32]
    out = q * scale  # [N, B, 32]
    out = out.reshape(N, D)
    return out.reshape(*prefix, D)


def main() -> None:
    torch.manual_seed(0)
    dev = torch.device("cuda:0")

    # vLLM-like параметры (minimax: Hq=40, Hkv=8, D=128, block_size=16)
    Hq = 40
    Hkv = 8
    D = 128
    block_size = 16
    num_blocks = 64

    # Один sequence, записываем 128 токенов префилла
    num_tokens = 128
    num_seqs = 1
    Sk = num_tokens

    # Key — fp16, vLLM-layout [num_tokens, Hkv, D]
    key = torch.randn(num_tokens, Hkv, D, dtype=torch.float16, device=dev) * 0.5

    # k_cache_q8 — [num_blocks, block_size, Hkv, (D/32)*34]
    bytes_per_row = (D // 32) * 34
    k_cache_q8 = torch.zeros(
        (num_blocks, block_size, Hkv, bytes_per_row),
        dtype=torch.uint8, device=dev,
    )

    # block_table: последовательно блоки 0..7 для sequence 0 (128 toks / 16 = 8 блоков)
    num_used_blocks = (Sk + block_size - 1) // block_size
    block_table = torch.arange(num_used_blocks, dtype=torch.int32, device=dev).view(1, num_used_blocks)

    # slot_mapping: pos i → block_table[0, i//bs]*bs + (i%bs)
    pos = torch.arange(num_tokens, device=dev)
    blk = block_table[0, pos // block_size].long()
    slot_mapping = blk * block_size + (pos % block_size)
    slot_mapping = slot_mapping.to(torch.int64)

    seq_lens = torch.tensor([Sk], dtype=torch.int32, device=dev)

    # --- 1) Запись через reshape_and_cache_q8 ---
    gfx906_fa.reshape_and_cache_q8(key, slot_mapping, k_cache_q8)

    # --- 2) Чтение через gather_paged_kv_q8 ---
    # value_cache нужен для API, но нас интересует K
    value_cache = torch.zeros(
        (num_blocks, block_size, Hkv, D),
        dtype=torch.float16, device=dev,
    )
    # Sk_pad — cr gather хочет кратно 32 (судя по bench_prefill)
    Sk_pad = ((Sk + 31) // 32) * 32

    K_q8, V_fp16 = gfx906_fa.gather_paged_kv_q8(
        k_cache_q8, value_cache, block_table, seq_lens, Sk_pad,
    )
    # K_q8: [num_seqs, Hkv, Sk_pad, bytes_per_row]
    print(f"K_q8 shape: {tuple(K_q8.shape)}, dtype={K_q8.dtype}")

    # --- 3) Обратно dequantize → [num_seqs, Hkv, Sk_pad, D] ---
    K_recon = dequantize_q8_0(K_q8[..., : bytes_per_row], D)  # fp32
    print(f"K_recon shape: {tuple(K_recon.shape)}")

    # Berем только первые Sk токенов
    K_recon = K_recon[:, :, :Sk, :]  # [1, Hkv, Sk, D]
    # Приводим к [Sk, Hkv, D]
    K_recon_thd = K_recon[0].permute(1, 0, 2).contiguous()

    # --- 4) Reference path: quantize_q8_0(key) → dequantize ---
    # quantize_q8_0 ждёт [N, D]. Приводим ключ к [num_tokens*Hkv, D]
    key_flat = key.reshape(num_tokens * Hkv, D).contiguous()
    K_q8_ref = gfx906_fa.quantize_q8_0(key_flat)  # [N, (D/32)*34] uint8
    K_ref_flat = dequantize_q8_0(K_q8_ref, D)  # [N, D] fp32
    K_ref = K_ref_flat.reshape(num_tokens, Hkv, D)

    # --- compare ---
    diff = (K_recon_thd - K_ref).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    print(f"\n=== Q8 roundtrip vs dense-quantize ===")
    print(f"max_err  = {max_err:.6e}")
    print(f"mean_err = {mean_err:.6e}")

    # Также сравним с FP16 исходником (dequantize ≈ key)
    diff_fp = (K_recon_thd - key.float()).abs()
    print(f"\n=== Dequantized vs fp16 source ===")
    print(f"max_err  = {diff_fp.max().item():.6e}")
    print(f"mean_err = {diff_fp.mean().item():.6e}")

    status = "PASS" if max_err < 1e-6 else "FAIL (roundtrip != dense-quantize)"
    print(f"\nStatus: {status}")

    if max_err >= 1e-6:
        # Покажем где различие
        print("\n-- Sample diff on token 0, head 0, dim 0..16 --")
        print("recon :", K_recon_thd[0, 0, :16].cpu().numpy())
        print("ref   :", K_ref[0, 0, :16].cpu().numpy())


if __name__ == "__main__":
    main()
