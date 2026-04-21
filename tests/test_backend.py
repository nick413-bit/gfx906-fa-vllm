"""
Smoke-тест интеграции gfx906_fa_backend с vLLM v1 AttentionBackend API.

Проверяет:
 1. Backend корректно регистрируется как CUSTOM.
 2. Gfx906FABackend.get_kv_cache_shape() возвращает совместимый с
    TritonAttentionBackend layout.
 3. Gfx906FAImpl.do_kv_cache_update() пишет K/V в paged cache.
 4. Gfx906FAImpl.forward() считает attention и результат близок
    к torch.nn.functional.scaled_dot_product_attention по slot_mapping'у.

Не требует запуска полного vLLM (engine, models и т.д.).
"""

import sys
import os
import torch

# Убедимся что наш модуль доступен
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../extension")

import gfx906_fa_backend  # автоматически регистрирует backend
from gfx906_fa_backend import (
    Gfx906FABackend,
    Gfx906FAImpl,
    Gfx906FAMetadata,
)


def test_registration() -> None:
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    cls = AttentionBackendEnum.CUSTOM.get_class()
    assert cls is Gfx906FABackend, f"CUSTOM registered to {cls}, expected Gfx906FABackend"
    print("[OK] CUSTOM → Gfx906FABackend")


def test_kv_cache_shape_compat() -> None:
    # Должен совпадать с TritonAttentionBackend для совместимости allocator'а
    from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend

    args = (128, 16, 8, 128, "auto")  # num_blocks, block_size, Hkv, D
    ours = Gfx906FABackend.get_kv_cache_shape(*args)
    theirs = TritonAttentionBackend.get_kv_cache_shape(*args)
    assert ours == theirs, f"shape mismatch: ours={ours}, triton={theirs}"
    print(f"[OK] kv_cache_shape == TritonAttentionBackend: {ours}")


def _build_fake_layer(head_size: int, device: torch.device):
    """Минимальный stub AttentionLayer для do_kv_cache_update."""
    class _FakeLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._k_scale = torch.tensor(1.0, device=device)
            self._v_scale = torch.tensor(1.0, device=device)
            self._q_scale_float = 1.0
    return _FakeLayer()


def test_forward_smoke() -> None:
    device = torch.device("cuda:0")
    torch.manual_seed(0)

    # Параметры
    Hq = 8
    Hkv = 4
    D = 128
    block_size = 16
    num_blocks = 8

    # Одна последовательность, Sq=1 (decode), Sk=64
    num_seqs = 1
    Sq = 1
    Sk = 64
    num_tokens = Sq

    # KV cache в vLLM layout: (num_blocks, 2, block_size, Hkv, D)
    kv_cache = torch.zeros(
        (num_blocks, 2, block_size, Hkv, D),
        dtype=torch.float16, device=device,
    )
    # Уже заполненный префикс (Sk-1 = 63 токенов). Новый key/value на текущем шаге —
    # один токен, который будет записан через do_kv_cache_update.
    # Для простоты: заполним cache случайными числами напрямую.
    num_used_blocks = (Sk + block_size - 1) // block_size  # 4
    kv_cache[:num_used_blocks].normal_(0, 0.5)

    # block_table: [B, max_num_blocks], соответствует layout'у vLLM
    block_table = torch.arange(
        num_used_blocks, dtype=torch.int32, device=device
    ).view(1, num_used_blocks)

    seq_lens = torch.tensor([Sk], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, Sq], dtype=torch.int32, device=device)
    # slot_mapping: куда писать новые K/V для каждого входного токена
    # Для decode с Sq=1: последний слот в sequence.
    # Позиция = Sk - 1 (0-indexed). Block = pos // block_size, offset = pos % block_size.
    pos = Sk - 1
    slot = block_table[0, pos // block_size].item() * block_size + (pos % block_size)
    slot_mapping = torch.tensor([slot], dtype=torch.int64, device=device)

    # Создаём Q/K/V на текущем шаге (шаг длиной Sq=1)
    query = torch.randn(num_tokens, Hq, D, dtype=torch.float16, device=device)
    key = torch.randn(num_tokens, Hkv, D, dtype=torch.float16, device=device)
    value = torch.randn(num_tokens, Hkv, D, dtype=torch.float16, device=device)

    # Инстанцируем Impl
    scale = 1.0 / (D ** 0.5)
    impl = Gfx906FAImpl(
        num_heads=Hq, head_size=D, scale=scale, num_kv_heads=Hkv,
        alibi_slopes=None, sliding_window=None, kv_cache_dtype="auto",
    )
    layer = _build_fake_layer(D, device)

    # 1) Записать новые K/V в paged cache
    impl.do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)

    # 2) Forward
    output = torch.empty(num_tokens, Hq * D, dtype=torch.float16, device=device)
    metadata = Gfx906FAMetadata(
        num_actual_tokens=num_tokens,
        max_query_len=Sq,
        max_seq_len=Sk,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        block_table=block_table,
        slot_mapping=slot_mapping,
    )

    result = impl.forward(layer, query, key, value, kv_cache, metadata, output=output)

    # Reference: собрать всё Sk токенов из cache и прогнать SDPA
    # K/V из cache в layout (num_blocks, block_size, Hkv, D) после unbind(1)
    key_cache, value_cache = kv_cache.unbind(1)
    # Gather последние Sk токенов sequence=0
    flat_key = key_cache.reshape(num_blocks * block_size, Hkv, D)
    flat_val = value_cache.reshape(num_blocks * block_size, Hkv, D)
    # positions: block_table[0, i//block_size]*block_size + i%block_size, i∈[0, Sk)
    pos_idx = torch.arange(Sk, device=device)
    block_idx = block_table[0, pos_idx // block_size].long()
    slot_idx = block_idx * block_size + (pos_idx % block_size)
    k_full = flat_key[slot_idx]  # [Sk, Hkv, D]
    v_full = flat_val[slot_idx]

    # SDPA expects [B, Hq, Sq, D]
    q_bhsd = query.permute(1, 0, 2).unsqueeze(0)  # [1, Hq, Sq, D]
    # GQA: повторить K/V
    rep = Hq // Hkv
    k_bhsd = k_full.permute(1, 0, 2).unsqueeze(0).repeat_interleave(rep, dim=1)  # [1, Hq, Sk, D]
    v_bhsd = v_full.permute(1, 0, 2).unsqueeze(0).repeat_interleave(rep, dim=1)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q_bhsd.float(), k_bhsd.float(), v_bhsd.float(), scale=scale,
    )  # [1, Hq, Sq, D]
    ref_flat = ref.squeeze(0).permute(1, 0, 2).reshape(num_tokens, Hq * D)

    out_f = result.float()
    err = (out_f - ref_flat).abs().max().item()
    mse = (out_f - ref_flat).pow(2).mean().item()
    print(
        f"[OK] forward_smoke: max_err={err:.5f}, mse={mse:.6f}"
    )
    assert err < 0.05, f"forward diverged too much from SDPA: max_err={err}"


def main() -> None:
    test_registration()
    test_kv_cache_shape_compat()
    test_forward_smoke()
    print("\nAll backend tests passed.")


if __name__ == "__main__":
    main()
