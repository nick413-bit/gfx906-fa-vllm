"""
Воспроизводит test_backend.py сценарий, но СРАВНИВАЕТ:
 Path A (LEGACY=1): forward квантует K_cache (fp16) на лету → attention
 Path B (LEGACY=0): forward читает K из Q8 side-buffer → attention

Ожидаемо: обе погрешности vs SDPA должны быть ~1e-3 (Q8 roundtrip).
Если Path B даёт 0.5 — значит _k_cache_q8 неконсистентен с key_cache (fp16).
"""

import os, sys, torch

sys.path.insert(0, "/gfx906_fa")
import gfx906_fa  # type: ignore
import gfx906_fa_backend as bk  # type: ignore


def build_case(Hq=8, Hkv=4, D=128, block_size=16, num_blocks=8, Sk=64, num_seqs=1, Sq=1, seed=0):
    torch.manual_seed(seed)
    dev = torch.device("cuda:0")
    num_tokens = Sq * num_seqs

    # fp16 K/V cache заполнены случайными данными в первых Sk слотах
    kv_cache = torch.zeros((num_blocks, 2, block_size, Hkv, D), dtype=torch.float16, device=dev)
    num_used_blocks = (Sk + block_size - 1) // block_size
    kv_cache[:num_used_blocks].normal_(0, 0.5)

    block_table = torch.arange(num_used_blocks, dtype=torch.int32, device=dev).view(1, num_used_blocks)

    seq_lens = torch.tensor([Sk], dtype=torch.int32, device=dev)
    query_start_loc = torch.tensor([0, Sq], dtype=torch.int32, device=dev)
    pos = Sk - 1
    slot = block_table[0, pos // block_size].item() * block_size + (pos % block_size)
    slot_mapping = torch.tensor([slot], dtype=torch.int64, device=dev)

    query = torch.randn(num_tokens, Hq, D, dtype=torch.float16, device=dev)
    key = torch.randn(num_tokens, Hkv, D, dtype=torch.float16, device=dev)
    value = torch.randn(num_tokens, Hkv, D, dtype=torch.float16, device=dev)

    class _FakeLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._k_scale = torch.tensor(1.0, device=dev)
            self._v_scale = torch.tensor(1.0, device=dev)
            self._q_scale_float = 1.0
    layer = _FakeLayer()

    metadata = bk.Gfx906FAMetadata(
        num_actual_tokens=num_tokens,
        max_query_len=Sq,
        max_seq_len=Sk,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        block_table=block_table,
        slot_mapping=slot_mapping,
    )
    return kv_cache, layer, query, key, value, metadata, dev


def run_path(legacy: int, kv_cache, layer, query, key, value, metadata):
    os.environ["GFX906_FA_LEGACY"] = str(legacy)
    scale = 1.0 / (query.shape[-1] ** 0.5)
    impl = bk.Gfx906FAImpl(
        num_heads=query.shape[1], head_size=query.shape[2], scale=scale,
        num_kv_heads=key.shape[1], alibi_slopes=None, sliding_window=None,
        kv_cache_dtype="auto",
    )
    kv = kv_cache.clone()
    impl.do_kv_cache_update(layer, key, value, kv, metadata.slot_mapping)

    # После do_kv_cache_update в legacy=0 у impl есть _k_cache_q8.
    # ВАЖНО: до первого do_kv_cache_update _k_cache_q8 не существует, и он только один слот из Sk=64 был записан (slot=63).
    # Остальные 63 слота остались нулями. Это **и есть** reproducer вопроса!

    out = torch.empty(query.shape[0], query.shape[1] * query.shape[2], dtype=torch.float16, device=query.device)
    return impl.forward(layer, query, key, value, kv, metadata, output=out), kv, impl


def sdpa_ref(kv_cache_after, query, metadata):
    key_cache, value_cache = kv_cache_after.unbind(1)
    num_blocks, block_size, Hkv, D = key_cache.shape
    Sk = metadata.max_seq_len
    block_table = metadata.block_table

    flat_key = key_cache.reshape(num_blocks * block_size, Hkv, D)
    flat_val = value_cache.reshape(num_blocks * block_size, Hkv, D)
    pos_idx = torch.arange(Sk, device=query.device)
    block_idx = block_table[0, pos_idx // block_size].long()
    slot_idx = block_idx * block_size + (pos_idx % block_size)
    k_full = flat_key[slot_idx]
    v_full = flat_val[slot_idx]

    Hq = query.shape[1]
    q_bhsd = query.permute(1, 0, 2).unsqueeze(0)
    rep = Hq // Hkv
    k_bhsd = k_full.permute(1, 0, 2).unsqueeze(0).repeat_interleave(rep, dim=1)
    v_bhsd = v_full.permute(1, 0, 2).unsqueeze(0).repeat_interleave(rep, dim=1)
    scale = 1.0 / (query.shape[-1] ** 0.5)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q_bhsd.float(), k_bhsd.float(), v_bhsd.float(), scale=scale,
    )
    return ref.squeeze(0).permute(1, 0, 2).reshape(query.shape[0], Hq * D).float()


def main():
    kv_cache, layer, query, key, value, metadata, dev = build_case()

    # Path A: legacy
    outA, kvA, implA = run_path(1, kv_cache, layer, query, key, value, metadata)
    refA = sdpa_ref(kvA, query, metadata)
    errA = (outA.float() - refA).abs().max().item()

    # Path B: Q8 side-buffer
    outB, kvB, implB = run_path(0, kv_cache, layer, query, key, value, metadata)
    refB = sdpa_ref(kvB, query, metadata)
    errB = (outB.float() - refB).abs().max().item()

    print(f"LEGACY=1 (on-the-fly Q8):  max_err vs SDPA = {errA:.5f}")
    print(f"LEGACY=0 (Q8 side-buffer): max_err vs SDPA = {errB:.5f}")

    # Критическая проверка: какие значения в _k_cache_q8?
    if implB._k_cache_q8 is not None:
        q8 = implB._k_cache_q8
        nz = (q8 != 0).any(dim=-1)  # [num_blocks, block_size, Hkv]
        nz_slots = nz.any(dim=-1).reshape(-1)  # [num_blocks * block_size]
        nz_count = nz_slots.sum().item()
        print(f"_k_cache_q8: total slots={nz_slots.numel()}, non-zero slots={nz_count}")
        # Должно быть ровно 1 (slot=63) — т.к. мы только его писали через do_kv_cache_update
        if nz_count == 1:
            print("  → Q8 side-buffer has ONLY slot 63. fp16 K_cache has 64 slots (slots 0..63 — random.normal, slot 63 overwritten by new key).")
            print("  → Forward reads Q8=0 for slots 0..62, but SDPA ref reads fp16=random for those. MISMATCH")


if __name__ == "__main__":
    main()
