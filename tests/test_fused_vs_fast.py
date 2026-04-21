"""Sanity: FUSED gather vs FAST (torch fancy index) на batched-decode.

Генерируем batched decode сценарий vLLM-like и проверяем, что oба пути дают
одинаковый output на forward_paged.  Если разница есть → баг в gather_paged_kv_q8.
"""
import os, sys, math
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/extension")
from gfx906_fa_paged import forward_paged

dev = "cuda"
torch.manual_seed(0)

num_seqs, Hq, Hkv, D = 4, 48, 8, 128
seq_lens = torch.tensor([1024, 2048, 512, 256], dtype=torch.int32, device=dev)
max_Sk = int(seq_lens.max().item())
bs = 16
max_blocks = (max_Sk + bs - 1) // bs
num_blocks = num_seqs * max_blocks + 8

key_cache = (torch.randn(num_blocks, bs, Hkv, D, dtype=torch.float16, device=dev) * 0.3)
value_cache = (torch.randn(num_blocks, bs, Hkv, D, dtype=torch.float16, device=dev) * 0.3)

block_table = torch.zeros((num_seqs, max_blocks), dtype=torch.int32, device=dev)
for s in range(num_seqs):
    block_table[s] = torch.arange(s * max_blocks + 1, s * max_blocks + 1 + max_blocks,
                                  dtype=torch.int32, device=dev)

import gfx906_fa
k_cache_q8 = gfx906_fa.quantize_q8_0(key_cache.view(num_blocks * bs * Hkv, D)).view(
    num_blocks, bs, Hkv, (D // 32) * 34
)

query = torch.randn((num_seqs, Hq, D), dtype=torch.float32, device=dev) * 0.5
cu = torch.arange(0, num_seqs + 1, dtype=torch.int32, device=dev)

def run(fused: str):
    os.environ["GFX906_FA_FUSED"] = fused
    # Нужно перезагрузить модуль paged, чтобы он перечитал env на init.
    import importlib, gfx906_fa_paged as _m
    importlib.reload(_m)
    return _m.forward_paged(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        cu_seqlens_q=cu,
        max_seqlen_q=1,
        max_seqlen_k=max_Sk,
        scale=1.0 / math.sqrt(D),
        key_cache_q8=k_cache_q8,
    )

out_fused = run("1")
out_fast  = run("0")

diff = (out_fused.float() - out_fast.float()).abs()
print(f"FUSED vs FAST: max_err={diff.max().item():.6f} mean={diff.mean().item():.6f}")
per_seq = diff.view(num_seqs, Hq, D).amax(dim=(1,2))
for s in range(num_seqs):
    print(f"  seq[{s}] Sk={int(seq_lens[s])} max_err={per_seq[s].item():.6f}")

if diff.max().item() > 0.05:
    print("FAIL: fused diverges from fast-path!")
else:
    print("PASS")
