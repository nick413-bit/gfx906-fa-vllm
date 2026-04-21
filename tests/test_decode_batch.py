"""Decode-batch correctness test (имитация vLLM minimax batched decode).

num_seqs=4, Sq=1 each, variable Sk, Hq=48, Hkv=8 — точно как MiniMax-m2.7
в режиме decode при batch=4. Этот тест воспроизводит баг Sq_pad=2 если
он действительно в integration с vLLM.
"""
import math
import sys
import torch

import gfx906_fa
from gfx906_fa_paged import forward_paged

dev = 'cuda'
torch.manual_seed(0)

num_seqs = 4
Hq, Hkv, D = 48, 8, 128
seq_lens = torch.tensor([1024, 2048, 512, 256], dtype=torch.int32, device=dev)
max_Sk = int(seq_lens.max().item())
block_size = 16
max_blocks = (max_Sk + block_size - 1) // block_size
num_blocks = num_seqs * max_blocks + 8

kv_cache = torch.randn((num_blocks, 2, block_size, Hkv, D),
                       dtype=torch.float16, device=dev) * 0.1
key_cache, value_cache = kv_cache.unbind(1)
block_table = torch.zeros((num_seqs, max_blocks), dtype=torch.int32, device=dev)
K_ref = torch.zeros((num_seqs, Hkv, max_Sk, D), dtype=torch.float16, device=dev)
V_ref = torch.zeros((num_seqs, Hkv, max_Sk, D), dtype=torch.float16, device=dev)

next_block = 0
for s in range(num_seqs):
    sk = int(seq_lens[s].item())
    nb = (sk + block_size - 1) // block_size
    for b in range(nb):
        block_table[s, b] = next_block
        toks = min(block_size, sk - b * block_size)
        for t in range(toks):
            K_ref[s, :, b * block_size + t, :] = key_cache[next_block, t]
            V_ref[s, :, b * block_size + t, :] = value_cache[next_block, t]
        next_block += 1

bytes_per_row = (D // 32) * 34
k_cache_q8 = torch.zeros((num_blocks, block_size, Hkv, bytes_per_row),
                         dtype=torch.uint8, device=dev)
for b in range(num_blocks):
    k_cache_q8[b] = gfx906_fa.quantize_q8_0(key_cache[b:b + 1]).squeeze(0)

query = torch.randn((num_seqs, Hq, D), dtype=torch.float32, device=dev) * 0.5
cu = torch.arange(0, num_seqs + 1, dtype=torch.int32, device=dev)

out = forward_paged(
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
out_fp16 = out.view(num_seqs, Hq, D).to(torch.float16)

ref = torch.zeros((num_seqs, Hq, D), dtype=torch.float16, device=dev)
for s in range(num_seqs):
    sk = int(seq_lens[s].item())
    Q = query[s:s + 1].unsqueeze(2)  # [1, Hq, 1, D] fp32
    K = K_ref[s:s + 1, :, :sk, :].repeat_interleave(Hq // Hkv, dim=1).to(torch.float32)
    V = V_ref[s:s + 1, :, :sk, :].repeat_interleave(Hq // Hkv, dim=1).to(torch.float32)
    r = torch.nn.functional.scaled_dot_product_attention(
        Q, K, V, is_causal=False).squeeze(2).to(torch.float16)
    ref[s] = r[0]

diff = (out_fp16 - ref).abs()
max_err = diff.max().item()
mse = ((out_fp16.float() - ref.float()) ** 2).mean().item()
print(f'batch-decode B={num_seqs} Sk={seq_lens.tolist()}')
print(f'  max_err={max_err:.5f}  mse={mse:.6f}')
per_seq_err = diff.view(num_seqs, -1).max(dim=1).values
for s, e in enumerate(per_seq_err.tolist()):
    print(f'  seq[{s}] max_err={e:.5f}')
print('PASS' if max_err < 0.05 else 'FAIL')
sys.exit(0 if max_err < 0.05 else 1)
