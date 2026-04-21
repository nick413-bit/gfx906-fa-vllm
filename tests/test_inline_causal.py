"""Level 3a validation: inline causal vs. materialised causal mask.

Сценарий prefill chunk:
    Batch = 2, Hq = 8, Hkv = 2 (GQA=4), D = 128.
    Sequence 0: seq_len = 512, query-chunk длиной n_q = 256 (prefill chunk).
    Sequence 1: seq_len = 384, query-chunk длиной n_q = 128.
    Ожидаем, что kernel с q_abs_offset даёт такой же вывод что и SDPA reference.

Запуск на сервере (в Docker vllm_fa_test):
    docker exec vllm_fa_test python3 /gfx906_fa/tests/test_inline_causal.py
"""

import math
import os
import sys
import torch
import torch.nn.functional as F

import gfx906_fa  # type: ignore


def build_paged_kv(seq_lens: list[int], num_kv_heads: int, D: int,
                   block_size: int = 16, device: str = "cuda"):
    """Собрать vLLM-совместимый paged KV cache + block_table + Q8 side-buffer."""
    n_blocks_per = [(sk + block_size - 1) // block_size for sk in seq_lens]
    max_blocks = max(n_blocks_per)
    total_blocks = sum(n_blocks_per) + 2

    K_fp16 = torch.randn(
        (total_blocks, block_size, num_kv_heads, D),
        dtype=torch.float16, device=device,
    )
    V_fp16 = torch.randn(
        (total_blocks, block_size, num_kv_heads, D),
        dtype=torch.float16, device=device,
    )

    bpr = (D // 32) * 34
    K_q8 = torch.empty(
        (total_blocks, block_size, num_kv_heads, bpr),
        dtype=torch.uint8, device=device,
    )
    for b in range(total_blocks):
        # [block_size, Hkv, D] fp16 → uint8 bytes
        for t in range(block_size):
            for h in range(num_kv_heads):
                q8 = gfx906_fa.quantize_q8_0(K_fp16[b, t, h].unsqueeze(0))
                K_q8[b, t, h] = q8[0]

    block_table = torch.full(
        (len(seq_lens), max_blocks), -1, dtype=torch.int32, device=device
    )
    block_cursor = 0
    for s, sk in enumerate(seq_lens):
        nb = n_blocks_per[s]
        for j in range(nb):
            block_table[s, j] = block_cursor
            block_cursor += 1

    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    return K_fp16, V_fp16, K_q8, block_table, seq_lens_t


def gather_seq_kv(K_fp16: torch.Tensor, V_fp16: torch.Tensor,
                  block_table: torch.Tensor, seq_lens: torch.Tensor,
                  s: int, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Для sequence s вернуть [Sk, Hkv, D] fp16 K/V без padding."""
    sk = int(seq_lens[s].item())
    nb = (sk + block_size - 1) // block_size
    blks = block_table[s, :nb].long()
    Ks = K_fp16[blks].reshape(nb * block_size, *K_fp16.shape[2:])[:sk]
    Vs = V_fp16[blks].reshape(nb * block_size, *V_fp16.shape[2:])[:sk]
    return Ks, Vs


def sdpa_ref(query_s: torch.Tensor, K_s: torch.Tensor, V_s: torch.Tensor,
             Hq: int, Hkv: int, scale: float, n_q: int) -> torch.Tensor:
    """Reference attention для одного sequence.

    query_s: [n_q, Hq, D] fp32
    K_s/V_s: [sk, Hkv, D] fp16
    """
    D = query_s.shape[-1]
    sk = K_s.shape[0]
    gqa = Hq // Hkv
    # broadcast Hkv → Hq
    K_s = K_s.repeat_interleave(gqa, dim=1)  # [sk, Hq, D]
    V_s = V_s.repeat_interleave(gqa, dim=1)
    # Reshape → [Hq, n_q/sk, D]
    q = query_s.permute(1, 0, 2).contiguous().float()  # [Hq, n_q, D]
    k = K_s.permute(1, 0, 2).contiguous().float()  # [Hq, sk, D]
    v = V_s.permute(1, 0, 2).contiguous().float()

    scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # [Hq, n_q, sk]
    # Causal: абс. q position = (sk - n_q) + j; k position = i; allowed if i <= q_abs
    q_abs = torch.arange(sk - n_q, sk, device=q.device).unsqueeze(-1)  # [n_q, 1]
    k_pos = torch.arange(sk, device=q.device).unsqueeze(0)  # [1, sk]
    mask = k_pos <= q_abs  # [n_q, sk]
    scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v)  # [Hq, n_q, D]
    return out.permute(1, 0, 2).contiguous()  # [n_q, Hq, D]


def main():
    torch.manual_seed(0)
    device = "cuda"
    block_size = 16

    Hq, Hkv, D = 8, 2, 128
    seq_lens = [512, 384]
    n_q_list = [256, 128]   # prefill chunk sizes

    K_fp16, V_fp16, K_q8, block_table, seq_lens_t = build_paged_kv(
        seq_lens, Hkv, D, block_size=block_size, device=device,
    )

    # query [total_q, Hq, D] + cu_seqlens
    total_q = sum(n_q_list)
    query = torch.randn((total_q, Hq, D), dtype=torch.float32, device=device)
    cu = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(n_q_list), dim=0).tolist()),
        dtype=torch.int32, device=device,
    )

    max_seqlen_q = max(n_q_list)
    max_seqlen_k = max(seq_lens)
    scale = 1.0 / math.sqrt(D)

    # --- 1) Our path: inline causal ---
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, "/gfx906_fa")
    from gfx906_fa_paged import forward_paged  # noqa: E402

    out_flat = forward_paged(
        query=query,
        key_cache=K_fp16,
        value_cache=V_fp16,
        block_table=block_table,
        seq_lens=seq_lens_t,
        cu_seqlens_q=cu,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        scale=scale,
        key_cache_q8=K_q8,
    )  # [total_q, Hq*D] fp32

    our = out_flat.view(total_q, Hq, D)

    # --- 2) Reference per-sequence ---
    max_err = 0.0
    for s, (n_q, sk) in enumerate(zip(n_q_list, seq_lens)):
        q_s = query[cu[s]:cu[s] + n_q]
        K_s, V_s = gather_seq_kv(K_fp16, V_fp16, block_table, seq_lens_t, s, block_size)
        ref = sdpa_ref(q_s, K_s, V_s, Hq, Hkv, scale, n_q)
        diff = (our[cu[s]:cu[s] + n_q].float() - ref.float()).abs()
        err = diff.max().item()
        mean_err = diff.mean().item()
        print(f"seq{s}: n_q={n_q} sk={sk}  max_err={err:.4e} mean={mean_err:.4e}")
        max_err = max(max_err, err)

    TOL = 2e-2
    ok = max_err < TOL
    print(f"\noverall max_err={max_err:.4e} → {'PASS' if ok else 'FAIL'} (tol={TOL})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
