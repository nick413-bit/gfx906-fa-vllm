"""Bit-exact correctness test для gather v1 vs v2.

Проверяет что оптимизация gather_paged_kv_q8_kernel_v2 (flat-iteration)
не изменила математический результат decode относительно v1.

Поскольку в launcher env cached через static — нужны 2 отдельных процесса.

Usage:
    # save mode: создаёт детерминированные inputs, запускает forward_paged,
    # сохраняет output тензор на диск.
    GFX906_FA_GATHER_V=1 python test_gather_bitexact.py save --sk=4096 --out=/tmp/v1_4k.pt
    GFX906_FA_GATHER_V=2 python test_gather_bitexact.py save --sk=4096 --out=/tmp/v2_4k.pt

    # compare mode: сравнивает два .pt файла.
    python test_gather_bitexact.py compare /tmp/v1_4k.pt /tmp/v2_4k.pt
"""
import argparse
import math
import os
import sys
import torch

sys.path.insert(0, '/work/src/extension')
import gfx906_fa  # type: ignore
from gfx906_fa_paged import forward_paged  # type: ignore


def make_inputs(SK, BATCH=4, Hq=40, Hkv=8, D=128, block_size=16, seed=42):
    torch.manual_seed(seed)
    device = "cuda"
    n_blocks_per = (SK + block_size - 1) // block_size
    total_blocks = BATCH * n_blocks_per + 4
    K_fp16 = torch.randn((total_blocks, block_size, Hkv, D), dtype=torch.float16, device=device)
    V_fp16 = torch.randn((total_blocks, block_size, Hkv, D), dtype=torch.float16, device=device)
    bpr = (D // 32) * 34
    K_q8 = torch.empty((total_blocks, block_size, Hkv, bpr), dtype=torch.uint8, device=device)
    for b in range(total_blocks):
        for t in range(block_size):
            for h in range(Hkv):
                q8 = gfx906_fa.quantize_q8_0(K_fp16[b, t, h].unsqueeze(0))
                K_q8[b, t, h] = q8[0]
    block_table = torch.full((BATCH, n_blocks_per), -1, dtype=torch.int32, device=device)
    cursor = 0
    for s in range(BATCH):
        for j in range(n_blocks_per):
            block_table[s, j] = cursor
            cursor += 1
    seq_lens = torch.full((BATCH,), SK, dtype=torch.int32, device=device)
    query = torch.randn((BATCH, Hq, D), dtype=torch.float32, device=device)
    cu = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device=device)
    scale = 1.0 / math.sqrt(D)
    return dict(query=query, key_cache=K_fp16, value_cache=V_fp16,
                block_table=block_table, seq_lens=seq_lens,
                cu_seqlens_q=cu, max_seqlen_q=1, max_seqlen_k=SK,
                scale=scale, key_cache_q8=K_q8)


def cmd_save(args):
    inputs = make_inputs(SK=args.sk, seed=args.seed)
    with torch.no_grad():
        for _ in range(2):
            out = forward_paged(**inputs)
        torch.cuda.synchronize()
    env_v = os.environ.get('GFX906_FA_GATHER_V', '(unset)')
    print(f"[save] env GFX906_FA_GATHER_V={env_v} SK={args.sk} "
          f"out.shape={tuple(out.shape)} out.dtype={out.dtype} "
          f"mean={out.float().mean().item():.6f} "
          f"std={out.float().std().item():.6f}")
    torch.save(out.cpu(), args.out)
    print(f"[save] wrote {args.out}")


def cmd_compare(args):
    a = torch.load(args.path_a, map_location='cpu')
    b = torch.load(args.path_b, map_location='cpu')
    if a.shape != b.shape:
        print(f"[compare] FAIL: shapes differ {a.shape} vs {b.shape}")
        sys.exit(1)
    af, bf = a.float(), b.float()
    diff = (af - bf).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    eps = 1e-3
    n_over_eps = (diff > eps).sum().item()
    total = diff.numel()
    denom = bf.abs().clamp_min(1e-6)
    max_rel = (diff / denom).max().item()
    a_eq_b = torch.equal(a, b)
    print(f"[compare] {args.path_a} vs {args.path_b}")
    print(f"[compare]   shape       = {tuple(a.shape)}")
    print(f"[compare]   bit-equal   = {a_eq_b}")
    print(f"[compare]   max_abs_diff= {max_abs:.3e}")
    print(f"[compare]   mean_abs_diff= {mean_abs:.3e}")
    print(f"[compare]   max_rel_diff= {max_rel:.3e}")
    print(f"[compare]   elems>{eps} = {n_over_eps}/{total} ({100*n_over_eps/total:.4f}%)")
    if a_eq_b:
        print("[compare] VERDICT: BIT-EXACT IDENTICAL (perfect)")
        return
    if max_abs < 1e-4:
        print("[compare] VERDICT: OK (within fp16 ULP)")
    elif max_abs < 1e-3:
        print("[compare] VERDICT: ACCEPTABLE (within fp16 epsilon)")
    elif max_abs < 1e-2:
        print("[compare] VERDICT: SUSPICIOUS (non-trivial drift)")
        sys.exit(2)
    else:
        print("[compare] VERDICT: FAIL (significant numerical drift)")
        sys.exit(3)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    ps = sub.add_parser('save')
    ps.add_argument('--sk', type=int, default=4096)
    ps.add_argument('--seed', type=int, default=42)
    ps.add_argument('--out', required=True)
    pc = sub.add_parser('compare')
    pc.add_argument('path_a')
    pc.add_argument('path_b')
    args = p.parse_args()
    if args.cmd == 'save':
        cmd_save(args)
    else:
        cmd_compare(args)


if __name__ == "__main__":
    main()
