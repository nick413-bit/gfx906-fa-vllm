"""Bit-exact correctness test: Level 3c direct-paged FA vs gather+FA.

Runs the same deterministic paged-attention decode step through two code
paths and verifies the outputs are numerically identical (or within a
negligible tolerance).

Paths compared:
  A) Gather path (fused, Level 1): GFX906_FA_FUSED=1 GFX906_FA_DIRECT_PAGED=0
     → HIP gather kernel → flash_attn_tile_q8 on contiguous buffers.
  B) Direct-paged (Level 3c):      GFX906_FA_DIRECT_PAGED=1
     → flash_attn_tile_q8_paged directly on paged cache.

Usage:
    python test_direct_paged_bitexact.py --seq-len 4096
    python test_direct_paged_bitexact.py --seq-len 16384 32768 61440

A passing run:
  * max_abs_diff  <= 1e-4
  * max_rel_diff  <= 1e-3 (for values above 1e-3 magnitude)
  * no NaN / Inf on either side
"""
from __future__ import annotations

import argparse
import os
import sys

import torch


def build_paged_decode_inputs(
    seq_len: int,
    *,
    batch: int = 1,
    heads_q: int = 64,
    heads_kv: int = 8,
    head_dim: int = 128,
    block_size: int = 16,
    device: str = "cuda",
    seed: int = 12345,
) -> dict:
    """Build a deterministic paged KV-cache decode step.

    Returns a dict with tensors matching the gfx906_fa_paged.forward_paged signature.
    """
    torch.manual_seed(seed)

    import gfx906_fa

    n_blocks_per_seq = (seq_len + block_size - 1) // block_size
    num_blocks = batch * n_blocks_per_seq + 4

    key_cache_fp16 = torch.randn(
        num_blocks, block_size, heads_kv, head_dim,
        dtype=torch.float16, device=device,
    )
    key_cache_q8 = gfx906_fa.quantize_q8_0(key_cache_fp16).contiguous()

    value_cache = torch.randn(
        num_blocks, block_size, heads_kv, head_dim,
        dtype=torch.float16, device=device,
    )

    # block_table: sequential mapping (simplest case).
    block_table = torch.zeros((batch, n_blocks_per_seq), dtype=torch.int32, device=device)
    for b in range(batch):
        for i in range(n_blocks_per_seq):
            block_table[b, i] = b * n_blocks_per_seq + i

    seq_lens = torch.full((batch,), seq_len, dtype=torch.int32, device=device)

    # Q input: decode step (Sq=1 per sequence).
    num_tokens = batch
    query = torch.randn(num_tokens, heads_q, head_dim, dtype=torch.float32, device=device)

    cu_seqlens_q = torch.arange(0, batch + 1, dtype=torch.int32, device=device)

    return dict(
        query=query,
        key_cache=value_cache,         # unused in fast/direct path but required
        value_cache=value_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        max_seqlen_k=seq_len,
        key_cache_q8=key_cache_q8,
    )


def run_path(inputs: dict, use_direct_paged: bool) -> torch.Tensor:
    """Run forward_paged with the requested path toggled via env."""
    # toggles must be set BEFORE importing the module (module reads env at import).
    os.environ["GFX906_FA_DIRECT_PAGED"] = "1" if use_direct_paged else "0"
    os.environ["GFX906_FA_FUSED"] = "1"  # always use fused gather for the reference path

    # Force fresh module to pick up env.
    for modname in list(sys.modules.keys()):
        if modname.startswith("gfx906_fa_paged"):
            del sys.modules[modname]

    import importlib
    import gfx906_fa_paged
    importlib.reload(gfx906_fa_paged)

    out = gfx906_fa_paged.forward_paged(**inputs)
    torch.cuda.synchronize()
    return out


def compare(out_a: torch.Tensor, out_b: torch.Tensor, *, tag: str) -> dict:
    assert out_a.shape == out_b.shape, f"shape mismatch: {out_a.shape} vs {out_b.shape}"

    diff = (out_a - out_b).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()

    mask = out_a.abs() > 1e-3
    rel = diff[mask] / out_a[mask].abs()
    max_rel = rel.max().item() if rel.numel() > 0 else 0.0

    nan_a = torch.isnan(out_a).any().item()
    nan_b = torch.isnan(out_b).any().item()
    inf_a = torch.isinf(out_a).any().item()
    inf_b = torch.isinf(out_b).any().item()

    # Element-wise bit-exact check.
    bit_equal = bool((out_a.view(torch.int32) == out_b.view(torch.int32)).all().item())

    info = dict(
        tag=tag,
        shape=tuple(out_a.shape),
        max_abs_diff=max_abs,
        mean_abs_diff=mean_abs,
        max_rel_diff=max_rel,
        bit_equal=bit_equal,
        nan=(nan_a, nan_b),
        inf=(inf_a, inf_b),
        norm_a=out_a.norm().item(),
        norm_b=out_b.norm().item(),
    )
    return info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, nargs="+", default=[4096, 16384, 32768, 61440])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads-q", type=int, default=64)
    parser.add_argument("--heads-kv", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--tol-abs", type=float, default=1e-4)
    parser.add_argument("--tol-rel", type=float, default=1e-3)
    args = parser.parse_args()

    all_pass = True
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Running bit-exact test: direct-paged vs gather+FA")
    print("=" * 80)

    for sl in args.seq_len:
        print(f"\n--- seq_len = {sl} ---")
        try:
            inputs = build_paged_decode_inputs(
                sl,
                batch=args.batch,
                heads_q=args.heads_q,
                heads_kv=args.heads_kv,
                head_dim=args.head_dim,
                seed=args.seed,
            )

            # Reference (gather path).
            out_gather = run_path({**inputs}, use_direct_paged=False)

            # Rebuild inputs (tensors may have been held as views).
            inputs2 = build_paged_decode_inputs(
                sl,
                batch=args.batch,
                heads_q=args.heads_q,
                heads_kv=args.heads_kv,
                head_dim=args.head_dim,
                seed=args.seed,
            )
            # Direct-paged.
            out_direct = run_path({**inputs2}, use_direct_paged=True)

            info = compare(out_gather, out_direct, tag=f"sl={sl}")
            print(f"  shape          : {info['shape']}")
            print(f"  norm_a / norm_b: {info['norm_a']:.6f} / {info['norm_b']:.6f}")
            print(f"  max_abs_diff   : {info['max_abs_diff']:.6e}")
            print(f"  mean_abs_diff  : {info['mean_abs_diff']:.6e}")
            print(f"  max_rel_diff   : {info['max_rel_diff']:.6e} "
                  f"(for |a|>1e-3)")
            print(f"  bit_equal      : {info['bit_equal']}")
            print(f"  nan (a, b)     : {info['nan']}")
            print(f"  inf (a, b)     : {info['inf']}")

            passed = (
                info["max_abs_diff"] <= args.tol_abs
                and info["max_rel_diff"] <= args.tol_rel
                and not any(info["nan"])
                and not any(info["inf"])
            )
            status = "PASS" if passed else "FAIL"
            if info["bit_equal"]:
                status += " (bit-exact)"
            print(f"  status         : {status}")
            all_pass = all_pass and passed

        except Exception as e:
            print(f"  ERROR: {e!r}")
            import traceback
            traceback.print_exc()
            all_pass = False

    print("\n" + "=" * 80)
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
