"""Microbenchmark: gfx906_fa vs torch.nn.functional.scaled_dot_product_attention.

Что меряем:
  - Decode attention (Sq=1, разные Skv): 128, 512, 1k, 4k, 8k, 16k, 32k, 64k, 94k, 128k
  - Prefill attention (Sq=512, Skv=Sq): 512, 1k, 2k, 4k, 8k
  - GQA ratios для MiniMax: 48Q/8KV

Методика:
  - batch=1 (single request, как в decode в vLLM)
  - D=128 (MiniMax)
  - Warm-up: 10 итераций (не меряем)
  - Measure: N=50 итераций, report median + p5/p95
  - GPU-sync между итерациями
  - Исключаем CPU-Q8 quantization (делаем 1 раз pre-test)

Запуск:
  cd /work/src && PYTHONPATH=/work/src/extension python3 -u tests/bench.py
"""
import math
import statistics
import sys
import time
import torch

try:
    import gfx906_fa
except ImportError as e:
    print(f"FAIL: {e}", file=sys.stderr)
    sys.exit(1)

torch.manual_seed(0)
dev = 'cuda'
D = 128

# ------------------------------------------------------------------
# Bench matrix
# ------------------------------------------------------------------
# Каждый кортеж: (name, Hq, Hkv, Sq, Skv)
BENCH_DECODE = [
    ("dec_128",   8, 4, 1,    128),
    ("dec_512",   8, 4, 1,    512),
    ("dec_1k",    8, 4, 1,   1024),
    ("dec_4k",    8, 4, 1,   4096),
    ("dec_8k",    8, 4, 1,   8192),
    ("dec_16k",   8, 4, 1,  16384),
    ("dec_32k",   8, 4, 1,  32768),
    ("dec_64k",   8, 4, 1,  65536),
    ("dec_94k",   8, 4, 1,  94208),  # актуальный target контекст MiniMax
    ("dec_128k",  8, 4, 1, 131072),
]

BENCH_PREFILL = [
    ("pre_512",   8, 4,   512,   512),
    ("pre_1k",    8, 4,  1024,  1024),
    ("pre_2k",    8, 4,  2048,  2048),
    ("pre_4k",    8, 4,  4096,  4096),
    ("pre_8k",    8, 4,  8192,  8192),
]

BENCH_MINIMAX_GQA = [
    # MiniMax M2.7: 48 Q heads, 8 KV heads, D=128
    # (Hq, Hkv реально как в модели; но батч=1 и fp16 numerics)
    ("mmx_dec_1k",   48, 8, 1,   1024),
    ("mmx_dec_8k",   48, 8, 1,   8192),
    ("mmx_dec_32k",  48, 8, 1,  32768),
    ("mmx_dec_94k",  48, 8, 1,  94208),
]


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------
def bench_one(name, Hq, Hkv, Sq, Skv, n_warmup=10, n_measure=50):
    B = 1
    assert Hq % Hkv == 0
    scale = 1.0 / math.sqrt(D)

    # Data
    q_fp32 = torch.randn(B, Hq,  Sq,  D, dtype=torch.float32, device=dev)
    k_fp16 = torch.randn(B, Hkv, Skv, D, dtype=torch.float16, device=dev) * 0.1
    v_fp16 = torch.randn(B, Hkv, Skv, D, dtype=torch.float16, device=dev) * 0.1

    # Pre-quantize K (once)
    k_q8 = gfx906_fa.quantize_q8_0(k_fp16)

    # For SDPA reference
    q_fp16 = q_fp32.to(torch.float16)
    gqa = Hq // Hkv
    k_rep = k_fp16.repeat_interleave(gqa, dim=1) if gqa > 1 else k_fp16
    v_rep = v_fp16.repeat_interleave(gqa, dim=1) if gqa > 1 else v_fp16

    # ---------- gfx906_fa ----------
    # warmup
    for _ in range(n_warmup):
        _ = gfx906_fa.forward(q_fp32, k_q8, v_fp16, scale)
    torch.cuda.synchronize()

    fa_times = []
    for _ in range(n_measure):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = gfx906_fa.forward(q_fp32, k_q8, v_fp16, scale)
        torch.cuda.synchronize()
        fa_times.append((time.perf_counter() - t0) * 1e6)  # usec

    # ---------- SDPA ----------
    for _ in range(n_warmup):
        _ = torch.nn.functional.scaled_dot_product_attention(q_fp16, k_rep, v_rep, scale=scale)
    torch.cuda.synchronize()

    sdpa_times = []
    for _ in range(n_measure):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = torch.nn.functional.scaled_dot_product_attention(q_fp16, k_rep, v_rep, scale=scale)
        torch.cuda.synchronize()
        sdpa_times.append((time.perf_counter() - t0) * 1e6)  # usec

    def stats(xs):
        xs_s = sorted(xs)
        return {
            'median': statistics.median(xs),
            'p5':     xs_s[int(0.05 * len(xs_s))],
            'p95':    xs_s[min(int(0.95 * len(xs_s)), len(xs_s)-1)],
            'mean':   statistics.mean(xs),
        }

    fa = stats(fa_times)
    sd = stats(sdpa_times)
    speedup = sd['median'] / fa['median'] if fa['median'] > 0 else 0.0

    return dict(name=name, Hq=Hq, Hkv=Hkv, Sq=Sq, Skv=Skv,
                fa_p50=fa['median'], fa_p5=fa['p5'], fa_p95=fa['p95'],
                sd_p50=sd['median'], sd_p5=sd['p5'], sd_p95=sd['p95'],
                speedup=speedup)


def run_group(title, matrix):
    print(f"\n{'='*110}")
    print(f"  {title}")
    print(f"{'='*110}")
    print(f"{'name':<14} {'Hq':>3} {'Hkv':>3} {'Sq':>5} {'Skv':>7} "
          f"{'fa_p50(us)':>11} {'fa_p5..p95':>14} "
          f"{'sdpa_p50(us)':>13} {'sdpa_p5..p95':>16} "
          f"{'speedup':>9}")
    print("-" * 110)
    for cfg in matrix:
        try:
            r = bench_one(*cfg)
        except Exception as e:
            print(f"{cfg[0]:<14} FAIL: {e}")
            continue
        print(f"{r['name']:<14} {r['Hq']:>3} {r['Hkv']:>3} {r['Sq']:>5} {r['Skv']:>7} "
              f"{r['fa_p50']:>11.1f} [{r['fa_p5']:>5.1f}..{r['fa_p95']:<5.1f}] "
              f"{r['sd_p50']:>13.1f} [{r['sd_p5']:>6.1f}..{r['sd_p95']:<6.1f}] "
              f"{r['speedup']:>8.2f}x")


run_group("DECODE (Sq=1, Hq=8/Hkv=4)", BENCH_DECODE)
run_group("PREFILL (Sq=Skv, Hq=8/Hkv=4)", BENCH_PREFILL)
run_group("MINIMAX GQA (Hq=48/Hkv=8)", BENCH_MINIMAX_GQA)

print(f"\n{'='*110}")
print("  Summary")
print(f"{'='*110}")
print("  Kernel = gfx906 FA Q8_0 (K-only quantized, V fp16)")
print("  Reference = torch.nn.functional.scaled_dot_product_attention (fp16)")
print("  batch = 1, D = 128, measurement = median of 50 iters, warmup = 10")
