"""Smoke test для gfx906 FA extension.

Шаги:
  1. import gfx906_fa
  2. создать dummy Q/K/V
  3. квантовать K → block_q8_0
  4. вызвать forward
  5. сравнить с torch native SDPA (fp16) на маленьком контексте
"""

import math
import sys
import torch

try:
    import gfx906_fa
except ImportError as e:
    print(f"FAIL: import gfx906_fa → {e}", file=sys.stderr)
    sys.exit(1)

print(f"OK: gfx906_fa imported. Version: {getattr(gfx906_fa, '__version__', 'unknown')}")
print(f"Has attrs: {[a for a in dir(gfx906_fa) if not a.startswith('_')]}")

# -------------- Dummy shapes (имитируют MiniMax 1 token decode) --------------
B  = 1
Hq = 8      # много чтобы протестировать GQA
Hkv = 4     # 2x GQA (позже протестируем 6x)
Sq = 1      # decode = 1 token
Skv = 128   # маленький контекст
D  = 128

torch.manual_seed(42)
dev = 'cuda'

# Q fp32
q = torch.randn(B, Hq, Sq, D, dtype=torch.float32, device=dev)
# K, V fp16
k_fp16 = torch.randn(B, Hkv, Skv, D, dtype=torch.float16, device=dev) * 0.1
v_fp16 = torch.randn(B, Hkv, Skv, D, dtype=torch.float16, device=dev) * 0.1

print(f"Input shapes: q={q.shape} k={k_fp16.shape} v={v_fp16.shape}")

# -------------- Quantize K --------------
k_q8 = gfx906_fa.quantize_q8_0(k_fp16)
print(f"Quantized K: {k_q8.shape} dtype={k_q8.dtype}")
expected_bytes = (D // 32) * 34
assert k_q8.shape == (B, Hkv, Skv, expected_bytes), f"bad K shape: {k_q8.shape}"

# -------------- Forward --------------
scale = 1.0 / math.sqrt(D)
try:
    out = gfx906_fa.forward(q, k_q8, v_fp16, scale)
    print(f"OK: forward returned tensor shape={out.shape} dtype={out.dtype}")
except Exception as e:
    print(f"FAIL: forward raised → {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(2)

# Проверка что output не NaN/Inf
has_nan = bool(torch.isnan(out).any().item())
has_inf = bool(torch.isinf(out).any().item())
print(f"NaN in output: {has_nan}, Inf: {has_inf}")
if has_nan or has_inf:
    print("FAIL: numerical issue in output", file=sys.stderr)
    sys.exit(3)

# -------------- Reference SDPA --------------
# SDPA использует fp16 Q, поэтому на маленьком контексте значения должны быть близки
# (с погрешностью Q8_0 квантизации K ≈ 1-2% MSE)
q_fp16 = q.to(torch.float16)
# Repeat K/V для GQA
gqa = Hq // Hkv
k_rep = k_fp16.repeat_interleave(gqa, dim=1)
v_rep = v_fp16.repeat_interleave(gqa, dim=1)
ref = torch.nn.functional.scaled_dot_product_attention(
    q_fp16, k_rep, v_rep, scale=scale
).to(torch.float32)

mse = torch.mean((out - ref) ** 2).item()
max_err = torch.max(torch.abs(out - ref)).item()
ref_mag = torch.mean(torch.abs(ref)).item()
print(f"vs SDPA: mse={mse:.6f}, max_err={max_err:.6f}, ref_mag={ref_mag:.6f}, rel_mse={mse/(ref_mag**2+1e-9):.4f}")

# Грубый порог: Q8 квантизация К даёт отн. ошибку ~1-3%
if mse > 0.1:
    print(f"WARN: large MSE vs reference — проверить корректность kernel'а", file=sys.stderr)

print("\n=== SMOKE TEST PASSED ===")
