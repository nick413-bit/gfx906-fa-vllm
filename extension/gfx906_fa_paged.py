"""Paged KV cache wrapper для gfx906_fa.

Два пути:
  * fast-path: key_cache_q8 передан → читаем уже квантованный K напрямую,
    квантование не выполняется. Используется когда backend держит side-buffer.
  * legacy-path: key_cache_q8 = None → gather FP16 K → quantize_q8_0 (device).
    Медленнее (лишний gather + квантование каждый шаг), но корректно.

API-уровневая обёртка: собирает вход из vLLM paged layout в contiguous
тензоры и вызывает gfx906_fa.forward().
"""
from __future__ import annotations

import math
import os as _os
import time as _time
from typing import Optional, Tuple

import torch

import gfx906_fa

_DBG   = _os.environ.get("GFX906_FA_FWD_DEBUG", "0") == "1"
# Level 1: по умолчанию используем fused gather kernel. Отключить можно
# через GFX906_FA_FUSED=0 — тогда работает старый путь через fancy-indexing
# (для A/B-замеров и как быстрый safety-fallback при регрессиях).
_FUSED = _os.environ.get("GFX906_FA_FUSED", "1") != "0"
# Level 3c: direct-paged FA — FA kernel читает K/V напрямую из paged cache
# через block_table indirection, без промежуточного gather.
#
# Режимы (GFX906_FA_DIRECT_PAGED):
#   "0"    → всегда gather+FA (baseline, заведомо корректный путь).
#   "1"    → всегда direct-paged (для A/B-замеров и benchmarking).
#   "auto" → (default) адаптивный выбор по {batch, max_seqlen_q}:
#            1) B < GFX906_FA_DIRECT_PAGED_MIN_BATCH   → gather (single-user decode).
#            2) max_seqlen_q > GFX906_FA_DIRECT_PAGED_MAX_SQ → gather
#               (длинный prefill, ncols1=64 — direct paged spill'ит VGPR и проигрывает).
#            3) иначе                                  → direct.
#
# Обоснование auto-default (MI50 / gfx906, bench_ab2.py + bench_prefill.py):
#   Decode (Sq=1, ncols1=2):
#     * B=1: gather быстрее на ~3-6% (compact access, direct добавляет block_table indirection).
#     * B=2: direct быстрее на ~4-7%.
#     * B≥3: direct быстрее на 7-35%.
#     * B=8, Sk=61K: gather → CUDA OOM (24 GiB peak); direct работает (~13 ms/step).
#   Prefill (bench_prefill.py, occupancy-fix применён):
#     * B=1 Sq=16 (ncols1=16): direct WIN -5..-27% (0 spill).
#     * B=2 Sq=32 (ncols1=32): direct LOSS +13% даже при 0 spill (block_table lookup latency).
#     * B=2 Sq=64 (ncols1=64): direct LOSS +34% (197 spill остался, 1 wave/EU).
#     * B=4 Sq=32/64: direct LOSS +0.5..+27%.
# → threshold {min_batch=2, max_sq=16} закрывает:
#     - регрессию B=1 (gather),
#     - регрессию ncols1=32/64 prefill (gather),
#     - включает direct для decode multi-batch (Sq=1) и короткого chunked prefill (Sq≤16),
#     - спасает от OOM на длинном Sk (mode=1 явный override).
_DIRECT_PAGED_MODE = _os.environ.get("GFX906_FA_DIRECT_PAGED", "auto").lower()
_DIRECT_PAGED_MIN_BATCH = int(_os.environ.get("GFX906_FA_DIRECT_PAGED_MIN_BATCH", "2"))
_DIRECT_PAGED_MAX_SQ = int(_os.environ.get("GFX906_FA_DIRECT_PAGED_MAX_SQ", "16"))

def _should_use_direct_paged(num_seqs: int, max_seqlen_q: int) -> bool:
    """Решает, использовать ли direct-paged FA для текущего batch/seq_q."""
    if _DIRECT_PAGED_MODE == "0":
        return False
    if _DIRECT_PAGED_MODE == "1":
        return True
    if num_seqs < _DIRECT_PAGED_MIN_BATCH:
        return False
    if max_seqlen_q > _DIRECT_PAGED_MAX_SQ:
        return False
    return True

def _fwdlog(msg: str) -> None:
    if not _DBG:
        return
    try:
        pid = _os.getpid()
        with open(f"/tmp/gfx906_fa_debug/fwd-{pid}.log", "a") as f:
            f.write(f"[{_time.time():.3f}] {msg}\n")
    except Exception:
        pass


def _gather_kv(
    key_cache: torch.Tensor,        # [num_blocks, block_size, Hkv, D]  fp16
    value_cache: torch.Tensor,      # [num_blocks, block_size, Hkv, D]  fp16
    block_table: torch.Tensor,      # [num_seqs, max_blocks]            int32
    seq_lens: torch.Tensor,         # [num_seqs]                        int32
    max_seqlen_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Собирает K, V из paged cache в contiguous layout.

    Возвращает:
      K: [B, Hkv, max_seqlen_k, D]  fp16
      V: [B, Hkv, max_seqlen_k, D]  fp16

    Позиции token'ов за пределами seq_lens[i] получают нулевые значения
    (чтобы softmax с дополнительным KV_max-срезом в kernel видел их как
     out-of-bounds; kernel уже умеет KV_max).

    Текущая реализация — через fancy indexing torch. Будет заменена
    custom HIP kernel'ом в v2 для снижения overhead'а (оценка 2-3x на prefill).
    """
    num_blocks, block_size, num_kv_heads, head_size = key_cache.shape
    num_seqs = block_table.shape[0]

    assert key_cache.shape == value_cache.shape, "K/V shape mismatch"
    assert block_table.dtype in (torch.int32, torch.int64), \
        f"block_table must be int, got {block_table.dtype}"
    assert seq_lens.shape == (num_seqs,), \
        f"seq_lens shape {seq_lens.shape} vs num_seqs={num_seqs}"

    # Количество блоков на последовательность (округление вверх)
    max_blocks_needed = (max_seqlen_k + block_size - 1) // block_size
    assert block_table.shape[1] >= max_blocks_needed, \
        f"block_table columns {block_table.shape[1]} < {max_blocks_needed}"

    bt = block_table[:, :max_blocks_needed].to(torch.long)  # [B, n_blocks]

    # Fancy indexing: выбираем block'и и reshape'им в [B, n_blocks*block_size, Hkv, D]
    # key_cache[bt] → [B, n_blocks, block_size, Hkv, D]
    k_gathered = key_cache[bt]    # fp16
    v_gathered = value_cache[bt]

    # Flatten block-dim: → [B, n_blocks*block_size, Hkv, D]
    k_gathered = k_gathered.view(num_seqs, -1, num_kv_heads, head_size)
    v_gathered = v_gathered.view(num_seqs, -1, num_kv_heads, head_size)

    # Обрезать до max_seqlen_k
    k_gathered = k_gathered[:, :max_seqlen_k].contiguous()
    v_gathered = v_gathered[:, :max_seqlen_k].contiguous()

    # Маскировать «хвост» за пределами seq_lens[i] нулями.
    # Создаём маску положения: [1, max_seqlen_k] → broadcast до [B, max_seqlen_k]
    positions = torch.arange(max_seqlen_k, device=seq_lens.device, dtype=seq_lens.dtype)
    mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)           # [B, Sk]
    mask_f = mask.view(num_seqs, max_seqlen_k, 1, 1).to(k_gathered.dtype)

    k_gathered = k_gathered * mask_f
    v_gathered = v_gathered * mask_f

    # Переставляем в [B, Hkv, Sk, D] — именно этот layout ждёт наш FA kernel
    k_bhsd = k_gathered.permute(0, 2, 1, 3).contiguous()
    v_bhsd = v_gathered.permute(0, 2, 1, 3).contiguous()

    return k_bhsd, v_bhsd


def _gather_kv_q8(
    key_cache_q8: torch.Tensor,    # [num_blocks, block_size, Hkv, (D/32)*34]  uint8
    value_cache:  torch.Tensor,    # [num_blocks, block_size, Hkv, D]           fp16
    block_table:  torch.Tensor,    # [num_seqs, max_blocks]                     int
    seq_lens:     torch.Tensor,    # [num_seqs]                                 int
    max_seqlen_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fast-path gather: K уже квантован в side-buffer, V — fp16.

    Возвращает:
      K_q8 : [B, Hkv, Sk, (D/32)*34]  uint8
      V    : [B, Hkv, Sk, D]           fp16

    Позиции за seq_lens[i] в V обнуляются. K — оставляем как есть
    (kernel отсекает по KV_max, доп. маска не нужна).
    """
    num_blocks, block_size, Hkv, bytes_per_row = key_cache_q8.shape
    _, _, _, head_size = value_cache.shape
    num_seqs = block_table.shape[0]
    assert value_cache.shape[0] == num_blocks and value_cache.shape[1] == block_size, \
        "key_cache_q8 / value_cache layout mismatch"

    max_blocks_needed = (max_seqlen_k + block_size - 1) // block_size
    bt = block_table[:, :max_blocks_needed].to(torch.long)

    # Fancy indexing: [B, n_blocks, bs, Hkv, bytes]
    k_gathered = key_cache_q8[bt]              # uint8
    v_gathered = value_cache[bt]               # fp16

    # Flatten block-dim: [B, n_blocks*bs, Hkv, ...]
    k_gathered = k_gathered.view(num_seqs, -1, Hkv, bytes_per_row)
    v_gathered = v_gathered.view(num_seqs, -1, Hkv, head_size)

    k_gathered = k_gathered[:, :max_seqlen_k].contiguous()
    v_gathered = v_gathered[:, :max_seqlen_k].contiguous()

    # V-маска tail: обнулим позиции за seq_lens (для безопасности).
    positions = torch.arange(max_seqlen_k, device=seq_lens.device, dtype=seq_lens.dtype)
    mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)   # [B, Sk]
    mask_f = mask.view(num_seqs, max_seqlen_k, 1, 1).to(v_gathered.dtype)
    v_gathered = v_gathered * mask_f
    # K_q8 — хвост мусор, но kernel отсекает через KV_max_d, так что OK.

    # Permute → [B, Hkv, Sk, ...]
    k_bhsd = k_gathered.permute(0, 2, 1, 3).contiguous()
    v_bhsd = v_gathered.permute(0, 2, 1, 3).contiguous()
    return k_bhsd, v_bhsd


def forward_paged(
    query: torch.Tensor,            # [num_tokens, Hq, D]     fp32
    key_cache: torch.Tensor,        # [num_blocks, block_size, Hkv, D]  fp16
    value_cache: torch.Tensor,      # [num_blocks, block_size, Hkv, D]  fp16
    block_table: torch.Tensor,      # [num_seqs, max_blocks]            int32
    seq_lens: torch.Tensor,         # [num_seqs]                        int32
    cu_seqlens_q: torch.Tensor,     # [num_seqs+1]                      int32
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: Optional[float] = None,
    key_cache_q8: Optional[torch.Tensor] = None,  # fast-path: [num_blocks,bs,Hkv,(D/32)*34]
    q_pad_buf: Optional[torch.Tensor] = None,
    mask_buf:  Optional[torch.Tensor] = None,
    k_gather_buf: Optional[torch.Tensor] = None,  # [B,Hkv,Sk_pad,bytes_per_row] uint8
    v_gather_buf: Optional[torch.Tensor] = None,  # [B,Hkv,Sk_pad,D]             fp16
) -> torch.Tensor:
    """vLLM-совместимый paged-attention forward.

    Вход / выход в flat layout num_tokens (как в vLLM forward signature).
    Внутри собирает KV из paged layout в BHSD, квантует K → Q8_0
    и вызывает gfx906_fa.forward().

    Возвращает out: [num_tokens, Hq*D] fp32 (для совместимости с vLLM).
    """
    num_tokens, Hq, D = query.shape
    assert query.dtype == torch.float32, "query must be fp32 for FA-q8 path"
    num_seqs = block_table.shape[0]

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # -------- Sq / Sk padding --------
    # Sq_pad должен быть кратен ncols1 — размеру tile колонки в kernel.
    # launcher выбирает ncols1 по seq_q = Sq_pad (launcher.hip switch_ncols1):
    #   Sq>32→64,  Sq>16→32,  Sq>8→16,  Sq>4→8,  Sq>2→4,  Sq≤2→2.
    # Ранее Sq_pad всегда округлялся до 64, из-за чего на decode (Sq=1)
    # kernel прогонял 64 query tile'а вместо 1 — лишняя работа.
    # Важное условие: pad только если это НЕ prefill (нет causal-маски в kernel
    # бьёт по pad-позициям некорректно при ncols1<64 на реальных данных).
    # Для prefill всегда используем ncols1=64 (как до фикса).
    if   max_seqlen_q >  32: ncols1 = 64
    elif max_seqlen_q >  16: ncols1 = 32
    elif max_seqlen_q >   8: ncols1 = 16
    elif max_seqlen_q >   4: ncols1 = 8
    elif max_seqlen_q >   2: ncols1 = 4
    else:                    ncols1 = 2
    Sq_pad = ((max_seqlen_q + ncols1 - 1) // ncols1) * ncols1
    Sk_pad = ((max_seqlen_k + 31) // 32) * 32

    # -------- Level 3c: Direct-paged FA (обходит gather полностью) --------
    # Работает только когда:
    #   * key_cache_q8 передан (есть side-buffer с Q8_0)
    #   * block_size == 16 (захардкожено в kernel)
    #   * _should_use_direct_paged(num_seqs, max_seqlen_q):
    #       mode=1 или (mode=auto AND B≥min_batch AND Sq≤max_sq)
    #
    # В этом режиме НЕ делаем gather/pad для K/V; kernel сам читает paged
    # cache по block_table. Только Q padding и q_abs_offset готовятся здесь.
    if (key_cache_q8 is not None
            and _should_use_direct_paged(num_seqs, max_seqlen_q)
            and key_cache_q8.dim() == 4
            and key_cache_q8.shape[1] == 16):
        bt_i32 = block_table if block_table.dtype == torch.int32 else block_table.to(torch.int32)
        sl_i32 = seq_lens   if seq_lens.dtype   == torch.int32 else seq_lens.to(torch.int32)
        bt_i32 = bt_i32.contiguous()
        sl_i32 = sl_i32.contiguous()

        if (q_pad_buf is not None
                and q_pad_buf.shape[0] >= num_seqs
                and q_pad_buf.shape[1] >= Hq
                and q_pad_buf.shape[2] >= Sq_pad
                and q_pad_buf.shape[3] == D
                and q_pad_buf.dtype == query.dtype):
            q_padded = q_pad_buf[:num_seqs, :Hq, :Sq_pad, :].contiguous()
            q_padded.zero_()
        else:
            q_padded = torch.zeros(
                (num_seqs, Hq, Sq_pad, D),
                dtype=query.dtype, device=query.device
            )

        cu = cu_seqlens_q.to(torch.long)
        if max_seqlen_q == 1 and num_tokens == num_seqs:
            q_padded[:, :, :1, :] = query.unsqueeze(2)
        else:
            for s in range(num_seqs):
                n = int(cu[s + 1] - cu[s])
                if n > 0:
                    q_seq = query[cu[s]:cu[s] + n]
                    q_padded[s, :, :n, :] = q_seq.permute(1, 0, 2)

        # Inline causal (same as gather path).
        need_causal = max_seqlen_q > 1
        q_abs_offset_tensor = None
        if need_causal:
            sl_i64 = sl_i32.to(torch.int64)
            cu_i64 = cu_seqlens_q.to(torch.int64) if cu_seqlens_q.dtype != torch.int64 else cu_seqlens_q
            n_q_per_seq = cu_i64[1:num_seqs + 1] - cu_i64[:num_seqs]
            q_abs_offset_tensor = (sl_i64 - n_q_per_seq).to(torch.int32).contiguous()

        if _DBG:
            _fwdlog(f"forward_paged DIRECT_PAGED: num_tokens={num_tokens} Hq={Hq} "
                    f"D={D} num_seqs={num_seqs} Sq_max={max_seqlen_q} "
                    f"Sk_max={max_seqlen_k} q_padded={tuple(q_padded.shape)} "
                    f"causal={'inline' if need_causal else 'none'}")
            torch.cuda.synchronize()

        try:
            out_padded = gfx906_fa.forward_paged_direct(
                q_padded,
                key_cache_q8,   # [num_blocks, 16, Hkv, (D/32)*34] uint8
                value_cache,    # [num_blocks, 16, Hkv, D] fp16
                bt_i32, sl_i32,
                float(scale),
                None,                     # mask
                q_abs_offset_tensor,      # inline causal
            )
            if _DBG:
                torch.cuda.synchronize()
                _fwdlog(f"forward_paged DIRECT_PAGED OK: out={tuple(out_padded.shape)}")
        except Exception as e:
            if _DBG:
                _fwdlog(f"forward_paged DIRECT_PAGED FAILED: {e!r}")
            raise

        if max_seqlen_q == 1 and num_tokens == num_seqs:
            return out_padded[:, :, 0, :].reshape(num_tokens, Hq * D).contiguous()

        out_flat = torch.empty((num_tokens, Hq * D), dtype=torch.float32, device=query.device)
        for s in range(num_seqs):
            n = int(cu[s + 1] - cu[s])
            if n > 0:
                out_flat[cu[s]:cu[s] + n] = out_padded[s, :, :n, :].permute(1, 0, 2).reshape(n, Hq * D)
        return out_flat

    # -------- gather KV + (возможно) quantize K --------
    if key_cache_q8 is not None and _FUSED:
        # Level 1 fused path: gather K_q8 + V_fp16 одним HIP kernel'ом.
        # Возвращает tensors с Sk=Sk_pad (хвост в V уже обнулён, K — мусор).
        bt_i32 = block_table if block_table.dtype == torch.int32 else block_table.to(torch.int32)
        sl_i32 = seq_lens   if seq_lens.dtype   == torch.int32 else seq_lens.to(torch.int32)
        bt_i32 = bt_i32.contiguous()
        sl_i32 = sl_i32.contiguous()
        # pre-allocated буферы (если подходят по ТОЧНОМУ shape) — zero-copy reuse.
        # Это критично на длинных контекстах: без этого каждая attention layer
        # аллоцирует 24-200+ MiB в HBM → peak VRAM spike → OOM.
        bytes_per_row_expected = (D // 32) * 34
        kbuf = k_gather_buf if (
            k_gather_buf is not None
            and k_gather_buf.dtype == torch.uint8
            and k_gather_buf.dim() == 4
            and k_gather_buf.shape == (num_seqs, key_cache_q8.shape[2], Sk_pad, bytes_per_row_expected)
            and k_gather_buf.is_contiguous()
        ) else None
        vbuf = v_gather_buf if (
            v_gather_buf is not None
            and v_gather_buf.dtype == torch.float16
            and v_gather_buf.dim() == 4
            and v_gather_buf.shape == (num_seqs, value_cache.shape[2], Sk_pad, D)
            and v_gather_buf.is_contiguous()
        ) else None
        K_q8, V_bhsd = gfx906_fa.gather_paged_kv_q8(
            key_cache_q8, value_cache, bt_i32, sl_i32, Sk_pad,
            k_out=kbuf, v_out=vbuf,
        )
        # K_q8: [B, Hkv, Sk_pad, bytes]; V_bhsd: [B, Hkv, Sk_pad, D] — уже padded.
        gathered_sk = Sk_pad
    elif key_cache_q8 is not None:
        # Fast-path (старый): K уже квантован в side-buffer, но gather через torch.
        K_q8, V_bhsd = _gather_kv_q8(
            key_cache_q8, value_cache, block_table, seq_lens, max_seqlen_k
        )
        gathered_sk = max_seqlen_k
    else:
        # Legacy-path: gather FP16 → quantize on the fly.
        K_bhsd, V_bhsd = _gather_kv(
            key_cache, value_cache, block_table, seq_lens, max_seqlen_k
        )
        K_q8 = gfx906_fa.quantize_q8_0(K_bhsd)
        gathered_sk = max_seqlen_k

    # Переиспользуем buffer если подходит по размеру; иначе создаём новый.
    if (q_pad_buf is not None
            and q_pad_buf.shape[0] >= num_seqs
            and q_pad_buf.shape[1] >= Hq
            and q_pad_buf.shape[2] >= Sq_pad
            and q_pad_buf.shape[3] == D
            and q_pad_buf.dtype == query.dtype):
        q_padded = q_pad_buf[:num_seqs, :Hq, :Sq_pad, :].contiguous()
        q_padded.zero_()
    else:
        q_padded = torch.zeros(
            (num_seqs, Hq, Sq_pad, D),
            dtype=query.dtype, device=query.device
        )

    cu = cu_seqlens_q.to(torch.long)
    # Сложить Q в [B, Hq, Sq_pad, D].
    # При Sq=1 (decode) — максимально частый случай: не гоняем Python-цикл
    # если все sequences имеют Sq=1 и num_tokens == num_seqs.
    if max_seqlen_q == 1 and num_tokens == num_seqs:
        # query: [num_seqs, Hq, D] → [num_seqs, Hq, 1, D] → паддинг по Sq_pad
        q_padded[:, :, :1, :] = query.unsqueeze(2)
    else:
        for s in range(num_seqs):
            n = int(cu[s + 1] - cu[s])
            if n > 0:
                q_seq = query[cu[s]:cu[s] + n]
                q_padded[s, :, :n, :] = q_seq.permute(1, 0, 2)

    # -------- Causal + kv_max --------
    # Level 3a: материализованная fp16 mask [B, Sq_pad, Sk_pad] заменена на
    # inline causal в kernel через q_abs_offset[B]. Для контекста 60K с Sq=4096
    # mask занимал бы ~480 MB — недопустимо на 32GB MI50.
    #
    # q_abs_offset[s] = seq_lens[s] - n_q[s] — абс. позиция query-chunk в
    # sequence. Kernel считает: k_pos > (q_abs_offset[s] + col_Q_0 + j) → -INF.
    kv_max_tensor = seq_lens.to(torch.int32).contiguous()

    need_causal = max_seqlen_q > 1
    q_abs_offset_tensor = None
    if need_causal:
        sl_i64 = seq_lens.to(torch.int64) if seq_lens.dtype != torch.int64 else seq_lens
        cu_i64 = cu_seqlens_q.to(torch.int64) if cu_seqlens_q.dtype != torch.int64 else cu_seqlens_q
        n_q_per_seq = cu_i64[1:num_seqs + 1] - cu_i64[:num_seqs]
        # shape [B]. Для padded row (j >= n_q[s]) не используется — kernel таких
        # колонок не вызывает (k_VKQ_max=seq_lens[s], + col_Q_0+j > seq_len в
        # padding всё равно обрезано через Sq_pad и bounds-check в quantize_Q).
        q_abs_offset_tensor = (sl_i64 - n_q_per_seq).to(torch.int32).contiguous()

    if _DBG:
        _path = ("FUSED" if (key_cache_q8 is not None and _FUSED)
                 else "FAST" if key_cache_q8 is not None
                 else "LEGACY")
        _fwdlog(f"forward_paged pre: path={_path} "
                f"num_tokens={num_tokens} Hq={Hq} D={D} num_seqs={num_seqs} "
                f"Sq_max={max_seqlen_q} Sk_max={max_seqlen_k} "
                f"Sq_pad={Sq_pad} Sk_pad={Sk_pad} "
                f"q_padded={tuple(q_padded.shape)} "
                f"K_q8={tuple(K_q8.shape)} V={tuple(V_bhsd.shape)} "
                f"causal={'inline' if need_causal else 'none'} scale={scale}")
        torch.cuda.synchronize()
    try:
        out_padded = gfx906_fa.forward(
            q_padded, K_q8, V_bhsd, float(scale),
            kv_max=kv_max_tensor,
            mask=None,
            q_abs_offset=q_abs_offset_tensor,
        )
        if _DBG:
            torch.cuda.synchronize()
            _fwdlog(f"forward_paged OK: out={tuple(out_padded.shape)}")
    except Exception as e:
        if _DBG:
            _fwdlog(f"forward_paged FAILED: {e!r}")
        raise

    # -------- распаковать обратно в flat [num_tokens, Hq*D] --------
    if max_seqlen_q == 1 and num_tokens == num_seqs:
        # Быстрый путь: out_padded[:, :, 0, :] → [B, Hq, D] → [B, Hq*D]
        return out_padded[:, :, 0, :].reshape(num_tokens, Hq * D).contiguous()

    out_flat = torch.empty((num_tokens, Hq * D), dtype=torch.float32, device=query.device)
    for s in range(num_seqs):
        n = int(cu[s + 1] - cu[s])
        if n > 0:
            out_flat[cu[s]:cu[s] + n] = out_padded[s, :, :n, :].permute(1, 0, 2).reshape(n, Hq * D)

    return out_flat
