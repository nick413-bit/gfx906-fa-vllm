# Level 0 — Q8 side-buffer + device quantize

## Что изменилось

### Новые файлы
- `extension/gfx906_fa_quant.cu` — HIP-kernel-ы `quantize_q8_0_dense_kernel` и
  `reshape_and_cache_q8_kernel` для device-side квантования FP16 → block_q8_0.
- `tests/test_quant.py` — unit-тесты корректности новых kernel-ов (3 теста).

### Изменённые файлы
- `extension/setup.py` — добавлен `gfx906_fa_quant.cu` в sources.
- `extension/gfx906_fa.cpp`:
  - `quantize_q8_0` переписан на device-path (был CPU, ~100ms на 8k-контекст
    KV через PCIe-трансферы).
  - Добавлен PYBIND `reshape_and_cache_q8(key, slot_mapping, k_cache_q8)` —
    квантует новые K-токены и пишет в paged Q8 cache по `slot_mapping`.
- `extension/gfx906_fa_backend.py`:
  - Импл держит `self._k_cache_q8` — Q8 side-buffer параллельно FP16 cache.
  - Lazy-аллокация в `_ensure_q8_sidebuffer()` при первом `do_kv_cache_update`.
  - `do_kv_cache_update` теперь делает **две** записи:
    1. `triton_reshape_and_cache_flash(...)` — штатный FP16-путь (для V + legacy K).
    2. `reshape_and_cache_q8(...)` — инкрементально квантует и кладёт новый
       K-токен в Q8 side-buffer.
  - Добавлены `_q_pad_buf` и `_mask_buf` — предаллоцированные буферы для
    `forward_paged`, grow-логика в `_ensure_forward_buffers()`.
  - Env-флаг `GFX906_FA_LEGACY=1` → обратная совместимость (старый путь с
    quantize на лету, без side-buffer).
- `extension/gfx906_fa_paged.py`:
  - Новая функция `_gather_kv_q8(...)` — fast-path gather из Q8 side-buffer.
  - `forward_paged(...)` принимает 3 новых optional-параметра:
    `key_cache_q8`, `q_pad_buf`, `mask_buf`.
    - Если `key_cache_q8` передан → fast-path (gather uint8, **без** quantize).
    - Иначе → legacy-path (gather FP16 + quantize_q8_0 на лету).
  - Быстрый путь для decode (Sq=1): убран Python-цикл по sequences, `unsqueeze`
    и `reshape` вместо per-seq копирований.
  - Причинная маска **не строится** при Sq=1 (убирает 2 Python-цикла + NEG_INF
    заливку на каждом decode-step).

## Что это даёт (ожидаемо)

**Устранены три overhead-а на горячем пути decode:**

| Источник | Было (на шаг) | Стало |
|---|---|---|
| `_gather_kv` FP16 (K+V, 8k·8·128·2 B) | ~200 MB / 0.2 ms | только V ~100 MB / 0.1 ms |
| `quantize_q8_0` CPU → GPU copy | ~100 ms (PCIe) | 0 (kernel для только 1 токена в `do_kv_cache_update`) |
| Python Sq-loop, NEG_INF mask | ~5 ms | 0 (fast-path для Sq=1) |

**Прогноз:** decode 4.3 tok/s → **15–20 tok/s** после Level 0.

## Что **не** изменилось (осознанно)

- `flash_attn_tile_q8` kernel — не тронут.
- Layout основного FP16 KV-cache — не тронут (vLLM allocator продолжает
  распоряжаться как раньше).
- Legacy-путь `forward_paged` (через `_gather_kv` + `quantize_q8_0`) остаётся
  доступным под `GFX906_FA_LEGACY=1` для сравнения / A-B.

## Объём VRAM

Side-buffer Q8 для K: `K_fp16_bytes × 34/64 ≈ 0.531 × K_fp16`. При
`max-model-len=65536`, TP=8, MiniMax (8 KV-heads × 128 × 24 layer / 8 GPU =
24K KV-cache per GPU per layer) это добавит **~0.5×** размера K-кэша. **V не
дублируется** — остаётся FP16 native.

Если GPU on-mem упирается в 32 GB (MI50-32GB) и было `--gpu-memory-utilization 0.95`,
то **реально лучше снизить до 0.9** или уменьшить `max-model-len`:

```
--gpu-memory-utilization 0.9
```

## Как проверить

### На удалённом хосте с MI50:

```bash
# 1. Пересобрать extension внутри контейнера:
cd /gfx906_fa/extension
python setup.py build_ext --inplace

# 2. Unit-тесты:
cd /gfx906_fa
PYTHONPATH=extension python tests/test_quant.py
PYTHONPATH=extension python tests/test_paged.py   # должен по-прежнему проходить

# 3. A/B bench (скрипт ab_bench.py делает HTTP-запросы):
bash scripts/run_vllm_fa.sh           # запустить vLLM с нашим backend
python scripts/ab_bench.py --url http://localhost:40044 --ctx 1024,8192 --gen 128
```

## Rollback

Если что-то не так — `GFX906_FA_LEGACY=1` вернёт старый путь без изменений
в Python-логике (side-buffer по-прежнему аллоцируется но не используется).

Полный rollback: `git revert` коммит Level 0.
