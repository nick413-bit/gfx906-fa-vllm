# gfx906-fa-vllm

[![Docker](https://img.shields.io/badge/docker-nickoptimal%2Fgfx906--fa--vllm-blue?logo=docker)](https://hub.docker.com/r/nickoptimal/gfx906-fa-vllm)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![ROCm](https://img.shields.io/badge/ROCm-7.2.1-red?logo=amd)](https://rocm.docs.amd.com/)

**FlashAttention-style attention backend для vLLM на AMD MI50 / MI60 / Radeon VII (gfx906).**

Custom HIP kernel с paged-KV cache и Q8_0 quantization для K, даёт +20-40%
к `TRITON_ATTN` на decode/prefill длинных контекстов. Тестирован на
8× MI50 с MiniMax-M2.7-AWQ-4bit (196K context).

---

## Зачем

Стоковый vLLM на ROCm использует `TRITON_ATTN`. Его автосгенерированные
ядра на gfx906 дают посредственный throughput на длинных контекстах
из-за неоптимальной occupancy и отсутствия paged-KV fast-path
для Vega 20. `gfx906-fa-vllm`:

- **Paged-KV attention прямо в kernel** (Level 3c direct-paged) — без
  промежуточного `gather → forward` пайплайна.
- **Q8_0 quantization для K** в KV-cache — сохраняет HBM bandwidth на
  decode.
- **Occupancy-aware launch bounds** под gfx906 (128 threads × 4 wavefronts,
  VGPR=128 sweet spot).
- **Адаптивный выбор пути** (short Sq → tile kernel, long Sq → paged-direct).

## Производительность

Бенчмарк `bench_vllm.py` на MiniMax-M2.7-AWQ-4bit, 8× MI50 TP=8, BS=1:

### Короткий профиль (`start_vllm_fa.sh`, CUDA Graphs ON)

| prompt ctx | TTFT (ms) | ITL (ms) | **TG (tok/s)** |
|---:|---:|---:|---:|
| 1K  |  89 |  36 | **27.7** |
| 8K  | 263 |  64 | **15.6** |
| 32K | 360 | 130 |  **7.7** |

### Долгий профиль (`start_vllm_fa_ngram.sh`, EAGER + ngram K=5)

| prompt ctx | TTFT (ms) | ITL (ms) | **TG (tok/s)** | Δ vs baseline |
|---:|---:|---:|---:|---:|
| 32K  |  638 | 135 |  **7.4** | +6% |
| 65K  |  647 | 225 |  **4.0** | — |
| 100K | 1711 | 256 |  **3.9** | **+32%** |
| 130K | 1246 | 332 |  **3.0** | **+29%** |

Ngram spec decoding эффективен на повторяющемся тексте (RAG, код, пересказ
документа) — там acceptance rate 100%, Mean acceptance length = 4 из 5.

---

## Quick start

### 1. Pre-built Docker image

```bash
docker pull nickoptimal/gfx906-fa-vllm:mvp-0.1

# Запуск (короткий профиль, ≤32K)
docker run -d --name vllm \
    --network host --ipc host --shm-size 16g \
    --device /dev/kfd --device /dev/dri \
    --group-add video --group-add render \
    --cap-add CAP_SYS_PTRACE --security-opt seccomp=unconfined \
    -v /path/to/your/models:/models \
    -e NCCL_P2P_DISABLE=1 -e HIP_ENABLE_PEER_ACCESS=0 \
    nickoptimal/gfx906-fa-vllm:mvp-0.1 \
    vllm serve /models/your-awq-model \
        --tensor-parallel-size 8 \
        --attention-backend CUSTOM \
        --disable-custom-all-reduce \
        --max-model-len 65536 \
        --gpu-memory-utilization 0.78
```

Или используйте готовые скрипты:

```bash
git clone https://github.com/nick413/gfx906-fa-vllm && cd gfx906-fa-vllm
MODEL=/models/MiniMax-M2.7-AWQ-4bit bash scripts/start_vllm_fa.sh
```

### 2. Для длинных контекстов (≥ 32K, RAG/code)

```bash
MODEL=/models/MiniMax-M2.7-AWQ-4bit bash scripts/start_vllm_fa_ngram.sh
```

### 3. Сборка из исходников

```bash
# Собрать Docker-образ (работает в любом ROCm-capable окружении):
docker build -f docker/Dockerfile -t gfx906-fa-vllm:dev .

# Либо собрать extension локально, внутри базового контейнера:
docker run -it --rm --device /dev/kfd --device /dev/dri \
    -v $PWD:/src \
    docker.io/nalanzeyu/vllm-gfx906:v0.12.0-rocm6.3-ct-upgraded-mi50p2p bash
# inside:
cd /src/extension && pip install -e . --no-deps
```

---

## Архитектура

```
vLLM  →  --attention-backend CUSTOM  →  Gfx906FAImpl (extension/gfx906_fa_backend.py)
                                       │
                                       ├─ do_kv_cache_update:  fp16 key/value → paged KV cache (fp16)
                                       │
                                       └─ forward:  ─┬─ adaptive path select (extension/gfx906_fa_paged.py):
                                                     │    • seq_q ≤ 32  → direct-paged FA (paged kernel)
                                                     │    • seq_q >  32 → gather + flat FA (tile kernel)
                                                     │
                                                     ├─ gfx906_fa.reshape_and_cache_q8 (inline fp16→Q8 на лету)
                                                     ├─ gfx906_fa.gather_paged_kv_q8   (когда нужен gather)
                                                     └─ gfx906_fa.forward              (core FA kernel)

HIP kernels (kernel/):
  fattn-q8.cuh         — flat FlashAttention kernel (Q8_0 K, fp16 V)
  fattn-q8-paged.cuh   — direct-paged вариант (без gather)
  gfx906-common.cuh    — wavefront utilities, shuffle, reductions
```

### Ключевые особенности

1. **Q8_0 K quantization** (side-buffer временно отключён из-за сложности
   синхронизации с vLLM warmup pass'ами — используется inline-quantize в
   forward). См. `BASELINE.md §14` в upstream репо.
2. **Paged KV cache format** совместим с vLLM stock layout
   (`(num_blocks, 2, block_size, num_kv_heads, head_dim)`).
3. **Launch bounds** настроены под gfx906: `__launch_bounds__(128, 4)` —
   128 threads (= 4 wavefronts × 32 per WF на gfx906 wavefront=64? — см.
   [обсуждение occupancy](#occupancy-notes)).

### Occupancy notes

gfx906 wavefront = 64, но PyTorch cpp_extension на ROCm использует
`blockDim.x = 128`, что даёт 2 wavefronts / block. Для нашего FA ядра это
оптимум — больше wavefronts упираются в VGPR=128 (register file).

---

## Спека совместимости

| Component | Version |
|---|---|
| GPU | AMD MI50 / MI60 / Radeon VII (gfx906) |
| ROCm | 7.2.1 |
| PyTorch | 2.x (в базовом образе) |
| vLLM | 0.12.0 (в базовом образе) |
| Models tested | MiniMax-M2.7-AWQ-4bit |

### Протестированные топологии

- 8× MI50 на AMD EPYC 7002, PCIe Gen4 x16, Microsemi switches — **OK**
- TP=8, disable-custom-all-reduce, P2P OFF — дефолтная конфигурация

Не тестировано на MI100/MI200 (gfx908/gfx90a). Код gfx906-specific,
потребует adaptation для других архитектур.

---

## Переменные окружения

| Env | Default | Назначение |
|---|---|---|
| `GFX906_FA_LEGACY` | `1` | `1` — inline fp16→Q8 (стабильный путь), `0` — Q8 side-buffer (экспериментальный) |
| `GFX906_FA_FUSED` | `1` | `1` — HIP gather kernel, `0` — Python fancy-index |
| `GFX906_FA_DIRECT_PAGED` | `auto` | `auto` / `1` / `0` — direct-paged FA path |
| `GFX906_FA_GATHER_V` | `2` | `1` — per-token, `2` — paged-block-coalesced |
| `NCCL_P2P_DISABLE` | `1` | P2P — на нашей топологии прирост 3-6%, не стоит сложности |
| `HIP_ENABLE_PEER_ACCESS` | `0` | парный к NCCL_P2P_DISABLE |

---

## Ограничения и TODO

- [ ] **Q8 V quantization** (сейчас V = fp16 в KV cache) — самый большой потенциал
      для улучшения decode на 100K+ (+20-30%).
- [ ] **EAGLE draft model** — +2-3× на любых промптах, требует обучения drafter.
- [ ] **Persistent kernel** — убрать launch overhead 62 layers. High risk.
- [ ] Q8 K side-buffer — починить синхронизацию с vLLM warmup pass'ами.

---

## Тесты

```bash
# Unit (внутри контейнера):
pytest tests/test_smoke.py       # базовая работоспособность
pytest tests/test_backend.py     # интеграция с vLLM AttentionImpl
pytest tests/test_backend_vs_legacy.py  # reproducer Q8 side-buffer bug
pytest tests/test_gather.py      # gather paged KV → Q8
pytest tests/test_fa_mixed_batch.py  # mixed batch (prefill + decode)
pytest tests/test_direct_paged_bitexact.py  # direct-paged FA vs flat path

# E2E benchmark (server должен быть запущен):
python tests/bench_vllm.py --url http://localhost:40044/v1 --model minimax \
    --lengths 1000 8000 32000 --max-new-tokens 32 --label baseline
```

---

## License

Apache-2.0 — см. [LICENSE](LICENSE).

Ядро базируется на идеях [llama.cpp FlashAttention Q8_0](https://github.com/ggerganov/llama.cpp)
(MIT), адаптированных под vLLM paged-KV layout и gfx906 occupancy.

---

## Authors

- **Nick** — [nick413@gmail.com](mailto:nick413@gmail.com)

Base Docker image: [@nalanzeyu/vllm-gfx906](https://hub.docker.com/r/nalanzeyu/vllm-gfx906) — thanks!
(содержит MI50 P2P-патч и прогретый Triton/ROCm 6.3 стек для gfx906).
