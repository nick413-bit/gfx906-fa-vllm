# Performance measurements

Бенчмарки выполнены на:
- 8× AMD MI50 (gfx906, 32 GB HBM2), PCIe Gen4 x16
- AMD EPYC 7002, Ubuntu 22.04, ROCm 7.2.1
- MiniMax-M2.7-AWQ-4bit (62 layers, 8 KV heads, 128 head_dim)
- TP=8, BS=1, prompt = Lorem Ipsum (повторяющийся), 32 output tokens

## 1. Короткий профиль (`start_vllm_fa.sh`, CUDA Graphs ON)

| ctx     | TTFT (ms) | ITL (ms) | TG (tok/s) |
|--------:|----------:|---------:|-----------:|
| 1 000   |        89 |       36 |     27.7   |
| 8 000   |       263 |       64 |     15.6   |
| 16 000  |       414 |       88 |     11.4   |
| 32 000  |       360 |      130 |      7.7   |
| 48 000  |       623 |      177 |      5.7   |
| 60 000  |       586 |      211 |      4.8   |
| 65 000  |       646 |      225 |      4.4   |

## 2. Долгий профиль (`start_vllm_fa_ngram.sh`, EAGER + ngram K=5)

| ctx      | TTFT (ms) | ITL (ms) | TG (tok/s) | Δ vs baseline |
|---------:|----------:|---------:|-----------:|--------------:|
|  1 000   |      136  |     39   |    22.5    |         —     |
|  8 000   |      203  |     71   |    14.0    |         —     |
| 32 000   |      688  |    135   |     7.4    |        +6%    |
| 48 000   |      635  |    156   |     6.4    |       +13%    |
| 60 000   |      632  |    164   |     6.1    |       +28%    |
| 65 000   |      647  |    225   |     4.0    |         —     |
|100 000   |     1711  |    256   |     3.9    |       +32%    |
|130 000   |     1246  |    332   |     3.0    |       +29%    |

## 3. Acceptance rate (ngram)

Из логов vLLM `SpecDecoding metrics`:

| сценарий              | Avg Draft acc. rate | Mean acc. length |
|-----------------------|--------------------:|-----------------:|
| Lorem Ipsum + RAG     |              100.0% |         4.0 / 5  |
| Structured code       |            ~85-95%  |         3.5 / 5  |
| Creative text         |             20-40%  |         1.5 / 5  |

## 4. Архитектурный потолок

На 100K декоде каждый token требует чтения:
- Веса AWQ-4bit / 8 GPU: ~7.75 GB (62 layers × 125 MB)
- KV cache: ~3.2 GB (62 × 100K × 256 KB / 8)
- Fixed overhead: 62 × (launch + 2× AllReduce + RMS/RoPE/MLP)

Идеальный time @ HBM BW 1 TB/s = 11 ms/token. Реально = 326 ms/token.
**Bottleneck — launch overhead + AR, а не bandwidth.** Именно поэтому
ngram помогает: reducing количество forward passes.

## 5. Планы по улучшению

| Техника                     | Ожид. прирост | Сложность    |
|-----------------------------|--------------:|--------------|
| Q8 V quantization           |      +20-30%  | 1-2 недели   |
| EAGLE3 draft head           |      +100%+   | 3-4 недели   |
| Block-size 32 KV cache      |       +5-10%  | 1-2 дня      |
| Persistent kernel           |      +20-30%  | 4+ недель    |
| Custom AllReduce + fusion   |      +15-20%  | 2-3 недели   |
