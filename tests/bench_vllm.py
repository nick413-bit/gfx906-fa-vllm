"""
End-to-end A/B бенчмарк vLLM для MiniMax-M2.7-AWQ-4bit на gfx906.

Измеряет:
  - TTFT (Time To First Token) — критично для пользователя
  - ITL  (Inter-Token Latency)  — decode speed в steady state
  - TG   (Tokens/sec в decode)  = 1/ITL

на разных длинах prompt'а: 1k / 8k / 32k / 64k* токенов (*32k если
max-model-len=32k, 94k недоступен если max-model-len=65536).

Usage:
    python3 bench_vllm.py \\
        --url http://localhost:40044/v1 \\
        --model minimax \\
        --lengths 1024 8192 32768 \\
        --max-new-tokens 64 \\
        --label triton_baseline \\
        --out bench_triton.json

Результаты сохраняются в JSON, чтобы их потом можно было сравнить
между backend'ами.
"""

import argparse
import json
import time
from dataclasses import dataclass, asdict

import requests


# Стабильный текст для набора токенов (latin, низкоэнтропийный)
_FILLER = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
    "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris "
    "nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in "
    "reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla "
    "pariatur. Excepteur sint occaecat cupidatat non proident, sunt in "
    "culpa qui officia deserunt mollit anim id est laborum. "
)
_QUESTION = "\n\nBased on the text above, answer in one short sentence: what is the main topic? "


def build_prompt_with_length(
    url: str, model: str, target_tokens: int
) -> tuple[str, int]:
    """Builds a prompt of approximately `target_tokens` tokens using the
    server's tokenizer. Returns (prompt_text, actual_tokens).
    """
    # Бинпоиск по числу повторений
    # /tokenize висит на корне, а не на /v1 → срезаем /v1 при необходимости
    root = url[:-3] if url.endswith("/v1") else url

    def tokenize_len(text: str) -> int:
        r = requests.post(
            f"{root}/tokenize", json={"model": model, "prompt": text}
        )
        r.raise_for_status()
        return r.json()["count"]

    # Сначала оценим token/char ratio
    sample_tokens = tokenize_len(_FILLER)
    ratio = len(_FILLER) / sample_tokens  # chars per token
    approx_repeats = max(1, int(target_tokens * ratio / len(_FILLER)))

    prompt = _FILLER * approx_repeats
    cur = tokenize_len(prompt + _QUESTION)

    # Корректируем (в большую сторону, потом отсекаем по символам)
    while cur < target_tokens - 16:
        approx_repeats += 1
        prompt = _FILLER * approx_repeats
        cur = tokenize_len(prompt + _QUESTION)
    while cur > target_tokens + 64:
        approx_repeats -= 1
        prompt = _FILLER * approx_repeats
        cur = tokenize_len(prompt + _QUESTION)

    full = prompt + _QUESTION
    final_tokens = tokenize_len(full)
    return full, final_tokens


@dataclass
class BenchResult:
    label: str
    prompt_tokens: int
    output_tokens: int
    ttft_ms: float
    total_ms: float
    itl_ms: float
    tg_toks_per_sec: float


def bench_once(
    url: str,
    model: str,
    prompt: str,
    max_new_tokens: int,
    label: str,
) -> BenchResult:
    """Выполняет один streaming completion, возвращает замеры."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_new_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

    t0 = time.perf_counter()
    first_token_t: float | None = None
    last_token_t: float = t0
    output_tokens = 0
    prompt_tokens_reported = 0

    with requests.post(
        f"{url}/completions", json=payload, headers=headers, stream=True
    ) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if not raw.startswith("data:"):
                continue
            data_str = raw[len("data:"):].strip()
            if data_str == "[DONE]":
                break
            now = time.perf_counter()
            try:
                evt = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            # Usage блок приходит в конце (include_usage)
            usage = evt.get("usage")
            if usage:
                output_tokens = usage.get("completion_tokens", output_tokens)
                prompt_tokens_reported = usage.get(
                    "prompt_tokens", prompt_tokens_reported
                )
                continue
            # Обычный chunk
            choices = evt.get("choices") or []
            if choices and choices[0].get("text", "") != "":
                if first_token_t is None:
                    first_token_t = now
                last_token_t = now

    t_end = time.perf_counter()
    if first_token_t is None:
        raise RuntimeError("No output tokens received")

    ttft_ms = (first_token_t - t0) * 1000.0
    total_ms = (t_end - t0) * 1000.0
    # ITL: время на генерацию (N-1) оставшихся токенов после первого
    gen_ms = (last_token_t - first_token_t) * 1000.0
    num_remaining = max(1, output_tokens - 1)
    itl_ms = gen_ms / num_remaining if num_remaining > 0 else 0.0
    tg = 1000.0 / itl_ms if itl_ms > 0 else 0.0

    return BenchResult(
        label=label,
        prompt_tokens=prompt_tokens_reported,
        output_tokens=output_tokens,
        ttft_ms=ttft_ms,
        total_ms=total_ms,
        itl_ms=itl_ms,
        tg_toks_per_sec=tg,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:40044/v1")
    p.add_argument("--model", default="minimax")
    p.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[1024, 8192, 32768],
    )
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--label", default="baseline")
    p.add_argument("--warmup", action="store_true", help="Запустить короткий прогрев")
    p.add_argument("--out", default="bench_result.json")
    args = p.parse_args()

    if args.warmup:
        print("[warmup] sending small request...", flush=True)
        short_prompt, _ = build_prompt_with_length(args.url, args.model, 256)
        _ = bench_once(args.url, args.model, short_prompt, 8, "warmup")
        print("[warmup] done.", flush=True)

    results: list[BenchResult] = []
    print(
        f"{'len_tgt':>8} {'len_real':>8} {'out':>4} "
        f"{'TTFT_ms':>10} {'total_ms':>10} {'ITL_ms':>8} {'TG_tps':>8}",
        flush=True,
    )
    for length in args.lengths:
        prompt, actual = build_prompt_with_length(args.url, args.model, length)
        # 2 прогона, берём второй (warm) для каждой длины
        _ = bench_once(args.url, args.model, prompt, args.max_new_tokens, args.label)
        res = bench_once(args.url, args.model, prompt, args.max_new_tokens, args.label)
        print(
            f"{length:>8} {res.prompt_tokens:>8} {res.output_tokens:>4} "
            f"{res.ttft_ms:>10.1f} {res.total_ms:>10.1f} "
            f"{res.itl_ms:>8.2f} {res.tg_toks_per_sec:>8.2f}",
            flush=True,
        )
        results.append(res)

    # Сохраняем в JSON
    with open(args.out, "w") as f:
        json.dump(
            {
                "label": args.label,
                "model": args.model,
                "url": args.url,
                "max_new_tokens": args.max_new_tokens,
                "runs": [asdict(r) for r in results],
            },
            f,
            indent=2,
        )
    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()
