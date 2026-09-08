#!/usr/bin/env python3
"""Repeatable client-side TTFT and TPS measurement for an OpenAI-compatible vLLM server.

No third-party packages are required. Authentication and endpoint details are read from
environment variables by default so result files never contain API keys.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_PROMPT = (
    "Explain in two concise paragraphs why streaming latency and generation "
    "throughput are different measurements for an LLM inference server."
)


@dataclass
class Sample:
    sample: int
    ttft_ms: float
    e2e_ms: float
    generation_ms: float
    completion_tokens: int
    prompt_tokens: int
    generation_tps: float


def percentile(values: list[float], percent: float) -> float:
    """Linear-interpolated percentile with no NumPy dependency."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def make_payload(args: argparse.Namespace, prompt: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    if args.disable_thinking:
        # Qwen's chat template supports this and makes a useful, consistent
        # serving benchmark: output tokens are the requested answer, not a
        # variable-length hidden reasoning trace.
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return payload


def request_once(args: argparse.Namespace, prompt: str, sample_number: int) -> Sample:
    request = Request(
        f"{args.base_url.rstrip('/')}/chat/completions",
        data=json.dumps(make_payload(args, prompt)).encode(),
        headers={
            "Authorization": f"Bearer {args.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    started = time.perf_counter()
    first_token_at: float | None = None
    completed_at: float | None = None
    usage: dict[str, Any] | None = None

    try:
        with urlopen(request, timeout=args.timeout_seconds) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                event = line[6:]
                if event == "[DONE]":
                    continue
                chunk = json.loads(event)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    # Some Qwen responses stream a reasoning field before content.
                    text = delta.get("content") or delta.get("reasoning_content") or ""
                    if text and first_token_at is None:
                        first_token_at = time.perf_counter()
                completed_at = time.perf_counter()
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"Could not reach vLLM: {error.reason}") from error

    if first_token_at is None:
        raise RuntimeError("The stream completed without a text token.")
    if usage is None or usage.get("completion_tokens") is None:
        raise RuntimeError("vLLM did not return usage; cannot calculate token-accurate TPS.")

    finished = completed_at or time.perf_counter()
    generation_seconds = finished - first_token_at
    completion_tokens = int(usage["completion_tokens"])
    if generation_seconds <= 0 or completion_tokens <= 0:
        raise RuntimeError("Invalid generation duration or token count from response.")
    return Sample(
        sample=sample_number,
        ttft_ms=(first_token_at - started) * 1_000,
        e2e_ms=(finished - started) * 1_000,
        generation_ms=generation_seconds * 1_000,
        completion_tokens=completion_tokens,
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        generation_tps=completion_tokens / generation_seconds,
    )


def summary(samples: list[Sample]) -> dict[str, Any]:
    def distribution(name: str) -> dict[str, float]:
        values = [getattr(sample, name) for sample in samples]
        return {
            "mean": statistics.fmean(values),
            "p50": percentile(values, 50),
            "p95": percentile(values, 95),
            "min": min(values),
            "max": max(values),
        }

    return {
        "samples": len(samples),
        "ttft_ms": distribution("ttft_ms"),
        "generation_tps": distribution("generation_tps"),
        "e2e_ms": distribution("e2e_ms"),
        "completion_tokens": distribution("completion_tokens"),
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL"), help="e.g. https://pod-8000.proxy.runpod.net/v1")
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY"), help="defaults to VLLM_API_KEY")
    parser.add_argument("--model", default=os.getenv("VLLM_MODEL", "cyankiwi/Qwen3.8-27B-AWQ-INT4"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/raw/vllm-bench"))
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--cache-mode",
        choices=("cold", "warm"),
        default="cold",
        help="cold appends a unique suffix per request to avoid prefix-cache hits; warm reuses the exact prompt.",
    )
    return parser.parse_args()


def main() -> int:
    args = arguments()
    if not args.base_url or not args.api_key:
        print("Set VLLM_BASE_URL and VLLM_API_KEY (or pass --base-url and --api-key).", file=sys.stderr)
        return 2
    if args.runs < 1 or args.warmup_runs < 0:
        print("--runs must be positive and --warmup-runs cannot be negative.", file=sys.stderr)
        return 2

    def prompt_for(index: int) -> str:
        if args.cache_mode == "warm":
            return args.prompt
        return f"{args.prompt}\n\nBenchmark nonce: {index:04d}."

    print(f"Warm-up: {args.warmup_runs} request(s), cache mode: {args.cache_mode}")
    for number in range(args.warmup_runs):
        request_once(args, prompt_for(-number - 1), -number - 1)

    samples: list[Sample] = []
    for number in range(1, args.runs + 1):
        sample = request_once(args, prompt_for(number), number)
        samples.append(sample)
        print(
            f"{number:02d}/{args.runs}: TTFT {sample.ttft_ms:.1f} ms | "
            f"TPS {sample.generation_tps:.1f} | {sample.completion_tokens} output tokens"
        )

    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "endpoint": args.base_url,
        "model": args.model,
        "settings": {
            "runs": args.runs,
            "warmup_runs": args.warmup_runs,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "disable_thinking": args.disable_thinking,
            "cache_mode": args.cache_mode,
        },
        "summary": summary(samples),
        "samples": [asdict(sample) for sample in samples],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir / f"{args.label}-{timestamp}.json"
    output.write_text(json.dumps(result, indent=2) + "\n")

    csv_output = output.with_suffix(".csv")
    with csv_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(samples[0])))
        writer.writeheader()
        writer.writerows(asdict(sample) for sample in samples)

    report = result["summary"]
    print(
        "\nSummary: "
        f"TTFT p50/p95 {report['ttft_ms']['p50']:.1f}/{report['ttft_ms']['p95']:.1f} ms; "
        f"TPS p50/p95 {report['generation_tps']['p50']:.1f}/{report['generation_tps']['p95']:.1f}."
    )
    print(f"Saved: {output} and {csv_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
