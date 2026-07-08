"""Privacy-aware request logging + latency/cost tracking.

NON-NEGOTIABLE for this project: the logger NEVER stores raw personal financial values
(income, expenses, debt, savings) — that's exactly the sensitive-data-at-rest problem the
ephemeral MVP scope was built to avoid. It logs only:
  - which tool(s) were called (name), success/failure, latency
  - whether context was provided (boolean), NOT the values
  - a coarse income BUCKET only if explicitly needed for debugging (never the exact figure)
  - estimated token cost

Logs are JSON lines in logs/requests.jsonl. Safe to inspect / commit-ignore.
"""
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "requests.jsonl"

# Rough Groq pricing (USD per 1M tokens) for cost estimation — update if models change.
PRICE_PER_1M = {
    "llama-3.1-8b-instant": {"in": 0.05, "out": 0.08},
    "llama-3.3-70b-versatile": {"in": 0.59, "out": 0.79},
}


def income_bucket(income: float | None) -> str | None:
    """Coarsen an income figure to a bucket. Only ever call this if a numeric signal is
    genuinely needed for debugging — never log the exact value."""
    if income is None:
        return None
    for hi, label in [(10000, "<10k"), (20000, "10k-20k"), (40000, "20k-40k"),
                      (75000, "40k-75k"), (float("inf"), "75k+")]:
        if income < hi:
            return label


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price = PRICE_PER_1M.get(model, {"in": 0.0, "out": 0.0})
    return round((prompt_tokens * price["in"] + completion_tokens * price["out"]) / 1_000_000, 6)


@dataclass
class RequestLog:
    timestamp: str
    latency_ms: int
    tools_called: list[str]
    context_provided: bool
    asked_for_missing: bool
    degraded: bool
    cache_hit: bool
    model: str
    estimated_cost_usd: float
    # explicitly NO raw query text with personal numbers, NO tool arguments, NO field values


def log_request(log: RequestLog, path: Path = LOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(asdict(log)) + "\n")


class Timer:
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed_ms = int((time.perf_counter() - self._start) * 1000)


def build_log(
    latency_ms: int,
    tools_called: list[str],
    context_provided: bool,
    asked_for_missing: bool,
    degraded: bool,
    cache_hit: bool,
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> RequestLog:
    return RequestLog(
        timestamp=datetime.now(timezone.utc).isoformat(),
        latency_ms=latency_ms,
        tools_called=tools_called,
        context_provided=context_provided,
        asked_for_missing=asked_for_missing,
        degraded=degraded,
        cache_hit=cache_hit,
        model=model,
        estimated_cost_usd=estimate_cost(model, prompt_tokens, completion_tokens),
    )


def read_logs(path: Path = LOG_PATH) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
