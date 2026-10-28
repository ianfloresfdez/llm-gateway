"""Summarise one or more locust runs (JSONL written by locustfile.py).

  python loadtest/report.py baseline=RAW/baseline.jsonl cached=RAW/cached.jsonl

Writes loadtest/results/summary.json and prints a Markdown table. Cost avoided is
valued at the reference price configured for the upstream model in config/gateway.yaml.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
LOCAL_MODEL = "gemma4:12b"


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def reference_price() -> tuple[str, dict]:
    pricing = yaml.safe_load((ROOT / "config/gateway.yaml").read_text())["pricing"]
    ref = pricing[LOCAL_MODEL]["reference"]
    return ref, pricing[ref]


def summarise(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    ok = [r for r in rows if r["status"] == 200]
    hits = [r for r in ok if (r["cache"] or "").startswith("hit")]
    misses = [r for r in ok if r not in hits]
    duration = max(r["t"] for r in rows) - min(r["t"] for r in rows)
    ref, price = reference_price()

    def cost(rs):
        return (
            sum(
                r["usage"]["prompt_tokens"] * price["input"]
                + r["usage"]["completion_tokens"] * price["output"]
                for r in rs
            )
            / 1e6
        )

    def lat(rs, p):
        return round(percentile([r["latency_s"] for r in rs], p) * 1000, 1)

    # Semantic hit precision: a hit is correct if the cached answer was generated for the same
    # question (any phrasing). Answers generated on misses tell us which question owns each answer.
    owner = {r["answer"]: r["question"] for r in misses}
    semantic = [r for r in ok if r["cache"] == "hit_semantic"]
    wrong = [r for r in semantic if owner.get(r["answer"], r["question"]) != r["question"]]
    return {
        "requests": len(rows),
        "errors": len(rows) - len(ok),
        "duration_s": round(duration, 1),
        "throughput_rps": round(len(rows) / duration, 2) if duration else None,
        "hit_rate": round(len(hits) / len(ok), 4) if ok else 0,
        "hits_exact": sum(r["cache"] == "hit_exact" for r in ok),
        "hits_semantic": len(semantic),
        "semantic_false_hits": len(wrong),
        "p50_ms": lat(ok, 50),
        "p95_ms": lat(ok, 95),
        "p50_hit_ms": lat(hits, 50) if hits else None,
        "p95_hit_ms": lat(hits, 95) if hits else None,
        "p50_miss_ms": lat(misses, 50),
        "p95_miss_ms": lat(misses, 95),
        "mean_miss_ms": round(statistics.mean(r["latency_s"] for r in misses) * 1000, 1) if misses else None,
        "tokens_total": sum(r["usage"]["total_tokens"] for r in ok),
        "tokens_saved": sum(r["usage"]["total_tokens"] for r in hits),
        "reference_model": ref,
        "cost_at_reference_usd": round(cost(ok), 6),
        "cost_avoided_usd": round(cost(hits), 6),
    }


def main(args: list[str]) -> None:
    runs = {}
    for arg in args:
        name, _, path = arg.partition("=")
        runs[name] = summarise(Path(path))
    out = ROOT / "loadtest/results/summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(runs, indent=2))
    keys = list(next(iter(runs.values())))
    print("| metric | " + " | ".join(runs) + " |")
    print("|---|" + "---|" * len(runs))
    for k in keys:
        print(f"| {k} | " + " | ".join(str(r[k]) for r in runs.values()) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
