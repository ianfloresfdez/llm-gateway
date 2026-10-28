"""Locust load test for llm-gateway.

  locust -f loadtest/locustfile.py --headless -u 4 -r 1 -t 6m --host http://localhost:8090

Env vars:
  LG_MODEL     model alias to call (default: local)
  LG_BYPASS    1 = send Cache-Control: no-cache (baseline without cache)
  LG_PARAPHRASE  share of requests using a paraphrase (default 0.3)
  LG_OUT       JSONL file with one line per request (default loadtest/results/raw/requests.jsonl)
  LG_SEED      RNG seed (default 42)
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from pathlib import Path

from locust import HttpUser, between, events, task
from prompts import QUESTIONS, SYSTEM

MODEL = os.environ.get("LG_MODEL", "local")
BYPASS = os.environ.get("LG_BYPASS") == "1"
PARAPHRASE = float(os.environ.get("LG_PARAPHRASE", "0.3"))
OUT = Path(os.environ.get("LG_OUT", "loadtest/results/raw/requests.jsonl"))
RNG = random.Random(int(os.environ.get("LG_SEED", "42")))
# Zipf(s=1) popularity over the 40 questions.
WEIGHTS = [1 / (rank + 1) for rank in range(len(QUESTIONS))]

_lock = threading.Lock()
_fh = None


@events.test_start.add_listener
def _open(**_):
    global _fh
    OUT.parent.mkdir(parents=True, exist_ok=True)
    _fh = OUT.open("w", encoding="utf-8")


@events.test_stop.add_listener
def _close(**_):
    if _fh:
        _fh.close()


def _record(row: dict) -> None:
    with _lock:
        _fh.write(json.dumps(row) + "\n")
        _fh.flush()


class ChatUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task
    def chat(self):
        with _lock:
            q = RNG.choices(range(len(QUESTIONS)), weights=WEIGHTS)[0]
            variant = RNG.choice((1, 2)) if RNG.random() < PARAPHRASE else 0
        prompt = QUESTIONS[q][variant]
        headers = {"cache-control": "no-cache"} if BYPASS else {}
        body = {
            "model": MODEL,
            "max_tokens": 64,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        }
        start = time.perf_counter()
        with self.client.post(
            "/v1/chat/completions", json=body, headers=headers, name=MODEL, catch_response=True
        ) as r:
            latency = time.perf_counter() - start
            row = {
                "t": time.time(),
                "question": q,
                "variant": variant,
                "status": r.status_code,
                "latency_s": latency,
                "cache": r.headers.get("x-gateway-cache"),
                "provider": r.headers.get("x-gateway-provider"),
            }
            if r.status_code == 200:
                data = r.json()
                row["usage"] = data["usage"]
                row["answer"] = data["choices"][0]["message"]["content"][:200]
                r.success()
            else:
                r.failure(f"HTTP {r.status_code}")
            _record(row)
