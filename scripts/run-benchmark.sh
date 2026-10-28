#!/usr/bin/env bash
# Two locust runs against the same deployment with the same seed:
#   baseline: every request sends Cache-Control: no-cache
#   cached:   empty cache at start, semantic cache on
# Usage: scripts/run-benchmark.sh [base_url] [duration] [users]
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=${1:-http://localhost:8090}
DURATION=${2:-6m}
USERS=${3:-4}
PY=${PY:-$( [[ -x .venv/Scripts/python ]] && echo .venv/Scripts/python || echo python )}
RAW=loadtest/results/raw
mkdir -p "$RAW"

flush_cache() {
  if kubectl -n llm get deploy llm-gateway-redis >/dev/null 2>&1; then
    kubectl -n llm exec deploy/llm-gateway-redis -- redis-cli flushall
  else
    docker compose exec -T redis redis-cli flushall
  fi
}

# Warm-up so the first measured request does not include loading the model into VRAM.
curl -sS -o /dev/null "$BASE/v1/chat/completions" -H 'content-type: application/json' -H 'cache-control: no-cache' \
  -d '{"model":"local","max_tokens":8,"messages":[{"role":"user","content":"warm up"}]}'

for run in baseline cached; do
  flush_cache
  echo "== $run start $(date -u +%FT%TZ)" | tee -a "$RAW/timeline.txt"
  LG_BYPASS=$([[ $run == baseline ]] && echo 1 || echo 0) LG_OUT="$RAW/$run.jsonl" \
    "$PY" -m locust -f loadtest/locustfile.py --headless -u "$USERS" -r 1 -t "$DURATION" \
      --host "$BASE" --only-summary --csv "$RAW/$run" >"$RAW/$run.log" 2>&1
  echo "== $run end $(date -u +%FT%TZ)" | tee -a "$RAW/timeline.txt"
done

"$PY" loadtest/report.py baseline="$RAW/baseline.jsonl" cached="$RAW/cached.jsonl" | tee loadtest/results/summary.md
