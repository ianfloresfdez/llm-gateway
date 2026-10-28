#!/usr/bin/env bash
# End-to-end checks against a running gateway (compose, kind or CI). Needs no LLM: uses the mock models.
# Usage: scripts/smoke-test.sh [base_url]
set -euo pipefail
BASE=${1:-http://localhost:8080}
RUN=$RANDOM$RANDOM   # unique prompt so a previous run's cache entry cannot pass the test

fail() { echo "FAIL: $*" >&2; exit 1; }

chat() {  # chat <model> <prompt>  -> prints response headers
  curl -sS -D - -o /dev/null "$BASE/v1/chat/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"$1\",\"messages\":[{\"role\":\"user\",\"content\":\"$2\"}]}"
}
header() { grep -i "^$1:" | head -1 | cut -d' ' -f2 | tr -d '\r'; }

echo "== readiness"
curl -sSf "$BASE/readyz" | grep -q '"redis":true' || fail "redis not reachable from the gateway"

echo "== first call is a miss, second an exact hit"
[[ $(chat mock "smoke $RUN" | header x-gateway-cache) == miss ]] || fail "expected miss"
[[ $(chat mock "smoke $RUN" | header x-gateway-cache) == hit_exact ]] || fail "expected hit_exact"

echo "== fallback: flaky mock -> healthy mock"
for i in $(seq 1 10); do
  status=$(curl -sS -o /dev/null -w '%{http_code}' "$BASE/v1/chat/completions" -H 'content-type: application/json' \
    -H 'cache-control: no-cache' -d "{\"model\":\"demo-fallback\",\"messages\":[{\"role\":\"user\",\"content\":\"f$i\"}]}")
  [[ $status == 200 ]] || fail "demo-fallback returned $status"
done

echo "== metrics exposed"
metrics=$(curl -sSf "$BASE/metrics")
grep -q 'gateway_cache_lookups_total{result="hit_exact"}' <<<"$metrics" || fail "cache metric missing"
grep -q 'gateway_requests_total' <<<"$metrics" || fail "request metric missing"

echo "smoke test passed"
