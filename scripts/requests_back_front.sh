#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# API request demo: fetch the BACK slice first, then fill the FRONT gap.
#
# Prereq:  docker compose up --build          (serves http://localhost:8000)
#       or uvicorn app.main:app --port 8000
#
# Usage:   bash scripts/requests_back_front.sh [BASE_URL]
#
# Sequence:
#   create -> wait for 8 buffered/paused -> GET back [4,8) -> ack 7 => 409
#          -> GET front [0,4) -> ack 7 => 200 (frees 8) -> refetch original
#          -> duplicate ack => credit_freed 0
# ---------------------------------------------------------------------------
set -u
BASE="${1:-http://localhost:8000}"
OID="DF$(date +%s)"   # unique order id so the demo is re-runnable

hr(){ printf '\n==================== %s ====================\n' "$1"; }

hr "1. create order $OID"
curl -sS -X POST "$BASE/api/orders" \
  -H 'Content-Type: application/json' \
  -d "{\"order_id\":\"$OID\",\"prefix\":\"The quick brown fox\",\"length\":40}"
echo

hr "2. wait until 8 unconfirmed tokens are buffered (status=paused)"
for _ in $(seq 1 100); do
  J="$(curl -sS "$BASE/api/orders/$OID")"
  G="$(printf '%s' "$J" | sed -n 's/.*"generated":\([0-9]*\).*/\1/p')"
  [ "${G:-0}" -ge 8 ] && { printf '%s\n' "$J"; break; }
  sleep 0.2
done

hr "3. fetch ONLY the back slice seq [4,8)  (先取后段)"
curl -sS "$BASE/api/orders/$OID/tokens?start=4&end=8"; echo

hr "4. ack seq=7 with the front missing -> expect HTTP 409"
curl -sS -i -X POST "$BASE/api/orders/$OID/ack" \
  -H 'Content-Type: application/json' -d '{"seq":7}' | sed -n '1p;/detail/p'

hr "5. fill the front gap seq [0,4)  (再补前段)"
curl -sS "$BASE/api/orders/$OID/tokens?start=0&end=4"; echo

hr "6. one ack to 7 now confirms front + the early back slice (frees 8)"
curl -sS -X POST "$BASE/api/orders/$OID/ack" \
  -H 'Content-Type: application/json' -d '{"seq":7}'; echo

hr "7. re-fetch [0,8): still the byte-identical original text"
curl -sS "$BASE/api/orders/$OID/tokens?start=0&end=8"; echo

hr "8. duplicate ack seq=7 -> credit_freed 0"
curl -sS -X POST "$BASE/api/orders/$OID/ack" \
  -H 'Content-Type: application/json' -d '{"seq":7}'; echo
