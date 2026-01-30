#!/usr/bin/env bash
set -e

ROLE="${ROLE:-api}"

case "$ROLE" in
api)
  exec uv run uvicorn main:app \
    --host 0.0.0.0 \
    --port 8090
  ;;
worker-fast)
  exec uv run rq worker eda_fast
  ;;
worker-deep)
  exec uv run rq worker eda_deep
  ;;
*)
  echo "[!] Unknown ROLE=$ROLE"
  exit 1
  ;;
esac
