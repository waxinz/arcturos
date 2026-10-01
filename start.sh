#!/usr/bin/env bash
# start.sh — run the Arcturos dashboard.
#
# Serves on 0.0.0.0:24816 (LAN-visible). Any OpenAI-compatible inference
# server on your network works as a bench target; point arcturos.local.toml
# (or the Kick off form) at yours after starting.
set -euo pipefail

PORT="${ARCTUROS_PORT:-24816}"

cd "$(dirname "$0")"

if [ ! -d venv ]; then
  echo "→ creating venv (first run)"
  python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate

if ! python -c "import fastapi" 2>/dev/null; then
  echo "→ installing dependencies (first run)"
  pip install -q -r requirements.txt
fi

echo "→ Arcturos on http://0.0.0.0:${PORT} (Ctrl+C to stop)"
exec python -m uvicorn arcturos.main:app --host 0.0.0.0 --port "$PORT"