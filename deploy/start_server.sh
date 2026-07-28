#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/home/paulsun/face_bib_photo_matcher}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
LOG_DIR="${LOG_DIR:-${APP_DIR}/logs}"

mkdir -p "${LOG_DIR}"
cd "${APP_DIR}"
. .venv/bin/activate

exec .venv/bin/uvicorn --app-dir src/website asgi:app --host "${HOST}" --port "${PORT}" --workers 1 >> "${LOG_DIR}/uvicorn.log" 2>&1
