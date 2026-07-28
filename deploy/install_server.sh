#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/home/paulsun/face_bib_photo_matcher}"

cd "${APP_DIR}"
python3 -m venv .venv
. .venv/bin/activate

pip install --upgrade pip
pip install -r requirements-server.txt
pip install --no-deps insightface
