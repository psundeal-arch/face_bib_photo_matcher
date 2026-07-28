#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-69.48.203.86}"
REMOTE_USER="${REMOTE_USER:-paulsun}"
REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_APP_DIR="${REMOTE_APP_DIR:-/home/paulsun/face_bib_photo_matcher}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_NAME="${SERVICE_NAME:-face-bib-photo-matcher}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
USE_SUDO="${USE_SUDO:-1}"

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required but not installed." >&2
  exit 1
fi

if ! command -v ssh >/dev/null 2>&1; then
  echo "ssh is required but not installed." >&2
  exit 1
fi

echo "Syncing project to ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_APP_DIR}"
rsync -az \
  --delete \
  --exclude '.git/' \
  --exclude '.DS_Store' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.pytest_cache/' \
  --exclude '.venv/' \
  --exclude 'downloads/' \
  "${LOCAL_ROOT}/" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_APP_DIR}/"

echo "Installing Python dependencies and restarting ${SERVICE_NAME}"
ssh -p "${REMOTE_PORT}" "${REMOTE_USER}@${REMOTE_HOST}" \
  "if [ '${USE_SUDO}' = '1' ]; then SUDO='sudo'; else SUDO=''; fi && \
   mkdir -p '${REMOTE_APP_DIR}' && \
   cd '${REMOTE_APP_DIR}' && \
   ./deploy/install_server.sh && \
   \$SUDO install -D -m 0644 deploy/${SERVICE_NAME}.service /etc/systemd/system/${SERVICE_NAME}.service && \
   \$SUDO systemctl daemon-reload && \
   \$SUDO systemctl enable ${SERVICE_NAME} && \
   \$SUDO systemctl restart ${SERVICE_NAME} && \
   \$SUDO systemctl --no-pager --full status ${SERVICE_NAME}"
