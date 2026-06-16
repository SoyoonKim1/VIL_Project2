#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SAVE_DIR="${SAVE_DIR:-./outputs/swav_stl10_gpu0_seed42}"
LOGDIR="${LOGDIR:-${SAVE_DIR}/tensorboard}"
PORT="${PORT:-6006}"
HOST="${HOST:-0.0.0.0}"

echo "Starting TensorBoard..."
echo "  logdir: ${LOGDIR}"
echo "  host:   ${HOST}"
echo "  port:   ${PORT}"

tensorboard --logdir "${LOGDIR}" --host "${HOST}" --port "${PORT}"
