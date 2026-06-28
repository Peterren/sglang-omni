#!/usr/bin/env bash
# Dependency consistency check for uv-managed Omni CI venvs (no python -m pip).
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <python-path>" >&2
  exit 1
fi

PYTHON="$1"
if [ ! -x "${PYTHON}" ]; then
  echo "python not executable: ${PYTHON}" >&2
  exit 1
fi

if ! uv pip check --python "${PYTHON}" >/dev/null 2>&1; then
  echo "uv pip check reported broken dependencies:" >&2
  uv pip check --python "${PYTHON}" >&2 || true
  exit 1
fi

echo "uv pip check ok for ${PYTHON}"
