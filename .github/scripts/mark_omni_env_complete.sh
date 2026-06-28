#!/usr/bin/env bash
# Record that setup finished successfully for this PR-scoped CI home.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <venv-name>" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_NAME="$1"

bash "${SCRIPT_DIR}/validate_omni_env_complete.sh" "${VENV_NAME}"

DEPS_HASH="$(sha256sum pyproject.toml | awk '{print $1}')"
MARKER="${OMNI_CI_HOME}/.omni-env-complete"
MARKED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

cat > "${MARKER}" <<EOF
deps_hash=${DEPS_HASH}
venv_name=${VENV_NAME}
marked_at=${MARKED_AT}
EOF

echo "Marked OMNI CI environment complete: ${MARKER}"
