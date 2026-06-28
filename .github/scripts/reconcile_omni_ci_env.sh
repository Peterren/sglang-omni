#!/usr/bin/env bash
# Align OMNI_CI_HOME with the checked-out tree: refresh if reusable, else rebuild.
#
# Always exits 0 only when validate_omni_env_reusable passes.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <venv-name>" >&2
  exit 1
fi

if [ -z "${OMNI_CI_HOME:-}" ]; then
  echo "OMNI_CI_HOME is not set" >&2
  exit 1
fi

VENV_NAME="$1"
HOST="${OMNI_CI_HOME}/${VENV_NAME}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

_link_workspace_venv() {
  rm -rf "./${VENV_NAME}"
  ln -sfn "${HOST}" "./${VENV_NAME}"
  source "${VENV_NAME}/bin/activate"
}

_refresh_editable_install() {
  _link_workspace_venv
  uv pip install --upgrade -e .
}

if bash "${SCRIPT_DIR}/validate_omni_env_reusable.sh" "${VENV_NAME}"; then
  echo "Reusing ${HOST}; refreshing editable install"
  _refresh_editable_install
  if bash "${SCRIPT_DIR}/validate_omni_env_reusable.sh" "${VENV_NAME}"; then
    echo "Environment reuse succeeded for ${OMNI_CI_HOME}"
    exit 0
  fi
  echo "Environment drift detected after refresh; rebuilding ${OMNI_CI_HOME}"
fi

bash "${SCRIPT_DIR}/prepare_omni_venv.sh" "${VENV_NAME}"
