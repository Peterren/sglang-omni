#!/usr/bin/env bash
# Strict gate for reusing an existing OMNI_CI_HOME venv across workflow runs.
#
# Checks path safety, pyproject.toml fingerprint, import probe, and exact == pins.
# Does not run uv pip check (override-dependencies produce expected resolver warnings).
# Does not require .omni-env-complete (downstream jobs use this gate; setup writes marker).
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <venv-name>" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_NAME="$1"

bash "${SCRIPT_DIR}/validate_omni_ci_home.sh"

if [ ! -f pyproject.toml ]; then
  echo "pyproject.toml not found in $(pwd); run from repository root" >&2
  exit 1
fi

# shellcheck source=omni_ci_deps_hash.sh
source "${SCRIPT_DIR}/omni_ci_deps_hash.sh"
DEPS_HASH="$(omni_ci_deps_hash)"
DEPS_HASH_FILE="${OMNI_CI_HOME}/.deps-hash"

if [ ! -f "${DEPS_HASH_FILE}" ]; then
  echo "missing ${DEPS_HASH_FILE}" >&2
  exit 1
fi

STORED_HASH="$(tr -d '[:space:]' < "${DEPS_HASH_FILE}")"
if [ "${STORED_HASH}" != "${DEPS_HASH}" ]; then
  echo "deps-hash mismatch: stored=${STORED_HASH} current=${DEPS_HASH}" >&2
  echo "pyproject.toml changed; full environment rebuild required" >&2
  exit 1
fi

if ! bash "${SCRIPT_DIR}/validate_omni_venv_imports.sh" "${VENV_NAME}"; then
  exit 1
fi

if ! bash "${SCRIPT_DIR}/verify_omni_installed_pins.sh" "${VENV_NAME}"; then
  echo "installed dependency pins do not match pyproject.toml" >&2
  exit 1
fi

# uv pip check is not a CI gate: this stack uses [tool.uv] override-dependencies
# (e.g. protobuf>=6.x) that uv resolves intentionally but pip check still flags.

echo "OMNI CI environment reusable: ${OMNI_CI_HOME} (venv=${VENV_NAME}, deps_hash=${DEPS_HASH})"
