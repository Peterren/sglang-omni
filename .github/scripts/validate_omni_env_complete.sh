#!/usr/bin/env bash
# Verify a PR-scoped CI home is complete and safe to reuse across workflow runs.
#
# Completeness requires:
#   - allowed OMNI_CI_HOME path
#   - virtualenv import probe (torch, av, whisper normalizer)
#   - .deps-hash matches current pyproject.toml
#   - optional .omni-env-complete marker with matching deps_hash
#
# Runner-side TTL (e.g. 3-day PR home retention) is enforced outside this repo;
# if the home was removed, this script fails and setup rebuilds.
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

DEPS_HASH="$(sha256sum pyproject.toml | awk '{print $1}')"
DEPS_HASH_FILE="${OMNI_CI_HOME}/.deps-hash"
MARKER="${OMNI_CI_HOME}/.omni-env-complete"

if ! bash "${SCRIPT_DIR}/validate_omni_venv_cache.sh" "${VENV_NAME}"; then
  echo "venv incomplete or corrupt under ${OMNI_CI_HOME}/${VENV_NAME}" >&2
  exit 1
fi

if [ ! -f "${DEPS_HASH_FILE}" ]; then
  echo "missing ${DEPS_HASH_FILE}" >&2
  exit 1
fi

STORED_HASH="$(tr -d '[:space:]' < "${DEPS_HASH_FILE}")"
if [ "${STORED_HASH}" != "${DEPS_HASH}" ]; then
  echo "deps-hash mismatch: stored=${STORED_HASH} current=${DEPS_HASH}" >&2
  exit 1
fi

if [ -f "${MARKER}" ]; then
  MARKER_HASH="$(grep -E '^deps_hash=' "${MARKER}" | head -1 | cut -d= -f2- | tr -d '[:space:]')"
  MARKER_VENV="$(grep -E '^venv_name=' "${MARKER}" | head -1 | cut -d= -f2- | tr -d '[:space:]')"
  if [ -n "${MARKER_HASH}" ] && [ "${MARKER_HASH}" != "${DEPS_HASH}" ]; then
    echo "marker deps_hash mismatch: marker=${MARKER_HASH} current=${DEPS_HASH}" >&2
    exit 1
  fi
  if [ -n "${MARKER_VENV}" ] && [ "${MARKER_VENV}" != "${VENV_NAME}" ]; then
    echo "marker venv_name mismatch: marker=${MARKER_VENV} expected=${VENV_NAME}" >&2
    exit 1
  fi
fi

echo "OMNI CI environment complete: ${OMNI_CI_HOME} (venv=${VENV_NAME}, deps_hash=${DEPS_HASH})"
