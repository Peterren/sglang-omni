#!/usr/bin/env bash
# Tab B: follow every server log launched by the active pytest for one GPU group.
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "usage: watch_calibration_servers.sh <gpu-group> <run-dir> [<run-dir> ...]" >&2
  exit 2
fi

GPU_GROUP="$1"
shift
RUN_DIRS=("$@")
POLL_S="${CALIBRATION_SERVER_WATCH_POLL_S:-1}"

declare -A TAIL_PIDS=()
declare -A ACTIVE_LOGS=()

echo "[Tab B][$GPU_GROUP] dynamic server logs"
echo "[Tab B][$GPU_GROUP] discovers new server.log files under each active pytest basetemp"

stop_tail() {
  local log="$1" pid="${TAIL_PIDS[$1]:-}"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
  unset 'TAIL_PIDS[$log]'
}

cleanup() {
  local log
  for log in "${!TAIL_PIDS[@]}"; do
    stop_tail "$log"
  done
}
trap cleanup EXIT INT TERM

active_basetemps() {
  local run line
  for run in "${RUN_DIRS[@]}"; do
    while IFS= read -r line; do
      [[ "$line" == *" -m pytest "* ]] || continue
      if [[ "$line" =~ --basetemp=([^[:space:]]+) ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
      fi
    done < <(pgrep -af "[p]ython -m pytest.*$(basename "$run")" 2>/dev/null || true)
  done
}

while true; do
  ACTIVE_LOGS=()
  while IFS= read -r basetemp; do
    [[ -d "$basetemp" ]] || continue
    while IFS= read -r log; do
      ACTIVE_LOGS["$log"]=1
      pid="${TAIL_PIDS[$log]:-}"
      if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
        echo "[Tab B][$GPU_GROUP] attach -> $log"
        tail -n +1 -F "$log" | sed -u "s|^|[$(basename "$(dirname "$log")")] |" &
        TAIL_PIDS["$log"]=$!
      fi
    done < <(find "$basetemp" -type f -name 'server.log' -print 2>/dev/null | sort)
  done < <(active_basetemps | sort -u)

  for log in "${!TAIL_PIDS[@]}"; do
    if [[ -z "${ACTIVE_LOGS[$log]:-}" ]]; then
      echo "[Tab B][$GPU_GROUP] detach old server -> $log"
      stop_tail "$log"
    fi
  done
  sleep "$POLL_S"
done
