#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_RUNNER="${SCRIPT_DIR}/run_qwen_parallel_bench.py"

if [[ ! -f "${PY_RUNNER}" ]]; then
  echo "Runner not found: ${PY_RUNNER}" >&2
  exit 1
fi

exec python3 "${PY_RUNNER}" "$@"
