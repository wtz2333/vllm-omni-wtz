#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG="${SCRIPT_DIR}/resolution_steps_bench_cases.yaml"
CONFIG_PATH="${DEFAULT_CONFIG}"
MODEL_OVERRIDE=""

print_usage() {
  echo "Usage: $0 [config_path] [model_path]"
  echo "   or: $0 --config <config_path> --model <model_path>"
}

while (( $# > 0 )); do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --model)
      MODEL_OVERRIDE="$2"
      shift 2
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      if [[ "${CONFIG_PATH}" == "${DEFAULT_CONFIG}" ]]; then
        CONFIG_PATH="$1"
      elif [[ -z "${MODEL_OVERRIDE}" ]]; then
        MODEL_OVERRIDE="$1"
      else
        echo "Unknown argument: $1" >&2
        print_usage
        exit 1
      fi
      shift
      ;;
  esac
done

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "python is required to parse yaml config." >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required for readiness checks and requests." >&2
  exit 1
fi

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm command not found in PATH." >&2
  exit 1
fi

if ! python -c "import yaml" >/dev/null 2>&1; then
  echo "PyYAML is required. Install with: pip install pyyaml" >&2
  exit 1
fi

readarray -t GLOBAL_VARS < <(
  python - "${CONFIG_PATH}" "${MODEL_OVERRIDE}" <<'PY'
import json
import shlex
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
g = cfg.get("global", {})
model_override = sys.argv[2] if len(sys.argv) > 2 else ""
model = model_override if model_override else str(g.get("model", "Qwen/Qwen-Image"))

print(f"MODEL={shlex.quote(model)}")
print(f"HOST={shlex.quote(str(g.get('host', '127.0.0.1')))}")
print(f"PORT={int(g.get('port', 8091))}")
print(f"READY_PATH={shlex.quote(str(g.get('ready_path', '/v1/models')))}")
print(f"REQUEST_PATH={shlex.quote(str(g.get('request_path', '/v1/images/generations')))}")
print(f"READY_TIMEOUT_SEC={int(g.get('ready_timeout_sec', 600))}")
print(f"POLL_INTERVAL_SEC={int(g.get('poll_interval_sec', 2))}")
print(f"STOP_TIMEOUT_SEC={int(g.get('stop_timeout_sec', 60))}")
print(f"MAX_GPU_PER_CASE={int(g.get('max_gpu_per_case', 8))}")
print(f"LOG_ROOT={shlex.quote(str(g.get('log_root', './image_steps_results')))}")
print(f"PROMPT={shlex.quote(str(g.get('prompt', 'a photo of a cute cat in a garden, ultra detailed')))}")
print(f"NEGATIVE_PROMPT={shlex.quote(str(g.get('negative_prompt', '')))}")
print(f"NUM_OUTPUTS={int(g.get('num_outputs', 1))}")
print(f"WARMUP_RUNS_PER_COMBO={int(g.get('warmup_runs_per_combo', 1))}")
print(f"MEASURED_RUNS_PER_COMBO={int(g.get('measured_runs_per_combo', 5))}")
print(f"REQUEST_TIMEOUT_SEC={int(g.get('request_timeout_sec', 1800))}")
print(f"INTER_RUN_SLEEP_SEC={float(g.get('inter_run_sleep_sec', 0))}")
print(f"SEED_BASE={int(g.get('seed_base', 1234))}")
print(f"USE_FIXED_SEED={str(bool(g.get('use_fixed_seed', False))).lower()}")
print("RESOLUTIONS_JSON=" + shlex.quote(json.dumps(g.get("resolutions", ["512x512", "768x768", "1024x1024"]))))
print("STEP_VALUES_JSON=" + shlex.quote(json.dumps(g.get("step_values", [5, 10, 50]))))
print("EXTRA_CLI_ARGS_JSON=" + shlex.quote(json.dumps(g.get("extra_cli_args", []))))
PY
)

for kv in "${GLOBAL_VARS[@]}"; do
  eval "${kv}"
done

if [[ -n "${MODEL_OVERRIDE}" ]]; then
  echo "Using model override: ${MODEL}"
fi

RUN_ROOT="${SCRIPT_DIR}/${LOG_ROOT}"
mkdir -p "${RUN_ROOT}"

SUMMARY_CSV="${RUN_ROOT}/summary.csv"
if [[ ! -f "${SUMMARY_CSV}" ]]; then
  echo "case_id,case_name,gpu,sp,cfg,tp,vae_use_slicing,vae_use_tiling,resolution,steps,first_startup_s,warmup_runs,measured_runs,latency_mean_s,latency_std_s,status,note" > "${SUMMARY_CSV}"
fi

cleanup_server() {
  local pid="$1"
  local timeout_sec="$2"

  if kill -0 "${pid}" >/dev/null 2>&1; then
    kill -TERM "${pid}" >/dev/null 2>&1 || true
    local elapsed=0
    while kill -0 "${pid}" >/dev/null 2>&1; do
      sleep 1
      elapsed=$((elapsed + 1))
      if (( elapsed >= timeout_sec )); then
        kill -KILL "${pid}" >/dev/null 2>&1 || true
        break
      fi
    done
  fi
}

wait_ready() {
  local url="$1"
  local timeout_sec="$2"
  local interval_sec="$3"
  local start_ns
  start_ns="$(date +%s%N)"
  local waited=0

  while true; do
    local code
    code="$(curl -s -o /dev/null -w '%{http_code}' "${url}" || true)"
    if [[ "${code}" == "200" ]]; then
      local end_ns
      end_ns="$(date +%s%N)"
      python - <<PY
start_ns=${start_ns}
end_ns=${end_ns}
print((end_ns - start_ns) / 1e9)
PY
      return 0
    fi

    echo "  waiting ready: ${url} (http=${code:-NA}, waited=${waited}s/${timeout_sec}s)"

    sleep "${interval_sec}"
    waited=$((waited + interval_sec))
    if (( waited >= timeout_sec )); then
      return 1
    fi
  done
}

build_cmd_line() {
  local model="$1"
  local port="$2"
  local host="$3"
  local log_file="$4"
  local sp="$5"
  local cfg="$6"
  local tp="$7"
  local vae_use_slicing="$8"
  local vae_use_tiling="$9"
  local extra_cli_json="${10}"

  local cmd=(vllm serve --model "${model}" --omni --port "${port}" --host "${host}" --log-file "${log_file}" --ulysses-degree "${sp}" --cfg-parallel-size "${cfg}" --tensor-parallel-size "${tp}")

  if [[ "${vae_use_slicing}" == "true" ]]; then
    cmd+=(--vae-use-slicing)
  fi
  if [[ "${vae_use_tiling}" == "true" ]]; then
    cmd+=(--vae-use-tiling)
  fi

  mapfile -t extra_args < <(
    python - "${extra_cli_json}" <<'PY'
import json
import sys
for item in json.loads(sys.argv[1]):
    print(str(item))
PY
  )

  if (( ${#extra_args[@]} > 0 )); then
    cmd+=("${extra_args[@]}")
  fi

  printf '%q ' "${cmd[@]}"
  echo
}

start_server() {
  local cmd_line="$1"
  local run_log="$2"
  local clear_log="$3"

  if [[ "${clear_log}" == "true" ]]; then
    : > "${run_log}"
  fi

  eval "${cmd_line}" >> "${run_log}" 2>&1 &
  echo "$!"
}

request_once() {
  local endpoint="$1"
  local payload_json="$2"
  local timeout_sec="$3"

  local result
  result="$(curl -sS -X POST "${endpoint}" \
    -H 'Content-Type: application/json' \
    --max-time "${timeout_sec}" \
    --data "${payload_json}" \
    -o /dev/null \
    -w '%{http_code},%{time_total}' || true)"

  if [[ -z "${result}" ]]; then
    echo "000,0"
  else
    echo "${result}"
  fi
}

build_payload_json() {
  local prompt="$1"
  local negative_prompt="$2"
  local size="$3"
  local steps="$4"
  local seed="$5"
  local outputs="$6"

  python - "${prompt}" "${negative_prompt}" "${size}" "${steps}" "${seed}" "${outputs}" <<'PY'
import json
import sys

prompt = sys.argv[1]
negative_prompt = sys.argv[2]
size = sys.argv[3]
steps = int(sys.argv[4])
seed = int(sys.argv[5])
outputs = int(sys.argv[6])

payload = {
    "prompt": prompt,
    "size": size,
    "n": outputs,
    "num_inference_steps": steps,
    "seed": seed,
}
if negative_prompt:
    payload["negative_prompt"] = negative_prompt

print(json.dumps(payload, ensure_ascii=True))
PY
}

calc_latency_stats() {
  local csv_path="$1"
  python - "${csv_path}" <<'PY'
import csv
import statistics
import sys

vals = []
with open(sys.argv[1], newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get("status") == "PASS":
            vals.append(float(row["latency_s"]))

if not vals:
    print("NA,NA")
elif len(vals) == 1:
    print(f"{vals[0]},0.0")
else:
    print(f"{statistics.mean(vals)},{statistics.stdev(vals)}")
PY
}

readarray -t CASE_ROWS < <(
  python - "${CONFIG_PATH}" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
for case in cfg.get("cases", []):
    print("\t".join([
        str(case.get("id", "")),
        str(case.get("name", "")),
        str(bool(case.get("enabled", True))).lower(),
        str(int(case.get("sp", 1))),
        str(int(case.get("cfg", 1))),
        str(int(case.get("tp", 1))),
        str(bool(case.get("vae_use_slicing", False))).lower(),
        str(bool(case.get("vae_use_tiling", False))).lower(),
    ]))
PY
)

readarray -t RESOLUTION_ROWS < <(
  python - "${RESOLUTIONS_JSON}" <<'PY'
import json
import sys
for item in json.loads(sys.argv[1]):
    print(str(item))
PY
)

readarray -t STEP_ROWS < <(
  python - "${STEP_VALUES_JSON}" <<'PY'
import json
import sys
for item in json.loads(sys.argv[1]):
    print(int(item))
PY
)

for row in "${CASE_ROWS[@]}"; do
  IFS=$'\t' read -r case_id case_name enabled sp cfg tp vae_use_slicing vae_use_tiling <<<"${row}"

  if [[ "${enabled}" != "true" ]]; then
    continue
  fi

  gpu_count=$((sp * cfg * tp))
  if (( gpu_count > MAX_GPU_PER_CASE )); then
    echo "[SKIP] case ${case_id} (${case_name}) requires ${gpu_count} GPUs (> ${MAX_GPU_PER_CASE})"
    for size in "${RESOLUTION_ROWS[@]}"; do
      for steps in "${STEP_ROWS[@]}"; do
        echo "${case_id},${case_name},${gpu_count},${sp},${cfg},${tp},${vae_use_slicing},${vae_use_tiling},${size},${steps},,,${MEASURED_RUNS_PER_COMBO},,,SKIP,requires>${MAX_GPU_PER_CASE}gpus" >> "${SUMMARY_CSV}"
      done
    done
    continue
  fi

  case_dir="${RUN_ROOT}/case_${case_id}_${case_name}"
  mkdir -p "${case_dir}"
  run_csv="${case_dir}/runs.csv"
  run_log="${case_dir}/service.log"

  echo "case_id,case_name,gpu,sp,cfg,tp,vae_use_slicing,vae_use_tiling,resolution,steps,run_type,run_idx,seed,http_code,latency_s,status,error" > "${run_csv}"

  cmd_line="$(build_cmd_line "${MODEL}" "${PORT}" "${HOST}" "${run_log}" "${sp}" "${cfg}" "${tp}" "${vae_use_slicing}" "${vae_use_tiling}" "${EXTRA_CLI_ARGS_JSON}")"
  ready_url="http://${HOST}:${PORT}${READY_PATH}"
  infer_url="http://${HOST}:${PORT}${REQUEST_PATH}"

  echo "[CASE ${case_id}] ${case_name}"
  echo "  command: ${cmd_line}"
  echo "  ready check: ${ready_url}"
  echo "  infer endpoint: ${infer_url}"
  echo "  service log: ${run_log}"

  server_pid="$(start_server "${cmd_line}" "${run_log}" "true")"

  first_startup_s=""
  if ! first_startup_s="$(wait_ready "${ready_url}" "${READY_TIMEOUT_SEC}" "${POLL_INTERVAL_SEC}")"; then
    echo "[FAIL] case ${case_id} startup did not become ready in ${READY_TIMEOUT_SEC}s"
    cleanup_server "${server_pid}" "${STOP_TIMEOUT_SEC}" || true
    for size in "${RESOLUTION_ROWS[@]}"; do
      for steps in "${STEP_ROWS[@]}"; do
        echo "${case_id},${case_name},${gpu_count},${sp},${cfg},${tp},${vae_use_slicing},${vae_use_tiling},${size},${steps},,,${MEASURED_RUNS_PER_COMBO},,,FAIL,startup-timeout" >> "${SUMMARY_CSV}"
      done
    done
    continue
  fi

  echo "  first startup ready in ${first_startup_s}s"

  for size in "${RESOLUTION_ROWS[@]}"; do
    for steps in "${STEP_ROWS[@]}"; do
      echo "  [COMBO] size=${size} steps=${steps}"

      combo_failed=false
      combo_note=""

      for ((w = 1; w <= WARMUP_RUNS_PER_COMBO; w++)); do
        if [[ "${USE_FIXED_SEED}" == "true" ]]; then
          seed="${SEED_BASE}"
        else
          seed=$((SEED_BASE + w))
        fi
        payload_json="$(build_payload_json "${PROMPT}" "${NEGATIVE_PROMPT}" "${size}" "${steps}" "${seed}" "${NUM_OUTPUTS}")"
        result="$(request_once "${infer_url}" "${payload_json}" "${REQUEST_TIMEOUT_SEC}")"
        IFS=',' read -r http_code latency_s <<<"${result}"

        status="PASS"
        err=""
        if [[ "${http_code}" != "200" ]]; then
          status="FAIL"
          err="warmup-http-${http_code}"
          combo_failed=true
          combo_note="${err}"
        fi

        echo "${case_id},${case_name},${gpu_count},${sp},${cfg},${tp},${vae_use_slicing},${vae_use_tiling},${size},${steps},warmup,${w},${seed},${http_code},${latency_s},${status},${err}" >> "${run_csv}"
        echo "    warmup ${w}/${WARMUP_RUNS_PER_COMBO}: http=${http_code} latency=${latency_s}s"

        if [[ "${combo_failed}" == "true" ]]; then
          break
        fi
        sleep "${INTER_RUN_SLEEP_SEC}"
      done

      if [[ "${combo_failed}" == "true" ]]; then
        echo "${case_id},${case_name},${gpu_count},${sp},${cfg},${tp},${vae_use_slicing},${vae_use_tiling},${size},${steps},${first_startup_s},${WARMUP_RUNS_PER_COMBO},${MEASURED_RUNS_PER_COMBO},,,FAIL,${combo_note}" >> "${SUMMARY_CSV}"
        continue
      fi

      combo_csv="${case_dir}/combo_${size}_${steps}.csv"
      echo "run_idx,seed,http_code,latency_s,status,error" > "${combo_csv}"

      for ((run_idx = 1; run_idx <= MEASURED_RUNS_PER_COMBO; run_idx++)); do
        if [[ "${USE_FIXED_SEED}" == "true" ]]; then
          seed="${SEED_BASE}"
        else
          seed=$((SEED_BASE + WARMUP_RUNS_PER_COMBO + run_idx))
        fi

        payload_json="$(build_payload_json "${PROMPT}" "${NEGATIVE_PROMPT}" "${size}" "${steps}" "${seed}" "${NUM_OUTPUTS}")"
        result="$(request_once "${infer_url}" "${payload_json}" "${REQUEST_TIMEOUT_SEC}")"
        IFS=',' read -r http_code latency_s <<<"${result}"

        status="PASS"
        err=""
        if [[ "${http_code}" != "200" ]]; then
          status="FAIL"
          err="http-${http_code}"
          combo_failed=true
          combo_note="${err}"
        fi

        echo "${run_idx},${seed},${http_code},${latency_s},${status},${err}" >> "${combo_csv}"
        echo "${case_id},${case_name},${gpu_count},${sp},${cfg},${tp},${vae_use_slicing},${vae_use_tiling},${size},${steps},measure,${run_idx},${seed},${http_code},${latency_s},${status},${err}" >> "${run_csv}"
        echo "    measure ${run_idx}/${MEASURED_RUNS_PER_COMBO}: http=${http_code} latency=${latency_s}s"

        if [[ "${combo_failed}" == "true" ]]; then
          break
        fi
        sleep "${INTER_RUN_SLEEP_SEC}"
      done

      if [[ "${combo_failed}" == "true" ]]; then
        echo "${case_id},${case_name},${gpu_count},${sp},${cfg},${tp},${vae_use_slicing},${vae_use_tiling},${size},${steps},${first_startup_s},${WARMUP_RUNS_PER_COMBO},${MEASURED_RUNS_PER_COMBO},,,FAIL,${combo_note}" >> "${SUMMARY_CSV}"
        continue
      fi

      stats="$(calc_latency_stats "${combo_csv}")"
      IFS=',' read -r latency_mean latency_std <<<"${stats}"
      echo "${case_id},${case_name},${gpu_count},${sp},${cfg},${tp},${vae_use_slicing},${vae_use_tiling},${size},${steps},${first_startup_s},${WARMUP_RUNS_PER_COMBO},${MEASURED_RUNS_PER_COMBO},${latency_mean},${latency_std},PASS," >> "${SUMMARY_CSV}"
      echo "  [PASS] size=${size} steps=${steps} mean=${latency_mean}s std=${latency_std}s"
    done
  done

  cleanup_server "${server_pid}" "${STOP_TIMEOUT_SEC}" || true
  echo "[CASE ${case_id}] done"
done

echo "Done. summary: ${SUMMARY_CSV}"
