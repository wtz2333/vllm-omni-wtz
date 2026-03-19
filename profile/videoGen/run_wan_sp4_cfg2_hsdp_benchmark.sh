#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BENCH_SCRIPT="${REPO_ROOT}/benchmarks/diffusion/diffusion_benchmark_serving.py"

CONFIG_NAME="sp4_cfg2_hsdp"
BASE_URL="http://localhost:8091"
MODEL=""
BACKEND="vllm-omni"
TASK="t2v"
DATASET="trace"
DATASET_PATH="/home/mumura/omni/dataset/trace/cogvideox_trace.txt/cogvideox_trace.txt"

DURATION_SECONDS=1800
RPS_LIST="0.01,0.1,1"
MAX_CONCURRENCY=1

WARMUP_REQUESTS=1
WARMUP_NUM_INFERENCE_STEPS=1
WIDTH=""
HEIGHT=""
NUM_FRAMES=""
NUM_INFERENCE_STEPS=50
FPS=""
SEED=""

SLO_ENABLED=0
SLO_SCALE=3.0

OUTPUT_ROOT="${REPO_ROOT}/profile/videoGen/results"
RUN_TAG=""

usage() {
  cat <<'EOF'
Usage:
  bash profile/videoGen/run_wan_sp4_cfg2_hsdp_benchmark.sh --model <MODEL> [options]

Required:
  --model <MODEL>                 Model name passed to diffusion benchmark.

Common options:
  --base-url <URL>                Default: http://localhost:8091
  --duration-seconds <SEC>        Default: 1800 (30 min)
  --rps-list <LIST>               Comma-separated, default: 0.01,0.1,1
  --dataset-path <PATH>           Default: dataset/trace/cogvideox_trace.txt/cogvideox_trace.txt
  --output-root <DIR>             Default: profile/videoGen/results
  --run-tag <TAG>                 Optional custom run tag
  --max-concurrency <N>           Default: 1

Optional benchmark args passthrough:
  --task <TASK>                   Default: t2v
  --backend <BACKEND>             Default: vllm-omni
  --warmup-requests <N>           Default: 1
  --warmup-num-inference-steps <N>
  --num-inference-steps <N>
  --width <W> --height <H> --num-frames <F> --fps <FPS> --seed <SEED>
  --slo                           Enable SLO metrics
  --slo-scale <FLOAT>             Default: 3.0

Example:
  bash profile/videoGen/run_wan_sp4_cfg2_hsdp_benchmark.sh \
    --model /data2/group_谈海生/mumura/models/Wan2.2-T2V-A14B-Diffusers \
    --base-url http://localhost:8091 \
    --duration-seconds 1800 \
    --rps-list 0.01,0.1,1 \
    --max-concurrency 10000 
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --base-url)
      BASE_URL="$2"
      shift 2
      ;;
    --backend)
      BACKEND="$2"
      shift 2
      ;;
    --task)
      TASK="$2"
      shift 2
      ;;
    --dataset-path)
      DATASET_PATH="$2"
      shift 2
      ;;
    --duration-seconds)
      DURATION_SECONDS="$2"
      shift 2
      ;;
    --rps-list)
      RPS_LIST="$2"
      shift 2
      ;;
    --max-concurrency)
      MAX_CONCURRENCY="$2"
      shift 2
      ;;
    --warmup-requests)
      WARMUP_REQUESTS="$2"
      shift 2
      ;;
    --warmup-num-inference-steps)
      WARMUP_NUM_INFERENCE_STEPS="$2"
      shift 2
      ;;
    --num-inference-steps)
      NUM_INFERENCE_STEPS="$2"
      shift 2
      ;;
    --width)
      WIDTH="$2"
      shift 2
      ;;
    --height)
      HEIGHT="$2"
      shift 2
      ;;
    --num-frames)
      NUM_FRAMES="$2"
      shift 2
      ;;
    --fps)
      FPS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --slo)
      SLO_ENABLED=1
      shift
      ;;
    --slo-scale)
      SLO_SCALE="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --run-tag)
      RUN_TAG="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${MODEL}" ]]; then
  echo "Error: --model is required."
  usage
  exit 1
fi

if [[ ! -f "${BENCH_SCRIPT}" ]]; then
  echo "Error: benchmark script not found: ${BENCH_SCRIPT}"
  exit 1
fi

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Error: dataset path not found: ${DATASET_PATH}"
  exit 1
fi

if [[ -z "${RUN_TAG}" ]]; then
  RUN_TAG="$(date +%Y%m%d_%H%M%S)"
fi

RUN_DIR="${OUTPUT_ROOT}/wan_${CONFIG_NAME}_${RUN_TAG}"
mkdir -p "${RUN_DIR}"

MASTER_LOG="${RUN_DIR}/benchmark.log"
CSV_FILE="${RUN_DIR}/summary.csv"
RUN_CONFIG_JSON="${RUN_DIR}/run_config.json"

cat > "${RUN_CONFIG_JSON}" <<EOF
{
  "config_name": "${CONFIG_NAME}",
  "base_url": "${BASE_URL}",
  "model": "${MODEL}",
  "backend": "${BACKEND}",
  "dataset": "${DATASET}",
  "dataset_path": "${DATASET_PATH}",
  "task": "${TASK}",
  "duration_seconds": ${DURATION_SECONDS},
  "rps_list": "${RPS_LIST}",
  "max_concurrency": ${MAX_CONCURRENCY},
  "warmup_requests": ${WARMUP_REQUESTS},
  "warmup_num_inference_steps": ${WARMUP_NUM_INFERENCE_STEPS},
  "num_inference_steps": ${NUM_INFERENCE_STEPS},
  "slo_enabled": ${SLO_ENABLED},
  "slo_scale": ${SLO_SCALE}
}
EOF

echo "[$(date '+%F %T')] Run dir: ${RUN_DIR}" | tee -a "${MASTER_LOG}"
echo "[$(date '+%F %T')] Master log: ${MASTER_LOG}" | tee -a "${MASTER_LOG}"

printf '%s\n' "timestamp,config_name,backend,base_url,model,dataset,dataset_path,task,rps,duration_seconds,num_prompts,completed_requests,failed_requests,throughput_qps,latency_mean,latency_median,latency_p99,latency_p50,peak_memory_mb_max,peak_memory_mb_mean,peak_memory_mb_median,slo_attainment_rate,slo_met_success,slo_scale,metrics_json,log_file,exit_code" > "${CSV_FILE}"

rps_to_num_prompts() {
  local duration="$1"
  local rps="$2"
  python3 - <<PY
import math
duration = float(${duration})
rps = float(${rps})
print(max(1, int(math.ceil(duration * rps))))
PY
}

append_csv_row() {
  local metrics_json="$1"
  local run_log="$2"
  local rps="$3"
  local num_prompts="$4"
  local exit_code="$5"

  python3 - <<PY
import csv
import json
from datetime import datetime

csv_file = ${CSV_FILE@Q}
metrics_json = ${metrics_json@Q}
run_log = ${run_log@Q}
rps = ${rps@Q}
num_prompts = int(${num_prompts})
exit_code = int(${exit_code})

with open(metrics_json, "r", encoding="utf-8") as f:
    m = json.load(f)

row = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "config_name": ${CONFIG_NAME@Q},
    "backend": m.get("backend", ${BACKEND@Q}),
    "base_url": ${BASE_URL@Q},
    "model": m.get("model", ${MODEL@Q}),
    "dataset": m.get("dataset", ${DATASET@Q}),
    "dataset_path": ${DATASET_PATH@Q},
    "task": m.get("task", ${TASK@Q}),
    "rps": rps,
    "duration_seconds": ${DURATION_SECONDS},
    "num_prompts": num_prompts,
    "completed_requests": m.get("completed_requests", ""),
    "failed_requests": m.get("failed_requests", ""),
    "throughput_qps": m.get("throughput_qps", ""),
    "latency_mean": m.get("latency_mean", ""),
    "latency_median": m.get("latency_median", ""),
    "latency_p99": m.get("latency_p99", ""),
    "latency_p50": m.get("latency_p50", ""),
    "peak_memory_mb_max": m.get("peak_memory_mb_max", ""),
    "peak_memory_mb_mean": m.get("peak_memory_mb_mean", ""),
    "peak_memory_mb_median": m.get("peak_memory_mb_median", ""),
    "slo_attainment_rate": m.get("slo_attainment_rate", ""),
    "slo_met_success": m.get("slo_met_success", ""),
    "slo_scale": m.get("slo_scale", ""),
    "metrics_json": metrics_json,
    "log_file": run_log,
    "exit_code": exit_code,
}

fields = [
    "timestamp","config_name","backend","base_url","model","dataset","dataset_path","task",
    "rps","duration_seconds","num_prompts","completed_requests","failed_requests","throughput_qps",
    "latency_mean","latency_median","latency_p99","latency_p50","peak_memory_mb_max",
    "peak_memory_mb_mean","peak_memory_mb_median","slo_attainment_rate","slo_met_success","slo_scale",
    "metrics_json","log_file","exit_code"
]

with open(csv_file, "a", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writerow(row)
PY
}

IFS=',' read -r -a RPS_ARRAY <<< "${RPS_LIST}"

for raw_rps in "${RPS_ARRAY[@]}"; do
  RPS="$(echo "${raw_rps}" | xargs)"
  if [[ -z "${RPS}" ]]; then
    continue
  fi

  NUM_PROMPTS="$(rps_to_num_prompts "${DURATION_SECONDS}" "${RPS}")"
  RPS_SAFE="${RPS//./_}"
  RUN_LOG="${RUN_DIR}/run_rps_${RPS_SAFE}.log"
  METRICS_JSON="${RUN_DIR}/metrics_rps_${RPS_SAFE}.json"

  {
    echo "[$(date '+%F %T')] [START] config=${CONFIG_NAME} rps=${RPS} duration=${DURATION_SECONDS}s num_prompts=${NUM_PROMPTS}"
    echo "[$(date '+%F %T')] Command: python3 ${BENCH_SCRIPT} --base-url ${BASE_URL} --model ${MODEL} --backend ${BACKEND} --dataset ${DATASET} --task ${TASK} --dataset-path ${DATASET_PATH} --num-prompts ${NUM_PROMPTS} --request-rate ${RPS} --max-concurrency ${MAX_CONCURRENCY} --warmup-requests ${WARMUP_REQUESTS} --warmup-num-inference-steps ${WARMUP_NUM_INFERENCE_STEPS} --num-inference-steps ${NUM_INFERENCE_STEPS} --output-file ${METRICS_JSON} --disable-tqdm"
  } | tee -a "${MASTER_LOG}" >> "${RUN_LOG}"

  set +e
  CMD=(
    python3 "${BENCH_SCRIPT}"
    --base-url "${BASE_URL}"
    --model "${MODEL}"
    --backend "${BACKEND}"
    --dataset "${DATASET}"
    --task "${TASK}"
    --dataset-path "${DATASET_PATH}"
    --num-prompts "${NUM_PROMPTS}"
    --request-rate "${RPS}"
    --max-concurrency "${MAX_CONCURRENCY}"
    --warmup-requests "${WARMUP_REQUESTS}"
    --warmup-num-inference-steps "${WARMUP_NUM_INFERENCE_STEPS}"
    --num-inference-steps "${NUM_INFERENCE_STEPS}"
    --output-file "${METRICS_JSON}"
    --disable-tqdm
  )

  if [[ -n "${WIDTH}" ]]; then CMD+=(--width "${WIDTH}"); fi
  if [[ -n "${HEIGHT}" ]]; then CMD+=(--height "${HEIGHT}"); fi
  if [[ -n "${NUM_FRAMES}" ]]; then CMD+=(--num-frames "${NUM_FRAMES}"); fi
  if [[ -n "${FPS}" ]]; then CMD+=(--fps "${FPS}"); fi
  if [[ -n "${SEED}" ]]; then CMD+=(--seed "${SEED}"); fi
  if [[ "${SLO_ENABLED}" -eq 1 ]]; then CMD+=(--slo --slo-scale "${SLO_SCALE}"); fi

  "${CMD[@]}" >> "${RUN_LOG}" 2>&1
  EXIT_CODE=$?
  set -e

  if [[ -f "${METRICS_JSON}" ]]; then
    append_csv_row "${METRICS_JSON}" "${RUN_LOG}" "${RPS}" "${NUM_PROMPTS}" "${EXIT_CODE}"
  else
    echo "[$(date '+%F %T')] [WARN] Metrics JSON not found for rps=${RPS}: ${METRICS_JSON}" | tee -a "${MASTER_LOG}" >> "${RUN_LOG}"
  fi

  echo "[$(date '+%F %T')] [END] rps=${RPS} exit_code=${EXIT_CODE} metrics=${METRICS_JSON}" | tee -a "${MASTER_LOG}" >> "${RUN_LOG}"
done

echo "[$(date '+%F %T')] [DONE] CSV: ${CSV_FILE}" | tee -a "${MASTER_LOG}"
echo "[$(date '+%F %T')] [DONE] LOG: ${MASTER_LOG}" | tee -a "${MASTER_LOG}"
