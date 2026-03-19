# QwenGen Parallel Profiling（与 videoGen 同构）

该目录实现与 `profile/videoGen` 同构的离线并行压测流程：

- 主控：`run_qwen_parallel_bench.py`
- Worker：`offline_profile_worker.py`
- 聚合：`aggregate_qwen_parallel_results.py`
- 请求类型配置：`request_types_18.json`
- 并行矩阵配置：`parallel_matrix.json`

仅替换了默认模型与默认配置路径：

- 默认模型：`/data2/group_谈海生/mumura/models/Qwen-Image`
- 默认请求配置：`profile/qwenGen/request_types_18.json`
- 默认并行矩阵：`profile/qwenGen/parallel_matrix.json`
- 默认输出目录：`profile/qwenGen/results`

## 运行

在仓库根目录执行：

```bash
python profile/qwenGen/run_qwen_parallel_bench.py \
  --card-counts 1,2,4,8 \
  --gpu-device-ids 0,1,2,3,4,5,6,7 \
  --warmup-iters 1 \
  --repeats 3
```

或者：

```bash
bash profile/qwenGen/run_qwen_parallel_bench.sh \
  --card-counts 1,2,4,8 \
  --gpu-device-ids 0,1,2,3,4,5,6,7
```

## 聚合

```bash
python profile/qwenGen/aggregate_qwen_parallel_results.py \
  --summary-csv profile/qwenGen/results/<timestamp>/summary_runs.csv
```
