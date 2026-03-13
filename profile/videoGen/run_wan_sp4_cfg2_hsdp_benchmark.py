#!/usr/bin/env python3

import argparse
import asyncio
import csv
import json
import os
import sys
import time
import uuid
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any


class Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed-duration WAN serving benchmark for sp4_cfg2_hsdp across multiple RPS values, "
            "save full logs, and export CSV summary."
        )
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--backend", type=str, default="vllm-omni", choices=["vllm-omni", "openai"])
    parser.add_argument("--model", type=str, required=True, help="Serving model name exposed by the endpoint.")

    parser.add_argument("--dataset", type=str, default="trace", choices=["trace", "vbench", "random"])
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="dataset/trace/cogvideox_trace.txt/cogvideox_trace.txt",
        help="Path for trace/vbench dataset.",
    )
    parser.add_argument("--task", type=str, default="t2v", choices=["t2v", "i2v", "ti2v", "ti2i", "i2i", "t2i"])
    parser.add_argument("--num-prompts", type=int, default=None, help="Optional cap when loading dataset items.")

    parser.add_argument("--duration-seconds", type=int, default=1800, help="Per-RPS sending duration in seconds.")
    parser.add_argument("--rps-list", type=str, default="0.01,0.1,1", help="Comma-separated request rates.")
    parser.add_argument("--max-concurrency", type=int, default=1)

    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--warmup-num-inference-steps", type=int, default=1)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)

    parser.add_argument("--slo", action="store_true")
    parser.add_argument("--slo-scale", type=float, default=3.0)

    parser.add_argument("--config-name", type=str, default="sp4_cfg2_hsdp")
    parser.add_argument("--output-root", type=str, default="profile/videoGen/results")
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--disable-console-tee", action="store_true")
    return parser.parse_args()


def _parse_rps_list(raw: str) -> list[float]:
    values: list[float] = []
    for part in raw.split(","):
        text = part.strip().lower()
        if not text:
            continue
        if text in {"inf", "infinity"}:
            values.append(float("inf"))
            continue
        value = float(text)
        if value <= 0:
            raise ValueError(f"RPS must be positive, got {value}.")
        values.append(value)
    if not values:
        raise ValueError("No valid rps values were provided.")
    return values


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    output_root = (repo_root / args.output_root).resolve()
    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"wan_{args.config_name}_{run_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return repo_root, run_dir, run_dir / "benchmark.log"


def _load_benchmark_module(repo_root: Path):
    bench_dir = repo_root / "benchmarks" / "diffusion"
    if str(bench_dir) not in sys.path:
        sys.path.insert(0, str(bench_dir))
    import diffusion_benchmark_serving as dbs  # noqa: PLC0415

    return dbs


async def _run_single_rps(
    dbs,
    args: argparse.Namespace,
    requests_seed: list[Any],
    request_func,
    rps: float,
) -> dict[str, Any]:
    import aiohttp

    if args.max_concurrency is not None and args.max_concurrency > 0:
        semaphore = asyncio.Semaphore(args.max_concurrency)
    else:
        semaphore = None

    async def limited_request(req, session):
        if semaphore:
            async with semaphore:
                return await request_func(req, session, None)
        return await request_func(req, session, None)

    api_start = time.perf_counter()
    sent_requests: list[Any] = []
    tasks: list[asyncio.Task] = []
    warmup_pairs: list[tuple[Any, Any]] = []

    async with aiohttp.ClientSession() as session:
        if args.warmup_requests > 0 and requests_seed:
            print(
                f"[RPS={rps}] warmup requests={args.warmup_requests}, "
                f"warmup_num_inference_steps={args.warmup_num_inference_steps}"
            )
            for i in range(args.warmup_requests):
                warm_req = requests_seed[i % len(requests_seed)]
                if args.warmup_num_inference_steps is not None:
                    warm_req = replace(warm_req, num_inference_steps=args.warmup_num_inference_steps)
                warm_out = await limited_request(warm_req, session)
                warmup_pairs.append((warm_req, warm_out))

        if args.slo:
            requests_seed = dbs._populate_slo_ms_from_warmups(
                requests_list=requests_seed,
                warmup_pairs=warmup_pairs,
                args=args,
            )

        deadline = time.perf_counter() + float(args.duration_seconds)
        next_emit_time = time.perf_counter()
        req_idx = 0

        while True:
            now = time.perf_counter()
            if now >= deadline:
                break

            if rps != float("inf"):
                sleep_s = max(0.0, next_emit_time - now)
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)
                now = time.perf_counter()
                if now >= deadline:
                    break

            base_req = requests_seed[req_idx % len(requests_seed)]
            req_idx += 1
            req = replace(base_req, request_id=str(uuid.uuid4()))
            sent_requests.append(req)
            tasks.append(asyncio.create_task(limited_request(req, session)))

            if rps != float("inf"):
                next_emit_time += 1.0 / float(rps)

        outputs = await asyncio.gather(*tasks) if tasks else []

    total_duration = time.perf_counter() - api_start
    metrics = dbs.calculate_metrics(outputs, total_duration, sent_requests, args, args.slo)

    if sent_requests and outputs:
        raw_send_duration = max(min(total_duration, float(args.duration_seconds)), 1e-9)
        achieved_send_rps = len(sent_requests) / raw_send_duration
    else:
        achieved_send_rps = 0.0

    metrics.update(
        {
            "configured_rps": rps,
            "duration_seconds": float(args.duration_seconds),
            "requests_sent": len(sent_requests),
            "achieved_send_rps": achieved_send_rps,
            "backend": args.backend,
            "model": args.model,
            "dataset": args.dataset,
            "task": args.task,
            "config_name": args.config_name,
        }
    )
    return metrics


def _write_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    fields = [
        "timestamp",
        "config_name",
        "backend",
        "base_url",
        "model",
        "dataset",
        "dataset_path",
        "task",
        "configured_rps",
        "duration_seconds",
        "requests_sent",
        "completed_requests",
        "failed_requests",
        "achieved_send_rps",
        "throughput_qps",
        "latency_mean",
        "latency_median",
        "latency_p99",
        "latency_p50",
        "slo_attainment_rate",
        "slo_met_success",
        "slo_scale",
        "peak_memory_mb_max",
        "peak_memory_mb_mean",
        "peak_memory_mb_median",
        "duration",
        "metrics_json",
        "log_file",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


async def _async_main(args: argparse.Namespace, repo_root: Path, run_dir: Path, log_file: Path) -> int:
    dbs = _load_benchmark_module(repo_root)

    if args.base_url is None:
        args.base_url = f"http://{args.host}:{args.port}"

    request_func, api_suffix = dbs.backends_function_mapping[args.backend]
    api_url = f"{args.base_url}{api_suffix}"

    if args.dataset == "vbench":
        dataset = dbs.VBenchDataset(args, api_url, args.model)
    elif args.dataset == "trace":
        dataset = dbs.TraceDataset(args, api_url, args.model)
    elif args.dataset == "random":
        dataset = dbs.RandomDataset(args, api_url, args.model)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    requests_seed = dataset.get_requests()
    if not requests_seed:
        raise ValueError("Loaded dataset is empty.")

    rps_values = _parse_rps_list(args.rps_list)
    print(f"[Init] config_name={args.config_name}")
    print(f"[Init] model={args.model}")
    print(f"[Init] base_url={args.base_url}")
    print(f"[Init] dataset={args.dataset}, dataset_path={args.dataset_path}, task={args.task}")
    print(f"[Init] loaded_requests={len(requests_seed)}, rps_values={rps_values}")
    print(f"[Init] duration_seconds_per_rps={args.duration_seconds}")

    all_rows: list[dict[str, Any]] = []
    csv_path = run_dir / "summary.csv"

    for rps in rps_values:
        rps_label = "inf" if rps == float("inf") else str(rps)
        print(f"[Run] start rps={rps_label}")
        metrics = await _run_single_rps(
            dbs=dbs,
            args=args,
            requests_seed=requests_seed,
            request_func=request_func,
            rps=rps,
        )

        metrics_json_path = run_dir / f"metrics_rps_{rps_label.replace('.', '_')}.json"
        metrics_json_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

        print(
            "[Run] done rps={} sent={} success={} failed={} throughput={:.4f} qps "
            "lat_mean={:.4f}s lat_p99={:.4f}s".format(
                rps_label,
                metrics.get("requests_sent", 0),
                metrics.get("completed_requests", 0),
                metrics.get("failed_requests", 0),
                float(metrics.get("throughput_qps", 0.0)),
                float(metrics.get("latency_mean", 0.0)),
                float(metrics.get("latency_p99", 0.0)),
            )
        )

        all_rows.append(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "config_name": args.config_name,
                "backend": args.backend,
                "base_url": args.base_url,
                "model": args.model,
                "dataset": args.dataset,
                "dataset_path": args.dataset_path,
                "task": args.task,
                "configured_rps": metrics.get("configured_rps", rps),
                "duration_seconds": metrics.get("duration_seconds", args.duration_seconds),
                "requests_sent": metrics.get("requests_sent", 0),
                "completed_requests": metrics.get("completed_requests", 0),
                "failed_requests": metrics.get("failed_requests", 0),
                "achieved_send_rps": metrics.get("achieved_send_rps", 0.0),
                "throughput_qps": metrics.get("throughput_qps", 0.0),
                "latency_mean": metrics.get("latency_mean", 0.0),
                "latency_median": metrics.get("latency_median", 0.0),
                "latency_p99": metrics.get("latency_p99", 0.0),
                "latency_p50": metrics.get("latency_p50", 0.0),
                "slo_attainment_rate": metrics.get("slo_attainment_rate", ""),
                "slo_met_success": metrics.get("slo_met_success", ""),
                "slo_scale": metrics.get("slo_scale", args.slo_scale),
                "peak_memory_mb_max": metrics.get("peak_memory_mb_max", 0.0),
                "peak_memory_mb_mean": metrics.get("peak_memory_mb_mean", 0.0),
                "peak_memory_mb_median": metrics.get("peak_memory_mb_median", 0.0),
                "duration": metrics.get("duration", 0.0),
                "metrics_json": str(metrics_json_path),
                "log_file": str(log_file),
            }
        )
        _write_csv(all_rows, csv_path)

    print(f"[Done] summary csv: {csv_path}")
    return 0


def main() -> int:
    args = parse_args()
    repo_root, run_dir, log_file = _resolve_paths(args)

    if args.dataset_path and not os.path.isabs(args.dataset_path):
        args.dataset_path = str((repo_root / args.dataset_path).resolve())

    plan_json = run_dir / "run_config.json"
    plan_json.write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")

    with log_file.open("w", encoding="utf-8") as lf:
        if args.disable_console_tee:
            tee_out = lf
        else:
            tee_out = Tee(sys.stdout, lf)

        with redirect_stdout(tee_out), redirect_stderr(tee_out):
            print(f"[Log] file={log_file}")
            print(f"[RunDir] {run_dir}")
            return asyncio.run(_async_main(args, repo_root, run_dir, log_file))


if __name__ == "__main__":
    raise SystemExit(main())
