from __future__ import annotations

import argparse
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from vllm_omni.version import __version__

from .config import GlobalSchedulerConfig, load_config


def _build_health_payload(config: Any) -> tuple[int, dict[str, Any]]:
    checks = {
        "config_loaded": isinstance(config, GlobalSchedulerConfig),
        "has_instances": False,
        "scheduler_type_valid": False,
    }

    scheduler_type = None
    instance_count = 0
    if isinstance(config, GlobalSchedulerConfig):
        scheduler_type = config.scheduler.type
        instance_count = len(config.instances)
        checks["has_instances"] = instance_count > 0
        checks["scheduler_type_valid"] = scheduler_type in {"baseline_sp1", "ondisc_sp1"}

    healthy = all(checks.values())
    payload: dict[str, Any] = {
        "status": "ok" if healthy else "degraded",
        "scheduler": scheduler_type,
        "instance_count": instance_count,
        "checks": checks,
        "version": __version__,
    }
    return (200 if healthy else 503), payload


def create_app(config: GlobalSchedulerConfig) -> FastAPI:
    app = FastAPI(title="vLLM-Omni Global Scheduler", version=__version__)
    app.state.global_scheduler_config = config

    @app.get("/health")
    async def health() -> JSONResponse:
        status_code, payload = _build_health_payload(getattr(app.state, "global_scheduler_config", None))
        return JSONResponse(
            status_code=status_code,
            content=payload,
        )

    return app


def run_server(config_path: str) -> None:
    config = load_config(config_path)
    app = create_app(config)
    uvicorn.run(app, host=config.server.host, port=config.server.port)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run vLLM-Omni global scheduler server")
    parser.add_argument("--config", required=True, help="Path to global scheduler YAML config")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_server(args.config)


if __name__ == "__main__":
    main()
