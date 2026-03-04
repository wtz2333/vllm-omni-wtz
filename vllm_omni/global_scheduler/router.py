from __future__ import annotations

from vllm_omni.global_scheduler.config import GlobalSchedulerConfig
from vllm_omni.global_scheduler.policies import BaselinePolicy


def build_policy(config: GlobalSchedulerConfig):
    if config.scheduler.type == "baseline_sp1":
        return BaselinePolicy(
            algorithm=config.policy.baseline_sp1.algorithm,
            tie_breaker=config.scheduler.tie_breaker,
        )
    raise ValueError("Unsupported scheduler.type. expected one of: baseline_sp1, ondisc_sp1")
