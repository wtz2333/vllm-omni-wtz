from __future__ import annotations

from vllm_omni.global_scheduler.policies.base import BasePolicy
from vllm_omni.global_scheduler.policies.baseline_fcfs import BaselineFCFSPolicy
from vllm_omni.global_scheduler.types import InstanceSpec, RequestMeta, RouteDecision, RuntimeStats


class BaselinePolicy(BasePolicy):
    def __init__(self, algorithm: str, tie_breaker: str = "random") -> None:
        super().__init__(tie_breaker=tie_breaker)
        self._algorithm = algorithm
        self._delegate: BasePolicy
        if algorithm == "fcfs":
            self._delegate = BaselineFCFSPolicy(tie_breaker=tie_breaker)
        else:
            raise ValueError(
                "Unsupported baseline algorithm. expected one of: fcfs, short_queue_runtime, estimated_completion_time"
            )

    def select_instance(
        self,
        request: RequestMeta,
        instances: list[InstanceSpec],
        runtime_stats: dict[str, RuntimeStats],
    ) -> RouteDecision:
        decision = self._delegate.select_instance(request=request, instances=instances, runtime_stats=runtime_stats)
        decision.reason = f"algorithm={self._algorithm};{decision.reason}"
        return decision
