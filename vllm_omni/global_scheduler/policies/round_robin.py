from __future__ import annotations

from vllm_omni.global_scheduler.policies.policy_base import PolicyBase
from vllm_omni.global_scheduler.types import InstanceSpec, RequestMeta, RouteDecision, RuntimeStats


class RoundRobinPolicy(PolicyBase):
    """Route requests in round-robin order among available instances."""

    def __init__(self, tie_breaker: str = "random") -> None:
        super().__init__(tie_breaker=tie_breaker)
        self._cursor = 0

    def select_instance(
        self,
        request: RequestMeta,
        instances: list[InstanceSpec],
        runtime_stats: dict[str, RuntimeStats],
    ) -> RouteDecision:
        """Select next instance by rotating cursor.

        Prefers available instances. If all are busy, still rotates across all
        instances to preserve fairness rather than collapsing to a fixed host.
        """
        del request
        if not instances:
            raise ValueError("No instances configured")

        total = len(instances)
        available = [item for item in instances if self._is_available(item, runtime_stats)]
        candidates = available if available else instances

        start = self._cursor % total
        selected = instances[start]
        if selected not in candidates:
            # Cursor points to a busy instance; pick the first available in order.
            for offset in range(total):
                probe = instances[(start + offset) % total]
                if probe in candidates:
                    selected = probe
                    break

        self._cursor = (instances.index(selected) + 1) % total
        selected_stats = runtime_stats[selected.id]

        return RouteDecision(
            instance_id=selected.id,
            endpoint=selected.endpoint,
            reason=f"algorithm=round_robin,available={str(bool(available)).lower()}",
            score=float(selected_stats.inflight),
        )
