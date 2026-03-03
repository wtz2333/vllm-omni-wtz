from __future__ import annotations

from dataclasses import replace
from threading import RLock

from .types import InstanceSpec, RuntimeStats


class RuntimeStateStore:
    """Thread-safe runtime state store for global scheduler instances."""

    def __init__(
        self,
        instances: list[InstanceSpec],
        ewma_alpha: float = 0.2,
        default_ewma_service_time_s: float = 1.0,
    ) -> None:
        if not 0.0 < ewma_alpha <= 1.0:
            raise ValueError("ewma_alpha must be in (0, 1]")
        if default_ewma_service_time_s <= 0.0:
            raise ValueError("default_ewma_service_time_s must be > 0")

        self._ewma_alpha = ewma_alpha
        self._lock = RLock()
        self._stats: dict[str, RuntimeStats] = {
            instance.id: RuntimeStats(
                queue_len=0,
                inflight=0,
                ewma_service_time_s=default_ewma_service_time_s,
            )
            for instance in instances
        }

        if not self._stats:
            raise ValueError("instances must not be empty")

    def snapshot(self) -> dict[str, RuntimeStats]:
        """Return an immutable snapshot copy of all instance runtime stats."""
        with self._lock:
            return {instance_id: replace(stats) for instance_id, stats in self._stats.items()}

    def on_request_start(self, instance_id: str) -> RuntimeStats:
        with self._lock:
            stats = self._get_stats(instance_id)
            stats.queue_len += 1
            stats.inflight += 1
            return replace(stats)

    def on_request_finish(self, instance_id: str, latency_s: float, ok: bool) -> RuntimeStats:
        del ok
        with self._lock:
            stats = self._get_stats(instance_id)
            stats.queue_len = max(stats.queue_len - 1, 0)
            stats.inflight = max(stats.inflight - 1, 0)

            if latency_s >= 0.0:
                stats.ewma_service_time_s = (
                    self._ewma_alpha * latency_s + (1.0 - self._ewma_alpha) * stats.ewma_service_time_s
                )

            return replace(stats)

    def _get_stats(self, instance_id: str) -> RuntimeStats:
        if instance_id not in self._stats:
            raise KeyError(f"Unknown instance id: {instance_id}")
        return self._stats[instance_id]
