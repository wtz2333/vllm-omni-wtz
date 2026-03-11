# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time
from collections import deque
from dataclasses import dataclass

import zmq
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.scheduler import Scheduler

logger = init_logger(__name__)


@dataclass
class _QueuedRequest:
    request: OmniDiffusionRequest
    enqueue_time: float


class Stage1Scheduler(Scheduler):
    """Stage-1 FCFS scheduler with queue observability and normalized failures."""

    def initialize(self, od_config: OmniDiffusionConfig):
        super().initialize(od_config)
        self._queue_cv = threading.Condition()
        self._waiting_queue: deque[_QueuedRequest] = deque()
        self._active_request: _QueuedRequest | None = None

    @staticmethod
    def _request_label(request: OmniDiffusionRequest) -> str:
        request_ids = getattr(request, "request_ids", None) or []
        if request_ids:
            return ",".join(request_ids)
        return "<missing-request-id>"

    def _build_generate_rpc_request(self, request: OmniDiffusionRequest) -> dict:
        return {
            "type": "rpc",
            "method": "generate",
            "args": (request,),
            "kwargs": {},
            "output_rank": 0,
            "exec_all_ranks": True,
        }

    def _annotate_output(
        self,
        output: DiffusionOutput,
        request: OmniDiffusionRequest,
        queue_wait_ms: float,
        execute_latency_ms: float,
    ) -> DiffusionOutput:
        request_label = self._request_label(request)
        metrics = dict(getattr(output, "metrics", {}) or {})
        metrics.update(
            {
                "scheduler_policy": "fcfs",
                "queue_wait_ms": queue_wait_ms,
                "scheduler_execute_ms": execute_latency_ms,
                "scheduler_latency_ms": queue_wait_ms + execute_latency_ms,
                "queue_len": len(self._waiting_queue),
            }
        )
        output.metrics = metrics
        output.request_id = output.request_id or request_label
        return output

    def _normalize_error_output(self, request: OmniDiffusionRequest, error: str, error_code: str) -> DiffusionOutput:
        request_label = self._request_label(request)
        return DiffusionOutput(
            error=error,
            error_code=error_code,
            request_id=request_label,
            metrics={"scheduler_policy": "fcfs", "queue_len": len(self._waiting_queue)},
        )

    def add_req(self, request: OmniDiffusionRequest) -> DiffusionOutput:
        queued_request = _QueuedRequest(request=request, enqueue_time=time.monotonic())
        request_label = self._request_label(request)

        with self._queue_cv:
            self._waiting_queue.append(queued_request)
            logger.info("QUEUE_ENQUEUE request_id=%s queue_len=%d", request_label, len(self._waiting_queue))
            while self._waiting_queue[0] is not queued_request or self._active_request is not None:
                self._queue_cv.wait()
            self._waiting_queue.popleft()
            self._active_request = queued_request
            queue_wait_ms = (time.monotonic() - queued_request.enqueue_time) * 1000
            logger.info("QUEUE_DEQUEUE request_id=%s queue_len=%d", request_label, len(self._waiting_queue))

        execute_start = time.monotonic()
        try:
            with self._lock:
                self.mq.enqueue(self._build_generate_rpc_request(request))
                if self.result_mq is None:
                    output = self._normalize_error_output(
                        request=request,
                        error="Result queue not initialized",
                        error_code="RESULT_QUEUE_NOT_INITIALIZED",
                    )
                else:
                    raw_output = self.result_mq.dequeue()
                    if isinstance(raw_output, dict) and raw_output.get("status") == "error":
                        output = self._normalize_error_output(
                            request=request,
                            error=raw_output.get("error", "worker error"),
                            error_code="WORKER_EXEC_FAILED",
                        )
                    else:
                        output = raw_output
        except zmq.error.Again as exc:
            logger.error("REQUEST_FAIL request_id=%s error_code=SCHEDULER_TIMEOUT", request_label)
            raise TimeoutError("Scheduler did not respond in time.") from exc
        finally:
            with self._queue_cv:
                self._active_request = None
                self._queue_cv.notify_all()

        execute_latency_ms = (time.monotonic() - execute_start) * 1000
        output = self._annotate_output(output, request, queue_wait_ms, execute_latency_ms)

        if output.error:
            if output.error_code is None:
                output.error_code = "REQUEST_EXEC_FAILED"
            logger.error(
                "REQUEST_FAIL request_id=%s queue_len=%d latency_ms=%.2f error_code=%s error=%s",
                output.request_id,
                output.metrics.get("queue_len", -1),
                output.metrics.get("scheduler_latency_ms", -1.0),
                output.error_code,
                output.error,
            )
            return output

        logger.info(
            "REQUEST_DONE request_id=%s queue_len=%d latency_ms=%.2f",
            output.request_id,
            output.metrics.get("queue_len", -1),
            output.metrics.get("scheduler_latency_ms", -1.0),
        )
        return output
