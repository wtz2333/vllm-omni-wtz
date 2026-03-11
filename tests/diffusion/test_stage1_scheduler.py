# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import threading
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.stage1_scheduler import Stage1Scheduler

pytestmark = [pytest.mark.diffusion]


def _tagged_output(tag: str) -> DiffusionOutput:
    return DiffusionOutput(output=torch.tensor([0]), error=tag)


def _mock_request(tag: str) -> Mock:
    req = Mock()
    req.request_ids = [tag]
    return req


def _make_stage1_scheduler():
    sched = Stage1Scheduler()
    sched.num_workers = 1
    sched.od_config = SimpleNamespace(num_gpus=1)
    sched._lock = threading.Lock()
    sched._queue_cv = threading.Condition()
    sched._waiting_queue = deque()
    sched._active_request = None

    req_q: queue.Queue = queue.Queue()
    res_q: queue.Queue = queue.Queue()

    mock_mq = Mock()
    mock_mq.enqueue = req_q.put

    mock_rmq = Mock()
    mock_rmq.dequeue = lambda timeout=None: res_q.get(timeout=timeout if timeout else 10)

    sched.mq = mock_mq
    sched.result_mq = mock_rmq
    return sched, req_q, res_q


def test_stage1_scheduler_attaches_metrics_on_success():
    sched, req_q, res_q = _make_stage1_scheduler()
    req = _mock_request("req-success")

    def _worker():
        req_q.get(timeout=5)
        res_q.put(DiffusionOutput(output=torch.tensor([1])))

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    output = sched.add_req(req)
    worker.join(5)

    assert output.error is None
    assert output.request_id == "req-success"
    assert output.metrics["scheduler_policy"] == "fcfs"
    assert output.metrics["scheduler_latency_ms"] >= 0
    assert output.metrics["queue_wait_ms"] >= 0
    assert output.metrics["scheduler_execute_ms"] >= 0


def test_stage1_scheduler_normalizes_worker_error_dict():
    sched, req_q, res_q = _make_stage1_scheduler()
    req = _mock_request("req-fail")

    def _worker():
        req_q.get(timeout=5)
        res_q.put({"status": "error", "error": "boom"})

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    output = sched.add_req(req)
    worker.join(5)

    assert output.error == "boom"
    assert output.error_code == "WORKER_EXEC_FAILED"
    assert output.request_id == "req-fail"
    assert output.metrics["scheduler_policy"] == "fcfs"


def test_stage1_scheduler_preserves_fcfs_order():
    sched, req_q, res_q = _make_stage1_scheduler()
    seen_request_ids: list[str] = []

    def _worker():
        for _ in range(2):
            rpc_request = req_q.get(timeout=5)
            seen_request_ids.append(rpc_request["args"][0].request_ids[0])
            request_id = rpc_request["args"][0].request_ids[0]
            res_q.put(DiffusionOutput(output=torch.tensor([0]), request_id=request_id))

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    outputs: dict[str, DiffusionOutput] = {}

    def _run(tag: str):
        outputs[tag] = sched.add_req(_mock_request(tag))

    t1 = threading.Thread(target=_run, args=("req-1",), daemon=True)
    t2 = threading.Thread(target=_run, args=("req-2",), daemon=True)
    t1.start()
    t2.start()
    t1.join(5)
    t2.join(5)
    worker.join(5)

    assert seen_request_ids == ["req-1", "req-2"]
    assert outputs["req-1"].request_id == "req-1"
    assert outputs["req-2"].request_id == "req-2"
