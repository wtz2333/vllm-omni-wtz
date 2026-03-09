"""Process controller tests for lifecycle command execution."""

import pytest

from vllm_omni.global_scheduler.process_controller import (
    LifecycleExecutionError,
    LifecycleUnsupportedError,
    LocalProcessController,
)
from vllm_omni.global_scheduler.types import InstanceSpec

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_local_process_controller_requires_command():
    """Missing launch config should raise LifecycleUnsupportedError."""
    controller = LocalProcessController()
    instance = InstanceSpec(id="worker-0", endpoint="http://127.0.0.1:9001")

    with pytest.raises(LifecycleUnsupportedError, match="launch config not provided"):
        controller.start(instance)


def test_local_process_controller_exec_error():
    """Failing command should raise LifecycleExecutionError with details."""
    controller = LocalProcessController()
    instance = InstanceSpec(
        id="worker-0",
        endpoint="http://127.0.0.1:9001",
        stop_executable="sh",
        stop_args=["-c", "echo boom 1>&2; exit 7"],
        launch_executable="sh",
        launch_model="ignored-model",
        launch_args=["-c", "exit 0"],
    )

    with pytest.raises(LifecycleExecutionError, match="stop failed"):
        controller.restart(instance)


def test_local_process_controller_success_command():
    """Successful command should return without raising."""
    controller = LocalProcessController()
    instance = InstanceSpec(
        id="worker-0",
        endpoint="http://127.0.0.1:9001",
        stop_executable="sh",
        stop_args=["-c", "exit 0"],
    )

    controller.stop(instance)


def test_local_process_controller_start_with_structured_launch():
    """Start should execute structured launch command with endpoint port."""
    controller = LocalProcessController()
    instance = InstanceSpec(
        id="worker-0",
        endpoint="http://127.0.0.1:9001",
        launch_executable="true",
        launch_model="ignored-model",
        launch_args=["--ulysses-degree", "2"],
    )

    controller.start(instance)
