from __future__ import annotations

import os
import subprocess
from abc import ABC, abstractmethod
from urllib.parse import urlparse

from .types import InstanceSpec


class LifecycleUnsupportedError(ValueError):
    """Raised when lifecycle operation is not configured for an instance."""


class LifecycleExecutionError(RuntimeError):
    """Raised when lifecycle operation command execution fails."""


class ProcessController(ABC):
    """Abstract process controller for instance lifecycle operations."""

    @abstractmethod
    def stop(self, instance: InstanceSpec) -> None:
        raise NotImplementedError

    @abstractmethod
    def start(self, instance: InstanceSpec) -> None:
        raise NotImplementedError

    @abstractmethod
    def restart(self, instance: InstanceSpec) -> None:
        raise NotImplementedError


class LocalProcessController(ProcessController):
    """Execute structured local lifecycle commands."""

    def stop(self, instance: InstanceSpec) -> None:
        if instance.stop_executable is None:
            raise LifecycleUnsupportedError(f"stop config not provided for instance {instance.id}")
        if not instance.stop_args:
            raise LifecycleUnsupportedError(f"stop.args is empty for instance {instance.id}")
        self._run(operation="stop", instance=instance, argv=[instance.stop_executable, *instance.stop_args], env=None)

    def start(self, instance: InstanceSpec) -> None:
        argv, env = self._build_start_argv_and_env(instance)
        self._run(operation="start", instance=instance, argv=argv, env=env)

    def restart(self, instance: InstanceSpec) -> None:
        self.stop(instance)
        self.start(instance)

    @staticmethod
    def _build_start_argv_and_env(instance: InstanceSpec) -> tuple[list[str], dict[str, str] | None]:
        if instance.launch_executable is None or instance.launch_model is None:
            raise LifecycleUnsupportedError(f"launch config not provided for instance {instance.id}")
        parsed = urlparse(instance.endpoint)
        if parsed.port is None:
            raise LifecycleUnsupportedError(f"endpoint has no port for instance {instance.id}")
        argv = [
            instance.launch_executable,
            "serve",
            instance.launch_model,
            "--port",
            str(parsed.port),
            *instance.launch_args,
        ]
        if not argv[0].strip():
            raise LifecycleUnsupportedError(f"launch executable is empty for instance {instance.id}")
        env = os.environ.copy()
        env.update(instance.launch_env)
        return argv, env

    @staticmethod
    def _run(operation: str, instance: InstanceSpec, argv: list[str], env: dict[str, str] | None) -> None:
        try:
            subprocess.run(
                argv,
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            detail = stderr or stdout or f"exit code {exc.returncode}"
            raise LifecycleExecutionError(f"{operation} failed for {instance.id}: {detail}") from exc
