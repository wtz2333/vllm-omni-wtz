import textwrap

import pytest

from vllm_omni.global_scheduler.config import load_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_load_config_success(tmp_path):
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            server:
              host: 0.0.0.0
              port: 8089
            scheduler:
              type: baseline_sp1
            instances:
              - id: worker-0
                endpoint: http://127.0.0.1:9001
                sp_size: 1
                max_concurrency: 2
              - id: worker-1
                endpoint: http://127.0.0.1:9002
                sp_size: 1
                max_concurrency: 2
            """
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.server.port == 8089
    assert config.scheduler.type == "baseline_sp1"
    assert len(config.instances) == 2


def test_load_config_duplicate_instance_id(tmp_path):
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            instances:
              - id: worker-0
                endpoint: http://127.0.0.1:9001
                sp_size: 1
                max_concurrency: 2
              - id: worker-0
                endpoint: http://127.0.0.1:9002
                sp_size: 1
                max_concurrency: 2
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="globally unique"):
        load_config(config_path)


def test_load_config_invalid_endpoint(tmp_path):
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            instances:
              - id: worker-0
                endpoint: https://127.0.0.1:9001
                sp_size: 1
                max_concurrency: 2
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="http://host:port"):
        load_config(config_path)


def test_load_config_baseline_algorithm_success(tmp_path):
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            scheduler:
              type: baseline_sp1
            policy:
              baseline_sp1:
                algorithm: fcfs
            instances:
              - id: worker-0
                endpoint: http://127.0.0.1:9001
                sp_size: 1
                max_concurrency: 2
            """
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.policy.baseline_sp1.algorithm == "fcfs"


def test_load_config_mode_alias_maps_to_algorithm(tmp_path):
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            policy:
              baseline_sp1:
                mode: short_queue_runtime
            instances:
              - id: worker-0
                endpoint: http://127.0.0.1:9001
                sp_size: 1
                max_concurrency: 2
            """
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.policy.baseline_sp1.algorithm == "short_queue_runtime"


def test_load_config_invalid_baseline_algorithm(tmp_path):
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            policy:
              baseline_sp1:
                algorithm: unknown_algo
            instances:
              - id: worker-0
                endpoint: http://127.0.0.1:9001
                sp_size: 1
                max_concurrency: 2
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="policy.baseline_sp1.algorithm"):
        load_config(config_path)
