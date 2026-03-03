import textwrap

import pytest
from fastapi.testclient import TestClient

from vllm_omni.global_scheduler.config import load_config
from vllm_omni.global_scheduler.server import create_app

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_health_endpoint_returns_scheduler_and_ok(tmp_path):
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            scheduler:
              type: ondisc_sp1
            instances:
              - id: worker-0
                endpoint: http://127.0.0.1:9001
                sp_size: 1
                max_concurrency: 1
            """
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    app = create_app(config)
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["scheduler"] == "ondisc_sp1"
    assert payload["instance_count"] == 1
    assert payload["checks"]["config_loaded"] is True
    assert payload["checks"]["has_instances"] is True
    assert payload["checks"]["scheduler_type_valid"] is True
    assert "version" in payload


def test_health_endpoint_returns_503_when_config_missing(tmp_path):
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            scheduler:
              type: baseline_sp1
            instances:
              - id: worker-0
                endpoint: http://127.0.0.1:9001
                sp_size: 1
                max_concurrency: 1
            """
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    app = create_app(config)
    app.state.global_scheduler_config = None
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["checks"]["config_loaded"] is False
    assert payload["checks"]["has_instances"] is False
    assert payload["checks"]["scheduler_type_valid"] is False


def test_load_config_missing_file_raises_clear_error(tmp_path):
    missing = tmp_path / "not_found.yaml"

    with pytest.raises(ValueError, match="Config file not found"):
        load_config(missing)
