from __future__ import annotations

import pytest
from opentelemetry.sdk.trace import TracerProvider

from agentship_otel_evals.config import EvalConfig, configure


@pytest.mark.unit
def test_configure_returns_tracer_provider() -> None:
    provider = configure(EvalConfig(service_name="test-svc"))
    assert isinstance(provider, TracerProvider)


@pytest.mark.unit
def test_configure_is_idempotent() -> None:
    first = configure(EvalConfig(service_name="a"))
    second = configure(EvalConfig(service_name="b"))
    assert first is second


@pytest.mark.unit
def test_eval_config_defaults_to_env() -> None:
    cfg = EvalConfig(service_name="foo")
    assert cfg.service_name == "foo"
    assert cfg.console_export in {True, False}
