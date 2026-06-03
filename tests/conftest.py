"""Pytest fixtures.

Installs a single global ``TracerProvider`` once per test session and gives
each test an ``InMemorySpanExporter`` attached as an extra processor. OTel's
``set_tracer_provider`` is set-once globally, so we cannot swap providers
between tests — instead we share one provider and clear the exporter.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agentship_otel_evals import config as eval_config


@pytest.fixture(scope="session", autouse=True)
def _install_provider() -> TracerProvider:
    """Install a fresh ``TracerProvider`` for the test session."""
    provider = TracerProvider()
    trace.set_tracer_provider(provider)
    eval_config._reset_for_testing()
    return provider


@pytest.fixture
def span_exporter(_install_provider: TracerProvider) -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)
    _install_provider.add_span_processor(processor)
    try:
        yield exporter
    finally:
        processor.shutdown()
        exporter.clear()
