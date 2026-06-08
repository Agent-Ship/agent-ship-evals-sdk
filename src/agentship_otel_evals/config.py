"""SDK configuration and OpenTelemetry provider bootstrap."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

_DEFAULT_SERVICE_NAME = "agentship-evals"
_configure_lock = Lock()
_configured = False


@dataclass
class EvalConfig:
    """Runtime configuration for the eval SDK.

    Any field can be overridden by an environment variable so the SDK plays
    nicely with container deployments where the binary is shipped pre-built.
    """

    service_name: str = field(
        default_factory=lambda: os.getenv("OTEL_SERVICE_NAME", _DEFAULT_SERVICE_NAME)
    )
    otlp_endpoint: str | None = field(
        default_factory=lambda: os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    )
    otlp_headers: dict[str, str] = field(default_factory=dict)
    console_export: bool = field(
        default_factory=lambda: os.getenv("AGENTSHIP_EVAL_CONSOLE", "").lower() in {"1", "true"}
    )
    resource_attributes: dict[str, Any] = field(default_factory=dict)

    def build_resource(self) -> Resource:
        attrs: dict[str, Any] = {"service.name": self.service_name}
        attrs.update(self.resource_attributes)
        return Resource.create(attrs)


def configure(config: EvalConfig | None = None) -> TracerProvider:
    """Install a ``TracerProvider`` with eval-appropriate exporters.

    Idempotent — calling ``configure()`` twice returns the already-installed
    provider. OTel's ``set_tracer_provider`` is set-once globally, so we
    cannot replace it after first install; instead we add this call's
    exporters onto the existing provider.
    """
    global _configured
    config = config or EvalConfig()
    processors = _build_processors(config)

    with _configure_lock:
        current = trace.get_tracer_provider()
        if _configured and isinstance(current, TracerProvider):
            for processor in processors:
                current.add_span_processor(processor)
            return current

        if isinstance(current, TracerProvider):
            # Provider was installed outside the SDK (e.g. by a test harness).
            for processor in processors:
                current.add_span_processor(processor)
            _configured = True
            return current

        provider = TracerProvider(resource=config.build_resource())
        for processor in processors:
            provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        _configured = True
        return provider


def _build_processors(config: EvalConfig) -> list[SpanProcessor]:
    processors: list[SpanProcessor] = []

    if config.otlp_endpoint:
        exporter = OTLPSpanExporter(
            endpoint=config.otlp_endpoint,
            headers=config.otlp_headers or None,
        )
        processors.append(BatchSpanProcessor(exporter))

    if config.console_export or not processors:
        # Always have at least one processor so spans go somewhere visible
        # during development.
        processors.append(SimpleSpanProcessor(ConsoleSpanExporter()))

    return processors


def _reset_for_testing() -> None:
    """Reset module-level config state. Test-only helper."""
    global _configured
    with _configure_lock:
        _configured = False
