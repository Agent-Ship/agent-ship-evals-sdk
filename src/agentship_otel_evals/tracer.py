"""Tracer accessor.

Wraps ``opentelemetry.trace.get_tracer`` so callers don't need to depend on
OTel directly and so we can swap implementations later without breaking the
public API.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from opentelemetry import trace
from opentelemetry.trace import Tracer

_INSTRUMENTATION_NAME = "agentship-otel-evals"

try:
    _INSTRUMENTATION_VERSION = version(_INSTRUMENTATION_NAME)
except PackageNotFoundError:
    _INSTRUMENTATION_VERSION = "0.0.0+unknown"


def get_tracer() -> Tracer:
    return trace.get_tracer(_INSTRUMENTATION_NAME, _INSTRUMENTATION_VERSION)
