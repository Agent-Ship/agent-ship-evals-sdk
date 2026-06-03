"""Decorators that wrap eval runs / steps in OTel spans.

Two entry points:

* :func:`eval_run` — wraps a top-level eval invocation (one dataset item).
  Emits an ``eval.run`` span, runs all attached scorers, attaches their
  results as child ``eval.scorer`` spans, and sets span status based on
  whether all scorers passed.
* :func:`eval_step` — wraps any sub-step inside an eval run (LLM call, tool
  call, retrieval, etc.). Emits an ``eval.step`` span using OTel GenAI
  semconv where applicable.

Both decorators support sync and async functions.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import time
import uuid
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from opentelemetry.trace import Status, StatusCode

from agentship_otel_evals import semconv
from agentship_otel_evals.scorers.base import Scorer
from agentship_otel_evals.tracer import get_tracer

F = TypeVar("F", bound=Callable[..., Any])


def eval_run(
    name: str,
    *,
    scorers: Iterable[Scorer] | None = None,
    expected: Any | None = None,
    dataset: str | None = None,
    dataset_item_id: str | None = None,
    capture_io: bool = True,
) -> Callable[[F], F]:
    """Wrap a function in an ``eval.run`` span.

    The wrapped function may be sync or async. The span captures latency,
    runs any attached scorers, and reports overall pass/fail via OTel span
    status so dashboards can filter on failures directly.
    """
    scorers_list = list(scorers or [])

    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await _run_async(
                    func, args, kwargs, name, scorers_list,
                    expected, dataset, dataset_item_id, capture_io,
                )

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            return _run_sync(
                func, args, kwargs, name, scorers_list,
                expected, dataset, dataset_item_id, capture_io,
            )

        return sync_wrapper  # type: ignore[return-value]

    return decorator


def eval_step(
    name: str,
    *,
    operation: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> Callable[[F], F]:
    """Wrap a function in an ``eval.step`` span.

    ``operation`` populates ``gen_ai.operation.name`` (e.g. ``chat``,
    ``embeddings``, ``tool_call``) for downstream OTel GenAI consumers.
    """
    base_attrs = dict(attributes or {})
    if operation:
        base_attrs[semconv.GEN_AI_OPERATION_NAME] = operation

    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with get_tracer().start_as_current_span(
                    f"{semconv.SpanKind.EVAL_STEP}.{name}"
                ) as span:
                    for k, v in base_attrs.items():
                        span.set_attribute(k, _safe_attr(v))
                    try:
                        return await func(*args, **kwargs)
                    except Exception as exc:
                        span.record_exception(exc)
                        span.set_status(Status(StatusCode.ERROR, str(exc)))
                        raise

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with get_tracer().start_as_current_span(
                f"{semconv.SpanKind.EVAL_STEP}.{name}"
            ) as span:
                for k, v in base_attrs.items():
                    span.set_attribute(k, _safe_attr(v))
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise

        return sync_wrapper  # type: ignore[return-value]

    return decorator


def _run_sync(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    name: str,
    scorers: list[Scorer],
    expected: Any | None,
    dataset: str | None,
    dataset_item_id: str | None,
    capture_io: bool,
) -> Any:
    tracer = get_tracer()
    run_id = str(uuid.uuid4())
    with tracer.start_as_current_span(f"{semconv.SpanKind.EVAL_RUN}.{name}") as span:
        _set_run_attrs(span, name, run_id, dataset, dataset_item_id)
        if capture_io:
            span.set_attribute(semconv.EVAL_INPUT, _serialize_io(args, kwargs))
            if expected is not None:
                span.set_attribute(semconv.EVAL_EXPECTED, _safe_attr(expected))

        start = time.perf_counter()
        try:
            output = func(*args, **kwargs)
        except Exception as exc:
            span.record_exception(exc)
            span.set_attribute(semconv.EVAL_STATUS, "error")
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise

        latency_ms = (time.perf_counter() - start) * 1000
        span.set_attribute(semconv.EVAL_LATENCY_MS, latency_ms)
        if capture_io:
            span.set_attribute(semconv.EVAL_OUTPUT, _safe_attr(output))

        passed = _run_scorers(tracer, scorers, output, expected, {"latency_ms": latency_ms})
        span.set_attribute(semconv.EVAL_STATUS, "passed" if passed else "failed")
        if not passed:
            span.set_status(Status(StatusCode.ERROR, "One or more scorers failed"))
        return output


async def _run_async(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    name: str,
    scorers: list[Scorer],
    expected: Any | None,
    dataset: str | None,
    dataset_item_id: str | None,
    capture_io: bool,
) -> Any:
    tracer = get_tracer()
    run_id = str(uuid.uuid4())
    with tracer.start_as_current_span(f"{semconv.SpanKind.EVAL_RUN}.{name}") as span:
        _set_run_attrs(span, name, run_id, dataset, dataset_item_id)
        if capture_io:
            span.set_attribute(semconv.EVAL_INPUT, _serialize_io(args, kwargs))
            if expected is not None:
                span.set_attribute(semconv.EVAL_EXPECTED, _safe_attr(expected))

        start = time.perf_counter()
        try:
            output = await func(*args, **kwargs)
        except Exception as exc:
            span.record_exception(exc)
            span.set_attribute(semconv.EVAL_STATUS, "error")
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise

        latency_ms = (time.perf_counter() - start) * 1000
        span.set_attribute(semconv.EVAL_LATENCY_MS, latency_ms)
        if capture_io:
            span.set_attribute(semconv.EVAL_OUTPUT, _safe_attr(output))

        passed = _run_scorers(tracer, scorers, output, expected, {"latency_ms": latency_ms})
        span.set_attribute(semconv.EVAL_STATUS, "passed" if passed else "failed")
        if not passed:
            span.set_status(Status(StatusCode.ERROR, "One or more scorers failed"))
        return output


def _set_run_attrs(
    span: Any,
    name: str,
    run_id: str,
    dataset: str | None,
    dataset_item_id: str | None,
) -> None:
    span.set_attribute(semconv.EVAL_NAME, name)
    span.set_attribute(semconv.EVAL_RUN_ID, run_id)
    if dataset:
        span.set_attribute(semconv.EVAL_DATASET, dataset)
    if dataset_item_id:
        span.set_attribute(semconv.EVAL_DATASET_ITEM_ID, dataset_item_id)


def _run_scorers(
    tracer: Any,
    scorers: list[Scorer],
    output: Any,
    expected: Any | None,
    context: dict[str, Any],
) -> bool:
    all_passed = True
    for scorer in scorers:
        with tracer.start_as_current_span(
            f"{semconv.SpanKind.SCORER}.{scorer.name}"
        ) as span:
            try:
                result = scorer.score(output=output, expected=expected, context=context)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                all_passed = False
                continue

            span.set_attribute(semconv.SCORER_NAME, result.name)
            span.set_attribute(semconv.SCORER_VALUE, result.value)
            span.set_attribute(semconv.SCORER_PASSED, result.passed)
            if result.threshold is not None:
                span.set_attribute(semconv.SCORER_THRESHOLD, result.threshold)
            if result.reason:
                span.set_attribute(semconv.SCORER_REASON, result.reason)
            if not result.passed:
                all_passed = False
                span.set_status(Status(StatusCode.ERROR, result.reason or "Scorer failed"))
    return all_passed


def _safe_attr(value: Any) -> Any:
    """Coerce arbitrary Python values into OTel-attribute-compatible types."""
    if isinstance(value, (str, bool, int, float)):
        return value
    if value is None:
        return ""
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def _serialize_io(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    try:
        return json.dumps({"args": args, "kwargs": kwargs}, default=str)
    except (TypeError, ValueError):
        return str({"args": args, "kwargs": kwargs})


# Re-exported for downstream consumers needing the raw helper.
__all__ = ["eval_run", "eval_step", "inspect"]
