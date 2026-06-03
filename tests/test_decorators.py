from __future__ import annotations

import asyncio

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from agentship_otel_evals import semconv
from agentship_otel_evals.decorators import eval_run, eval_step
from agentship_otel_evals.scorers import ExactMatchScorer, LatencyScorer


def _spans_by_name(exporter: InMemorySpanExporter) -> dict[str, list]:
    out: dict[str, list] = {}
    for span in exporter.get_finished_spans():
        out.setdefault(span.name, []).append(span)
    return out


@pytest.mark.unit
class TestEvalRun:
    def test_sync_run_emits_span_with_attributes(
        self, span_exporter: InMemorySpanExporter
    ) -> None:
        @eval_run("greet", scorers=[ExactMatchScorer()], expected="hello", dataset="ds-v1")
        def greet(name: str) -> str:
            return "hello"

        result = greet("world")
        assert result == "hello"

        spans = _spans_by_name(span_exporter)
        run_spans = spans[f"{semconv.SpanKind.EVAL_RUN}.greet"]
        assert len(run_spans) == 1
        attrs = dict(run_spans[0].attributes or {})
        assert attrs[semconv.EVAL_NAME] == "greet"
        assert attrs[semconv.EVAL_DATASET] == "ds-v1"
        assert attrs[semconv.EVAL_STATUS] == "passed"
        assert attrs[semconv.EVAL_LATENCY_MS] >= 0

        scorer_spans = spans[f"{semconv.SpanKind.SCORER}.exact_match"]
        assert len(scorer_spans) == 1
        scorer_attrs = dict(scorer_spans[0].attributes or {})
        assert scorer_attrs[semconv.SCORER_PASSED] is True

    def test_failing_scorer_sets_error_status(
        self, span_exporter: InMemorySpanExporter
    ) -> None:
        @eval_run("greet", scorers=[ExactMatchScorer()], expected="goodbye")
        def greet() -> str:
            return "hello"

        greet()
        run_span = _spans_by_name(span_exporter)[f"{semconv.SpanKind.EVAL_RUN}.greet"][0]
        attrs = dict(run_span.attributes or {})
        assert attrs[semconv.EVAL_STATUS] == "failed"
        assert run_span.status.status_code == StatusCode.ERROR

    def test_exception_marks_error(self, span_exporter: InMemorySpanExporter) -> None:
        @eval_run("boom")
        def boom() -> str:
            raise RuntimeError("kaboom")

        with pytest.raises(RuntimeError):
            boom()

        run_span = _spans_by_name(span_exporter)[f"{semconv.SpanKind.EVAL_RUN}.boom"][0]
        assert run_span.status.status_code == StatusCode.ERROR
        assert dict(run_span.attributes or {})[semconv.EVAL_STATUS] == "error"

    def test_async_run(self, span_exporter: InMemorySpanExporter) -> None:
        @eval_run("agreet", scorers=[ExactMatchScorer()], expected="hi")
        async def agreet() -> str:
            await asyncio.sleep(0)
            return "hi"

        result = asyncio.run(agreet())
        assert result == "hi"

        attrs = dict(
            _spans_by_name(span_exporter)[f"{semconv.SpanKind.EVAL_RUN}.agreet"][0].attributes
            or {}
        )
        assert attrs[semconv.EVAL_STATUS] == "passed"

    def test_latency_scorer_uses_context(
        self, span_exporter: InMemorySpanExporter
    ) -> None:
        @eval_run("slow", scorers=[LatencyScorer(max_latency_ms=0.0)])
        def slow() -> str:
            return "done"

        slow()
        run_span = _spans_by_name(span_exporter)[f"{semconv.SpanKind.EVAL_RUN}.slow"][0]
        assert dict(run_span.attributes or {})[semconv.EVAL_STATUS] == "failed"


@pytest.mark.unit
class TestEvalStep:
    def test_step_sets_operation_attribute(
        self, span_exporter: InMemorySpanExporter
    ) -> None:
        @eval_step("call_llm", operation="chat", attributes={semconv.GEN_AI_SYSTEM: "openai"})
        def call_llm() -> str:
            return "ok"

        call_llm()
        step_span = _spans_by_name(span_exporter)[f"{semconv.SpanKind.EVAL_STEP}.call_llm"][0]
        attrs = dict(step_span.attributes or {})
        assert attrs[semconv.GEN_AI_OPERATION_NAME] == "chat"
        assert attrs[semconv.GEN_AI_SYSTEM] == "openai"
