"""Minimal example: evaluate a single function and emit OTel spans.

Run::

    python examples/basic_eval.py

Spans go to stdout because no OTLP endpoint is configured. To send to a
collector, set ``OTEL_EXPORTER_OTLP_ENDPOINT`` (e.g. ``http://localhost:4318/v1/traces``).
"""

from __future__ import annotations

from agentship_otel_evals import (
    EvalConfig,
    ExactMatchScorer,
    LatencyScorer,
    configure,
    eval_run,
    eval_step,
)

configure(EvalConfig(service_name="example-evals", console_export=True))


@eval_step("call_llm", operation="chat", attributes={"gen_ai.system": "openai"})
def call_llm(prompt: str) -> str:
    # Stand-in for a real LLM call.
    return "Paris"


@eval_run(
    "capital_of_france",
    scorers=[ExactMatchScorer(), LatencyScorer(max_latency_ms=500)],
    expected="Paris",
    dataset="geography-v1",
    dataset_item_id="q-001",
)
def answer_capital(country: str) -> str:
    return call_llm(f"What is the capital of {country}?")


if __name__ == "__main__":
    print(answer_capital("France"))
