# agentship-otel-evals

OpenTelemetry-based evaluation SDK for AI agents and LLMs. Emits evaluation
runs, steps, and scorer results as OTel spans using the
[GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/),
so eval results land in whatever OTel-compatible backend you already use
(Tempo, Honeycomb, Datadog, etc.) — no proprietary backend required.

> **Status:** alpha. APIs may change before 1.0.

## Why

Most eval frameworks bolt on their own dashboards. If you already ship traces
via OTel, you can get eval signals in the same place by using OTel-native
semantics. This SDK gives you:

- `@eval_run` / `@eval_step` decorators that wrap any function in spans
- Built-in scorers: exact-match, latency, cost, LLM-as-judge
- A `Scorer` base class for custom scorers
- OTel GenAI attributes (`gen_ai.system`, `gen_ai.operation.name`, etc.) on
  step spans, plus `agentship.eval.*` attributes on run spans for eval-specific
  signals (pass/fail, scorer values, dataset, latency, cost)

## Install

```bash
pip install agentship-otel-evals
```

For LLM-judge support:

```bash
pip install "agentship-otel-evals[llm-judge]"
```

## Quick start

```python
from agentship_otel_evals import (
    EvalConfig, configure, eval_run, eval_step,
    ExactMatchScorer, LatencyScorer,
)

configure(EvalConfig(service_name="my-evals"))

@eval_step("call_llm", operation="chat", attributes={"gen_ai.system": "openai"})
def call_llm(prompt: str) -> str:
    ...

@eval_run(
    "capital_of_france",
    scorers=[ExactMatchScorer(), LatencyScorer(max_latency_ms=500)],
    expected="Paris",
    dataset="geography-v1",
)
def answer_capital(country: str) -> str:
    return call_llm(f"What is the capital of {country}?")

answer_capital("France")
```

Set `OTEL_EXPORTER_OTLP_ENDPOINT` (e.g. `http://localhost:4318/v1/traces`)
to ship spans to your collector. Without it, spans print to stdout — useful
for local development.

## Span model

| Span name | When | Key attributes |
|---|---|---|
| `eval.run.<name>` | One eval invocation (one dataset item) | `agentship.eval.name`, `agentship.eval.run_id`, `agentship.eval.dataset`, `agentship.eval.status`, `agentship.eval.latency_ms`, `agentship.eval.input`, `agentship.eval.output`, `agentship.eval.expected` |
| `eval.step.<name>` | Any sub-step (LLM call, tool call, retrieval) | `gen_ai.operation.name`, `gen_ai.system`, `gen_ai.request.model`, plus any custom attributes you pass |
| `eval.scorer.<scorer_name>` | One scorer invocation | `agentship.scorer.name`, `agentship.scorer.value`, `agentship.scorer.passed`, `agentship.scorer.threshold`, `agentship.scorer.reason` |

Run and scorer spans set OTel `Status.ERROR` on failure, so dashboards can
filter on failed evals without parsing custom attributes.

## Built-in scorers

| Scorer | Pass condition |
|---|---|
| `ExactMatchScorer` | `str(output) == str(expected)` (case-insensitive, stripped by default) |
| `LatencyScorer(max_latency_ms=...)` | `latency_ms <= max_latency_ms` (from run context) |
| `CostScorer(max_cost_usd=...)` | `cost_usd <= max_cost_usd` (from run context) |
| `LLMJudgeScorer(judge_fn, threshold=...)` | `judge_fn(prompt, output, expected) >= threshold` |

`LLMJudgeScorer` does not bundle a specific LLM provider — pass your own
`judge_fn` so the core SDK stays provider-neutral.

## Custom scorers

```python
from agentship_otel_evals import Scorer, ScoreResult

class ContainsScorer(Scorer):
    def __init__(self, needle: str):
        super().__init__(name=f"contains:{needle}")
        self.needle = needle

    def score(self, *, output, expected=None, context=None) -> ScoreResult:
        passed = self.needle in str(output)
        return ScoreResult(name=self.name, value=1.0 if passed else 0.0, passed=passed)
```

## Configuration

```python
from agentship_otel_evals import EvalConfig, configure

configure(EvalConfig(
    service_name="my-evals",
    otlp_endpoint="http://collector:4318/v1/traces",
    otlp_headers={"x-api-key": "..."},
    resource_attributes={"deployment.environment": "staging"},
))
```

Environment variables (standard OTel):

- `OTEL_SERVICE_NAME`
- `OTEL_EXPORTER_OTLP_ENDPOINT`
- `AGENTSHIP_EVAL_CONSOLE=1` — also print spans to stdout

## Development

```bash
pip install -e ".[dev]"
pytest -v
ruff check .
mypy src
```

## License

MIT — see [LICENSE](LICENSE).
