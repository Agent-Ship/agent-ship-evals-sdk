"""Built-in scorers."""

from __future__ import annotations

from typing import Any

from agentship_otel_evals.scorers.base import Scorer, ScoreResult


class ExactMatchScorer(Scorer):
    """Pass when ``str(output) == str(expected)``.

    ``case_sensitive=False`` lowercases both sides before comparing; ``strip``
    trims whitespace. Both default to forgiving behavior because exact match
    is most useful as a coarse sanity check, not a strict assertion.
    """

    def __init__(
        self,
        *,
        case_sensitive: bool = False,
        strip: bool = True,
        name: str = "exact_match",
    ) -> None:
        super().__init__(name=name)
        self.case_sensitive = case_sensitive
        self.strip = strip

    def score(
        self,
        *,
        output: Any,
        expected: Any | None = None,
        context: dict[str, Any] | None = None,
    ) -> ScoreResult:
        if expected is None:
            return ScoreResult(
                name=self.name,
                value=0.0,
                passed=False,
                reason="No expected value provided",
            )

        a, b = str(output), str(expected)
        if self.strip:
            a, b = a.strip(), b.strip()
        if not self.case_sensitive:
            a, b = a.lower(), b.lower()

        passed = a == b
        return ScoreResult(
            name=self.name,
            value=1.0 if passed else 0.0,
            passed=passed,
        )


class LatencyScorer(Scorer):
    """Pass when ``latency_ms <= max_latency_ms``.

    Reads ``latency_ms`` from the run context, which is populated by the
    ``eval_run`` / ``eval_step`` decorators.
    """

    def __init__(self, *, max_latency_ms: float, name: str = "latency") -> None:
        super().__init__(name=name)
        self.max_latency_ms = max_latency_ms

    def score(
        self,
        *,
        output: Any,
        expected: Any | None = None,
        context: dict[str, Any] | None = None,
    ) -> ScoreResult:
        latency = float((context or {}).get("latency_ms", 0.0))
        passed = latency <= self.max_latency_ms
        return ScoreResult(
            name=self.name,
            value=latency,
            passed=passed,
            threshold=self.max_latency_ms,
            reason=f"{latency:.2f}ms vs threshold {self.max_latency_ms:.2f}ms",
        )


class CostScorer(Scorer):
    """Pass when ``cost_usd <= max_cost_usd``."""

    def __init__(self, *, max_cost_usd: float, name: str = "cost") -> None:
        super().__init__(name=name)
        self.max_cost_usd = max_cost_usd

    def score(
        self,
        *,
        output: Any,
        expected: Any | None = None,
        context: dict[str, Any] | None = None,
    ) -> ScoreResult:
        cost = float((context or {}).get("cost_usd", 0.0))
        passed = cost <= self.max_cost_usd
        return ScoreResult(
            name=self.name,
            value=cost,
            passed=passed,
            threshold=self.max_cost_usd,
            reason=f"${cost:.6f} vs threshold ${self.max_cost_usd:.6f}",
        )


class LLMJudgeScorer(Scorer):
    """LLM-as-judge scorer (stub).

    Calls a user-supplied ``judge_fn(prompt, output, expected) -> (score, reason)``
    instead of bundling a specific provider. This keeps the core SDK free of
    LLM-vendor dependencies; install the ``llm-judge`` extra and wire your own
    ``judge_fn`` (e.g. an OpenAI call) at the integration layer.
    """

    def __init__(
        self,
        judge_fn: Any,
        *,
        threshold: float = 0.5,
        prompt: str = "Rate the quality of OUTPUT vs EXPECTED on a 0-1 scale.",
        name: str = "llm_judge",
    ) -> None:
        super().__init__(name=name)
        self.judge_fn = judge_fn
        self.threshold = threshold
        self.prompt = prompt

    def score(
        self,
        *,
        output: Any,
        expected: Any | None = None,
        context: dict[str, Any] | None = None,
    ) -> ScoreResult:
        result = self.judge_fn(self.prompt, output, expected)
        if isinstance(result, tuple) and len(result) == 2:
            value, reason = result
        else:
            value, reason = float(result), None
        value = float(value)
        return ScoreResult(
            name=self.name,
            value=value,
            passed=value >= self.threshold,
            threshold=self.threshold,
            reason=reason,
        )
