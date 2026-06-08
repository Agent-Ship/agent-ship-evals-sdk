from __future__ import annotations

import pytest

from agentship_otel_evals.scorers import (
    CostScorer,
    ExactMatchScorer,
    LatencyScorer,
    LLMJudgeScorer,
)


@pytest.mark.unit
class TestExactMatch:
    def test_match_case_insensitive_by_default(self) -> None:
        scorer = ExactMatchScorer()
        result = scorer.score(output="Hello", expected="hello")
        assert result.passed is True
        assert result.value == 1.0

    def test_no_match(self) -> None:
        scorer = ExactMatchScorer()
        result = scorer.score(output="foo", expected="bar")
        assert result.passed is False
        assert result.value == 0.0

    def test_case_sensitive(self) -> None:
        scorer = ExactMatchScorer(case_sensitive=True)
        assert scorer.score(output="Hi", expected="hi").passed is False

    def test_missing_expected_fails(self) -> None:
        scorer = ExactMatchScorer()
        result = scorer.score(output="foo", expected=None)
        assert result.passed is False
        assert "No expected" in (result.reason or "")


@pytest.mark.unit
class TestLatency:
    def test_under_threshold_passes(self) -> None:
        scorer = LatencyScorer(max_latency_ms=100)
        result = scorer.score(output="x", context={"latency_ms": 50})
        assert result.passed is True
        assert result.value == 50.0
        assert result.threshold == 100

    def test_over_threshold_fails(self) -> None:
        scorer = LatencyScorer(max_latency_ms=10)
        result = scorer.score(output="x", context={"latency_ms": 50})
        assert result.passed is False


@pytest.mark.unit
class TestCost:
    def test_under_threshold_passes(self) -> None:
        scorer = CostScorer(max_cost_usd=0.01)
        result = scorer.score(output="x", context={"cost_usd": 0.005})
        assert result.passed is True

    def test_over_threshold_fails(self) -> None:
        scorer = CostScorer(max_cost_usd=0.001)
        result = scorer.score(output="x", context={"cost_usd": 0.01})
        assert result.passed is False


@pytest.mark.unit
class TestLLMJudge:
    def test_judge_above_threshold_passes(self) -> None:
        scorer = LLMJudgeScorer(judge_fn=lambda p, o, e: (0.9, "looks good"), threshold=0.7)
        result = scorer.score(output="x", expected="y")
        assert result.passed is True
        assert result.reason == "looks good"

    def test_judge_returns_scalar(self) -> None:
        scorer = LLMJudgeScorer(judge_fn=lambda p, o, e: 0.4, threshold=0.5)
        result = scorer.score(output="x", expected="y")
        assert result.passed is False
        assert result.value == 0.4
