"""agentship-otel-evals — OpenTelemetry-based evaluation SDK for AI agents and LLMs."""

from agentship_otel_evals.config import EvalConfig, configure
from agentship_otel_evals.decorators import eval_run, eval_step
from agentship_otel_evals.scorers.base import Scorer, ScoreResult
from agentship_otel_evals.scorers.builtin import (
    CostScorer,
    ExactMatchScorer,
    LatencyScorer,
    LLMJudgeScorer,
)
from agentship_otel_evals.tracer import get_tracer

__version__ = "0.1.0"

__all__ = [
    "EvalConfig",
    "configure",
    "eval_run",
    "eval_step",
    "Scorer",
    "ScoreResult",
    "CostScorer",
    "ExactMatchScorer",
    "LatencyScorer",
    "LLMJudgeScorer",
    "get_tracer",
    "__version__",
]
