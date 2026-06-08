"""Scorer base interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScoreResult:
    """Outcome of running a single scorer against an eval result."""

    name: str
    value: float
    passed: bool
    threshold: float | None = None
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Scorer(ABC):
    """Stateful scorer that produces a :class:`ScoreResult` from an eval result."""

    name: str

    def __init__(self, name: str | None = None) -> None:
        self.name = name or self.__class__.__name__

    @abstractmethod
    def score(
        self,
        *,
        output: Any,
        expected: Any | None = None,
        context: dict[str, Any] | None = None,
    ) -> ScoreResult:
        """Produce a score for the given output.

        ``context`` carries free-form metadata from the eval run (latency_ms,
        cost_usd, model, etc.) that a scorer may consult.
        """
