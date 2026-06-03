"""GenAI semantic convention attribute names.

Aligned with the OpenTelemetry GenAI semantic conventions
(https://opentelemetry.io/docs/specs/semconv/gen-ai/) and extended with
``agentship.eval.*`` attributes for eval-specific signals that do not yet
have a stable OTel namespace.

Centralizing strings here keeps span attribute keys consistent across the
SDK and lets downstream consumers (collectors, dashboards) rely on a single
source of truth.
"""

from __future__ import annotations

# --- OTel GenAI semconv (stable) ---
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_PROMPT = "gen_ai.prompt"
GEN_AI_COMPLETION = "gen_ai.completion"

# --- agentship.eval.* (SDK-specific) ---
EVAL_NAME = "agentship.eval.name"
EVAL_RUN_ID = "agentship.eval.run_id"
EVAL_DATASET = "agentship.eval.dataset"
EVAL_DATASET_ITEM_ID = "agentship.eval.dataset.item_id"
EVAL_INPUT = "agentship.eval.input"
EVAL_OUTPUT = "agentship.eval.output"
EVAL_EXPECTED = "agentship.eval.expected"
EVAL_LATENCY_MS = "agentship.eval.latency_ms"
EVAL_COST_USD = "agentship.eval.cost_usd"
EVAL_STATUS = "agentship.eval.status"  # passed | failed | error

# --- Scorer attributes ---
SCORER_NAME = "agentship.scorer.name"
SCORER_VALUE = "agentship.scorer.value"
SCORER_PASSED = "agentship.scorer.passed"
SCORER_THRESHOLD = "agentship.scorer.threshold"
SCORER_REASON = "agentship.scorer.reason"


class SpanKind:
    """Span ``name`` prefixes used by the SDK."""

    EVAL_RUN = "eval.run"
    EVAL_STEP = "eval.step"
    SCORER = "eval.scorer"
