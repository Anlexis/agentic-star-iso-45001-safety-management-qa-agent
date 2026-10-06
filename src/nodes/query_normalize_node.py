"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - Return the status as AgentStatus.<X>.value — the enum's string value,
#    not the enum object itself
#  - Never import from mediator/, api/, or other agents
#
# MFG-C2-021 — QueryNormalizeNode
# First step of the inner workflow: language detection and query normalization.
# Detects JA / VI / EN from the question via unicode-range heuristics,
# normalizes whitespace and applies NFC unicode normalization,
# and lowercases the question when the detected language is EN.
#
# Security notes:
#   No model call; deterministic heuristics only (no external side-effects).
#   Output fields (normalized_query / detected_language) contain no credentials
#   or personal data — the question was screened and masked upstream.

import logging
import unicodedata
from typing import Any, ClassVar, Dict

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Unicode-range helpers for language detection
# ---------------------------------------------------------------------------

# CJK Unified Ideographs and CJK-adjacent ranges that indicate Japanese text.
_CJK_RANGES = (
    (0x3000, 0x9FFF),  # CJK Unified Ideographs + Hiragana / Katakana / misc
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0x20000, 0x2A6DF),  # CJK Extension B
)

# Latin Extended and combining diacritic ranges common in Vietnamese.
# Vietnamese uses Latin script with a rich set of combining marks and precomposed
# letters in the Unicode Latin Extended Additional block.
_VI_RANGES = (
    (0x00C0, 0x024F),  # Latin Extended-A/B (includes accented letters)
    (0x1E00, 0x1EFF),  # Latin Extended Additional (core Vietnamese block)
    (0x0300, 0x036F),  # Combining Diacritical Marks
)


def _has_cjk(text: str) -> bool:
    """Return True if any character in *text* falls within a CJK range."""
    for ch in text:
        cp = ord(ch)
        for lo, hi in _CJK_RANGES:
            if lo <= cp <= hi:
                return True
    return False


def _vi_score(text: str) -> int:
    """Count how many characters fall in Vietnamese-specific unicode ranges."""
    count = 0
    for ch in text:
        cp = ord(ch)
        for lo, hi in _VI_RANGES:
            if lo <= cp <= hi:
                count += 1
                break
    return count


def _detect_language(text: str) -> str:
    """Detect language of *text* using unicode-range heuristics.

    Returns:
        "JA"  — Japanese (CJK characters detected)
        "VI"  — Vietnamese (Latin-extended / diacritic characters dominant)
        "EN"  — English / default (pure ASCII Latin)
    """
    if not text:
        return "EN"

    if _has_cjk(text):
        return "JA"

    # Vietnamese detection: if more than 5% of characters are VI-range, classify as VI.
    vi_count = _vi_score(text)
    if len(text) > 0 and vi_count / len(text) >= 0.05:
        return "VI"

    return "EN"


def _normalize_query(raw: str, language: str) -> str:
    """Apply NFC normalization, strip leading/trailing whitespace,
    collapse internal whitespace, and lowercase for EN queries.
    """
    # NFC normalization
    normalized = unicodedata.normalize("NFC", raw)
    # Strip and collapse internal whitespace
    normalized = " ".join(normalized.split())
    # Lowercase only for EN (preserve casing for JA/VI)
    if language == "EN":
        normalized = normalized.lower()
    return normalized


class QueryNormalizeNode(FunctionNode):
    """Detect query language and normalize the raw query string.

    First step of the inner workflow. Applies unicode-range-based language
    detection (JA / VI / EN), then writes the NFC-normalized form back.

    Input state keys:
        user_input: str — the question, as handed to the inner graph by the
            outer main slot. ``query`` is accepted as an alias so the node can
            be exercised directly with a pre-seeded field.

    Output state keys (partial dict):
        normalized_query:  str — NFC-normalized, whitespace-collapsed question
        detected_language: str — "JA", "VI", or "EN"
        status:            AgentStatus.SUCCESS or AgentStatus.ERROR
        error_log:         (on error) list of error messages
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # The inner graph receives the question as user_input: BaseGraph.invoke()
        # writes whatever the outer main slot handed over into that field. Reading
        # `query` alone left this node blank on every real invocation while unit
        # tests that seed `query` by hand kept passing.
        raw_query = state.get("query") or state.get("user_input") or ""

        if not isinstance(raw_query, str):
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [f"QueryNormalizeNode: question must be a string, " f"got {type(raw_query).__name__}"],
            }

        if not raw_query.strip():
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["QueryNormalizeNode: question is empty or blank"],
            }

        language = _detect_language(raw_query)
        normalized = _normalize_query(raw_query, language)

        logger.info(
            "QueryNormalizeNode: detected_language=%s normalized_length=%d",
            language,
            len(normalized),
        )
        emit_trace_event(
            "query_normalized",
            {"language": language, "length": len(normalized)},
            state,
        )

        return {
            "normalized_query": normalized,
            "detected_language": language,
            "status": AgentStatus.SUCCESS.value,
        }
