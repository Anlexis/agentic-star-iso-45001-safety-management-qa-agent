"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - Return the status as AgentStatus.<X>.value — the enum's string value,
#    not the enum object itself
#  - Never import from mediator/, api/, or other agents
#
# MFG-C2-021 — SafetyFlagCheckNode
# Life-safety escalation gate.
# Scans normalized_query and extracted_procedures for emergency keywords
# across EN / JA / VI vocabularies. On any match sets is_emergency_escalation=True
# and records the matched keywords in safety_flags.  Escalation is a *valid*
# output — status remains SUCCESS regardless of whether a match is found.
#
# Security notes:
#   The life-safety escalation decision is security-critical and must be
#   auditable: every execution emits a trace event recording the outcome,
#   including the execution that decides not to escalate.

import logging
from typing import Any, ClassVar, Dict, List, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel

# Audit trail — a free function, not a node method.
from shared.utils.audit_logger import emit_trace_event

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default emergency keyword lists (EN / JA / VI).
# Operators may override via node_config["emergency_keywords"].
# ---------------------------------------------------------------------------

_DEFAULT_KEYWORDS_EN: List[str] = [
    "emergency",
    "fatality",
    "fire",
    "evacuation",
    "explosion",
    "injury",
    "lockout",
    "hazmat",
]

_DEFAULT_KEYWORDS_JA: List[str] = [
    "緊急",
    "死亡",
    "火災",
    "避難",
    "爆発",
    "負傷",
    "立入禁止",
]

_DEFAULT_KEYWORDS_VI: List[str] = [
    "khẩn cấp",
    "tử vong",
    "hỏa hoạn",
    "sơ tán",
    "nổ",
    "thương tích",
]

_DEFAULT_KEYWORDS: List[str] = _DEFAULT_KEYWORDS_EN + _DEFAULT_KEYWORDS_JA + _DEFAULT_KEYWORDS_VI


def _question_text(state: Dict[str, Any]) -> str:
    """Return the caller's question, the text the escalation decision is made from."""
    normalized_query = state.get("normalized_query", "")
    return normalized_query if isinstance(normalized_query, str) else ""


def _procedure_text(state: Dict[str, Any]) -> str:
    """Flatten the extracted procedures into one scan string.

    ProcedureExtractNode emits a list of ``{"clause", "title", "content"}``
    mappings. Reading only strings and lists-of-strings meant this text was
    never scanned on any real invocation: the shape it looked for is a shape
    the pipeline never produces.
    """
    parts: List[str] = []
    extracted_procedures = state.get("extracted_procedures", "")
    if isinstance(extracted_procedures, str) and extracted_procedures:
        parts.append(extracted_procedures)
    elif isinstance(extracted_procedures, list):
        for item in extracted_procedures:
            if isinstance(item, str) and item:
                parts.append(item)
            elif isinstance(item, dict):
                for key in ("title", "content"):
                    value = item.get(key)
                    if isinstance(value, str) and value:
                        parts.append(value)
    return " ".join(parts)


def _find_matched_keywords(text: str, keywords: List[str]) -> List[str]:
    """Return the subset of keywords that appear (case-insensitive) in text."""
    text_lower = text.lower()
    matched: List[str] = []
    for kw in keywords:
        # For JA/VI keywords use simple substring match (no word-boundary).
        if kw.lower() in text_lower:
            matched.append(kw)
    return matched


class SafetyFlagCheckNode(FunctionNode):
    """Life-safety escalation gate for MFG-C2-021.

    Two scan surfaces with different consequences, and the difference matters:

    - The **question** decides escalation. A caller writing "there is a fire in
      bay 3" is reporting a live situation, and the agent answers with the
      escalation notice instead of a procedure summary.
    - The **retrieved procedures** contribute hazard terms to safety_flags but
      never trigger escalation on their own. Escalating on procedure text would
      escalate every fire-response question ever asked, because a fire-response
      clause contains the word fire — a fail-closed screen firing on exactly the
      domain text the agent exists to serve.

    Escalation is a valid output; status is always AgentStatus.SUCCESS unless an
    internal error occurs.

    Input state keys:
        normalized_query:      str  — cleaned question (from normalization)
        extracted_procedures:  list[dict] — {"clause", "title", "content"} entries

    Output state keys (partial dict):
        is_emergency_escalation: bool — True when the QUESTION carries a hazard term
        safety_flags:            list[str] — hazard terms from the question first,
                                 then any further terms found in the cited procedures
        status:                  AgentStatus.SUCCESS (always on normal execution)
    """

    # The entry gate is VERIFIED_EXTERNAL (the level this agent is registered
    # at). Declaring a higher level on an inner node does not add protection —
    # the caller's trust travels unchanged into the subgraph, so no caller who
    # cleared the entry gate can ever satisfy it, and the escalation check
    # denies itself on every request.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(self, keywords: Optional[List[str]] = None) -> None:
        """
        Args:
            keywords: Optional replacement for the built-in escalation
                vocabulary. A deployment with its own hazard terminology passes
                it at construction; the graph owns the wiring.
        """
        super().__init__()
        self._keywords: List[str] = list(keywords) if keywords else list(_DEFAULT_KEYWORDS)

    def execute(self, state: AgentState) -> Dict[str, Any]:
        keywords: List[str] = self._keywords

        question_flags = _find_matched_keywords(_question_text(state), keywords)
        procedure_flags = [
            keyword
            for keyword in _find_matched_keywords(_procedure_text(state), keywords)
            if keyword not in question_flags
        ]
        is_emergency = bool(question_flags)
        safety_flags = question_flags + procedure_flags

        # The escalation decision is security-critical and must be auditable on
        # every execution, including the one that decides not to escalate.
        emit_trace_event(
            "safety_flag_check",
            {
                "is_emergency": is_emergency,
                "question_flag_count": len(question_flags),
                "procedure_flag_count": len(procedure_flags),
            },
            state,
        )

        if is_emergency:
            logger.warning(
                "SafetyFlagCheckNode: life-safety terms in the question: %s",
                question_flags,
            )
        else:
            logger.info(
                "SafetyFlagCheckNode: no life-safety terms in the question "
                "(%d term(s) noted in the cited procedures)",
                len(procedure_flags),
            )

        return {
            "is_emergency_escalation": is_emergency,
            "safety_flags": safety_flags,
            "status": AgentStatus.SUCCESS.value,
        }
