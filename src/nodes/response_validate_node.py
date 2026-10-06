"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - Return the status as AgentStatus.<X>.value — the enum's string value,
#    not the enum object itself
#  - Never import from mediator/, api/, or other agents
#
# MFG-C2-021 — ResponseValidateNode
# The inner workflow's output gate. Validates the generated answer for:
#   1. Non-empty and within max_response_chars (from config/config.yaml)
#   2. Absence of credential-shaped content
# Out-of-scope passthrough:
#   When state["out_of_scope"] is True the canned reply is released unchecked
#   for length but still scanned — a canned string cannot be over-length, and a
#   scan that skips a path is a scan with a hole in it.

import logging
import re
from typing import Any, ClassVar, Dict, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from framework.security.credential_detector import detect_credentials
from shared.utils.audit_logger import emit_trace_event

logger = logging.getLogger(__name__)

DEFAULT_MAX_RESPONSE_CHARS: int = 10000

# Shapes the framework's own detector does not carry. The framework set is the
# floor, never the ceiling: a local set NARROWER than the framework's is a
# containment bypass, because a value the framework catches and this node
# misses makes the framework's own gate raise after execute() returns, and the
# wrapper then discards this node's whole delta — the clearing included. These
# patterns are additive only.
_EXTRA_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    # pk- / ak- prefixed keys (the framework covers sk- only)
    ("prefixed_key", re.compile(r"\b(?:pk|ak)-[A-Za-z0-9]{16,}")),
    # Credential assignment in prose: "password: ...", "api_key = ..."
    (
        "credential_assignment",
        re.compile(
            r"\b(?:password|passwd|secret|api[_-]?key|token|access[_-]?key|private[_-]?key)" r"\s*[:=]\s*\S{8,}",
            re.IGNORECASE,
        ),
    ),
)

# Text released in place of a withheld answer. It has to be TRUTHY: the
# framework resolves an agent's output as `formatted_output or result`, so an
# empty string falls straight through to the ungated answer it was meant to
# replace.
WITHHELD_NOTICE = (
    "The generated answer was withheld by the output gate and has not been released. "
    "No content from it is included in this response."
)

# The complete inventory of state fields this workflow can release content
# through, each with the empty value the gate overwrites it with. Pinned here so
# a test can assert it, and so a field added later that carries released content
# cannot join the state without a visible decision about whether it is cleared.
ANSWER_BEARING_FIELDS: Tuple[Tuple[str, Any], ...] = (
    ("response", ""),
    ("result", ""),
    ("extracted_procedures", []),
)


def _scan(content: str) -> Optional[str]:
    """Return a violation TYPE when *content* carries credential-shaped text.

    Only the type is returned — never the matched value. The framework's output
    gate scans every value a node returns, so quoting the match into an error
    message makes that gate raise and discard this node's delta, taking the
    clearing with it.
    """
    findings = detect_credentials(content)
    if findings:
        return str(findings[0]["type"])
    for name, pattern in _EXTRA_PATTERNS:
        if pattern.search(content):
            return name
    return None


class ResponseValidateNode(FunctionNode):
    """Output gate — validate the answer before it leaves the inner workflow.

    Responsibilities:
    - Verify state["response"] is non-empty.
    - Verify length does not exceed max_response_chars (from config.yaml).
    - Scan for credential-shaped content using the framework detector plus the
      additional shapes above.
    - On any violation: return ERROR **and clear every field carrying answer
      text**, replacing the released text with a truthy withheld notice.

    Returning ERROR without clearing is not containment. The framework resolves
    an agent's output as `formatted_output or result` with no status check, so
    an error status alone still ships the ungated answer inside the error
    envelope.

    Input state keys:
        response:      the generated answer to validate
        out_of_scope:  (optional) True when the question had no source to cite

    Output state keys (partial dict):
        is_valid_response: bool — True if validation passed
        status:            AgentStatus.SUCCESS or AgentStatus.ERROR
        error_message:     (on failure) reason string, naming no content
        response/result:   (on failure) cleared
        formatted_output:  (on failure) the withheld notice
    """

    # The entry gate is VERIFIED_EXTERNAL (the level this agent is registered
    # at). Declaring a higher level on an inner node does not add protection —
    # the caller's trust travels unchanged into the subgraph, so no caller who
    # cleared the entry gate can ever satisfy it, and the gate denies itself on
    # every request instead of gating anything.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Args:
            config: The ``response`` block of config/config.yaml.
                ``max_response_chars`` is read from it, so the declared value is
                the value enforced.
        """
        super().__init__()
        settings: Dict[str, Any] = config or {}
        self._max_chars = int(settings.get("max_response_chars", DEFAULT_MAX_RESPONSE_CHARS))

    def _withhold(self, reason: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Refuse, and clear every field that carries answer text or a payload.

        ANSWER_BEARING_FIELDS is the complete inventory for this workflow's
        state, pinned so a field added later that carries released content has
        to make a visible decision about joining it. Inert provenance —
        safety_flags, is_valid_response, node_history — stays: it is a set of
        matched vocabulary terms and booleans, and it is what lets the caller
        see that a gate fired rather than that the pipeline vanished.
        """
        emit_trace_event(
            "response_withheld",
            {"reason": reason},
            state,
        )
        logger.error("ResponseValidateNode: OUTPUT WITHHELD — %s", reason)
        cleared: Dict[str, Any] = {
            "is_valid_response": False,
            "status": AgentStatus.ERROR.value,
            "error_message": f"ResponseValidateNode: output withheld ({reason})",
            "formatted_output": WITHHELD_NOTICE,
        }
        cleared.update(dict(ANSWER_BEARING_FIELDS))
        return cleared

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        response_raw = state.get("response")
        response: str = response_raw if isinstance(response_raw, str) else ""
        out_of_scope = bool(state.get("out_of_scope"))

        # --- Check 1: non-empty ---
        if not response.strip():
            logger.warning("ResponseValidateNode: response is empty")
            return {
                "is_valid_response": False,
                "status": AgentStatus.ERROR.value,
                "error_message": "ResponseValidateNode: response is empty",
            }

        # --- Check 2: length limit (a canned out-of-scope reply is exempt) ---
        if not out_of_scope and len(response) > self._max_chars:
            return self._withhold("length_cap", state)

        # --- Check 3: credential-shape scan, on every path ---
        violation = _scan(response)
        if violation:
            return self._withhold(f"credential_shape:{violation}", state)

        logger.info(
            "ResponseValidateNode: PASS — response %d chars, no credential shapes",
            len(response),
        )
        emit_trace_event(
            "response_validate",
            {"is_valid": True, "violation": None, "out_of_scope": out_of_scope},
            state,
        )
        return {
            "is_valid_response": True,
            "status": AgentStatus.SUCCESS.value,
        }
