"""AgentCore Platform v1.0"""

# MFG-C2-021 — PostProcessNode
# The outer backbone's output gate. It renders the answer the inner workflow
# released and scans it once more before it leaves the agent.
#
# Two layers, deliberately independent: the inner workflow's gate runs inside
# the subgraph and this one runs at the agent boundary. Neither is a substitute
# for the other — the inner one can be bypassed by a change to the inner
# topology, and this one never sees a run the inner graph refused.

import re
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from framework.security.credential_detector import detect_credentials
from shared.utils.audit_logger import emit_trace_event
from src.services.failure_message import EMPTY_INPUT, INPUT_REJECTED, INVALID_VALUE, TOO_LONG

# Shapes the framework's own detector does not carry. The framework set is the
# floor: a local set NARROWER than the framework's is a containment bypass —
# a value the framework catches and this node misses makes the framework's gate
# raise after execute() returns, and the wrapper then discards this node's whole
# delta, clearing included, leaving the ungated answer in state for the output
# resolver to fall back to. These patterns are additive only.
_EXTRA_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("prefixed_key", re.compile(r"\b(?:pk|ak)-[A-Za-z0-9]{16,}")),
    (
        "credential_assignment",
        re.compile(
            r"\b(?:password|passwd|secret|api[_-]?key|token|access[_-]?key|private[_-]?key)" r"\s*[:=]\s*\S{8,}",
            re.IGNORECASE,
        ),
    ),
)

# Released in place of a withheld answer. TRUTHY on purpose: the framework
# resolves an agent's output as `formatted_output or result` with no status
# check, so an empty string here falls straight through to the ungated answer.
WITHHELD_NOTICE = (
    "The answer was withheld by the output gate and has not been released. "
    "No content from it is included in this response."
)

# Every field in this agent's state that can carry released answer text.
# Pinned so a test can assert it and a new answer-bearing field cannot join the
# state without a visible decision about whether this gate clears it.
ANSWER_BEARING_FIELDS: List[str] = ["result", "response", "formatted_output"]


def _scan(text: str) -> Optional[str]:
    """Return the violation TYPE when *text* carries credential-shaped content.

    Only the type is returned — never the matched value. The framework's output
    gate scans every value this node returns, so quoting the match makes that
    gate raise and discard the delta, taking the clearing with it.
    """
    findings = detect_credentials(text)
    if findings:
        return str(findings[0]["type"])
    for name, pattern in _EXTRA_PATTERNS:
        if pattern.search(text):
            return name
    return None


# Reason code -> the sentence the caller reads. A code with no entry falls
# back to the generic one rather than leaking the code itself.
_DEGRADED_MESSAGES = {
    "EMPTY_INPUT": EMPTY_INPUT,
    "QUESTION_TOO_LONG": TOO_LONG,
    "INVALID_REQUEST": INVALID_VALUE,
}


class PostProcessNode(FunctionNode):
    """Render the released answer and gate it at the agent boundary."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # A run declined upstream has nothing to format. Render the reason as
        # the caller-facing body and carry the marker onward.
        marker = state.get("error_code")
        if marker:
            message = _DEGRADED_MESSAGES.get(marker, INPUT_REJECTED)
            emit_trace_event("post_process_degraded", {"reason": marker}, state)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": marker,
                "formatted_output": message,
                "result": message,
                "response": message,
            }
        result = state.get("result", "")

        # Genuine formatting work: render the result into the output field.
        formatted_output = result if isinstance(result, str) else str(result)

        violation = _scan(formatted_output)
        if violation:
            emit_trace_event(
                "output_withheld",
                {"reason": f"credential_shape:{violation}"},
                state,
            )
            # Returning ERROR is not enough on its own. Every field carrying
            # answer text is overwritten here, because the framework resolves
            # the agent's output as `formatted_output or result` with no status
            # check — an error status alone still ships the ungated answer
            # inside the error envelope.
            return {
                "status": AgentStatus.ERROR.value,
                "result": "",
                "response": "",
                "formatted_output": WITHHELD_NOTICE,
                "error_log": [f"PostProcessNode: output withheld (credential_shape:{violation})"],
            }

        emit_trace_event(
            "output_released",
            {"length": len(formatted_output)},
            state,
        )
        return {
            "formatted_output": formatted_output,
            "status": AgentStatus.SUCCESS.value,
        }
