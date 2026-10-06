"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - Return the status as AgentStatus.<X>.value — the enum's string value,
#    not the enum object itself
#  - Never import from mediator/, api/, or other agents
#
# MFG-C2-021 — ResponseGenerateNode
# Slot: main  (the primary response node in the inner domain graph)
#
# Three execution paths:
#   (a) Emergency escalation: state["is_emergency_escalation"] == True
#       -> prefixed escalation message with detected language support (JA/VI/EN)
#   (b) Scope exceeded: state["out_of_scope"] == True
#       -> canned reply for queries outside the ISO 45001 knowledge base
#   (c) Normal: deterministic synthesis of state["extracted_procedures"]
#       -> structured Q&A response citing ISO 45001 clauses
#
# Deterministic synthesis only — no model call. The answer is assembled from
# the clauses the extractor found, so it is reproducible and every sentence in
# it is traceable to a source document.

import logging
from typing import Any, ClassVar, Dict, List, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

# Audit trail — a free function, not a node method.
from shared.utils.audit_logger import emit_trace_event

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language-localised response templates
# ---------------------------------------------------------------------------
_ESCALATION_TEMPLATES: Dict[str, str] = {
    "JA": ("⚠️ 緊急事態が検出されました。" "直ちに安全担当者に連絡してください。" "【安全フラグ: {safety_flags}】"),
    "VI": (
        "⚠️ Phát hiện tình huống khẩn cấp."
        " Vui lòng liên hệ ngay với nhân viên an toàn của bạn."
        " [Cờ an toàn: {safety_flags}]"
    ),
    "EN": (
        "⚠️ Emergency detected." " Please contact your safety officer immediately." " [Safety flags: {safety_flags}]"
    ),
}

# Canned replies for queries that fall outside the knowledge base scope.
_SCOPE_EXCEEDED_TEMPLATES: Dict[str, str] = {
    "JA": "このクエリは ISO 45001 知識ベースの範囲外です。",
    "VI": "Truy vấn này nằm ngoài phạm vi của cơ sở tri thức ISO 45001.",
    "EN": "This query is outside the scope of the ISO 45001 knowledge base.",
}

_SUPPORTED_LANGUAGES = {"JA", "VI", "EN"}
_DEFAULT_LANGUAGE = "EN"


def _resolve_language(detected_language: Optional[str]) -> str:
    """Return a supported language code, defaulting to EN."""
    if detected_language and detected_language.upper() in _SUPPORTED_LANGUAGES:
        return detected_language.upper()
    return _DEFAULT_LANGUAGE


def _format_safety_flags(safety_flags: Any) -> str:
    """Convert safety_flags (list or string) to a compact display string."""
    if isinstance(safety_flags, list):
        return ", ".join(str(f) for f in safety_flags) if safety_flags else "(none)"
    return str(safety_flags) if safety_flags else "(none)"


def _synthesize_procedures(
    extracted_procedures: Any,
    user_input: str,
    language: str,
) -> str:
    """Deterministically synthesise a Q&A response from extracted_procedures.

    Args:
        extracted_procedures: list of procedure dicts with keys
            "clause", "title", "content" (all str) — matches the output of
            ProcedureExtractNode.
        user_input: original query from the user.
        language: resolved language code (JA/VI/EN).

    Returns:
        Structured markdown answer string.
    """
    procedures: List[Dict[str, Any]] = []
    if isinstance(extracted_procedures, list):
        procedures = [p for p in extracted_procedures if isinstance(p, dict)]

    if not procedures:
        _no_result: Dict[str, str] = {
            "JA": (
                "ご質問に関連する ISO 45001 条項が見つかりませんでした。" "詳細については安全担当者にご確認ください。"
            ),
            "VI": (
                "Không tìm thấy điều khoản ISO 45001 liên quan đến truy vấn của bạn."
                " Vui lòng tham khảo nhân viên an toàn để biết thêm chi tiết."
            ),
            "EN": (
                "No relevant ISO 45001 clauses were found for your query."
                " Please consult your safety officer for further details."
            ),
        }
        return _no_result.get(language, _no_result[_DEFAULT_LANGUAGE])

    # Build structured answer header.
    _headers: Dict[str, str] = {
        "JA": f"**ISO 45001 安全管理システム — Q&A**\n\n**質問:** {user_input}\n",
        "VI": (f"**ISO 45001 Hệ thống quản lý an toàn — Q&A**\n\n" f"**Câu hỏi:** {user_input}\n"),
        "EN": f"**ISO 45001 Safety Management System — Q&A**\n\n**Query:** {user_input}\n",
    }
    header = _headers.get(language, _headers[_DEFAULT_LANGUAGE])

    # Build per-clause blocks.
    clause_blocks: List[str] = []
    for proc in procedures:
        clause = str(proc.get("clause", "")).strip()
        title = str(proc.get("title", "")).strip()
        content = str(proc.get("content", "")).strip()
        if not content:
            continue
        if clause and title:
            heading = f"### {clause} — {title}"
        elif clause:
            heading = f"### {clause}"
        elif title:
            heading = f"### {title}"
        else:
            heading = "### ISO 45001 Reference"
        clause_blocks.append(f"{heading}\n\n{content}")

    if not clause_blocks:
        _no_content: Dict[str, str] = {
            "JA": "抽出された手順にコンテンツが含まれていません。安全担当者にご確認ください。",
            "VI": ("Các quy trình được trích xuất không chứa nội dung." " Vui lòng tham khảo nhân viên an toàn."),
            "EN": ("The extracted procedures contain no content." " Please consult your safety officer."),
        }
        return header + "\n" + _no_content.get(language, _no_content[_DEFAULT_LANGUAGE])

    clauses_section = "\n\n".join(clause_blocks)

    _footers: Dict[str, str] = {
        "JA": (
            f"\n\n---\n*参照: {len(clause_blocks)} 件の ISO 45001 条項 | " "本回答は知識ベースの内容に基づいています。*"
        ),
        "VI": (
            f"\n\n---\n*Tham chiếu: {len(clause_blocks)} điều khoản ISO 45001 | "
            "Câu trả lời này dựa trên nội dung cơ sở tri thức.*"
        ),
        "EN": (
            f"\n\n---\n*References: {len(clause_blocks)} ISO 45001 clause(s) cited | "
            "This answer is based on the knowledge base content.*"
        ),
    }
    footer = _footers.get(language, _footers[_DEFAULT_LANGUAGE])

    return header + "\n" + clauses_section + footer


class ResponseGenerateNode(FunctionNode):
    """Generate the final Q&A response for ISO 45001 / Safety Management System queries.

    Slot: main (inner domain graph, MFG-C2-021).

    Three execution paths:

    (a) Emergency escalation (``state["is_emergency_escalation"] == True``):
        Returns a language-localised prefix-marked escalation message listing
        ``state["safety_flags"]``.

    (b) Scope exceeded (``state["out_of_scope"] == True``):
        Returns a canned reply for queries outside the knowledge base.

    (c) Normal:
        Deterministically synthesises ``state["extracted_procedures"]`` into
        a structured markdown Q&A response citing ISO 45001 clauses.

    Input state keys:
        user_input:               the question, as handed to the inner graph
        is_emergency_escalation:  bool — trigger path (a)
        safety_flags:             list[str] | str — flags for path (a)
        detected_language:        str — "JA" | "VI" | "EN"
        out_of_scope:             bool — trigger path (b)
        extracted_procedures:     list[dict] — used in path (c)

    Output state keys (partial dict):
        response:  str — the generated response text
        result:    str — same as response (agent result slot)
        status:    AgentStatus.SUCCESS or AgentStatus.ERROR
        error_log: list[str] — populated only on ERROR
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Resolve language early — used across all paths.
        detected_language: str = _resolve_language(state.get("detected_language"))

        # ------------------------------------------------------------------
        # Path (a): Emergency escalation
        # ------------------------------------------------------------------
        is_emergency: bool = bool(state.get("is_emergency_escalation", False))
        if is_emergency:
            safety_flags_raw = state.get("safety_flags", [])
            flags_str = _format_safety_flags(safety_flags_raw)
            template = _ESCALATION_TEMPLATES.get(detected_language, _ESCALATION_TEMPLATES[_DEFAULT_LANGUAGE])
            response_text = template.format(safety_flags=flags_str)
            logger.info(
                "ResponseGenerateNode: emergency path — language=%s flags=%s",
                detected_language,
                flags_str,
            )
            # Audit trail: record which answer path produced the response.
            emit_trace_event(
                "response_generated",
                {"path": "emergency", "language": detected_language},
                state,
            )
            return {
                "response": response_text,
                "result": response_text,
                "status": AgentStatus.SUCCESS.value,
            }

        # ------------------------------------------------------------------
        # Path (b): Query outside knowledge base scope
        # ------------------------------------------------------------------
        is_beyond_scope: bool = bool(state.get("out_of_scope", False))
        if is_beyond_scope:
            response_text = _SCOPE_EXCEEDED_TEMPLATES.get(
                detected_language, _SCOPE_EXCEEDED_TEMPLATES[_DEFAULT_LANGUAGE]
            )
            logger.info(
                "ResponseGenerateNode: beyond-scope path — language=%s",
                detected_language,
            )
            # Audit trail: record which answer path produced the response.
            emit_trace_event(
                "response_generated",
                {"path": "out_of_scope", "language": detected_language},
                state,
            )
            return {
                "response": response_text,
                "result": response_text,
                "status": AgentStatus.SUCCESS.value,
            }

        # ------------------------------------------------------------------
        # Path (c): Normal — deterministic synthesis from the extracted clauses
        # ------------------------------------------------------------------
        user_input: str = str(state.get("user_input", "")).strip()
        extracted_procedures: Any = state.get("extracted_procedures", [])

        if not user_input:
            err = "ResponseGenerateNode: user_input is empty — cannot generate response"
            logger.error(err)
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [err],
            }

        try:
            response_text = _synthesize_procedures(
                extracted_procedures=extracted_procedures,
                user_input=user_input,
                language=detected_language,
            )
        except Exception as exc:  # noqa: BLE001
            err = f"ResponseGenerateNode: synthesis failed — {exc}"
            logger.error(err)
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [err],
            }

        logger.info(
            "ResponseGenerateNode: normal path — language=%s procedures=%d response_len=%d",
            detected_language,
            len(extracted_procedures) if isinstance(extracted_procedures, list) else 0,
            len(response_text),
        )
        # Audit trail: record which answer path produced the response.
        emit_trace_event(
            "response_generated",
            {
                "path": "normal",
                "language": detected_language,
                "procedure_count": (len(extracted_procedures) if isinstance(extracted_procedures, list) else 0),
            },
            state,
        )
        return {
            "response": response_text,
            "result": response_text,
            "status": AgentStatus.SUCCESS.value,
        }
