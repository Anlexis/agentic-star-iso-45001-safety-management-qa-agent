"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - Return the status as AgentStatus.<X>.value — the enum's string value,
#    not the enum object itself
#  - Read input_context via state.get("input_context", {}) — read-only
#  - Never import from mediator/, api/, or other agents
#
# MFG-C2-021 — PreProcessNode
# The single gate for caller data. Everything downstream — the outer main slot,
# the whole inner workflow, the renderer — consumes only what this node has
# validated. Nothing here echoes a rejected value: an error names the field and
# the reason, never the content that failed.

import json
import math
import re
import unicodedata
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from framework.security.injection_policy import evaluate_untrusted_content
from framework.security.pii_detector import detect_pii
from framework.security.pii_masking import mask_pii
from shared.utils.audit_logger import emit_trace_event
from src.services.failure_message import INPUT_REJECTED
from src.services.progress import emit_progress

# Upper bound on accepted query length (manufacturing-safety questions are
# short; anything past this is treated as abuse or an accidental paste).
MAX_QUERY_LEN: int = 2000

# Fallback bounds. The live values come from config/config.yaml through the
# constructor; these apply only when a bound is absent from that file.
DEFAULT_MAX_CALLER_DOCUMENTS: int = 20
DEFAULT_MAX_DOCUMENT_CHARS: int = 20000

# Cap on the JSON request envelope accepted through `user_input`. A client with
# no structured channel — the chat box is one — carries the whole request in the
# text field, so the question bound alone cannot also bound the documents.
MAX_REQUEST_ENVELOPE_CHARS: int = 65536

# Widest top_k a caller may ask for. Beyond this the request is refused rather
# than clamped, so a caller never silently gets a different search than the one
# they asked for.
MAX_CALLER_TOP_K: int = 20

# Caller strings that reach the rendered answer are locked to an inert
# identifier alphabet. Free text in those positions is output injection.
_INERT_IDENTIFIER_RE = re.compile(r"^[a-z0-9_]{1,32}$")

# Personal-data categories masked in caller reference documents.
#
# The framework's input gate covers user_input only, so nothing masks this
# channel unless the template does — that part is purely additive.
#
# The generic title-case "name" category is deliberately not applied to
# reference documents. Its pattern is a run of capitalised words separated by
# whitespace, and whitespace includes a newline, so a standards heading
# followed by a sentence — "Competence\nThe organization shall …" — is read as
# a two-word personal name and masked. Measured on this agent's own corpus: the
# clause heading vanished, the extractor could no longer find the clause, and
# the answer silently dropped it. The category is retained everywhere it works
# (labelled names, and the framework's own scan of the question), and every
# structured identifier category below — card numbers, national IDs, e-mail,
# phone — still applies here.
_DOCUMENT_PII_TYPES = frozenset({"credit_card", "ssn_us", "my_number_jp", "email", "phone_jp", "phone_us", "name_jp"})

# Every context key this agent consumes. Anything else a CALLER sends is
# dropped before the payload travels: validators that merely ignore an unknown
# key leave it in play for whatever reads the mapping next.
_ALLOWED_CONTEXT_KEYS = frozenset({"channel", "kb_documents", "top_k"})
_ALLOWED_DOCUMENT_KEYS = frozenset({"doc_id", "content"})

# Fields the hosting runtime places on the context channel itself. They are not
# caller parameters, this agent consumes none of them, and they arrive on every
# invocation served that way — refusing them would refuse all of those, and the
# caller cannot remove a field it never added. They are accepted and ignored,
# which is sound precisely because no constraint is attached to them: nothing
# is read from them and nothing is promised about them, so there is no wrong
# expectation for the refusal above to correct.
#
# They are also held out of the structural screen below, and for the same
# reason: that screen guards a channel this agent reads. Screening prose the
# agent never looks at would turn an ordinary earlier turn into a refusal
# without protecting anything — the field is dropped, not consumed.
_RUNTIME_CONTEXT_KEYS = frozenset({"conversation_history"})

# Control-character strip: remove C0 control chars EXCEPT tab (\x09) and
# newline (\x0a). Keeps the full JA / VI / EN unicode range intact.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

# Zero-width and soft-hyphen characters, used to split a directive so a naive
# substring screen no longer sees it.
_ZERO_WIDTH_RE = re.compile(r"[​‌‍⁠﻿­]")

# ---------------------------------------------------------------------------
# Refusal screen
# ---------------------------------------------------------------------------
#
# Two families, refused rather than stripped. Removing a marker and forwarding
# what surrounded it converts a detectable attack into undetectable plain text:
# the directive still reads as an instruction, and nothing downstream can tell
# it apart from the caller's question.
#
# Family 1 — chat-template control tokens. These forge a turn boundary in a
# prompt, and no manufacturing-safety question contains one. Screened as a
# class, not as a list of literals, so a template dialect we have not seen is
# still caught.
_CONTROL_TOKEN_PATTERNS: Tuple[re.Pattern[str], ...] = (
    # <|im_start|>, <|im_end|>, <|endoftext|> and any other <|...|> token
    re.compile(r"<\|[^|>]{0,64}\|>"),
    # [INST] / [/INST] / [SYS] / [/SYS]
    re.compile(r"\[/?(?:INST|SYS)\]", re.IGNORECASE),
    # <<SYS>> / <</SYS>>
    re.compile(r"<</?SYS>>", re.IGNORECASE),
    # <system> / </assistant> style role markers
    re.compile(r"<\s*/?\s*(?:system|user|assistant)\s*>", re.IGNORECASE),
    # ChatML-adjacent heading form
    re.compile(r"###\s*(?:system|instruction)\b", re.IGNORECASE),
)

# Family 2 — instruction-override directives. Anchored on a verb plus its
# object so ordinary safety prose survives: a procedure may legitimately say
# "ignore the previous revision of this checklist", and it does here, because
# the object has to be an instruction/prompt/rule word.
_DIRECTIVE_PATTERNS: Tuple[re.Pattern[str], ...] = (
    # EN
    re.compile(
        r"(?i)\b(?:ignore|disregard|forget|override)\s+(?:all\s+|any\s+|the\s+)*"
        r"(?:previous|prior|above|earlier|system)?\s*"
        r"(?:instruction|instructions|prompt|prompts|rule|rules|direction|directions)\b"
    ),
    re.compile(r"(?i)\byou\s+are\s+now\s+(?:a|an|the)\b"),
    re.compile(r"(?i)\bact\s+as\s+(?:a|an)\s+[a-z ]{0,24}\b(?:jailbroken|unrestricted|unfiltered|dan)\b"),
    re.compile(r"(?i)\b(?:reveal|print|show|repeat)\s+(?:me\s+)?(?:your|the)\s+system\s+prompt\b"),
    # JA
    re.compile(r"(?:以前|前|上記|これまで)の(?:指示|命令|ルール)を?(?:無視|忘れ)"),
    re.compile(r"システムプロンプトを?(?:表示|教え|出力)"),
    # VI
    re.compile(r"(?i)\bbỏ\s+qua\s+(?:tất\s+cả\s+)?(?:các\s+)?(?:hướng\s+dẫn|chỉ\s+dẫn|quy\s+tắc)\b"),
)


def _strip_control_chars(text: str) -> str:
    """Remove NUL and non-printable C0 control chars (keep tab and newline)."""
    return _CONTROL_CHARS_RE.sub("", text)


def _screen_forms(text: str) -> List[str]:
    """Return the forms of *text* an injection screen has to inspect.

    A single pass over the raw string is not enough. Zero-width characters and
    NFKC-foldable look-alikes are both ways to write a directive that reads
    normally to a person and matches nothing literal, so the same screen runs
    over the raw text, the text with those characters removed, and the
    compatibility-normalised form.
    """
    forms = [text]
    stripped = _ZERO_WIDTH_RE.sub("", text)
    if stripped != text:
        forms.append(stripped)
    folded = unicodedata.normalize("NFKC", stripped)
    if folded not in forms:
        forms.append(folded)
    return forms


def _screen_reason(text: str) -> Optional[str]:
    """Return a fixed reason code when *text* must be refused, else None.

    The reason codes are constants. They never carry the matched text: the
    framework's output gate scans every value a node returns for credential
    shapes, and quoting caller content into an error makes that gate raise and
    discard the node's whole delta — including any clearing it performed.
    """
    for form in _screen_forms(text):
        for pattern in _CONTROL_TOKEN_PATTERNS:
            if pattern.search(form):
                return "control_token"
        for pattern in _DIRECTIVE_PATTERNS:
            if pattern.search(form):
                return "instruction_override"
    return None


def _screen_structure(value: Any, path: str, depth: int = 0) -> Optional[str]:
    """Depth-first screen over a parsed structure, KEYS included.

    Keys are caller data too, and a directive hidden in a key name reaches the
    same places the values do. Escaped forms cannot evade this: the payload has
    already been parsed by the time it arrives, so `\\u0069gnore` is the plain
    string by now.

    Returns "<path>:<reason>" for the first offending position, else None.
    """
    if depth > 6:
        return f"{path}:nesting_depth"
    if isinstance(value, str):
        reason = _screen_reason(value)
        return f"{path}:{reason}" if reason else None
    if isinstance(value, dict):
        for index, (key, item) in enumerate(value.items(), start=1):
            key_label = key if isinstance(key, str) and _INERT_IDENTIFIER_RE.match(key) else f"#{index}"
            if isinstance(key, str):
                reason = _screen_reason(key)
                if reason:
                    return f"{path}.{key_label}:key_{reason}"
            found = _screen_structure(item, f"{path}.{key_label}", depth + 1)
            if found:
                return found
        return None
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = _screen_structure(item, f"{path}[{index}]", depth + 1)
            if found:
                return found
        return None
    return None


def _finite_in_range(value: Any, low: float, high: float) -> Optional[float]:
    """Parse *value* as a finite number inside [low, high], else None.

    Rejects bools (``isinstance(True, int)`` is True in Python), non-numerics,
    and NaN / +-Infinity. Those three parse through ``float()`` without
    complaint, and every comparison against NaN is False — so a NaN bound would
    silently disable the very check it configures. Fails CLOSED.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except (TypeError, ValueError):
            return None
    else:
        return None
    if not math.isfinite(number):
        return None
    if number < low or number > high:
        return None
    return number


class PreProcessNode(FunctionNode):
    """Validate the caller's question and structured context before processing.

    Input state keys:
        user_input:    str  — the raw question
        input_context: dict — optional structured caller parameters:
            channel      str            — inert identifier, recorded for audit
            kb_documents list[dict]     — {"doc_id": inert id, "content": text}
            top_k        int            — documents to carry forward, 1..20
        A field the hosting runtime places on that channel itself is accepted
        and dropped; it is not a caller parameter and nothing here reads it.

    Output state keys (partial dict):
        validated_input:  str        — screened, normalised question
        caller_documents: list[dict] — validated, masked knowledge-base entries
        caller_top_k:     int | None — validated caller override, absent if none
        enriched_context: dict       — audit-facing provenance
        status / error_log
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Bind the runtime bounds this node enforces.

        The bounds arrive from config/config.yaml through the graph, so a value
        declared there is the value enforced here rather than a constant that
        happens to agree with it.
        """
        super().__init__()
        retrieval: Dict[str, Any] = (config or {}).get("retrieval") or {}
        self._max_documents = int(retrieval.get("max_caller_documents", DEFAULT_MAX_CALLER_DOCUMENTS))
        self._max_document_chars = int(retrieval.get("max_document_chars", DEFAULT_MAX_DOCUMENT_CHARS))

    def _reject(self, reason: str, code: str = "INVALID_REQUEST") -> Dict[str, Any]:
        """Build the rejection delta. *reason* is a field name plus a code.

        Two ways to stop, and the caller can act on only one of them. A value
        the caller can correct completes the run carrying ``code``, so the
        reason reaches the caller and a corrected request can be sent on the
        same conversation. Content the agent refuses outright passes ``code=""``
        and terminates, so a refusal is never presented as something a reworded
        request would get past.

        The branch is chosen by the call site through ``code``, never by
        reading *reason*: the screening sites pass ``code=""`` and
        ``_validate_documents`` returns its own reason class for the same
        purpose.
        """
        if code:
            # A value the caller can correct: the run COMPLETES carrying the
            # reason so the request can be sent again on the same conversation.
            emit_progress(INPUT_REJECTED)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": code,
                "error_log": [f"PreProcessNode: input rejected ({reason})"],
            }
        return {
            "status": AgentStatus.ERROR.value,
            "error_log": [f"PreProcessNode: input rejected ({reason})"],
        }

    def _validate_documents(
        self, raw: Any, state: Dict[str, Any]
    ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str], str]:
        """Validate kb_documents; return (documents, rejection reason, reason class).

        The third element is the reason CLASS, carried alongside the reason so
        the caller of this method never has to read the reason text to tell the
        two kinds of rejection apart. A malformed document is a value the caller
        can correct and carries a reason code; content this agent refuses to
        act on carries ``""`` and terminates the run. Deciding that by matching
        on the reason string would put a refusal one careless edit away from
        being reported as a correctable value.
        """
        if not isinstance(raw, list):
            return None, "input_context.kb_documents:not_a_list", "INVALID_REQUEST"
        if len(raw) > self._max_documents:
            return None, "input_context.kb_documents:entry_cap", "INVALID_REQUEST"

        documents: List[Dict[str, Any]] = []
        for index, entry in enumerate(raw):
            label = f"input_context.kb_documents[{index}]"
            if not isinstance(entry, dict):
                return None, f"{label}:not_an_object", "INVALID_REQUEST"
            unknown = set(entry) - _ALLOWED_DOCUMENT_KEYS
            if unknown:
                return None, f"{label}:unknown_field", "INVALID_REQUEST"
            doc_id = entry.get("doc_id")
            if not isinstance(doc_id, str) or not _INERT_IDENTIFIER_RE.match(doc_id):
                return None, f"{label}.doc_id:not_an_identifier", "INVALID_REQUEST"
            content = entry.get("content")
            if not isinstance(content, str) or not content.strip():
                return None, f"{label}.content:empty", "INVALID_REQUEST"
            if len(content) > self._max_document_chars:
                return None, f"{label}.content:size_cap", "INVALID_REQUEST"

            # Caller documents become actionable content the moment the
            # extractor parses them, so they are evaluated at that boundary —
            # the framework's own input gate covers user_input only.
            checked = evaluate_untrusted_content(
                content,
                source="kb_documents",
                state=state,
                node_name=self.__class__.__name__,
            )
            if checked.get("status") == AgentStatus.ERROR.value:
                # Terminal: content this agent refuses to act on is not a value
                # the caller can correct, so it does not travel as one.
                return None, f"{label}.content:untrusted_content", ""

            # The framework masks personal data in user_input; nothing masks
            # the context channel, and a maintenance record plausibly carries
            # the contact details of the worker who filed it.
            findings = [f for f in detect_pii(content) if f.get("type") in _DOCUMENT_PII_TYPES]
            safe_content = mask_pii(content, findings) if findings else content

            documents.append({"doc_id": doc_id, "content": safe_content})
        return documents, None, "INVALID_REQUEST"

    @staticmethod
    def _split_envelope(user_input: str) -> Tuple[str, Dict[str, Any], Optional[Tuple[str, str]]]:
        """Split a JSON request envelope carried in ``user_input``.

        A caller with no structured channel sends one text field, so the request
        object travels there instead. Both channels then go through the identical
        validation below: the question is screened as a question and the context
        fields as context fields, so this adds a transport, not a second contract.

        Text that is not a JSON object is an ordinary question and is returned
        unchanged — only a caller who clearly intended an envelope gets an
        envelope error.

        Returns ``(question, context, rejection)`` where ``rejection`` is
        ``(reason, code)``.
        """
        stripped = user_input.strip()
        if not (stripped.startswith("{") and stripped.endswith("}")):
            return user_input, {}, None
        if len(stripped) > MAX_REQUEST_ENVELOPE_CHARS:
            # The envelope as a whole is oversized, which is a different fix from
            # a question past its own bound: shorten the documents, not the ask.
            return user_input, {}, ("user_input:envelope_length_cap", "QUESTION_TOO_LONG")
        try:
            parsed = json.loads(stripped)
        except (TypeError, ValueError):
            return user_input, {}, None
        if not isinstance(parsed, dict):
            return user_input, {}, None
        question = parsed.get("question")
        if not isinstance(question, str):
            return user_input, {}, ("user_input:envelope_without_question", "INVALID_REQUEST")
        context = {k: v for k, v in parsed.items() if k != "question"}
        return question, context, None

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_input = state.get("user_input", "")
        raw_context = state.get("input_context", {})  # read-only

        if not isinstance(user_input, str) or not user_input.strip():
            return self._reject("user_input:empty", code="EMPTY_INPUT")

        # A caller with no structured channel carries the whole request in the
        # text field. The envelope is read first so the question is a question
        # on both channels; where both carry the same context field the
        # structured channel wins, because it is the declared contract.
        if not isinstance(raw_context, dict):
            return self._reject("input_context:not_an_object")
        user_input, envelope_context, rejection = self._split_envelope(user_input)
        if rejection is not None:
            return self._reject(rejection[0], code=rejection[1])
        if envelope_context:
            raw_context = {**envelope_context, **raw_context}
        if not user_input.strip():
            return self._reject("user_input:empty", code="EMPTY_INPUT")

        if len(user_input) > MAX_QUERY_LEN:
            return self._reject("user_input:length_cap", code="QUESTION_TOO_LONG")

        # Strip control characters first, then screen. Screening before a
        # transformation is not screening: the transformation is what decides
        # the string every later reader sees.
        sanitized = _strip_control_chars(user_input)
        for candidate in (user_input, sanitized):
            reason = _screen_reason(candidate)
            if reason:
                emit_trace_event(
                    "input_screen_refused",
                    {"field": "user_input", "reason": reason},
                    state,
                )
                # Terminal: a refusal, not a correctable value. Rewording the
                # request must not be presented as a route past it.
                return self._reject(f"user_input:{reason}", code="")

        sanitized = " ".join(sanitized.split())
        if not sanitized:
            return self._reject("user_input:empty_after_normalisation", code="EMPTY_INPUT")

        if not isinstance(raw_context, dict):
            return self._reject("input_context:not_an_object")

        # Only the caller-authored portion is screened and validated; the
        # runtime's own fields are neither read nor carried forward.
        caller_context = {k: v for k, v in raw_context.items() if k not in _RUNTIME_CONTEXT_KEYS}

        # The screen runs BEFORE the unknown-key check, because both rules match
        # a directive written into a key name and only one of them is the true
        # reason. Checked the other way round, a key carrying an override reads
        # as "a field this agent does not read" - a correctable value - and the
        # refusal is never reported as one.
        offending = _screen_structure(caller_context, "input_context")
        if offending:
            emit_trace_event(
                "input_screen_refused",
                {"field": offending.rsplit(":", 1)[0], "reason": offending.rsplit(":", 1)[1]},
                state,
            )
            # Terminal for the same reason as the user_input screen: a
            # directive hidden in the structured channel is the same refusal.
            return self._reject(offending, code="")

        unknown = set(caller_context) - _ALLOWED_CONTEXT_KEYS
        if unknown:
            # Named rather than silently dropped: a caller who sent a field this
            # agent does not read has a wrong expectation about the answer, and
            # a silent drop hides that until the answer is already wrong.
            return self._reject("input_context:unknown_field")

        channel = caller_context.get("channel", "unknown")
        if not isinstance(channel, str) or not _INERT_IDENTIFIER_RE.match(channel):
            if "channel" in caller_context:
                return self._reject("input_context.channel:not_an_identifier")
            channel = "unknown"

        caller_top_k: Optional[int] = None
        if "top_k" in caller_context:
            parsed = _finite_in_range(caller_context["top_k"], 1, MAX_CALLER_TOP_K)
            if parsed is None or parsed != int(parsed):
                return self._reject("input_context.top_k:out_of_bounds")
            caller_top_k = int(parsed)

        documents: List[Dict[str, Any]] = []
        if "kb_documents" in caller_context:
            validated, reason, reason_code = self._validate_documents(caller_context["kb_documents"], state)
            if reason is not None:
                return self._reject(reason, code=reason_code)
            documents = validated or []

        emit_trace_event(
            "input_validated",
            {
                "channel": channel,
                "document_count": len(documents),
                "top_k_override": caller_top_k is not None,
            },
            state,
        )

        return {
            "validated_input": sanitized,
            "caller_documents": documents,
            "caller_top_k": caller_top_k,
            "enriched_context": {
                "source": "ManufacturingSafetyQAAgent",
                "channel": channel,
                "document_count": len(documents),
            },
            "status": AgentStatus.SUCCESS.value,
        }
