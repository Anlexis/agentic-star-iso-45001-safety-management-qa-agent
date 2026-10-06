"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - Return the status as AgentStatus.<X>.value — the enum's string value,
#    not the enum object itself
#  - Never import from mediator/, api/, or other agents
#
# MFG-C2-021 — ProcedureExtractNode
# Inner domain graph node: extract structured ISO 45001 procedure items
# from retrieved documents using rule-based / regex parsing (no LLM in v1).
#
# Input:  state["retrieved_documents"] — list of document dicts or raw strings
# Output: state["extracted_procedures"] — list of {"clause", "title", "content"} dicts
#         Duplicates by clause number are removed (first-occurrence wins).

import logging
import re
from typing import Any, ClassVar, Dict, List, Set, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

# Audit trail — a free function, not a node method.
from shared.utils.audit_logger import emit_trace_event

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Patterns for ISO 45001 clause detection
# ---------------------------------------------------------------------------

# Matches clause numbers like "4", "4.1", "4.1.2", "10.3" at line start.
_CLAUSE_LINE_RE = re.compile(
    r"^\s*(?P<clause>\d+(?:\.\d+){0,3})\s+(?P<title>[^\n]+)",
    re.MULTILINE,
)

# Broader heading variant: "Clause 4.1 – Title" or "4.1 - Title"
_CLAUSE_HEADING_RE = re.compile(
    r"(?:clause\s+)?(?P<clause>\d+(?:\.\d+){0,3})\s*[-–—:]\s*(?P<title>[^\n]+)",
    re.IGNORECASE | re.MULTILINE,
)

# ISO 45001 well-known top-level clauses (4–10) for relevance filtering.
_ISO_45001_CLAUSES = frozenset(str(n) for n in range(4, 11))


def _is_iso_clause(clause: str) -> bool:
    """Return True if the clause number falls under ISO 45001 scope (4–10)."""
    top = clause.split(".")[0]
    return top in _ISO_45001_CLAUSES


def _extract_text(doc: Any) -> str:
    """Coerce a document entry to plain text."""
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict):
        # Accept common field names used by retrieval layers.
        for key in ("content", "text", "body", "page_content"):
            val = doc.get(key)
            if isinstance(val, str) and val.strip():
                return val
    return ""


def _extract_procedures_from_text(text: str) -> List[Dict[str, str]]:
    """Parse text and return a list of procedure dicts.

    Uses two-pass regex extraction:
    Pass 1 — find clause headings with _CLAUSE_LINE_RE (strict line-start form).
    Pass 2 — supplement with _CLAUSE_HEADING_RE (loose dash/colon form).
    Body content is the text between successive headings.
    """
    procedures: List[Dict[str, str]] = []
    if not text or not text.strip():
        return procedures

    # Collect all heading match positions (both patterns, dedup by position).
    hits: List[Tuple[int, int, str, str]] = []
    seen_starts: Set[int] = set()

    for pattern in (_CLAUSE_LINE_RE, _CLAUSE_HEADING_RE):
        for m in pattern.finditer(text):
            start = m.start()
            if start in seen_starts:
                continue
            clause = m.group("clause").strip()
            title = m.group("title").strip()
            if not _is_iso_clause(clause):
                continue
            seen_starts.add(start)
            hits.append((start, m.end(), clause, title))

    # Sort by position in document.
    hits.sort(key=lambda h: h[0])

    for i, (start, end, clause, title) in enumerate(hits):
        # Body = text from end-of-heading to start-of-next-heading (or EOF).
        next_start = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        body = text[end:next_start].strip()

        # Trim body to a reasonable length to avoid embedding entire documents.
        if len(body) > 2000:
            body = body[:2000].rsplit("\n", 1)[0] + " [...]"

        procedures.append(
            {
                "clause": clause,
                "title": title,
                "content": body,
            }
        )

    return procedures


def _dedup_by_clause(procedures: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Remove duplicates by clause number; first occurrence wins."""
    seen: Set[str] = set()
    result: List[Dict[str, str]] = []
    for proc in procedures:
        clause = proc["clause"]
        if clause not in seen:
            seen.add(clause)
            result.append(proc)
    return result


class ProcedureExtractNode(FunctionNode):
    """Extract structured ISO 45001 procedure items from retrieved documents.

    Inner domain graph node for MFG-C2-021 ManufacturingSafetyQAAgent.
    Uses rule-based / regex parsing (no LLM call in v1).

    Input state keys:
        retrieved_documents: list of document dicts or strings from the
            retrieval step.  Each entry may be a plain string or a dict
            with a "content"/"text"/"body"/"page_content" field.

    Output state keys (partial dict):
        extracted_procedures: list of dicts with keys:
            - clause  (str): ISO 45001 clause number, e.g. "4.3"
            - title   (str): clause heading text
            - content (str): body text under that clause
        status: AgentStatus.SUCCESS on success, AgentStatus.ERROR on failure
        error_log: list of error messages (only present on ERROR)
    """

    # The entry gate is VERIFIED_EXTERNAL (the level this agent is registered
    # at). Declaring a higher level on an inner node does not add protection —
    # the caller's trust travels unchanged into the subgraph, so no caller who
    # cleared the entry gate can ever satisfy it, and the whole pipeline denies
    # itself on every request.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Retrieve document list — tolerate missing key (returns empty list).
        raw_docs = state.get("retrieved_documents")

        if raw_docs is None:
            logger.info("ProcedureExtractNode: 'retrieved_documents' not in state — " "returning empty procedure list")
            return {
                "extracted_procedures": [],
                "status": AgentStatus.SUCCESS.value,
            }

        if not isinstance(raw_docs, (list, tuple)):
            return {
                "extracted_procedures": [],
                "status": AgentStatus.ERROR.value,
                "error_log": [
                    "ProcedureExtractNode: 'retrieved_documents' must be a list, " f"got {type(raw_docs).__name__}"
                ],
            }

        all_procedures: List[Dict[str, str]] = []

        for idx, doc in enumerate(raw_docs):
            text = _extract_text(doc)
            if not text:
                logger.debug(
                    "ProcedureExtractNode: doc[%d] yielded no extractable text — skipping",
                    idx,
                )
                continue
            try:
                procs = _extract_procedures_from_text(text)
                all_procedures.extend(procs)
            except Exception as exc:  # noqa: BLE001
                # Log but continue — one bad document should not abort extraction.
                logger.warning(
                    "ProcedureExtractNode: error processing doc[%d]: %s",
                    idx,
                    exc,
                )

        # Deduplicate across all documents (first occurrence wins).
        deduped = _dedup_by_clause(all_procedures)

        logger.info(
            "ProcedureExtractNode: extracted %d procedures from %d documents " "(%d before dedup)",
            len(deduped),
            len(raw_docs),
            len(all_procedures),
        )

        # Audit trail: record the extraction side-effect.
        emit_trace_event(
            "procedure_extract",
            {"procedure_count": len(deduped), "doc_count": len(raw_docs)},
            state,
        )

        return {
            "extracted_procedures": deduped,
            "status": AgentStatus.SUCCESS.value,
        }
