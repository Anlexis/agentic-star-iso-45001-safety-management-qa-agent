"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - Return the status as AgentStatus.<X>.value — the enum's string value,
#    not the enum object itself
#  - Read inputs via state.get(...) — read-only
#  - Never import from mediator/, api/, or other agents
#
# MFG-C2-021 — DocumentRetrieveNode
# Selects the safety-management source documents that answer the normalized
# question, ranked by term overlap, and carries at most top_k of them forward.
#
# The corpus is the set of knowledge-base entries the caller supplied for this
# request, already bounded and masked by the input contract. A request that
# carries none is not an error: retrieval returns nothing and the pipeline takes
# its out-of-scope path, which is the honest answer when there is no source to
# cite. An injectable retriever remains available for a deployment that wires a
# vector store in front of the same interface.

import logging
import re
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5

# Terms shorter than this carry no retrieval signal and match everywhere.
_MIN_TERM_LEN = 3

_TERM_RE = re.compile(r"[0-9a-z]+|[぀-ヿ一-鿿]+", re.IGNORECASE)


def _terms(text: str) -> List[str]:
    """Split *text* into scoring terms across latin and CJK scripts."""
    out: List[str] = []
    for token in _TERM_RE.findall(text.lower()):
        if token.isascii():
            if len(token) >= _MIN_TERM_LEN:
                out.append(token)
        else:
            # CJK runs carry signal at shorter lengths; index bigrams so a
            # two-character term still matches inside a longer compound.
            out.extend(token[i : i + 2] for i in range(max(len(token) - 1, 1)))
    return out


def _score(question_terms: List[str], content: str) -> int:
    """Count how many distinct question terms appear in *content*."""
    if not question_terms:
        return 0
    lowered = content.lower()
    return sum(1 for term in set(question_terms) if term in lowered)


class DocumentRetrieveNode(FunctionNode):
    """Rank the caller's safety documents against the normalized question.

    Input state keys:
        normalized_query:  str        — sanitized, normalized question
        detected_language: str        — (optional) "JA" | "VI" | "EN"
        caller_documents:  list[dict] — validated {"doc_id", "content"} entries
        caller_top_k:      int | None — validated caller override for top_k

    Output state keys (partial dict):
        retrieved_documents: list[dict] — ranked entries, highest score first
        out_of_scope:        bool       — True when retrieval returns 0 entries
        status:              AgentStatus.SUCCESS or AgentStatus.ERROR
        error_log:           list[str]  — populated only on ERROR
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(
        self,
        retriever: Optional[Callable[..., List[Dict[str, Any]]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            retriever: Optional callable
                ``retriever(query: str, top_k: int, language: str | None) -> list[dict]``.
                A deployment that wires a vector store passes it here; when
                absent the caller-supplied corpus is used.
            config:    The ``retrieval`` block of config/config.yaml. ``top_k``
                       is read from it, so the declared value is the value used.
        """
        super().__init__()
        self._retriever = retriever
        settings: Dict[str, Any] = config or {}
        self._top_k = int(settings.get("top_k", DEFAULT_TOP_K))

    def _resolve_top_k(self, state: Dict[str, Any]) -> int:
        """Configured top_k, narrowed by a validated caller override.

        The caller may ask for fewer or more entries within the bound the input
        contract already enforced; anything outside it never reaches this node.
        """
        caller_top_k = state.get("caller_top_k")
        if isinstance(caller_top_k, int) and not isinstance(caller_top_k, bool) and caller_top_k >= 1:
            return caller_top_k
        return max(self._top_k, 1)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        normalized_query = state.get("normalized_query", "")

        if not normalized_query or not isinstance(normalized_query, str):
            return {
                "retrieved_documents": [],
                "out_of_scope": True,
                "status": AgentStatus.ERROR.value,
                "error_log": ["DocumentRetrieveNode: normalized_query is missing or not a string"],
            }

        normalized_query = normalized_query.strip()
        if not normalized_query:
            return {
                "retrieved_documents": [],
                "out_of_scope": True,
                "status": AgentStatus.ERROR.value,
                "error_log": ["DocumentRetrieveNode: normalized_query is blank after strip"],
            }

        detected_language = state.get("detected_language")
        top_k = self._resolve_top_k(state)

        retrieved_documents: List[Dict[str, Any]] = []
        try:
            if self._retriever is not None:
                retrieved_documents = list(
                    self._retriever(
                        normalized_query,
                        top_k=top_k,
                        language=detected_language,
                    )
                )
            else:
                retrieved_documents = self._rank_caller_documents(state, normalized_query, top_k)
        except Exception as exc:  # noqa: BLE001
            logger.error("DocumentRetrieveNode: retrieval raised an unexpected error: %s", exc)
            return {
                "retrieved_documents": [],
                "out_of_scope": True,
                "status": AgentStatus.ERROR.value,
                # The reason is named without the source text: an exception
                # message from a caller-supplied corpus is caller data.
                "error_log": ["DocumentRetrieveNode: retrieval failed"],
            }

        emit_trace_event(
            "document_retrieve",
            {
                "doc_count": len(retrieved_documents),
                "top_k": top_k,
                "language": detected_language,
            },
            state,
        )

        if not retrieved_documents:
            logger.info(
                "DocumentRetrieveNode: no documents matched (lang=%s top_k=%d) — out_of_scope",
                detected_language,
                top_k,
            )
            return {
                "retrieved_documents": [],
                "out_of_scope": True,
                "status": AgentStatus.SUCCESS.value,
            }

        logger.info(
            "DocumentRetrieveNode: retrieved %d document(s) (lang=%s top_k=%d)",
            len(retrieved_documents),
            detected_language,
            top_k,
        )

        return {
            "retrieved_documents": retrieved_documents,
            "out_of_scope": False,
            "status": AgentStatus.SUCCESS.value,
        }

    def _rank_caller_documents(
        self,
        state: Dict[str, Any],
        normalized_query: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Score the caller's corpus by term overlap and keep the best top_k."""
        corpus = state.get("caller_documents") or []
        if not isinstance(corpus, list) or not corpus:
            return []

        question_terms = _terms(normalized_query)
        scored: List[Tuple[int, int, Dict[str, Any]]] = []
        for position, entry in enumerate(corpus):
            if not isinstance(entry, dict):
                continue
            content = entry.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            score = _score(question_terms, content)
            if score <= 0:
                continue
            # position keeps the order stable for equal scores.
            scored.append((-score, position, entry))

        scored.sort(key=lambda item: (item[0], item[1]))
        return [
            {
                "doc_id": entry.get("doc_id", ""),
                "content": entry.get("content", ""),
                "score": -negative_score,
            }
            for negative_score, _position, entry in scored[:top_k]
        ]
