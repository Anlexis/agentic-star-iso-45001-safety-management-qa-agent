# MFG-C2-021 — Proof-of-Boundary: Out-of-scope block
#
# Out-of-scope block : a zero-document retrieval sets out_of_scope=True and
#                      ResponseGenerateNode returns the canned scope-exceeded
#                      reply (no domain content leaks).
#
# Note: the out-of-scope path terminates with AgentStatus.SUCCESS.value (the canned
#       reply is a valid output) — there is no separate OUT_OF_SCOPE status.

from framework.schemas.agent_status import AgentStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _zero_doc_retriever(query, top_k=5, language=None):
    """A retriever stand-in that always returns zero documents (out-of-scope)."""
    return []


# ---------------------------------------------------------------------------
# Out-of-scope block
# ---------------------------------------------------------------------------


class TestPBOutOfScopeBlock:
    """A zero-document retrieval flags out_of_scope and yields a canned reply."""

    def setup_method(self):
        from src.nodes.document_retrieve_node import DocumentRetrieveNode
        from src.nodes.response_generate_node import ResponseGenerateNode

        self.retrieve_node = DocumentRetrieveNode(retriever=_zero_doc_retriever)
        self.response_node = ResponseGenerateNode()

    def test_zero_docs_sets_out_of_scope(self):
        """Zero documents → out_of_scope True, status SUCCESS, no documents."""
        state = {"normalized_query": "how to bake a cake", "detected_language": "EN"}

        result = self.retrieve_node.execute(state)

        assert result["out_of_scope"] is True, "out-of-scope: zero docs must set out_of_scope=True"
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["retrieved_documents"] == []

    def test_out_of_scope_returns_canned_reply(self):
        """ResponseGenerateNode returns the canned scope-exceeded reply, no content."""
        state = {"out_of_scope": True, "detected_language": "EN", "extracted_procedures": []}

        result = self.response_node.execute(state)

        assert result["status"] == AgentStatus.SUCCESS.value
        assert (
            "outside the scope" in result["response"].lower()
        ), "out-of-scope: response must be the canned scope-exceeded reply"
        # No ISO clause body should leak into the out-of-scope reply.
        assert "###" not in result["response"]
