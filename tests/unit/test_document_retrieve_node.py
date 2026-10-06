# MFG-C2-021 — Unit Tests: DocumentRetrieveNode
#
# Covers the injectable-retriever contract:
#   - 3 docs returned   → status SUCCESS, out_of_scope False
#   - empty result      → out_of_scope True, status SUCCESS
#   - missing query     → status ERROR


from src.nodes.document_retrieve_node import DocumentRetrieveNode
from framework.schemas.agent_status import AgentStatus


def _make_retriever(docs):
    """Return a retriever callable matching the node's expected signature."""

    def _retriever(query, top_k=5, language=None):
        return docs

    return _retriever


class TestDocumentRetrieveNode:
    """Unit tests for the ISO 45001 document-retrieval node."""

    def test_three_docs_returns_success(self):
        """A retriever yielding 3 docs → SUCCESS, out_of_scope False."""
        docs = [
            {"content": "ISO 45001 clause 4.1", "source": "iso45001.pdf"},
            {"content": "ISO 45001 clause 6.1", "source": "iso45001.pdf"},
            {"content": "ISO 45001 clause 8.1", "source": "iso45001.pdf"},
        ]
        node = DocumentRetrieveNode(retriever=_make_retriever(docs))
        state = {"normalized_query": "hazard identification", "detected_language": "EN"}

        result = node.execute(state)

        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["out_of_scope"] is False
        assert len(result["retrieved_documents"]) == 3

    def test_empty_result_sets_out_of_scope(self):
        """A retriever yielding 0 docs → out_of_scope True, status SUCCESS."""
        node = DocumentRetrieveNode(retriever=_make_retriever([]))
        state = {"normalized_query": "how to bake a cake", "detected_language": "EN"}

        result = node.execute(state)

        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["out_of_scope"] is True
        assert result["retrieved_documents"] == []

    def test_no_retriever_falls_back_to_out_of_scope(self):
        """No injected retriever → deterministic empty result → out_of_scope True."""
        node = DocumentRetrieveNode()
        state = {"normalized_query": "iso 45001 clause 6", "detected_language": "EN"}

        result = node.execute(state)

        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["out_of_scope"] is True

    def test_missing_query_returns_error(self):
        """Missing normalized_query → ERROR."""
        node = DocumentRetrieveNode(retriever=_make_retriever([{"content": "x"}]))
        state = {"detected_language": "EN"}

        result = node.execute(state)

        assert result["status"] == AgentStatus.ERROR.value
        assert result["out_of_scope"] is True
        assert "error_log" in result

    def test_blank_query_returns_error(self):
        """Whitespace-only normalized_query → ERROR."""
        node = DocumentRetrieveNode(retriever=_make_retriever([{"content": "x"}]))
        state = {"normalized_query": "    ", "detected_language": "EN"}

        result = node.execute(state)

        assert result["status"] == AgentStatus.ERROR.value
        assert "error_log" in result

    def test_retriever_exception_returns_error(self):
        """A retriever that raises → ERROR, out_of_scope True (graceful failure)."""

        def _boom(query, top_k=5, language=None):
            raise RuntimeError("vector store unavailable")

        node = DocumentRetrieveNode(retriever=_boom)
        state = {"normalized_query": "iso 45001", "detected_language": "EN"}

        result = node.execute(state)

        assert result["status"] == AgentStatus.ERROR.value
        assert result["out_of_scope"] is True
        assert "error_log" in result
