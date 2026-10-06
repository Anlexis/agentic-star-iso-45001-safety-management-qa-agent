# MFG-C2-021 — Unit Tests: QueryNormalizeNode
#
# Covers JA / VI / EN language detection and query normalization.


from src.nodes.query_normalize_node import QueryNormalizeNode
from framework.schemas.agent_status import AgentStatus


class TestQueryNormalizeNode:
    """Unit tests for the pre-process language-detection / normalization node."""

    def setup_method(self):
        self.node = QueryNormalizeNode()

    def test_detects_english(self):
        """ASCII Latin query → EN, lowercased and whitespace-collapsed."""
        state = {"query": "  What Does ISO 45001   Require?  "}
        result = self.node.execute(state)

        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["detected_language"] == "EN"
        # EN is lowercased and internal whitespace collapsed.
        assert result["normalized_query"] == "what does iso 45001 require?"

    def test_detects_japanese(self):
        """CJK characters → JA; casing preserved (no lowercasing for JA)."""
        state = {"query": "ISO 45001 の要件は何ですか"}
        result = self.node.execute(state)

        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["detected_language"] == "JA"
        # JA keeps original characters; whitespace is still collapsed.
        assert "要件" in result["normalized_query"]

    def test_detects_vietnamese(self):
        """Latin-extended diacritics → VI; casing preserved."""
        state = {"query": "Yêu cầu của ISO 45001 là gì"}
        result = self.node.execute(state)

        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["detected_language"] == "VI"
        # VI is not lowercased — the leading capital is preserved.
        assert result["normalized_query"].startswith("Yêu")

    def test_whitespace_is_collapsed(self):
        """Multiple internal spaces / tabs collapse to single spaces."""
        state = {"query": "hazard\t\tidentification    procedure"}
        result = self.node.execute(state)

        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["normalized_query"] == "hazard identification procedure"

    def test_empty_query_returns_error(self):
        """Blank query → ERROR with an error_log entry."""
        state = {"query": "   "}
        result = self.node.execute(state)

        assert result["status"] == AgentStatus.ERROR.value
        assert "error_log" in result
        assert result["error_log"]

    def test_non_string_query_returns_error(self):
        """Non-string question → ERROR (type guard)."""
        state = {"query": 12345}
        result = self.node.execute(state)

        assert result["status"] == AgentStatus.ERROR.value
        assert "error_log" in result
