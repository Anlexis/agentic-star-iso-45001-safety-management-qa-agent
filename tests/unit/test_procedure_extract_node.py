# MFG-C2-021 — Unit Tests: ProcedureExtractNode
#
# Covers ISO 45001 clause extraction, deduplication, and empty-doc passthrough.


from src.nodes.procedure_extract_node import ProcedureExtractNode
from framework.schemas.agent_status import AgentStatus


class TestProcedureExtractNode:
    """Unit tests for the rule-based ISO 45001 procedure-extraction node."""

    def setup_method(self):
        self.node = ProcedureExtractNode()

    def test_extracts_iso_clauses(self):
        """Clause headings in document text are extracted into procedure dicts."""
        doc = (
            "6.1 Actions to address risks and opportunities\n"
            "The organization shall plan actions to address OH&S risks.\n"
            "8.2 Emergency preparedness and response\n"
            "The organization shall establish processes for emergencies.\n"
        )
        state = {"retrieved_documents": [{"content": doc}]}

        result = self.node.execute(state)

        assert result["status"] == AgentStatus.SUCCESS.value
        clauses = {p["clause"] for p in result["extracted_procedures"]}
        assert "6.1" in clauses
        assert "8.2" in clauses
        # Each extracted procedure carries clause / title / content keys.
        for proc in result["extracted_procedures"]:
            assert set(proc.keys()) >= {"clause", "title", "content"}

    def test_dedup_by_clause(self):
        """The same clause appearing in multiple docs is kept once (first wins)."""
        doc_a = "6.1 Actions to address risks\nFirst occurrence body.\n"
        doc_b = "6.1 Actions to address risks (duplicate)\nSecond occurrence body.\n"
        state = {"retrieved_documents": [{"content": doc_a}, {"content": doc_b}]}

        result = self.node.execute(state)

        assert result["status"] == AgentStatus.SUCCESS.value
        clause_ids = [p["clause"] for p in result["extracted_procedures"]]
        assert clause_ids.count("6.1") == 1, "duplicate clause 6.1 must be deduplicated"

    def test_non_iso_clauses_are_filtered(self):
        """Clause numbers outside ISO 45001 scope (4–10) are not extracted."""
        doc = "1.1 Scope of this irrelevant document\nbody\n" "6.1 Actions to address risks\nbody\n"
        state = {"retrieved_documents": [{"content": doc}]}

        result = self.node.execute(state)

        clauses = {p["clause"] for p in result["extracted_procedures"]}
        assert "6.1" in clauses
        assert "1.1" not in clauses, "clause 1.1 is outside ISO 45001 scope (4–10)"

    def test_empty_doc_list_passthrough(self):
        """An empty document list → empty procedure list, status SUCCESS."""
        state = {"retrieved_documents": []}

        result = self.node.execute(state)

        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["extracted_procedures"] == []

    def test_missing_key_passthrough(self):
        """Missing retrieved_documents key → empty passthrough, status SUCCESS."""
        result = self.node.execute({})

        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["extracted_procedures"] == []

    def test_plain_string_documents_supported(self):
        """Documents supplied as plain strings are also parsed."""
        state = {"retrieved_documents": ["7.1 Resources\nThe organization shall provide resources.\n"]}

        result = self.node.execute(state)

        assert result["status"] == AgentStatus.SUCCESS.value
        clauses = {p["clause"] for p in result["extracted_procedures"]}
        assert "7.1" in clauses

    def test_non_list_documents_returns_error(self):
        """retrieved_documents as a non-list → ERROR."""
        state = {"retrieved_documents": "not a list"}

        result = self.node.execute(state)

        assert result["status"] == AgentStatus.ERROR.value
        assert result["extracted_procedures"] == []
        assert "error_log" in result
