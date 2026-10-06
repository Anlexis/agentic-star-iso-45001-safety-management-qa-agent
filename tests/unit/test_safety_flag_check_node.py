# MFG-C2-021 — Unit Tests: SafetyFlagCheckNode
#
# Covers EN / JA / VI emergency-term matching and the no-match path.
# The escalation decision comes from the QUESTION; terms found in the cited
# procedures are surfaced in safety_flags but do not escalate on their own.


from src.nodes.safety_flag_check_node import SafetyFlagCheckNode
from framework.schemas.agent_status import AgentStatus


class TestSafetyFlagCheckNode:
    """Unit tests for the life-safety escalation gate."""

    def setup_method(self):
        self.node = SafetyFlagCheckNode()

    def test_english_keyword_matches(self):
        """EN 'fire' → is_emergency_escalation True, keyword recorded."""
        state = {"normalized_query": "there is a fire near the press"}

        result = self.node.execute(state)

        assert result["is_emergency_escalation"] is True
        assert "fire" in result["safety_flags"]
        assert result["status"] == AgentStatus.SUCCESS.value

    def test_japanese_keyword_matches(self):
        """JA '火災' → is_emergency_escalation True."""
        state = {"normalized_query": "工場で火災が発生しました"}

        result = self.node.execute(state)

        assert result["is_emergency_escalation"] is True
        assert "火災" in result["safety_flags"]
        assert result["status"] == AgentStatus.SUCCESS.value

    def test_vietnamese_keyword_matches(self):
        """VI 'hỏa hoạn' → is_emergency_escalation True."""
        state = {"normalized_query": "có hỏa hoạn trong nhà máy"}

        result = self.node.execute(state)

        assert result["is_emergency_escalation"] is True
        assert "hỏa hoạn" in result["safety_flags"]
        assert result["status"] == AgentStatus.SUCCESS.value

    def test_keyword_match_is_case_insensitive(self):
        """Uppercase EN keyword still matches (case-insensitive scan)."""
        state = {"normalized_query": "EVACUATION required immediately"}

        result = self.node.execute(state)

        assert result["is_emergency_escalation"] is True
        assert "evacuation" in result["safety_flags"]

    def test_no_match_path(self):
        """A routine query → no escalation, empty safety_flags."""
        state = {"normalized_query": "what does iso 45001 clause 6 require"}

        result = self.node.execute(state)

        assert result["is_emergency_escalation"] is False
        assert result["safety_flags"] == []
        assert result["status"] == AgentStatus.SUCCESS.value

    def test_empty_state_no_escalation(self):
        """No scannable text → no escalation, status SUCCESS."""
        result = self.node.execute({})

        assert result["is_emergency_escalation"] is False
        assert result["safety_flags"] == []
        assert result["status"] == AgentStatus.SUCCESS.value

    def test_terms_in_extracted_procedures_are_surfaced_not_escalated(self):
        """Procedure text contributes flags; only the question escalates.

        The procedure branch used to read a shape the pipeline never produces —
        a list of strings, where the extractor emits a list of mappings — so it
        matched nothing on any real invocation. Making it read the real shape
        also required deciding what it may do: a fire-response clause contains
        the word fire, so escalating on it would escalate every fire-safety
        question the agent exists to answer.
        """
        state = {
            "normalized_query": "routine question",
            "extracted_procedures": [{"clause": "8.2", "title": "Response", "content": "explosion containment"}],
        }

        result = self.node.execute(state)

        assert result["is_emergency_escalation"] is False
        assert "explosion" in result["safety_flags"]

    def test_a_hazard_term_in_the_question_escalates_even_with_a_clean_corpus(self):
        state = {
            "normalized_query": "there is an explosion risk at the press right now",
            "extracted_procedures": [{"clause": "6.1", "title": "Planning", "content": "routine review"}],
        }

        result = self.node.execute(state)

        assert result["is_emergency_escalation"] is True
        assert result["safety_flags"][0] == "explosion"
