# MFG-C2-021 — Proof-of-Boundary: emergency escalation (life-safety)
#
# SafetyFlagCheckNode raises is_emergency_escalation for EN ("fire") /
# JA ("火災") / VI ("hỏa hoạn") life-safety terms in the QUESTION, and
# ResponseGenerateNode emits the language-localised escalation message.
#
# The decision is driven by the question, not by the retrieved procedures:
# escalating on procedure text would escalate every fire-response question ever
# asked, because a fire-response clause contains the word fire.
#
# Note: status is asserted against AgentStatus.<X>.value — the node returns the
#       enum's string value, not the enum object.

import pytest
from framework.schemas.agent_status import AgentStatus


class TestPBEmergencyEscalation:
    """Life-safety terms in the question raise is_emergency_escalation."""

    def setup_method(self):
        from src.nodes.response_generate_node import ResponseGenerateNode
        from src.nodes.safety_flag_check_node import SafetyFlagCheckNode

        self.safety_node = SafetyFlagCheckNode()
        self.response_node = ResponseGenerateNode()

    @pytest.mark.parametrize(
        "query,language",
        [
            ("There is a fire in the assembly line", "EN"),
            ("組立ラインで火災が発生しています", "JA"),
            ("Có hỏa hoạn trong dây chuyền lắp ráp", "VI"),
        ],
    )
    def test_emergency_term_sets_escalation_flag(self, query, language):
        state = {"normalized_query": query, "extracted_procedures": []}

        result = self.safety_node.execute(state)

        assert result["is_emergency_escalation"] is True, f"{language} emergency term must set is_emergency_escalation"
        assert result["safety_flags"], "the matched term must be recorded in safety_flags"
        # Escalation is a valid output — status stays SUCCESS.
        assert result["status"] == AgentStatus.SUCCESS.value

    @pytest.mark.parametrize("language", ["EN", "JA", "VI"])
    def test_escalation_response_is_emitted(self, language):
        state = {
            "is_emergency_escalation": True,
            "safety_flags": ["fire"],
            "detected_language": language,
        }

        result = self.response_node.execute(state)

        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["response"], "the escalation path must produce a non-empty response"
        # The escalation marker is language-independent and always present.
        assert "⚠️" in result["response"]

    def test_normal_query_does_not_escalate(self):
        state = {
            "normalized_query": "what does iso 45001 clause 6 require",
            "extracted_procedures": [],
        }

        result = self.safety_node.execute(state)

        assert result["is_emergency_escalation"] is False
        assert result["safety_flags"] == []
        assert result["status"] == AgentStatus.SUCCESS.value

    def test_a_fire_procedure_in_the_corpus_does_not_escalate_a_routine_question(self):
        # The screen must not fire on the domain text the agent exists to serve.
        state = {
            "normalized_query": "what does iso 45001 clause 8 require",
            "extracted_procedures": [
                {
                    "clause": "8.2",
                    "title": "Emergency preparedness and response",
                    "content": "Processes for responding to a fire, including evacuation drills.",
                }
            ],
        }

        result = self.safety_node.execute(state)

        assert result["is_emergency_escalation"] is False
        # The terms are still surfaced for the reader, they just do not escalate.
        assert "fire" in result["safety_flags"]
