# MFG-C2-021 — Unit Tests: ResponseGenerateNode
#
# Covers the three synthesis paths: emergency escalation, out-of-scope, normal.


from src.nodes.response_generate_node import ResponseGenerateNode
from framework.schemas.agent_status import AgentStatus


class TestResponseGenerateNode:
    """Unit tests for the main-slot response-generation node."""

    def setup_method(self):
        self.node = ResponseGenerateNode()

    def test_emergency_path(self):
        """is_emergency_escalation True → localised escalation message, SUCCESS."""
        state = {
            "is_emergency_escalation": True,
            "safety_flags": ["fire"],
            "detected_language": "EN",
        }

        result = self.node.execute(state)

        assert result["status"] == AgentStatus.SUCCESS.value
        assert "⚠️" in result["response"]
        assert "safety officer" in result["response"].lower()
        # result mirrors response for PostProcessNode compatibility.
        assert result["result"] == result["response"]

    def test_emergency_path_takes_priority_over_out_of_scope(self):
        """Emergency is evaluated before out_of_scope when both are set."""
        state = {
            "is_emergency_escalation": True,
            "out_of_scope": True,
            "safety_flags": ["explosion"],
            "detected_language": "EN",
        }

        result = self.node.execute(state)

        assert result["status"] == AgentStatus.SUCCESS.value
        assert "⚠️" in result["response"], "emergency path must win over out-of-scope"

    def test_out_of_scope_path(self):
        """out_of_scope True → canned scope-exceeded reply, SUCCESS."""
        state = {"out_of_scope": True, "detected_language": "EN", "extracted_procedures": []}

        result = self.node.execute(state)

        assert result["status"] == AgentStatus.SUCCESS.value
        assert "outside the scope" in result["response"].lower()

    def test_out_of_scope_localised_japanese(self):
        """out_of_scope reply is localised by detected_language."""
        state = {"out_of_scope": True, "detected_language": "JA", "extracted_procedures": []}

        result = self.node.execute(state)

        assert result["status"] == AgentStatus.SUCCESS.value
        assert "ISO 45001" in result["response"]
        assert "範囲外" in result["response"]

    def test_normal_synthesis_path(self):
        """Normal path synthesises extracted procedures into a structured answer.

        The fixture uses the real "clause" key emitted by ProcedureExtractNode
        ({"clause", "title", "content"}) — the synthesis must read it and render
        the ISO 45001 clause number in the per-clause heading.
        """
        state = {
            "user_input": "What does ISO 45001 require for emergency preparedness?",
            "detected_language": "EN",
            "extracted_procedures": [
                {
                    "clause": "8.2",
                    "title": "Emergency preparedness and response",
                    "content": "The organization shall establish emergency processes.",
                },
            ],
        }

        result = self.node.execute(state)

        assert result["status"] == AgentStatus.SUCCESS.value
        # The fixed key mapping must render the clause number in the heading.
        assert (
            "### 8.2 — Emergency preparedness and response" in result["response"]
        ), "normal synthesis must render the clause number from the 'clause' key"
        assert "8.2" in result["response"]
        assert "Emergency preparedness" in result["response"]
        assert "The organization shall establish emergency processes." in result["response"]
        assert result["result"] == result["response"]

    def test_normal_path_no_procedures_returns_guidance(self):
        """Normal path with no procedures still produces a SUCCESS guidance answer."""
        state = {
            "user_input": "Tell me about hazard controls",
            "detected_language": "EN",
            "extracted_procedures": [],
        }

        result = self.node.execute(state)

        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["response"]

    def test_normal_path_empty_user_input_returns_error(self):
        """Normal path with empty user_input → ERROR."""
        state = {
            "user_input": "",
            "detected_language": "EN",
            "extracted_procedures": [],
        }

        result = self.node.execute(state)

        assert result["status"] == AgentStatus.ERROR.value
        assert "error_log" in result
