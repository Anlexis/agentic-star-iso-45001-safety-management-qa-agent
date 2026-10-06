# MFG-C2-021 — Proof-of-Boundary: the inner workflow's output gate
#
# ResponseValidateNode is the mandatory output gate of the inner workflow.
# Oversize and credential-bearing answers must terminate with ERROR and release
# nothing; a clean, in-bounds answer must pass with SUCCESS.
#
# Note: import isolation (AST scan) and state msgpack-safety live in
#       test_import_isolation.py and test_state_safety.py respectively;
#       escalation lives in test_pb_emergency_escalation.py and the
#       out-of-scope block in test_pb_out_of_scope.py.

from framework.schemas.agent_status import AgentStatus
from src.nodes.response_validate_node import ANSWER_BEARING_FIELDS, ResponseValidateNode


class TestPBOutputGate:
    """The gate refuses AND withholds; those are two separate obligations."""

    def setup_method(self):
        self.node = ResponseValidateNode()

    def test_oversize_response_blocked_with_error(self):
        node = ResponseValidateNode(config={"max_response_chars": 100})
        state = {"response": "A" * 500, "result": "A" * 500, "out_of_scope": False}

        result = node.execute(state)

        assert result["status"] == AgentStatus.ERROR.value, "an oversize answer must terminate with ERROR"
        assert result["is_valid_response"] is False
        assert "error_message" in result

    def test_credential_bearing_response_blocked_and_withheld(self):
        leaked = "Here is the key: sk-ABCDEFGHIJKLMNOP1234567890"
        state = {"response": leaked, "result": leaked, "out_of_scope": False}

        result = self.node.execute(state)

        assert result["status"] == AgentStatus.ERROR.value
        assert result["is_valid_response"] is False
        # Refusing is not withholding: every answer-bearing field has to be
        # present in the delta AND empty, or the previous value survives the
        # merge and the output resolver falls back to it.
        for field, empty in ANSWER_BEARING_FIELDS:
            assert field in result and result[field] == empty
        assert bool(result["formatted_output"])
        assert "sk-ABCDEFGHIJKLMNOP1234567890" not in str(result)

    def test_valid_response_passes_with_success(self):
        valid = "ISO 45001 clause 6.1.2 requires hazard identification " "to be ongoing and proactive."
        state = {"response": valid, "out_of_scope": False}

        result = self.node.execute(state)

        assert result["status"] == AgentStatus.SUCCESS.value, "a valid answer must pass"
        assert result["is_valid_response"] is True
