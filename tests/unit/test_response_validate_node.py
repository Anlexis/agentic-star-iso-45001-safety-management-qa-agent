# MFG-C2-021 — Unit Tests: ResponseValidateNode (the inner workflow's output gate)
#
# Two properties, and they are not the same thing:
#   1. a violating answer is REFUSED, and
#   2. every field carrying that answer is CLEARED, with a truthy notice in its
#      place.
# Only the second withholds anything. The framework resolves an agent's output
# as `formatted_output or result` with no status check, so an error status alone
# still ships the ungated answer inside the error envelope, and a falsy notice
# falls straight through to it.

import pytest
from framework.schemas.agent_status import AgentStatus
from src.nodes.response_validate_node import (
    ANSWER_BEARING_FIELDS,
    WITHHELD_NOTICE,
    ResponseValidateNode,
    _scan,
)


class TestCleanPath:
    """The control: a refuse-everything gate must not be able to pass."""

    def setup_method(self):
        self.node = ResponseValidateNode()

    def test_valid_response_passes(self):
        result = self.node.execute(
            {
                "response": "ISO 45001 clause 6.1.2 requires ongoing hazard identification.",
                "out_of_scope": False,
            }
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["is_valid_response"] is True

    def test_out_of_scope_reply_passes(self):
        result = self.node.execute(
            {
                "response": "This query is outside the scope of the ISO 45001 knowledge base.",
                "out_of_scope": True,
            }
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["is_valid_response"] is True

    @pytest.mark.parametrize("response", ["", "   \n  ", None])
    def test_empty_response_fails(self, response):
        result = self.node.execute({"response": response, "out_of_scope": False})
        assert result["status"] == AgentStatus.ERROR.value
        assert result["is_valid_response"] is False

    def test_empty_out_of_scope_response_still_fails(self):
        # An out-of-scope run still has to produce the canned reply. Passing an
        # empty answer through as SUCCESS releases nothing while telling the
        # caller everything worked.
        result = self.node.execute({"response": "", "out_of_scope": True})
        assert result["status"] == AgentStatus.ERROR.value


class TestConfiguredLengthBound:
    def test_declared_bound_is_the_enforced_bound(self):
        node = ResponseValidateNode(config={"max_response_chars": 50})
        result = node.execute({"response": "B" * 200, "out_of_scope": False})
        assert result["status"] == AgentStatus.ERROR.value

    def test_within_the_declared_bound_passes(self):
        node = ResponseValidateNode(config={"max_response_chars": 500})
        result = node.execute({"response": "B" * 200, "out_of_scope": False})
        assert result["status"] == AgentStatus.SUCCESS.value

    def test_out_of_scope_reply_is_exempt_from_the_length_bound(self):
        node = ResponseValidateNode(config={"max_response_chars": 5})
        result = node.execute({"response": "B" * 200, "out_of_scope": True})
        assert result["status"] == AgentStatus.SUCCESS.value


class TestDetectorParity:
    """The local set must be a SUPERSET of the framework's, never narrower.

    A shape the framework catches and this node misses is a containment bypass:
    the framework's own gate raises after execute() returns and the wrapper
    discards this node's whole delta, clearing included.
    """

    @pytest.mark.parametrize(
        "sample",
        [
            "sk_live_ABCDEFGHIJKLMNOP1234",
            "sk_test_ABCDEFGHIJKLMNOP1234",
            "sk-ABCDEFGHIJKLMNOPQRSTUVWX",
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.signaturehere",
            "AKIAIOSFODNN7EXAMPLE",
            "Authorization: Bearer abcdef0123456789ABCDEF",
            "redis://cache.internal:6379/safety_kb",
        ],
    )
    def test_framework_shapes_are_all_detected(self, sample):
        assert _scan(sample) is not None

    @pytest.mark.parametrize(
        "sample",
        ["pk-ABCD1234abcd5678EFGH", "The system password: hunter2supersecret is configured."],
    )
    def test_additional_shapes_are_detected(self, sample):
        assert _scan(sample) is not None

    @pytest.mark.parametrize(
        "sample",
        [
            "プレス機の安全装置について",
            "ISO 45001 clause 8.2 covers emergency preparedness.",
            "A token of appreciation was given to the safety team.",
        ],
    )
    def test_ordinary_safety_text_is_not_flagged(self, sample):
        assert _scan(sample) is None


class TestContainment:
    LEAK = "Use this key sk-ABCDEFGHIJKLMNOP0123456789 to open the cabinet."

    def setup_method(self):
        self.node = ResponseValidateNode()

    def _violate(self):
        return self.node.execute({"response": self.LEAK, "result": self.LEAK, "out_of_scope": False})

    def test_violating_answer_is_refused(self):
        result = self._violate()
        assert result["status"] == AgentStatus.ERROR.value
        assert result["is_valid_response"] is False

    def test_every_answer_bearing_field_is_present_and_cleared(self):
        # Presence AND emptiness. LangGraph merges partial deltas, so a key
        # simply omitted from the delta leaves the OLD value in state — an
        # assertion that only checks falsiness passes on a gate that clears
        # nothing at all.
        result = self._violate()
        for field, empty in ANSWER_BEARING_FIELDS:
            assert field in result, f"{field} missing from the delta — nothing was cleared"
            assert result[field] == empty

    def test_the_notice_is_truthy(self):
        result = self._violate()
        assert bool(result["formatted_output"])
        assert result["formatted_output"] == WITHHELD_NOTICE

    def test_no_returned_field_carries_the_released_text(self):
        rendered = str(self._violate())
        assert "sk-ABCDEFGHIJKLMNOP0123456789" not in rendered
        assert "open the cabinet" not in rendered

    def test_violation_message_names_a_reason_not_a_value(self):
        result = self._violate()
        assert "credential_shape" in result["error_message"]
        assert "ABCDEFGHIJKLMNOP0123456789" not in result["error_message"]

    def test_over_length_answer_is_also_cleared(self):
        node = ResponseValidateNode(config={"max_response_chars": 10})
        result = node.execute({"response": "B" * 200, "result": "B" * 200, "out_of_scope": False})
        assert result["status"] == AgentStatus.ERROR.value
        for field, empty in ANSWER_BEARING_FIELDS:
            assert field in result and result[field] == empty
