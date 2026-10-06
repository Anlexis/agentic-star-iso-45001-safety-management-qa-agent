# MFG-C2-021 — Unit Tests: PostProcessNode (the agent-boundary output gate)
#
# Two properties are under test and they are not the same thing:
#   1. a violating answer is REFUSED, and
#   2. every field that could carry that answer forward is CLEARED, with a
#      truthy notice in its place.
# Only the second one actually withholds anything: the framework resolves an
# agent's output as `formatted_output or result` with no status check, so an
# error status alone still ships the ungated answer inside the error envelope,
# and a falsy notice falls straight through to it.

import pytest
from framework.schemas.agent_status import AgentStatus
from src.nodes.post_process_node import (
    ANSWER_BEARING_FIELDS,
    WITHHELD_NOTICE,
    PostProcessNode,
    _scan,
)


class TestCleanPath:
    """The control: a refuse-everything gate must not be able to pass."""

    def setup_method(self):
        self.node = PostProcessNode()

    def test_clean_answer_is_released_verbatim(self):
        answer = "ISO 45001 clause 6.1.2 requires hazard identification to be " "ongoing and proactive."
        result = self.node.execute({"result": answer})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["formatted_output"] == answer

    def test_clean_japanese_answer_is_released(self):
        answer = "プレス機の安全装置は、両手操作式の起動ボタンが必要です。"
        result = self.node.execute({"result": answer})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["formatted_output"] == answer


class TestDetectorParity:
    """The local set must be a SUPERSET of the framework's, never narrower.

    A shape the framework catches and this node misses is a containment
    bypass, not a smaller gate: the framework's own gate raises after execute()
    returns, and the wrapper then discards this node's whole delta — the
    clearing with it — leaving the ungated answer in state.
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
            "postgresql://db.internal:5432/safety_kb",
            "mongodb://db.internal/safety_kb",
        ],
    )
    def test_framework_shapes_are_all_detected(self, sample):
        assert _scan(sample) is not None

    @pytest.mark.parametrize(
        "sample",
        [
            "pk-ABCD1234abcd5678EFGH",
            "ak-ABCD1234abcd5678EFGH",
            "password: hunter2hunter2",
            "api_key = ABCD1234abcd5678",
        ],
    )
    def test_additional_shapes_are_detected(self, sample):
        assert _scan(sample) is not None

    @pytest.mark.parametrize(
        "sample",
        [
            "プレス機の安全装置について",
            "ISO 45001 clause 8.2 covers emergency preparedness and response.",
            "The token of appreciation was presented to the safety team.",
            "Refer to section 6.1 for hazard identification.",
        ],
    )
    def test_ordinary_safety_text_is_not_flagged(self, sample):
        assert _scan(sample) is None


class TestContainment:
    def setup_method(self):
        self.node = PostProcessNode()

    LEAK = "Use this key: sk-ABCDEFGHIJKLMNOPQRSTUVWX to open the panel."

    def test_violating_answer_is_refused(self):
        result = self.node.execute({"result": self.LEAK})
        assert result["status"] == AgentStatus.ERROR.value

    def test_every_answer_bearing_field_is_present_and_cleared(self):
        # Presence AND emptiness. LangGraph merges partial deltas, so a key
        # simply omitted from the delta leaves the OLD value in state — an
        # assertion that only checks falsiness passes on a gate that clears
        # nothing at all.
        result = self.node.execute({"result": self.LEAK})
        for field in ANSWER_BEARING_FIELDS:
            assert field in result, f"{field} missing from the delta — nothing was cleared"
        assert result["result"] == ""
        assert result["response"] == ""

    def test_the_notice_is_truthy(self):
        # A falsy notice re-activates the `formatted_output or result` fallback
        # and produces the exact leak the clearing exists to prevent.
        result = self.node.execute({"result": self.LEAK})
        assert bool(result["formatted_output"])
        assert result["formatted_output"] == WITHHELD_NOTICE

    def test_no_returned_field_carries_the_released_text(self):
        result = self.node.execute({"result": self.LEAK})
        rendered = str(result)
        assert "sk-ABCDEFGHIJKLMNOPQRSTUVWX" not in rendered
        assert "open the panel" not in rendered

    def test_violation_message_names_a_reason_not_a_value(self):
        result = self.node.execute({"result": self.LEAK})
        assert "credential_shape" in result["error_log"][0]
        assert "ABCDEFGHIJKLMNOPQRSTUVWX" not in result["error_log"][0]

    def test_non_string_result_is_rendered_before_scanning(self):
        # Screening before a transformation is not screening: the value the
        # caller receives is the rendered one, so that is what gets scanned.
        result = self.node.execute({"result": {"note": "sk-ABCDEFGHIJKLMNOPQRSTUVWX"}})
        assert result["status"] == AgentStatus.ERROR.value
        assert "sk-ABCDEFGHIJKLMNOPQRSTUVWX" not in str(result)
