# MFG-C2-021 — Containment, measured through a real run.
#
# The fault is injected on the DATA path, never on the gate: a caller document
# carries a credential shape the framework's own detector does not recognise
# ("password: …"), so it survives every framework gate, is parsed into a clause,
# is rendered into the answer, and reaches the workflow's output gate as a
# genuinely violating answer. Patching the gate to force a violation would test
# the patch, not the agent.
#
# The two boundaries are exercised separately because they are reachable
# separately: the inner workflow is a published class a caller can invoke
# directly, and the agent's HTTP entry point is the deployed surface.

import importlib

import pytest
from fastapi.testclient import TestClient
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from framework.security.credential_detector import detect_credentials
from src.graph.context_bridge import clear_caller_payload, set_caller_payload
from src.graph.domain_workflow_graph import DomainWorkflowGraph
from src.graph.graph import runtime_config
from src.nodes.response_validate_node import WITHHELD_NOTICE

TOKEN = "containment-token"
SECRET = "password: hunter2hunter2"

POISONED_DOCUMENT = {
    "doc_id": "maint_log",
    "content": (
        "6.1 Actions to address risks and opportunities\n"
        f"The control panel is opened with {SECRET} before hazard identification starts.\n"
    ),
}

CLEAN_DOCUMENT = {
    "doc_id": "iso45001_c6",
    "content": (
        "6.1 Actions to address risks and opportunities\n"
        "The organization shall establish processes for hazard identification.\n"
    ),
}


def test_the_probe_uses_a_shape_the_framework_itself_misses():
    """Otherwise the framework blocks it upstream and this file proves nothing."""
    assert detect_credentials(SECRET) == []


def _run_inner(documents):
    clear_caller_payload()
    set_caller_payload(documents, None)
    graph = DomainWorkflowGraph(config=runtime_config())
    graph.compile()
    try:
        return graph.invoke(
            "hazard identification",
            ctx=InvocationContext(session_id="s", caller_trust_level=TrustLevel.VERIFIED_EXTERNAL),
        )
    finally:
        clear_caller_payload()


class TestInnerBoundary:
    def test_clean_path_control(self):
        """A refuse-everything gate must not be able to pass this file."""
        result = _run_inner([CLEAN_DOCUMENT])
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "hazard identification" in result["result"]
        assert "ResponseValidateNode" in result["node_history"]

    def test_violating_answer_is_refused(self):
        result = _run_inner([POISONED_DOCUMENT])
        assert result["status"] == AgentStatus.ERROR.value

    def test_the_block_happened_at_the_gate_not_upstream(self):
        result = _run_inner([POISONED_DOCUMENT])
        assert "ResponseValidateNode" in result["node_history"]
        assert "ResponseGenerateNode" in result["node_history"]

    def test_the_boundary_releases_no_answer_text(self):
        result = _run_inner([POISONED_DOCUMENT])
        rendered = str(result)
        assert SECRET not in rendered
        assert "control panel is opened" not in rendered

    def test_no_output_bearing_field_survives(self):
        result = _run_inner([POISONED_DOCUMENT])
        assert result["result"] == ""
        assert result["extracted_procedures"] == []

    def test_the_error_carries_a_reason_not_a_traceback(self):
        result = _run_inner([POISONED_DOCUMENT])
        for entry in result.get("error_log", []):
            assert "Traceback" not in entry
            assert "/src/" not in entry


class TestAgentBoundary:
    @pytest.fixture()
    def client(self, monkeypatch):
        monkeypatch.setenv("INVOKE_AUTH_TOKEN", TOKEN)
        import src.api.server as server

        importlib.reload(server)
        return TestClient(server.app)

    def _ask(self, client, documents):
        return client.post(
            "/invoke",
            json={"input": "hazard identification", "input_context": {"kb_documents": documents}},
            headers={"Authorization": f"Bearer {TOKEN}"},
        ).json()

    def test_clean_path_control(self, client):
        body = self._ask(client, [CLEAN_DOCUMENT])
        assert body["status"] == "success"
        assert "hazard identification" in body["output"]

    def test_the_envelope_carries_no_released_text(self, client):
        body = self._ask(client, [POISONED_DOCUMENT])
        assert body["status"] == "error"
        rendered = str(body)
        assert SECRET not in rendered
        assert "control panel is opened" not in rendered

    def test_the_envelope_carries_no_traceback_or_source_path(self, client):
        rendered = str(self._ask(client, [POISONED_DOCUMENT]))
        assert "Traceback" not in rendered
        assert "site-packages" not in rendered
        assert "/src/nodes/" not in rendered


class TestTheNoticeIsTruthy:
    """A falsy notice re-activates the `formatted_output or result` fallback.

    Setting the withheld marker to "" or {} is the single change that turns this
    whole gate back into the leak it was written to prevent, so the property is
    asserted directly rather than inferred from the envelope being clean.
    """

    def test_notice_is_truthy(self):
        assert bool(WITHHELD_NOTICE)
