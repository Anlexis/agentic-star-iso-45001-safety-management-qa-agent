"""AgentCore Platform v1.0"""

# MFG-C2-021 — Outer graph (AgentBaseGraph; Cat 2 two-layer nested architecture)
#
# Architecture (Cat 2):
#
#   Outer backbone (fixed — identical to Cat 1, do NOT override add_edges()):
#     START → initialize → pre_process → main → {route} → post_process → finalize → END
#                                             ↓ (RETRY, max_retry)
#                                          pre_process
#
#   `main` slot is a GraphNode subclass (ManufacturingSafetyQAGraphNode) that
#   delegates the full domain workflow to DomainWorkflowGraph (inner BaseGraph).
#
#   Domain complexity is fully encapsulated inside the inner graph. The outer
#   backbone is never modified.
#
# Directory layout:
#   src/graph/graph.py                 ← outer graph (this file)
#   src/graph/domain_workflow_graph.py ← inner graph (multi-step topology)
#   src/graph/context_bridge.py        ← caller-document hand-off across the boundary
#
# Rules enforced:
#   ManufacturingSafetyQAAgent inherits AgentBaseGraph
#   super().register_nodes() called first (fills initialize + finalize)
#   ManufacturingSafetyQAGraphNode assigned to self._nodes["main"]
#   merge_output() returns only changed keys
#   add_edges() NOT overridden on the outer graph
#   No platform SDK imports

import pathlib
from typing import Any, ClassVar, Dict, List, Optional, cast

import yaml
from framework.graph.agent_base_graph import AgentBaseGraph
from framework.nodes.graph_node import GraphNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from src.graph.context_bridge import set_caller_payload
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State

_CONFIG_PATH = pathlib.Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def runtime_config() -> Dict[str, Any]:
    """Read the runtime parameters from config/config.yaml.

    This is the same file the platform registry loads and hands to the graph
    constructor. The standalone entry point reads it through this function so a
    registry-loaded agent and a directly deployed one see identical settings —
    without it, every value declared in config.yaml is inert in one of the two
    deployments and the difference is invisible: nothing fails, the framework
    simply falls back to its own defaults.

    Returns an empty mapping — never raises — when the file is absent or does
    not parse, so a missing file degrades to defaults rather than breaking
    start-up.
    """
    if not _CONFIG_PATH.exists():
        return {}
    try:
        loaded = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


class ManufacturingSafetyQAGraphNode(GraphNode):
    """GraphNode subclass assigned to the `main` slot of ManufacturingSafetyQAAgent.

    Wraps DomainWorkflowGraph (inner Cat 2 BaseGraph).
    Called by the backbone after pre_process and before post_process.

    Contracts:
      get_subgraph()    — instantiate and return DomainWorkflowGraph
      extract_input()   — pass the validated question into the inner invoke, and
                          stash the validated caller documents on the bridge
      merge_output()    — map sub_result fields into the outer state delta
      error_strategy    — "propagate": re-raise inner errors as SubgraphError

    Trust: ANONYMOUS at this boundary — the outer PreProcessNode already
    performed the caller check, and the inner nodes inherit the caller's
    InvocationContext unchanged.
    """

    error_strategy: ClassVar[str] = "propagate"

    # False: interrupts are handled inside the inner graph only.
    propagate_hitl: ClassVar[bool] = False

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def get_subgraph(self) -> Any:
        """Instantiate and return the inner domain workflow graph.

        Lazy import avoids circular-import risk at module load time. The runtime
        settings are forwarded so a value declared in config.yaml is live in the
        inner graph as well as the outer one.
        """
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        return DomainWorkflowGraph(config=self._parent_config())

    def _parent_config(self) -> Dict[str, Any]:
        """Forward the declared runtime settings to the inner graph.

        Reads the same config.yaml the outer graph was constructed from. Kept as
        its own method so the inner graph's configuration has one documented
        source rather than being reconstructed at each call site.
        """
        return runtime_config()

    def execute(self, state: AgentState) -> dict[str, Any]:
        """Skip the inner graph when the request was already found unacceptable.

        A request declined by pre_process has no validated input to act on, so
        running the inner graph would only produce a second, vaguer reason for
        the same rejection - and overwrite the specific one already settled.
        """
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}
        result: dict[str, Any] = super().execute(state)
        return result

    def extract_input(self, state: AgentState) -> str:
        """Return the question string passed into inner_graph.invoke().

        Also stashes the VALIDATED caller payload on the context bridge. The
        framework's GraphNode does not forward input_context into the inner
        graph, so without this the caller's knowledge-base entries and their
        retrieval override would be absent from every inner node and the agent
        would answer from its built-in defaults while the unit suite stayed
        green.

        What crosses is what PreProcessNode produced, never the raw context:
        caller data is bounded exactly once, upstream of this point.
        """
        set_caller_payload(
            cast(Optional[List[Dict[str, Any]]], state.get("caller_documents")),
            cast(Optional[int], state.get("caller_top_k")),
        )
        validated = state.get("validated_input")
        if isinstance(validated, str) and validated:
            return validated
        return cast(str, state.get("user_input", "") or "")

    def merge_output(self, state: AgentState, sub_result: Dict[str, Any]) -> Dict[str, Any]:
        """Map the inner graph's sub_result back into the outer state delta.

        Returns ONLY changed keys — never the full state.

        Key coupling (designed with DomainWorkflowGraph.get_output()):
          Inner get_output() emits:   "result", "status", "safety_flags",
                                      "extracted_procedures", "is_valid_response",
                                      "out_of_scope", "trace_id", "correlation_id"
          This merge_output() reads:  "result", "status", "safety_flags",
                                      "is_valid_response"

        A non-SUCCESS inner run contributes NO answer text. The inner graph
        already withholds it, but repeating the check here means neither layer
        alone has to be right for an unreleased answer to stay unreleased.
        """
        # Outer reason wins: a reason settled before the inner run is the real
        # one, and a plain sub_result.get() would erase it.
        marker = state.get("error_code") or sub_result.get("error_code", "")
        status = sub_result.get("status")
        if status != AgentStatus.SUCCESS.value:
            return {"result": None, "status": status, "error_code": marker}
        return {
            "result": sub_result.get("result"),
            "status": status,
            "safety_flags": sub_result.get("safety_flags"),
            "is_valid_response": sub_result.get("is_valid_response"),
            "error_code": marker,
        }


class ManufacturingSafetyQAAgent(AgentBaseGraph):
    """Outer graph for MFG-C2-021 (Cat 2).

    Inherits AgentBaseGraph directly (L1 Base). Domain logic is fully
    encapsulated in ManufacturingSafetyQAGraphNode (main slot), which delegates
    to DomainWorkflowGraph (inner BaseGraph).

    Construct with the runtime settings — `ManufacturingSafetyQAAgent(config=runtime_config())` —
    exactly as the platform registry does. The backbone consumes `max_retry`
    from that mapping for retry routing, and the caller-data bounds travel from
    the same mapping into the nodes that enforce them, so the declared values
    are live in both deployments.

    Backbone (fixed — identical to Cat 1):
        START → initialize → pre_process → main → post_process → finalize → END

    register_nodes() is the ONLY override:
      - super().register_nodes() fills: initialize, finalize (framework defaults)
      - pre_process: PreProcessNode (caller trust gate + full input contract)
      - main:        ManufacturingSafetyQAGraphNode (delegates to DomainWorkflowGraph)
      - post_process: PostProcessNode (output gate)

    add_edges() is NOT overridden — backbone wiring belongs to the framework.
    """

    @property
    def name(self) -> str:
        """Agent identifier registered with AgentRegistry."""
        return "mfg_c2_021"

    @property
    def state_schema(self) -> type:
        return State

    def register_nodes(self) -> None:
        """Fill all 5 backbone slots.

        super().register_nodes() MUST be called first — it injects the
        framework's default InitializeNode (sets schema_version, session_id,
        trust_level) and FinalizeNode (builds response_metadata, total_time_ms).
        """
        super().register_nodes()  # fills: initialize, finalize

        self._nodes["pre_process"] = PreProcessNode(config=self.config)
        self._nodes["main"] = ManufacturingSafetyQAGraphNode()
        self._nodes["post_process"] = PostProcessNode()

    # _invoke_impl() is NOT overridden — inherited from AgentBaseGraph.
    # add_edges() is NOT overridden — backbone wiring belongs to the framework.


# Back-compat alias — callers may reference either name.
Graph = ManufacturingSafetyQAAgent
