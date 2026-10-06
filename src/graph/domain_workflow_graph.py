"""AgentCore Platform v1.0"""

# MFG-C2-021 — DomainWorkflowGraph (inner BaseGraph)
#
# This is the INNER graph for the Cat 2 two-layer nested architecture.
# It encapsulates the full ISO 45001 / safety management system workflow:
#
#   START → query_normalize → document_retrieve → procedure_extract
#         → safety_flag_check → response_generate → response_validate → END
#
# Called by ManufacturingSafetyQAGraphNode.get_subgraph() (graph.py).
# get_output() shapes the sub_result dict consumed by merge_output() there.
#
# Rules enforced:
#   Inherits BaseGraph (fully custom topology — no forced backbone)
#   Implements all 7 BaseGraph ABC methods
#   register_nodes() does NOT call super() (abstract in BaseGraph)
#   Does NOT register initialize / finalize (outer backbone concerns)
#   get_output() designed together with ManufacturingSafetyQAGraphNode.merge_output()
#   _extra_initial_state() seeds the validated caller documents (context bridge)
#   No platform SDK imports

from typing import Any, Dict

from framework.errors import ConfigError
from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_status import AgentStatus
from langgraph.graph import END, START
from src.graph.context_bridge import get_caller_payload
from src.nodes.document_retrieve_node import DocumentRetrieveNode
from src.nodes.procedure_extract_node import ProcedureExtractNode
from src.nodes.query_normalize_node import QueryNormalizeNode
from src.nodes.response_generate_node import ResponseGenerateNode
from src.nodes.response_validate_node import ResponseValidateNode
from src.nodes.safety_flag_check_node import SafetyFlagCheckNode
from src.schemas.state import State


class DomainWorkflowGraph(BaseGraph):
    """Inner domain workflow graph for MFG-C2-021.

    Inherits BaseGraph directly for a fully custom node topology.
    Called by ManufacturingSafetyQAGraphNode.get_subgraph() in graph.py.

    Pipeline (linear):
        START
          → query_normalize      (QueryNormalizeNode)
          → document_retrieve    (DocumentRetrieveNode)
          → procedure_extract    (ProcedureExtractNode)
          → safety_flag_check    (SafetyFlagCheckNode)
          → response_generate    (ResponseGenerateNode)
          → response_validate    (ResponseValidateNode)
          → END

    All nodes are FunctionNode subclasses returning partial-dict state updates.
    initialize / finalize are outer backbone concerns — not registered here.
    """

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Unique identifier for this inner graph."""
        return "mfg_c2_021_safety_qa_workflow"

    @property
    def state_schema(self) -> type:
        """TypedDict subclass shared across inner and outer graph."""
        return State

    # ── Config validation ─────────────────────────────────────────────────────

    def _validate_config(self) -> None:
        """Reject a runtime configuration this graph cannot honour.

        The bounds below are handed to the nodes that enforce them, so an
        unusable value has to fail here rather than be silently replaced by a
        default — a bound that quietly reverts is a bound nobody is enforcing.
        """
        for block, key, low, high in (
            ("retrieval", "top_k", 1, 50),
            ("retrieval", "max_caller_documents", 1, 500),
            ("retrieval", "max_document_chars", 1, 1_000_000),
            ("response", "max_response_chars", 1, 1_000_000),
        ):
            section = self.config.get(block)
            if section is None:
                continue
            if not isinstance(section, dict):
                raise ConfigError(f"[{type(self).__name__}] '{block}' must be a mapping")
            if key not in section:
                continue
            value = section[key]
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ConfigError(f"[{type(self).__name__}] '{block}.{key}' must be an integer " f"in [{low}, {high}]")

    # ── Caller-data hand-off ──────────────────────────────────────────────────

    def _extra_initial_state(self) -> Dict[str, Any]:
        """Seed the validated caller payload into the inner initial state.

        The framework's GraphNode does not forward input_context into a
        subgraph, so this hook and the matching stash in the outer node's
        extract_input() are the whole bridge. Reading it here rather than in a
        node keeps the hand-off at the graph boundary, where the ordering is
        guaranteed: extract_input() runs before invoke(), invoke() calls this.

        Every validated field an inner node reads is seeded here. One left out
        is invisible: the node reads a key absent from its own state, quietly
        takes its default, and the caller's setting is ignored under a success
        status.
        """
        documents, top_k = get_caller_payload()
        return {"caller_documents": documents, "caller_top_k": top_k}

    # ── Node registration ─────────────────────────────────────────────────────

    def register_nodes(self) -> None:
        """Register all 6 domain nodes with the bounds declared in config.yaml.

        No super() call — BaseGraph.register_nodes() is abstract.
        Do NOT register initialize or finalize; those are outer backbone
        concerns handled by AgentBaseGraph in graph.py.
        Every key registered here is referenced in add_edges().
        """
        retrieval: Dict[str, Any] = self.config.get("retrieval") or {}
        response: Dict[str, Any] = self.config.get("response") or {}

        self._nodes["query_normalize"] = QueryNormalizeNode()
        self._nodes["document_retrieve"] = DocumentRetrieveNode(config=retrieval)
        self._nodes["procedure_extract"] = ProcedureExtractNode()
        self._nodes["safety_flag_check"] = SafetyFlagCheckNode()
        self._nodes["response_generate"] = ResponseGenerateNode()
        self._nodes["response_validate"] = ResponseValidateNode(config=response)

    # ── Edge wiring ───────────────────────────────────────────────────────────

    def add_edges(self) -> None:
        """Wire the linear safety Q&A domain topology.

        Each step passes its partial-dict output into the shared State.
        The topology is intentionally linear — no conditional branching between
        domain nodes. route() is implemented as required by the ABC but
        add_conditional_edges() is not used.

        Flow:
          QueryNormalize → DocumentRetrieve → ProcedureExtract
                        → SafetyFlagCheck → ResponseGenerate → ResponseValidate
        """
        self._sg.add_edge(START, "query_normalize")
        self._sg.add_edge("query_normalize", "document_retrieve")
        self._sg.add_edge("document_retrieve", "procedure_extract")
        self._sg.add_edge("procedure_extract", "safety_flag_check")
        self._sg.add_edge("safety_flag_check", "response_generate")
        self._sg.add_edge("response_generate", "response_validate")
        self._sg.add_edge("response_validate", END)

    # ── Routing ───────────────────────────────────────────────────────────────

    def route(self, state: State) -> str:
        """Conditional routing — required by BaseGraph ABC.

        Annotated with this graph's OWN State: LangGraph reads a path
        callable's annotation as its input schema and projects away every field
        the annotation does not carry, so a wider annotation here would hide
        the domain fields the decision is made from.

        For this linear topology add_conditional_edges() is not used, so the
        method is never called at runtime. It returns END on error so an
        unexpected call cannot re-enter a processing node.
        """
        if state.get("status") == AgentStatus.ERROR.value:
            return END
        return "response_validate"

    # ── Output shape ──────────────────────────────────────────────────────────

    def get_output(self, state: State) -> Dict[str, Any]:
        """Shape the output dict returned to the outer graph as sub_result.

        This dict is received by ManufacturingSafetyQAGraphNode.merge_output()
        in graph.py as the `sub_result` argument. Both methods are designed
        together to guarantee field-name consistency:

            Inner get_output() emits:  "result", "status", "safety_flags",
                                       "extracted_procedures", "is_valid_response",
                                       "out_of_scope", "trace_id", "correlation_id",
                                       "node_history", "error_log"
            Outer merge_output() reads: "result", "status", "safety_flags",
                                        "is_valid_response"

        This method reports the state FAITHFULLY and blanks nothing of its own.
        Withholding the answer on a non-success status is the gate node's job,
        and it does it by clearing the answer-bearing fields in the state this
        reads. A second status check here would contain the same leak on its
        own, and would therefore make the gate's clearing impossible to falsify
        — every test would still pass with the clearing removed. More defence
        can buy less assurance; the clearing is verified by removing it and
        watching this boundary leak.

        error_log travels so the outer boundary can say WHY a run failed. It
        carries reason codes and field paths only — the nodes that write it
        never put caller content or a matched value in it.
        """
        return {
            # the reason must leave the subgraph or the outer graph cannot report it
            "error_code": state.get("error_code"),
            "result": state.get("result"),
            "safety_flags": state.get("safety_flags"),
            "extracted_procedures": state.get("extracted_procedures"),
            "is_valid_response": state.get("is_valid_response"),
            "out_of_scope": state.get("out_of_scope"),
            "status": state.get("status"),
            "error_log": state.get("error_log", []),
            "trace_id": state.get("trace_id"),
            "correlation_id": state.get("correlation_id"),
            "node_history": state.get("node_history", []),
        }

    # ── Lifecycle helpers ─────────────────────────────────────────────────────

    def get_state_class(self) -> type:
        """Return the State TypedDict used by both inner and outer graphs."""
        return State
