"""AgentCore Platform v1.0"""

# State must be a flat TypedDict — never a Pydantic BaseModel.
# LangGraph checkpoints use msgpack serialization; Pydantic objects
# cause silent corruption.  Extend AgentState with agent-specific
# fields only.  Do NOT add credentials, secrets, or Pydantic models.
#
# MFG-C2-021 — Manufacturing ISO 45001 / Safety Management System Q&A Agent
# Two-layer nested Cat 2 graph: outer backbone (AgentBaseGraph) + inner
# domain workflow (BaseGraph).  Fields below cover both layers.
#
# Personal-data note: a question may name the worker it is about. The input
# contract masks structured personal identifiers in caller-supplied documents
# and the framework masks them in the question itself, so no field below
# carries an unmasked one.

from typing import Any, Dict, List, Optional

from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus


class State(AgentState):
    """Flat TypedDict for MFG-C2-021.

    All shared fields (user_input, status, session_id, node_history,
    error_log, hitl_*, etc.) are inherited from AgentState.
    """

    # ------------------------------------------------------------------
    # Caller-supplied data — written by PreProcessNode, carried into the inner
    # workflow by the context bridge (src/graph/context_bridge.py).
    # ------------------------------------------------------------------

    # Validated knowledge-base entries: {"doc_id": inert id, "content": text}.
    # Empty when the caller supplied none; the pipeline then answers
    # out-of-scope rather than inventing a source.
    caller_documents: List[Dict[str, Any]]

    # Validated caller override for the number of documents carried forward.
    # None when the caller did not ask for one.
    caller_top_k: Optional[int]

    # ------------------------------------------------------------------
    # Input / normalization layer — QueryNormalizeNode
    # ------------------------------------------------------------------

    # The question as handed to the inner workflow. Accepted as an alias for
    # user_input so the node can be exercised with a pre-seeded field.
    query: str

    # Normalized query text (lowercased, de-duplicated whitespace, etc.)
    # produced by QueryNormalizeNode.
    normalized_query: str

    # Language detected by QueryNormalizeNode: "JA", "VI", or "EN".
    detected_language: str

    # ------------------------------------------------------------------
    # Retrieval layer — DocumentRetrieveNode
    # ------------------------------------------------------------------

    # ISO 45001 documents retrieved from the vector store.
    # Each entry is a dict with at minimum {"content": str, "source": str}.
    retrieved_documents: List[Dict[str, Any]]

    # ------------------------------------------------------------------
    # Extraction layer — ProcedureExtractNode
    # ------------------------------------------------------------------

    # ISO 45001 procedure items extracted from retrieved_documents.
    # Each entry: {"procedure_id": str, "title": str, "description": str}
    extracted_procedures: List[Dict[str, str]]

    # ------------------------------------------------------------------
    # Safety gate — SafetyFlagCheckNode
    # ------------------------------------------------------------------

    # Life-safety / emergency keywords detected in the query or procedures.
    # Each entry: {"keyword": str, "context": str}
    safety_flags: List[str]

    # True when SafetyFlagCheckNode determines the query requires immediate
    # emergency escalation (e.g. imminent danger, fatality risk).
    is_emergency_escalation: bool

    # ------------------------------------------------------------------
    # Response generation — ResponseGenerateNode (main slot)
    # ------------------------------------------------------------------

    # Generated response text presented to the end user.
    # Required: the main-slot node always writes a non-None result here
    # and also maps it to state["result"] for PostProcessNode compatibility.
    response: str

    # ------------------------------------------------------------------
    # Validation gate — ResponseValidateNode
    # ------------------------------------------------------------------

    # True when the output gate released the answer: non-empty, within the
    # declared length bound, and carrying no credential-shaped content.
    is_valid_response: bool

    # ------------------------------------------------------------------
    # Final status — set by the last executing node
    # ------------------------------------------------------------------

    # Final agent status: AgentStatus.SUCCESS or AgentStatus.ERROR.
    # out-of-scope queries are NOT a separate status value; use out_of_scope
    # (bool flag below) combined with agent_status = AgentStatus.SUCCESS.
    agent_status: AgentStatus

    # True if the query falls outside the ISO 45001 / safety management
    # domain scope as determined by QueryNormalizeNode or RetrieveNode.
    out_of_scope: bool

    # Human-readable error detail when agent_status == AgentStatus.ERROR.
    error_message: Optional[str]
    error_code: Optional[str]
