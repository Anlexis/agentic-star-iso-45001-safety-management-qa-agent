"""AgentCore Platform v1.0"""

# src/graph/context_bridge.py — carries the caller's validated payload across
# the outer -> inner graph boundary.
#
# Why this exists: the framework's GraphNode invokes the inner graph as
# `subgraph.invoke(user_input, session_id=..., ctx=...)` and does NOT forward the
# outer state's input_context. Without a bridge, every inner-node read of the
# caller's documents sees nothing at all, through the whole nested graph, while
# unit tests that hand a node a populated list keep passing. The sanctioned
# subclass hooks bridge it:
#
#   ManufacturingSafetyQAGraphNode.extract_input(state)
#       [runs BEFORE subgraph.invoke]  -> set_caller_payload(...)
#   DomainWorkflowGraph._extra_initial_state()
#       [runs INSIDE subgraph.invoke]  -> returns the stashed payload
#
# EVERY validated caller field the inner graph reads has to travel here. A field
# left behind is not a compile error and not a test failure: the inner node
# reads a key that is absent from its own state, silently falls back to a
# default, and the caller's setting is ignored while everything reports success.
#
# What crosses is the VALIDATED payload the outer pre_process node produced —
# never the raw input_context. Caller data is bounded exactly once, upstream of
# this bridge.
#
# A ContextVar keeps the hand-off correct per thread and per task, so concurrent
# invocations in one process cannot see each other's caller data.

from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Tuple

_CALLER_DOCUMENTS: ContextVar[Optional[List[Dict[str, Any]]]] = ContextVar("mfg_c2_021_caller_documents", default=None)
_CALLER_TOP_K: ContextVar[Optional[int]] = ContextVar("mfg_c2_021_caller_top_k", default=None)


def set_caller_payload(
    documents: Optional[List[Dict[str, Any]]],
    top_k: Optional[int] = None,
) -> None:
    """Stash the validated caller payload for the imminent inner-graph invoke."""
    _CALLER_DOCUMENTS.set([dict(doc) for doc in documents] if documents else [])
    _CALLER_TOP_K.set(top_k)


def get_caller_payload() -> Tuple[List[Dict[str, Any]], Optional[int]]:
    """Read (without consuming) the stashed payload; empty when none was set."""
    return _CALLER_DOCUMENTS.get() or [], _CALLER_TOP_K.get()


def clear_caller_payload() -> None:
    """Drop the stashed payload. Used by tests to assert the absent-data path."""
    _CALLER_DOCUMENTS.set(None)
    _CALLER_TOP_K.set(None)
