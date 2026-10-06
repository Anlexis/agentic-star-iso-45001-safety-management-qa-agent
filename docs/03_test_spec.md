# Test Specification — MFG-C2-021

## Strategy

Three layers, each answering a question the others cannot:

- **Unit tests** pin the contracts a single node owns — what it accepts, what it
  refuses, and exactly what its returned delta contains.
- **Boundary tests** pin the properties of the framework contract the agent
  depends on: invocation order, trust denial, import isolation, state
  serialisability.
- **End-to-end tests** drive the real ASGI application against the real compiled
  graph. Several of this agent's properties are observable nowhere else — the
  caller trust boundary, the hand-off of caller documents into the inner graph,
  and whether a declared configuration value reaches the node that enforces it.

A node-level suite can be entirely green while the deployed agent answers
nothing, so a green unit run is never treated as evidence that the agent works.

Assertions are behavioural: an input is stopped and nothing is carried forward,
never a particular error wording. A stop is asserted by its KIND — a declined
request completes carrying a reason code, a refused one terminates — because
collapsing the two would let a refusal be quietly downgraded into a correctable
value with every test still green. Where an end-to-end assertion has to look at
the answer body, it imports the sentence constant rather than repeating the text,
so the wording can change without the assertion changing.

Run: `python -m pytest tests/ -v`

## Files that ship

| File | Count | Covers |
|---|---|---|
| `tests/unit/test_pre_process_node.py` | 68 | The caller-data contract: question bounds, injection screen both directions, context contract, finite-number matrix, the declined/refused split, no-echo of rejected values |
| `tests/unit/test_response_validate_node.py` | 27 | The inner output gate: clean path, configured length bound, detector parity, refusal AND clearing |
| `tests/unit/test_post_process_node.py` | 24 | The agent-boundary output gate: same properties at the outer layer |
| `tests/unit/test_safety_flag_check_node.py` | 8 | Escalation from the question; procedure terms surfaced without escalating |
| `tests/unit/test_procedure_extract_node.py` | 7 | Clause parsing and de-duplication |
| `tests/unit/test_response_generate_node.py` | 7 | The three answer paths and language selection |
| `tests/unit/test_document_retrieve_node.py` | 6 | Ranking, the `top_k` bound, and the zero-result path |
| `tests/unit/test_query_normalize_node.py` | 6 | Unicode normalisation and JA/VI/EN detection |
| `tests/unit/test_framework_compliance_tc06_tc07.py` | 2 | The framework's gates cannot be replaced by a subclass |
| `tests/integration/test_invoke_e2e.py` | 34 | The caller boundary, real work from caller documents, live configuration, bounded input, how each kind of stop is reported to the caller, credential-shaped context |
| `tests/integration/test_output_containment_e2e.py` | 11 | Containment at both boundaries, measured on a fault injected on the data path |
| `tests/proof_of_boundary/test_pb_emergency_escalation.py` | 8 | Escalation across EN/JA/VI and the no-escalation control |
| `tests/proof_of_boundary/test_pb_output_gate.py` | 3 | The output gate refuses and withholds |
| `tests/proof_of_boundary/test_pb_invoke_order.py` | 2 | PB-6 invocation order and trust denial |
| `tests/proof_of_boundary/test_pb_out_of_scope.py` | 2 | The out-of-scope reply is canned and cites nothing |
| `tests/proof_of_boundary/test_pb7_hitl_interrupt_propagation.py` | 2 | Human-review interrupt propagation (skipped: this agent does not interrupt) |
| `tests/proof_of_boundary/test_import_isolation.py` | 1 | PB-4 import isolation |
| `tests/proof_of_boundary/test_state_safety.py` | 1 | PB-2 state is serialisable primitives only |

## Framework compliance

| TC-ID | Test | Expected | Where |
|---|---|---|---|
| TC-01 | State is a flat TypedDict | No Pydantic or dataclass in state | `test_state_safety.py` |
| TC-02 | Invalid input is stopped | A correctable value: success status carrying a reason code. A control token or instruction-override directive: error status. Neither carries validated data forward | `test_pre_process_node.py` |
| TC-03 | No credential in state | Repo credential scan: 0 findings | `scripts/check_credentials.py` |
| TC-05 | No duplicate lifecycle events in `execute()` | `node_start`/`node_complete`/`node_error` absent from node bodies | `scripts/check_audit_trace.py` |
| TC-06 | The default input gate cannot be overridden | `TypeError` at class definition | `test_framework_compliance_tc06_tc07.py` |
| TC-07 | The default output gate cannot be overridden | `TypeError` at class definition | `test_framework_compliance_tc06_tc07.py` |
| TC-08 | `required_trust_level` is enforced | Insufficient trust is refused before `execute()` | `test_pb_invoke_order.py` |
| TC-11 | Every node emits a domain audit event | ≥1 per node, on a reachable path | `scripts/check_audit_trace.py` |

## Proof-of-boundary

| PB-ID | Boundary | Expected | Where |
|---|---|---|---|
| PB-2 | State serialisation | Post-invoke state is primitives only | `test_state_safety.py` |
| PB-4 | Import isolation | No platform-SDK imports (AST scan) | `test_import_isolation.py` |
| PB-6 | Invocation order | trust gate → `node_start` → input gate → `execute()` → output gate → `node_complete` | `test_pb_invoke_order.py` |
| PB-7 | Human-review interrupt | Skipped — this agent does not call `interrupt()` | `test_pb7_hitl_interrupt_propagation.py` |

## Business-logic coverage

| ID | Property | Where |
|---|---|---|
| BL-01 | A caller's documents produce a real answer citing their clauses | `test_invoke_e2e.py` |
| BL-02 | The answer changes when the question changes | `test_invoke_e2e.py` |
| BL-03 | No documents degrades to the out-of-scope reply, never an invented answer | `test_invoke_e2e.py` |
| BL-04 | A question describing a live hazard escalates instead of answering | `test_invoke_e2e.py`, `test_pb_emergency_escalation.py` |
| BL-05 | A fire-response clause in the corpus does NOT escalate a routine question | `test_pb_emergency_escalation.py` |
| BL-06 | A declared `top_k` visibly changes which clauses are cited | `test_invoke_e2e.py` |
| BL-07 | A declared answer-length bound withholds an answer that exceeds it | `test_invoke_e2e.py` |
| BL-08 | Unauthenticated and wrongly authenticated callers are refused identically | `test_invoke_e2e.py` |
| BL-09 | Injection terminates the run end to end, with no output; the same words in a real question are answered | `test_invoke_e2e.py`, `test_pre_process_node.py` |
| BL-10 | Non-finite and out-of-range numbers are declined — the run completes carrying the reason sentence and no answer — including a raw JSON `NaN` | `test_invoke_e2e.py`, `test_pre_process_node.py` |
| BL-11 | A credential-shaped context value is refused readably, naming the field only | `test_invoke_e2e.py` |
| BL-12 | Containment: a violating answer is refused AND every answer-bearing field cleared | `test_output_containment_e2e.py` |
| BL-13 | A correctable value declines and a screen hit terminates; the two kinds of stop never collapse into one, and a declined request carries a reason code rather than an answer | `test_pre_process_node.py`, `test_invoke_e2e.py` |

## How containment is tested

The fault is injected on the DATA path, never on the gate. A caller document
carries a credential shape the framework's own detector does not recognise, so
it survives every framework gate, is parsed into a clause, is rendered into the
answer, and reaches the output gate as a genuinely violating answer. Patching
the gate to force a violation would test the patch, not the agent.

The suite keeps a **clean-path control** alongside it: the same request with an
ordinary document still produces its real answer, so a gate that refused
everything could not pass. Each containment test also asserts that the gate node
appears in the node history, proving the block happened at the gate rather than
somewhere upstream.

Clearing is asserted as **presence AND emptiness**: the key must be in the
returned delta and hold the cleared value. Partial deltas are merged, so a key
simply omitted leaves the old value in state — an assertion that only checks
falsiness passes on a gate that clears nothing.
