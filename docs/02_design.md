# Design Specification — MFG-C2-021 Safety Management System Q&A Agent

## What the agent does

It answers questions about occupational health and safety management system
requirements from a corpus of safety documents, citing the clauses its answer is
built from. Three outcomes are possible for a well-formed request:

| Outcome | When | What the caller gets |
|---|---|---|
| Escalation | The question describes a live hazard | A short notice directing the caller to a safety officer, with the hazard terms that matched |
| Answer | Clauses were found for the question | A structured answer, one section per cited clause, with a reference count |
| Out of scope | Nothing in the corpus matched | A single sentence saying so — never an invented answer |

Answers are assembled deterministically from the retrieved clauses. There is no
model call, so the same question over the same documents always produces the
same answer, and every sentence in it is traceable to a source document.
Japanese, Vietnamese and English are detected and answered in kind.

A request that is not well-formed never reaches any of the three outcomes: it
stops before the workflow runs, and how it is reported depends on which kind of
stop it is — see *Two ways a request stops*.

## Position in the framework

| Aspect | This agent |
|---|---|
| Agent class | `ManufacturingSafetyQAAgent` (`src/graph/graph.py`) |
| L1 Base (framework base class) | AgentBaseGraph — direct framework inheritance |
| Shape | Cat 2 — an outer backbone whose `main` slot delegates to an inner workflow graph |
| State | A flat TypedDict (`src/schemas/state.py`), shared by both graphs |
| Nodes | `FunctionNode` subclasses implementing `execute(self, state) -> dict`, each returning only the keys it changes |
| Error propagation | `propagate` — an inner failure is raised at the boundary rather than partially merged |

`execute()` takes the state and nothing else. The framework's node wrapper calls
it with one argument, so a node that declares a second `config` parameter never
receives one: any value read from it is silently the default. Runtime settings
reach the nodes through their constructors instead — see *Runtime configuration*.

## Structure

```
src/graph/graph.py                  outer graph + the runtime-config reader
src/graph/domain_workflow_graph.py  inner workflow graph
src/graph/context_bridge.py         caller-data hand-off across the boundary
src/nodes/                          the workflow steps and both gates
src/schemas/state.py                the shared state
src/api/server.py                   the HTTP entry point
```

### Data flow

```
START → initialize → pre_process → main → {route} → post_process → finalize → END
                                            ↓ (retry)
                                         pre_process

main = ManufacturingSafetyQAGraphNode
         └── DomainWorkflowGraph:
             query_normalize → document_retrieve → procedure_extract
                             → safety_flag_check → response_generate
                             → response_validate
```

The outer backbone is the framework's and is not overridden. Note the `{route}`
edge: on any non-success status the run goes straight to `finalize`, skipping
`post_process` — so the outer output gate never sees a failed run, which is why
the inner workflow has a gate of its own.

A request declined for a value the caller can correct is not a failed run. It
carries a success status, so it keeps to the normal path and does reach
`post_process` — which is where the reason is rendered into the answer body.
Every node between the decline and that point checks the marker and does nothing.

### Nodes

| Node | Reads | Writes | Responsibility |
|---|---|---|---|
| `PreProcessNode` (pre_process) | `user_input`, `input_context` | `validated_input`, `caller_documents`, `caller_top_k`, `enriched_context`, or `error_code` when the request is declined | The single gate for caller data: length, injection screen, context contract, personal-data masking |
| `ManufacturingSafetyQAGraphNode` (main) | `validated_input`, `caller_documents`, `caller_top_k`, `error_code` | `result`, `safety_flags`, `is_valid_response`, `error_code` | Stashes the caller payload on the context bridge, runs the inner graph, merges its result back — or skips the inner graph outright when a reason was already settled |
| `QueryNormalizeNode` | `user_input` (`query` accepted as an alias) | `normalized_query`, `detected_language` | Unicode normalisation and JA/VI/EN detection |
| `DocumentRetrieveNode` | `normalized_query`, `caller_documents`, `caller_top_k` | `retrieved_documents`, `out_of_scope` | Ranks the corpus by term overlap and keeps the best `top_k` |
| `ProcedureExtractNode` | `retrieved_documents` | `extracted_procedures` | Parses clause headings and bodies out of the retrieved text |
| `SafetyFlagCheckNode` | `normalized_query`, `extracted_procedures` | `is_emergency_escalation`, `safety_flags` | Decides escalation from the question; surfaces hazard terms found in the cited procedures |
| `ResponseGenerateNode` | `is_emergency_escalation`, `out_of_scope`, `extracted_procedures` | `response`, `result` | Renders one of the three outcomes |
| `ResponseValidateNode` | `response`, `out_of_scope` | `is_valid_response`, and on refusal the cleared answer fields | The inner output gate |
| `PostProcessNode` (post_process) | `result`, `error_code` | `formatted_output`, and on a declined request `result`, `response`, `error_code` | Renders the released answer and gates it once more at the agent boundary; on a declined request renders the decline sentence instead, and nothing else |

### State

| Field | Type | Purpose |
|---|---|---|
| `caller_documents` | `list[dict]` | Validated `{doc_id, content}` entries; empty when the caller supplied none |
| `caller_top_k` | `int \| None` | Validated caller override for the retrieval reach |
| `query` | `str` | Alias for the question when a node is exercised directly |
| `normalized_query` | `str` | NFC-normalised, whitespace-collapsed question |
| `detected_language` | `str` | `JA`, `VI` or `EN` |
| `retrieved_documents` | `list[dict]` | Ranked corpus entries with their scores |
| `extracted_procedures` | `list[dict]` | `{clause, title, content}` entries |
| `safety_flags` | `list[str]` | Hazard terms — from the question first, then from the cited procedures |
| `is_emergency_escalation` | `bool` | True when the QUESTION carries a hazard term |
| `response` | `str` | The generated answer |
| `is_valid_response` | `bool` | True when the output gate released the answer |
| `out_of_scope` | `bool` | True when retrieval found nothing |
| `error_code` | `str \| None` | Set when the request was declined for a value the caller can correct; it selects the sentence the caller reads and is never returned in the response envelope |
| `error_message` | `str \| None` | Reason code from the output gate |

State constraints: flat TypedDict only, JSON-serialisable values, no credentials
and no Pydantic models — the checkpointer serialises with msgpack, which
silently corrupts arbitrary objects.

## The caller-data contract

`/invoke` accepts a question and an optional `input_context`:

| Field | Shape | Bound |
|---|---|---|
| `channel` | inert identifier | `[a-z0-9_]{1,32}` |
| `kb_documents` | list of `{doc_id, content}` | entry cap and per-document size cap from `config/config.yaml`; `doc_id` is an inert identifier |
| `top_k` | integer | finite, 1–20 |

A caller with no structured channel sends one text field, so the same request
object may travel there instead: a JSON object in `user_input` carrying
`question` plus any of the fields above. It is a transport, not a second
contract — the question is screened as a question and the context fields as
context fields, through the identical validation. Text that is not a JSON object
is an ordinary question. Where both channels carry the same field, the
structured channel wins, because it is the declared contract. The envelope as a
whole is bounded separately from the question, since the question bound alone
cannot also bound the documents.

Rules the contract enforces:

- **Unknown fields are refused, not ignored.** A validator that ignores an
  unknown key leaves it in the mapping for whatever reads it next, and leaves the
  caller believing it had an effect.
- **Every caller number goes through a finite, bounded parser.** `NaN` and the
  infinities parse cleanly through `float()` and arrive intact over raw JSON, and
  every comparison against `NaN` is False — a non-finite bound would silently
  disable the check it configures. The parser fails closed.
- **The audit record names the field and a reason code, never the value.** The
  framework's output gate scans every value a node returns for credential shapes,
  so quoting caller content into an error makes that gate raise and discard the
  node's entire result.
- **Absent context degrades to the baseline.** No documents means the
  out-of-scope answer, which is the honest response when there is no source.

### Two ways a request stops

A stop is not one event. Whether the caller can do anything about it decides how
it is reported, and the two kinds are kept apart deliberately.

| | Declined | Refused |
|---|---|---|
| Cause | A value the caller can correct: an empty question, a question past the length bound, a context field outside the contract | Content the agent will not act on: a chat-template control token or an instruction-override directive, in the question or anywhere in `input_context` |
| Status | success, carrying an internal reason code — `EMPTY_INPUT`, `QUESTION_TOO_LONG` or `INVALID_REQUEST` | error |
| What the caller gets | A fixed sentence naming what to correct, as the answer body | No output at all: the `{route}` edge sends a failed run straight to `finalize` |
| Work done | None — the inner workflow does not run and no answer is assembled | None — the framework skips every remaining `execute()` once the status is error |

A declined request completes so the caller can correct the value and send the
request again on the same conversation. Terminating instead ends the calling
surface's turn and surfaces an exception type alone, leaving the reason reachable
only from the audit record. A refusal terminates for the opposite reason:
rewording is not a route past it, and reporting it as a correctable value would
say that it is.

The reason code is internal. It selects the sentence and never appears in the
response envelope, so the caller reads the sentence rather than a code. The
sentences live in `src/services/failure_message.py`; each names WHAT to correct
and nothing else — never the rejected value, a field path or a gate message,
which stay in `error_log`.

A declined run still traverses the backbone, so every node after the decline has
to check the marker and do nothing. Without that check the inner workflow would
run on a request nothing validated and would overwrite the specific reason with a
vaguer one. A refused run needs no such check: the framework already skips
`execute()` on every node once the status is error.

Nothing structured is released on a declined run. The answer body is the sentence
and the clause sections, flags and reference counts are withheld — a run that did
not process the request must not hand back something shaped like a result.

### Injection screen

The screen refuses; it does not strip and forward. Removing a marker and
forwarding the surrounding text converts a detectable attack into undetectable
plain text — the directive still reads as an instruction and nothing downstream
can tell it apart from the question. A hit terminates the run: it is a refusal,
never a declined value. Two families are covered:

- **Chat-template control tokens** as a class: `<|…|>`, `[INST]`/`[/INST]`,
  `<<SYS>>`, `<system>`-style role markers, `### System:` headings.
- **Instruction-override directives** in EN, JA and VI, anchored on a verb plus
  an instruction/prompt/rule object so ordinary safety prose survives — a
  procedure may legitimately say "ignore the previous revision of the checklist".

Each string is screened raw, with zero-width characters removed, and after NFKC
folding, so a directive split by a soft hyphen or written in fullwidth
characters is still caught. Caller context is screened depth-first including
KEYS: a key name is caller data and reaches the same places a value does.

### Personal data

The framework's input gate masks personal data in the question. It does not
touch `input_context`, so the contract masks structured identifiers — card
numbers, national IDs, e-mail addresses, telephone numbers, labelled names — in
caller documents itself.

The generic title-case name category is deliberately not applied to reference
documents. Its pattern is a run of capitalised words separated by whitespace,
and whitespace includes a newline, so a clause heading followed by a sentence
("Competence\nThe organization shall …") reads as a two-word personal name.
Masking it removes the heading, the extractor can no longer find the clause, and
the answer silently drops it. Measured on this agent's own corpus.

### Credential shapes in `input_context`

A credential-shaped string anywhere in `input_context` fails the run at the
FIRST node: the backbone's initialize node returns `input_context` verbatim in
its own result, and the framework's output gate scans every value of every
result. The caller gets an error with no indication of what went wrong.

The request cannot succeed either way, so the entry point screens the assembled
context before invoking and refuses with a 400 naming the field. It calls the
same detector the framework's gate calls, on the same object, so the refusal set
and the block set are one set by construction. Field names are caller data too:
a name is echoed only when it is short, inert and carries no credential shape of
its own, otherwise the field is reported by position.

## Runtime configuration

`config/config.yaml` holds the runtime settings. The platform registry loads it
and passes it to the graph constructor; the HTTP entry point reads it through the
same function, so a declared value is live in both deployments.

| Key | Read by | Effect |
|---|---|---|
| `max_retry` | the framework backbone | retry routing |
| `timeout_s` | the framework backbone | invocation timeout |
| `retrieval.top_k` | `DocumentRetrieveNode` | how many documents reach extraction |
| `retrieval.max_caller_documents` | `PreProcessNode` | entry cap on caller documents |
| `retrieval.max_document_chars` | `PreProcessNode` | per-document size cap |
| `response.max_response_chars` | `ResponseValidateNode` | answers longer than this are withheld |

The inner graph rejects an unusable value rather than replacing it with a
default: a bound that quietly reverts is a bound nobody is enforcing.

## Trust

The manifest registers the agent at `VERIFIED_EXTERNAL`, and `PreProcessNode`
declares the same level, so an unauthenticated caller is refused at the first
node. The caller's trust travels unchanged into the inner graph — a subgraph
does not gain privilege by being nested — so every node inside declares the
level a legitimate caller can actually hold. Declaring a higher one there would
not add protection; it would make the pipeline deny itself on every request.

In a standalone deployment nothing else establishes the caller's trust, so the
entry point authenticates: with `INVOKE_AUTH_TOKEN` set on the server
environment, a caller no middleware vouched for must present it as a bearer
token to run as a verified external caller. Trust already established upstream is
never demoted.

## The output boundary

There are two gates: `ResponseValidateNode` inside the workflow and
`PostProcessNode` at the agent boundary. They are independent — the inner one
never runs at the agent boundary, and the outer one never sees a run the inner
workflow refused.

Each gate does two separate things, and only the second withholds anything:

1. **Refuse.** Return an error status on an empty answer, an answer over the
   declared length bound, or an answer carrying credential-shaped content.
2. **Clear.** Overwrite every field that could carry the answer forward, and put
   a **truthy** notice in `formatted_output`.

The second matters because the framework resolves an agent's output as
`formatted_output or result`, with no status check. An error status alone still
ships the ungated answer inside the error envelope, and an empty-string notice
falls straight through to it. `src/nodes/response_validate_node.py` pins the
complete inventory of answer-bearing fields so a field added later cannot quietly
stay out of the clearing.

Both gates call the framework's own credential detector and add two shapes it
does not carry (`pk-`/`ak-` prefixed keys, and credential assignment in prose).
The local set is a superset by construction. A local set *narrower* than the
framework's would be a containment bypass rather than a smaller gate: the
framework's gate raises after `execute()` returns, and the wrapper then discards
the node's whole result — the clearing with it.

The inner graph's `get_output()` reports state faithfully and blanks nothing of
its own. A second status check there would contain the same leak on its own and
make the gate's clearing impossible to falsify — every test would still pass with
the clearing removed.

**Numeric precision:** this agent renders no monetary aggregates. Its answers
carry clause numbers, section headings and reference counts, and it applies no
rounding or snapping to any of them, so a monetary-precision grid is not
applicable here. The invariant it does enforce is that every rendered clause is
traceable to a supplied document.

## Import isolation

The template imports `framework.*`, `shared.*` and `langgraph.*` only, never the
platform SDK underneath them. `tests/proof_of_boundary/test_import_isolation.py`
scans for this.
