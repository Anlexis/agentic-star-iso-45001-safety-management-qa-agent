# ISO 45001 Safety Management Q&A Agent

AI agent for answering ISO 45001 safety management system questions, built with Agentic Star.

- **Category**: Cat 2 (a domain pipeline built on the framework's base graph)
- **Industry**: Manufacturing
- **Template ID**: MFG-C2-021

## Overview

A question-answering agent for occupational health and safety management systems. It answers questions about ISO 45001 clause requirements from a knowledge base of standard and company safety documents, citing the clauses its answer is built from, and recognises a question that describes a live emergency so the caller is directed to a safety officer instead of a procedure summary.

Answers are assembled deterministically from the retrieved clauses — there is no model call, so the same question over the same documents always produces the same answer, and every sentence in it is traceable to a source. Questions and answers are handled in Japanese, Vietnamese and English.

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent connects to the platform at start-up. Without it, start-up fails immediately (see *Behaviour without the platform* below). Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Installed from PyPI as a dependency. |
| Python | >=3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run **only** on AGENTIC STAR. There is no fallback or degraded
mode. If the platform is unreachable or the SDK version does not match, the agent fails during
graph compile / start-up preflight rather than starting in a partially working state. This is
intentional — a half-running agent is worse than one that refuses to start.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection. Running the agent itself does not.

## Project Structure

```
src/graph/    outer agent graph, inner workflow graph, caller-data bridge
src/nodes/    the workflow steps and the input and output gates
src/schemas/  the state shared by both graphs
src/api/      the HTTP entry point
src/examples/ reference samples for each agent shape
tests/        unit, integration and boundary tests
config/       agent.yaml (the manifest) and config.yaml (runtime bounds)
docs/         design specification and test specification
```

See `docs/02_design.md` for the node-by-node design and the caller-data contract, and
`docs/03_test_spec.md` for what the suite covers.

## Customising

1. Adjust `config/config.yaml` for your own bounds — how many source documents a caller may
   supply, how large each may be, how many are carried into an answer, and how long an
   answer may be. Each value has a reader in `src/` and changes observable behaviour.
2. Supply your own safety documents. A caller passes them per request on `input_context`;
   to retrieve from a store instead, pass a retriever to `DocumentRetrieveNode`.
3. Adjust the escalation vocabulary in `src/nodes/safety_flag_check_node.py` to your own
   hazard terminology.
4. Re-run the test suite.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.
