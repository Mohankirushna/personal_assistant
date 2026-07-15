# ADR 0002: Native tool-calling over a hand-rolled JSON protocol

## Status

Accepted (Phase 5)

## Context

The planner needs the LLM to choose tools and supply arguments. The first
implementation asked the model (qwen2.5:3b) to emit a JSON object matching a
`{"action": "tool_call" | "respond", ...}` schema, using Ollama's
`format="json"` constrained decoding.

On the 3B model this failed in practice: measured outputs included malformed
JSON (`"args: {}` with a missing brace), chat-template tokens leaking into the
content (`</tool_call>`), and long multilingual hallucinations — for a plain
"what time is it?". Retk retries made turns slow (30s+) without fixing it.

## Decision

Use Ollama's native tool-calling API (`tools=[...]`, which maps tool specs
into the model's own trained function-call template) instead of a
hand-specified JSON protocol. Tool arguments still come back as structured
data, which we validate against each tool's Pydantic `args_model` before
execution.

## Consequences

- **Positive**: reliable tool selection and argument extraction on the 3B
  model; the model uses the format it was fine-tuned for. Integration tests
  (clock for time questions, finder_list among 26 tools) pass consistently.
- **Positive**: less prompt engineering; no JSON-repair retry loop.
- **Negative**: relies on the model being tool-calling-capable (qwen2.5 is;
  some small models aren't). Documented as a model requirement.
- **Neutral**: compound multi-step commands remain at the edge of 3B
  capability — handled by an empty-turn nudge + honest-inability fallback
  (never fabricated success), with 7B power mode for reliability.
