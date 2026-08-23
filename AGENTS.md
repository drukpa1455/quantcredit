# AGENTS

Build an executable consumer-credit research record with a tiny functional core
and an explicit notebook shell.

- `notebooks/credit_research.py` is the sole authored notebook.
- Stable operations live in `src/quantcredit/`; notebook cells call them.
- Make the common path memorable and fluid: minimal required arguments, safe
  defaults, explicit overrides, and helpers simple enough to rederive live.
- Preserve source states and causal time. Never infer outcomes from disappearance.
- Keep raw or sampled loan rows, notebook outputs, and personal SEC user agents
  out of Git.
- `quantcredit` may import released TinyMesh primitives. TinyMesh must never
  import credit-domain code.
- Do not add a model or abstraction before the prior research stage establishes
  its need.
