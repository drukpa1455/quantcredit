# quantcredit

Executable consumer-credit research from public records to classical and graph
models.

The project starts with SEC Form ABS-EE auto-loan disclosures. It first proves
what the records mean, then builds a shallow gradient-boosted baseline, and only
then asks whether relational structure adds information.

```text
SEC filings -> loan states -> causal target -> shallow GBM -> graph controls
```

## Notebook

[`notebooks/credit_research.py`](notebooks/credit_research.py) is the canonical
notebook. Zed executes each block between `# %%` markers through a Jupyter
kernel while Git and coding agents see ordinary Python.

```console
uv sync --locked --group notebook
uv run --locked python -m quantcredit.source
```

Open the repository in Zed, select its `.venv` toolchain and kernel, then run a
cell with `Ctrl-Shift-Enter`.

To acquire and checksum-pin the declared EX-102 files, provide the identifying
user agent required by SEC access policy without storing it in the repository:

```console
export QUANTCREDIT_SEC_USER_AGENT="your-name your-email"
uv run --locked python -m quantcredit.acquire
```

Raw loan records, notebook outputs, and personal SEC user-agent values are not
committed. This is research and research preparation, not underwriting or
investment advice.

## Reusable download

`fetch` provides the same bounded, atomic download boundary for an ordinary
HTTPS dataset:

```python
from pathlib import Path

from quantcredit.fetch import fetch

receipt = fetch(
  "https://data.example/loans.csv",
  Path("data/loans.csv"),
  timeout_seconds=30,
  max_bytes=100_000_000,
)
```

The receipt supplies the exact byte count and SHA-256 needed to verify later
cache reuse. analysis rules still decide whether personal helpers are allowed.

## Plan

The complete staged contract is in
[`plans/consumer-credit-research.md`](plans/consumer-credit-research.md).
