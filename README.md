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

The notebook API keeps the common path on the research values themselves:

```python
import quantcredit as qc
from quantcredit.source import load_manifest
from quantcredit.visuals import plot_examples

manifest = load_manifest()
audit = qc.audit(manifest)
audit.plot()

split = qc.split(manifest.report_periods)
split.plot()

examples = qc.examples(manifest, split)
examples.groupby(["fold", "target_status"]).size()
plot_examples(examples)

baseline = qc.fit(examples)
baseline.candidates.sort_values("log_loss").head(10)
baseline.surface()
baseline.plot()

test = qc.evaluate(baseline, examples, manifest, split)
test.summary()
test.plot()
```

Ordinary materialization exposes train and validation outcomes but marks every
eligible test row as `held_out` with a missing target. The later explicit test
operation owns derivation and evaluation of those outcomes, and returns only
aggregate metrics and calibration. September is out of time but not described
as blind because its marginal event rate was historically observed.

The default baseline maps a declared 36-candidate validation surface. During a
short assessment, narrow the same operation explicitly—for example,
`qc.fit(examples, depths=(2, 3), learning_rates=(0.05,), estimators=(120,))`.

`quantcredit.visuals.plot_audit`, `plot_split`, and `plot_examples` remain
available when explicit functional composition is preferable.

To acquire and checksum-pin the declared EX-102 files, provide the identifying
user agent required by SEC access policy without storing it in the repository:

```console
export QUANTCREDIT_SEC_USER_AGENT="your-name your-email"
uv run --locked python -m quantcredit.acquire
```

Raw loan records, notebook outputs, and personal SEC user-agent values are not
committed. This is research and interview preparation, not underwriting or
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
)
```

The defaults allow 30 seconds and 500 MB. Override either bound when the source
requires it. The receipt supplies the exact byte count and SHA-256 needed to
verify later cache reuse. Assessment rules still decide whether personal helpers
are allowed.

## Plan

The complete staged contract is in
[`plans/consumer-credit-research.md`](plans/consumer-credit-research.md).
