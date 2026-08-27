# quantcredit

Executable consumer-credit research from public records to classical, temporal,
and graph challengers.

The project starts with SEC Form ABS-EE auto-loan disclosures. It first proves
what the records mean, then builds a shallow gradient-boosted baseline, and only
then asks whether relational structure adds information.

```text
SEC filings -> causal target -> shallow GBM -> decision frontier -> cash waterfall
                                      |                       -> graph controls
                                      `-> loan history ------> next-report warning
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

decision = qc.decide(baseline, examples)
decision.frontier
decision.plot()

pool = qc.select(baseline, examples, 200_000_000, limits={"geography": 0.10})
pool.summary()

study = qc.challenge(baseline, examples)
study.summary()
study.deltas()
study.comparison
study.plot()

graph_test = qc.confirm(study, examples, manifest, split)
graph_test.summary()

early_split = qc.split(manifest.report_periods, horizon_reports=1)
history = qc.history(manifest, early_split)
history.summary()
history.plot()

forecast = qc.forecast(history)
forecast.summary()
forecast.comparison
forecast.drivers
forecast.plot()

forecast_test = qc.reveal(forecast, history, manifest, early_split)
forecast_test.summary()
```

Ordinary materialization exposes train and validation outcomes but marks every
eligible test row as `held_out` with a missing target. The later explicit test
operation owns derivation and evaluation of those outcomes, and returns only
aggregate metrics and calibration. September is out of time but not described
as blind because its marginal event rate was historically observed.

The decision frontier is retrospective validation evidence: it compares the
frozen GBM with simple rules at matched excluded balance. It is not a return,
price, or underwriting claim. The notebook then keeps a separate, fully
declared collateral and tranche scenario to demonstrate cash-flow mechanics.

The graph study tests one explicit cohort-incidence ontology:

```text
geography context --geography--> loan at cutoff
vintage context   --vintage----> loan at cutoff
vehicle context   --vehicle----> loan at cutoff
global context    --trust------> loan at cutoff
```

Loan nodes carry the declared causal loan features. Each context node carries a
past-only smoothed event rate and log sample count. Each directed edge carries
only its one-hot relation type, with exactly one edge of each type per loan.
`study.ontology.nodes` and `study.ontology.edges` expose this schema directly.

The enriched GBM, node-local, true GINE, erased-relation, and false-relation
arms receive the same train-owned cohort facts, while raw GBM remains the
incumbent. On the pinned panel, true topology did not beat the GBMs or its
topology controls, so the retained decision is
`no_value_from_current_ontology`. This is evidence about the tested ontology,
not a rejection of graph methods or richer borrower, account, payment, asset,
identity, or temporal relations. No TinyMesh primitive was added.

The temporal study asks a narrower early-warning question among loans observed
for three consecutive reports and fewer than 30 days delinquent at cutoff: will
the loan first reach 60+ delinquency or charge-off in the next report? The
snapshot and history arms use the same selected shallow-GBM capacity. A
shuffled-history arm preserves every cutoff's history distribution while
breaking loan alignment.

`forecast.drivers` keeps the selected history model fixed and jointly permutes
each feature's lag/change family on validation. It measures model reliance, not
causal importance or standalone feature value; correlated trajectories may
substitute for one another.

On validation, aligned history improved log loss from `0.014829` to `0.013788`
and average precision from `0.260214` to `0.323902`; shuffled history did not
improve the snapshot. The frozen test retained the direction: log loss
`0.012302` to `0.012083` and average precision `0.317180` to `0.335136`, while
Brier score moved adversely from `0.002259` to `0.002328`. The result supports
loan-history information for next-report warning, with mixed probability-error
evidence. It does not estimate legal default, an event exactly three months
ahead, or graph-recurrent value. The changing loan universe, short panel, and
lack of borrower-to-borrower relations leave a temporal graph arm unidentified.

The default baseline maps a declared 36-candidate validation surface. For a
quick exploratory run, narrow the same operation explicitly—for example,
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
committed. This is public research, not underwriting or investment advice.

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
verify later cache reuse.

## Plan

The complete staged contract is in
[`plans/consumer-credit-research.md`](plans/consumer-credit-research.md).
