"""Canonical executable narrative for the quantcredit research program."""

# %% 00 — Contract and environment
from importlib.metadata import version
from pathlib import Path
from platform import python_version

import quantcredit as qc
from quantcredit.populations import FEATURE_LINEAGE, LEAKAGE_FIELDS
from quantcredit.source import load_manifest
from quantcredit.visuals import plot_examples

ROOT = Path.cwd()
if not (ROOT / "pyproject.toml").is_file():
  raise RuntimeError("start the Zed kernel from the quantcredit repository root")

{"python": python_version(), "quantcredit": version("quantcredit")}


# %% 01 — Acquire and verify sources
# Acquisition is the explicit `python -m quantcredit.acquire` shell effect.
# The notebook reads only the resulting tracked declaration and ignored cache.
manifest = load_manifest()
manifest.summary()


# %% 02 — Schema, identity, and missingness
# Which fields exist, which states are distinct, and is asset identity stable?
audit = qc.audit(manifest)
{
  "panel": audit.panel,
  "states": audit.states,
  "most_missing_fields": sorted(
    audit.fields.items(), key=lambda item: item[1]["missing"], reverse=True
  )[:10],
}


# %% 03 — Loan-state transitions
# Which observed transitions are valid, terminal, reversible, or censored?
{
  "continuity": audit.continuity,
  "most_common_transitions": sorted(
    audit.transitions.items(), key=lambda item: item[1], reverse=True
  )[:15],
}


# %% 04 — Target and censoring decision
# Which outcome can be derived without treating disappearance as an event?
audit.targets


# %% 04a — Data audit overview
# How do population, states, transitions, and target disposition fit together?
audit.plot()


# %% 05 — Chronological split
# What would have been knowable at each prediction cutoff?
split = qc.split(manifest.report_periods)
split.summary()


# %% 05a — Causal timeline
# When are features measured, and when is each outcome fully observable?
split.plot()


# %% 05b — Modeling population
# Which eligible positions are visible without realizing held-out test outcomes?
examples = qc.examples(manifest, split)
population = examples.groupby(["fold", "target_status"], observed=True).size().unstack(
  fill_value=0
)
binary = examples.dropna(subset=["target"])
events = binary.groupby("fold", observed=True)["target"].agg(["count", "sum", "mean"])
missing = examples[list(FEATURE_LINEAGE)].isna().sum().sort_values(ascending=False)
{"population": population, "binary_events": events, "feature_missingness": missing}


# %% 05c — Feature boundary
# Which past-only fields enter the baseline, and which outcome fields stay out?
{"features": FEATURE_LINEAGE, "excluded_as_leakage": LEAKAGE_FIELDS}


# %% 05d — Population diagnostics
# What imbalance, missingness, and temporal drift should shape the baseline?
plot_examples(examples)


# %% 06 — Shallow GBM baseline
# How well does the smallest explainable nonlinear model rank future events?
baseline = qc.fit(examples)
baseline.candidates.sort_values("log_loss").head(10)
baseline.surface()


# %% 07 — Calibration and error analysis
# Do predicted probabilities match observed rates across time and cohorts?
{"train_rate_reference": baseline.reference, "selected": baseline.validation}
baseline.calibration
baseline.plot()


# %% 07a — Frozen out-of-time evaluation
# How does the unchanged validation-selected model perform at the later cutoff?
test = qc.evaluate(baseline, examples, manifest, split)
test.summary()
test.calibration
test.plot()


# %% 08 — Expected-loss interpretation
# Do probability, exposure, and realized severity support an economic claim?
exposure = test.exposure
exposure.summary()
exposure.bands
# Illustrative sensitivity only: this bounded panel does not estimate LGD.
exposure.scenario(lgd=0.60)
exposure.plot()


# %% 08a — Validation decision frontier
# Which past-only effects remain, and does the GBM beat simple rules at matched balance?
decision = qc.decide(baseline, examples)
decision.effects
decision.cohorts.head(20)
decision.frontier
decision.plot()


# %% 08b — Constrained validation pool
# What would a deterministic low-score pool look like under an explicit geography cap?
# This is retrospective selection evidence, not investment performance.
pool = qc.select(
  baseline,
  examples,
  budget=200_000_000,
  limits={"geography": 0.10},
)
pool.summary()
pool.allocations


# %% 08c — Illustrative collateral and tranche scenario
# How do declared hazards and recoveries flow through a transparent capital structure?
collateral = qc.project(
  balance=100_000_000,
  annual_rate=0.12,
  months=60,
  annual_default_rate=0.04,
  annual_prepayment_rate=0.15,
  recovery_rate=0.40,
  recovery_lag=3,
)
deal = qc.waterfall(
  collateral,
  (
    qc.Tranche("Senior", 75_000_000, 0.05, price=74_000_000),
    qc.Tranche("Mezzanine", 15_000_000, 0.08, price=13_500_000),
    qc.Tranche("Equity", 10_000_000, 0.00, price=8_000_000),
  ),
  annual_fee_rate=0.01,
)
deal.summary()
deal.plot()


# %% 09 — Matched cohort controls
# Does flattened past-only context improve the same loans and temporal folds?
study = qc.challenge(baseline, examples)
study.summary()
study.topology
study.deltas()
study.comparison


# %% 10 — Graph challenger and falsification
# Does true topology beat enriched tabular, erased, false, and node-local controls?
study.calibration
study.plot()


# %% 10a — One frozen out-of-time reveal
# Do the already-frozen validation arms move in the same direction on test?
graph_test = qc.confirm(study, examples, manifest, split)
graph_test.summary()
graph_test.plot()


# %% 11 — Conclusion and reopening conditions
# What does the evidence prove, reject, and require before another experiment?
{"validation": study.decision, "test": graph_test.decision}
