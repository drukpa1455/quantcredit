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
# Which eligible loan-cutoff positions remain observable in each fold?
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
baseline.candidates


# %% 07 — Calibration and error analysis
# Do predicted probabilities match observed rates across time and cohorts?
{"train_rate_reference": baseline.reference, "selected": baseline.validation}
baseline.calibration
baseline.plot()


# %% 08 — Expected-loss interpretation
# Do probability, exposure, and realized severity support an economic claim?


# %% 09 — Matched cohort controls
# Does flattened past-only context improve the same loans and temporal folds?


# %% 10 — Graph challenger and falsification
# Does true topology beat enriched tabular, erased, false, and node-local controls?


# %% 11 — Conclusion and reopening conditions
# What does the evidence prove, reject, and require before another experiment?
