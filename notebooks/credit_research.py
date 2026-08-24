"""Canonical executable narrative for the quantcredit research program."""

# %% 00 — Contract and environment
from importlib.metadata import version
from pathlib import Path
from platform import python_version

from quantcredit.audit import audit_sources
from quantcredit.source import load_manifest
from quantcredit.splits import causal_split
from quantcredit.visuals import plot_audit, plot_split

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
audit = audit_sources(manifest)
{
  "panel": audit["panel"],
  "states": audit["states"],
  "most_missing_fields": sorted(
    audit["fields"].items(), key=lambda item: item[1]["missing"], reverse=True
  )[:10],
}


# %% 03 — Loan-state transitions
# Which observed transitions are valid, terminal, reversible, or censored?
{
  "continuity": audit["continuity"],
  "most_common_transitions": sorted(
    audit["transitions"].items(), key=lambda item: item[1], reverse=True
  )[:15],
}


# %% 04 — Target and censoring decision
# Which outcome can be derived without treating disappearance as an event?
audit["targets"]


# %% 04a — Data audit overview
# How do population, states, transitions, and target disposition fit together?
plot_audit(audit)


# %% 05 — Chronological split
# What would have been knowable at each prediction cutoff?
split = causal_split(manifest.report_periods)
split.summary()


# %% 05a — Causal timeline
# When are features measured, and when is each outcome fully observable?
plot_split(split)


# %% 06 — Shallow GBM baseline
# How well does the smallest explainable nonlinear model rank future events?


# %% 07 — Calibration and error analysis
# Do predicted probabilities match observed rates across time and cohorts?


# %% 08 — Expected-loss interpretation
# Do probability, exposure, and realized severity support an economic claim?


# %% 09 — Matched cohort controls
# Does flattened past-only context improve the same loans and temporal folds?


# %% 10 — Graph challenger and falsification
# Does true topology beat enriched tabular, erased, false, and node-local controls?


# %% 11 — Conclusion and reopening conditions
# What does the evidence prove, reject, and require before another experiment?
