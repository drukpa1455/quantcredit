"""Canonical executable narrative for the quantcredit research program."""

# %% 00 — Contract and environment
from importlib.metadata import version
from pathlib import Path
from platform import python_version

from quantcredit.source import load_manifest

ROOT = Path.cwd()
if not (ROOT / "pyproject.toml").is_file():
  raise RuntimeError("start the Zed kernel from the quantcredit repository root")

{"python": python_version(), "quantcredit": version("quantcredit")}


# %% 01 — Acquire and verify sources
manifest = load_manifest(ROOT / "sources/ford-credit-auto-owner-trust-2024-a.json")
manifest.summary()


# %% 02 — Schema, identity, and missingness
# Which fields exist, which states are distinct, and is asset identity stable?


# %% 03 — Loan-state transitions
# Which observed transitions are valid, terminal, reversible, or censored?


# %% 04 — Target and censoring decision
# Which outcome can be derived without treating disappearance as an event?


# %% 05 — Chronological split
# What would have been knowable at each prediction cutoff?


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
