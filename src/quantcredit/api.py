"""Small lazy facade for interactive research."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from pandas import DataFrame

  from quantcredit.audits import Audit
  from quantcredit.baselines import Baseline
  from quantcredit.source import SourceManifest
  from quantcredit.splits import CausalSplit


def audit(
  manifest: SourceManifest,
  cache: Path | None = None,
  *,
  horizon_reports: int = 3,
) -> Audit:
  """Build aggregate evidence without eagerly loading the audit CLI module."""
  from quantcredit.audits import audit_sources

  if cache is None:
    return audit_sources(manifest, horizon_reports=horizon_reports)
  return audit_sources(manifest, cache, horizon_reports=horizon_reports)


def split(report_periods: tuple[date, ...], *, horizon_reports: int = 3) -> CausalSplit:
  """Build causal cutoffs without eagerly loading the split implementation."""
  from quantcredit.splits import causal_split

  return causal_split(report_periods, horizon_reports=horizon_reports)


def examples(
  manifest: SourceManifest,
  split: CausalSplit,
  cache: Path | None = None,
) -> DataFrame:
  """Materialize the causal modeling population from verified source bytes."""
  from quantcredit.populations import materialize_examples

  if cache is None:
    return materialize_examples(manifest, split)
  return materialize_examples(manifest, split, cache)


def fit(
  examples: DataFrame,
  *,
  depths: tuple[int, ...] = (1, 2, 3, 4),
  learning_rates: tuple[float, ...] = (0.02, 0.05, 0.10),
  estimators: tuple[int, ...] = (60, 120, 240),
  seed: int = 7,
) -> Baseline:
  """Fit the train-only shallow GBM sensitivity surface on validation."""
  from quantcredit.baselines import fit_baseline

  return fit_baseline(
    examples,
    depths=depths,
    learning_rates=learning_rates,
    estimators=estimators,
    seed=seed,
  )
