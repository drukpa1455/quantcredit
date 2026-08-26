"""Small lazy facade for interactive research."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from pandas import DataFrame

  from quantcredit.audits import Audit
  from quantcredit.baselines import Baseline, Evaluation
  from quantcredit.cashflows import Deal, Tranche
  from quantcredit.challengers import GraphEvaluation, GraphStudy
  from quantcredit.decisions import Decision, Pool
  from quantcredit.source import SourceManifest
  from quantcredit.splits import CausalSplit
  from quantcredit.temporal import Forecast, ForecastEvaluation, History


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


def history(
  manifest: SourceManifest,
  split: CausalSplit,
  cache: Path | None = None,
  *,
  lookback_reports: int = 3,
) -> History:
  """Materialize private causal histories with aggregate inspection."""
  from quantcredit.temporal import materialize_history

  if cache is None:
    return materialize_history(manifest, split, lookback_reports=lookback_reports)
  return materialize_history(
    manifest,
    split,
    cache,
    lookback_reports=lookback_reports,
  )


def forecast(
  history: History,
  *,
  depths: tuple[int, ...] = (1, 2, 3, 4),
  learning_rates: tuple[float, ...] = (0.02, 0.05, 0.10),
  estimators: tuple[int, ...] = (60, 120, 240),
  seed: int = 7,
) -> Forecast:
  """Test whether aligned loan history improves the snapshot GBM."""
  from quantcredit.temporal import forecast as run

  return run(
    history,
    depths=depths,
    learning_rates=learning_rates,
    estimators=estimators,
    seed=seed,
  )


def reveal(
  study: Forecast,
  history: History,
  manifest: SourceManifest,
  split: CausalSplit,
  cache: Path | None = None,
) -> ForecastEvaluation:
  """Open one exact held-out test for a validation-frozen temporal study."""
  from quantcredit.temporal import reveal as run

  if cache is None:
    return run(study, history, manifest, split)
  return run(study, history, manifest, split, cache)


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


def evaluate(
  baseline: Baseline,
  examples: DataFrame,
  manifest: SourceManifest,
  split: CausalSplit,
  cache: Path | None = None,
) -> Evaluation:
  """Evaluate one frozen baseline on the explicitly derived test fold."""
  from quantcredit.baselines import evaluate_baseline

  if cache is None:
    return evaluate_baseline(baseline, examples, manifest, split)
  return evaluate_baseline(baseline, examples, manifest, split, cache)


def challenge(
  baseline: Baseline,
  examples: DataFrame,
  *,
  seeds: tuple[int, ...] = (7, 19, 43),
  hidden: int = 16,
  steps: int = 60,
  learning_rate: float = 0.01,
) -> GraphStudy:
  """Run matched tabular, node-local, and graph validation controls."""
  from quantcredit.challengers import challenge as run

  return run(
    baseline,
    examples,
    seeds=seeds,
    hidden=hidden,
    steps=steps,
    learning_rate=learning_rate,
  )


def confirm(
  study: GraphStudy,
  examples: DataFrame,
  manifest: SourceManifest,
  split: CausalSplit,
  cache: Path | None = None,
) -> GraphEvaluation:
  """Open one exact held-out test for every frozen challenger."""
  from quantcredit.challengers import confirm as run

  if cache is None:
    return run(study, examples, manifest, split)
  return run(study, examples, manifest, split, cache)


def decide(
  baseline: Baseline,
  examples: DataFrame,
  *,
  effects: tuple[str, ...] | None = None,
  cohorts: tuple[str, ...] | None = None,
  bands: int = 10,
  min_cohort: int = 100,
  excluded_shares: tuple[float, ...] | None = None,
) -> Decision:
  """Explain and compare frozen-model validation selection policies."""
  from quantcredit.decisions import (
    DEFAULT_COHORTS,
    DEFAULT_EFFECTS,
    DEFAULT_EXCLUDED_SHARES,
    analyze_decisions,
  )

  return analyze_decisions(
    baseline,
    examples,
    effects=DEFAULT_EFFECTS if effects is None else effects,
    cohorts=DEFAULT_COHORTS if cohorts is None else cohorts,
    bands=bands,
    min_cohort=min_cohort,
    excluded_shares=(
      DEFAULT_EXCLUDED_SHARES if excluded_shares is None else excluded_shares
    ),
  )


def select(
  baseline: Baseline,
  examples: DataFrame,
  budget: float,
  *,
  limits: dict[str, float] | None = None,
) -> Pool:
  """Select a lowest-score validation pool under explicit concentration limits."""
  from quantcredit.decisions import select_pool

  return select_pool(baseline, examples, budget, limits=limits)


def project(
  *,
  balance: float,
  annual_rate: float,
  months: int,
  annual_default_rate: float,
  annual_prepayment_rate: float,
  recovery_rate: float,
  recovery_lag: int = 3,
) -> DataFrame:
  """Project an aggregate collateral scenario from explicit constant rates."""
  from quantcredit.cashflows import project_collateral

  return project_collateral(
    balance=balance,
    annual_rate=annual_rate,
    months=months,
    annual_default_rate=annual_default_rate,
    annual_prepayment_rate=annual_prepayment_rate,
    recovery_rate=recovery_rate,
    recovery_lag=recovery_lag,
  )


def waterfall(
  collateral: DataFrame,
  tranches: tuple[Tranche, ...],
  *,
  annual_fee_rate: float = 0.0,
) -> Deal:
  """Run a simple sequential waterfall over one collateral scenario."""
  from quantcredit.cashflows import run_waterfall

  return run_waterfall(collateral, tranches, annual_fee_rate=annual_fee_rate)
