"""Aggregate validation interpretation and deterministic loan-pool selection."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from pandas import DataFrame

from quantcredit.baselines import Baseline
from quantcredit.populations import FEATURE_COLUMNS, NUMERIC_FEATURES

if TYPE_CHECKING:
  from matplotlib.figure import Figure

DEFAULT_EFFECTS = (
  "next_payment_due",
  "remaining_term",
  "delinquency_days",
  "current_ltv",
  "original_interest_rate",
)
DEFAULT_COHORTS = (
  "geography",
  "vehicle_type",
  "credit_score_status",
  "income_verification",
)
DEFAULT_EXCLUDED_SHARES = (0.0, 0.05, 0.10, 0.20, 0.30)


@dataclass(frozen=True)
class Decision:
  """Aggregate validation evidence for interpretation and selection policies."""

  effects: DataFrame
  cohorts: DataFrame
  frontier: DataFrame

  def plot(self) -> Figure:
    """Render feature shape, cohort residual, and matched-balance policy evidence."""
    from quantcredit.visuals import plot_decision

    return plot_decision(self)


@dataclass(frozen=True)
class Pool:
  """Aggregate evidence for one deterministic validation-pool selection."""

  budget: float
  selected_exposure: float
  samples: int
  weighted_pd: float
  weighted_coupon: float
  events: int
  observed_event_exposure: float
  allocations: DataFrame

  @property
  def utilization(self) -> float:
    return self.selected_exposure / self.budget

  def summary(self) -> dict[str, int | float | str]:
    return {
      "budget": self.budget,
      "selected_exposure": self.selected_exposure,
      "budget_utilization": self.utilization,
      "samples": self.samples,
      "exposure_weighted_pd": self.weighted_pd,
      "weighted_coupon": self.weighted_coupon,
      "events": self.events,
      "observed_event_exposure": self.observed_event_exposure,
      "status": "retrospective validation selection, not investment performance",
    }


def analyze_decisions(
  baseline: Baseline,
  examples: DataFrame,
  *,
  effects: tuple[str, ...] = DEFAULT_EFFECTS,
  cohorts: tuple[str, ...] = DEFAULT_COHORTS,
  bands: int = 10,
  min_cohort: int = 100,
  excluded_shares: tuple[float, ...] = DEFAULT_EXCLUDED_SHARES,
) -> Decision:
  """Explain the frozen model and compare validation selection policies."""
  frame = _scored_validation(baseline, examples)
  return Decision(
    effects=_effects(frame, effects, bands),
    cohorts=_cohorts(frame, cohorts, min_cohort),
    frontier=_frontier(frame, excluded_shares),
  )


def select_pool(
  baseline: Baseline,
  examples: DataFrame,
  budget: float,
  *,
  limits: dict[str, float] | None = None,
) -> Pool:
  """Select lowest-score validation loans under whole-loan concentration limits."""
  if not isfinite(budget) or budget <= 0:
    raise ValueError("budget must be a finite positive value")
  limits = {} if limits is None else limits
  _validate_limits(limits)
  frame = _scored_validation(baseline, examples)
  frame = _observed_exposure(frame)
  if budget > float(frame["ending_balance"].sum()):
    raise ValueError("budget cannot exceed observed validation exposure")

  selected: list[Any] = []
  used = 0.0
  group_used: dict[tuple[str, str], float] = {}
  ordered = frame.sort_values(["score", "loan_id"], kind="stable")
  for index, row in ordered.iterrows():
    amount = float(row["ending_balance"])
    if used + amount > budget:
      continue
    groups = [(field, _group_value(row[field])) for field in limits]
    if any(group_used.get(group, 0.0) + amount > budget * limits[group[0]] for group in groups):
      continue
    selected.append(index)
    used += amount
    for group in groups:
      group_used[group] = group_used.get(group, 0.0) + amount

  if not selected:
    raise ValueError("constraints selected no loans")
  pool = frame.loc[selected]
  allocations = _allocations(pool, limits)
  if not allocations.empty and (allocations["share"] > allocations["limit"] + 1e-12).any():
    raise ValueError("constraints could not construct a compliant whole-loan pool")
  return Pool(
    budget=budget,
    selected_exposure=used,
    samples=len(pool),
    weighted_pd=_weighted(pool, "score"),
    weighted_coupon=_weighted(pool, "original_interest_rate"),
    events=int(pool["target"].sum()),
    observed_event_exposure=float((pool["target"] * pool["ending_balance"]).sum()),
    allocations=allocations,
  )


def _scored_validation(baseline: Baseline, examples: DataFrame) -> DataFrame:
  required = {"loan_id", "fold", "target", *FEATURE_COLUMNS}
  missing = sorted(required - set(examples.columns))
  if missing:
    raise ValueError(f"examples are missing required columns: {', '.join(missing)}")
  frame = examples.loc[
    (examples["fold"] == "validation") & examples["target"].notna()
  ].reset_index(drop=True)
  if frame.empty or frame["target"].nunique() != 2:
    raise ValueError("validation requires both target classes")
  features = frame[list(FEATURE_COLUMNS)]
  transformed = baseline.preprocessor.transform(features)
  frame["score"] = baseline.classifier.predict_proba(transformed)[:, 1]
  return frame


def _effects(frame: DataFrame, features: tuple[str, ...], bands: int) -> DataFrame:
  if not features:
    raise ValueError("effects must name at least one feature")
  if bands < 2:
    raise ValueError("bands must be at least 2")
  missing = sorted(set(features) - set(NUMERIC_FEATURES))
  if missing:
    raise ValueError(f"unknown effect features: {', '.join(missing)}")

  records: list[dict[str, Any]] = []
  for feature in features:
    values = pd.to_numeric(frame[feature], errors="coerce")
    observed = frame.loc[values.notna(), ["target", "score"]].copy()
    observed["value"] = values.loc[values.notna()]
    if not observed.empty:
      quantiles = pd.qcut(
        observed["value"],
        q=min(bands, int(observed["value"].nunique())),
        labels=False,
        duplicates="drop",
      )
      observed["band"] = 1 if quantiles.isna().all() else quantiles.astype(int) + 1
      for band, group in observed.groupby("band", observed=True):
        records.append(_effect_record(feature, str(int(cast(Any, band))), group))
    absent = frame.loc[values.isna(), ["target", "score"]].copy()
    if not absent.empty:
      absent["value"] = np.nan
      records.append(_effect_record(feature, "missing", absent))
  return DataFrame(records)


def _effect_record(feature: str, band: str, group: DataFrame) -> dict[str, Any]:
  event_rate = float(group["target"].mean())
  mean_score = float(group["score"].mean())
  return {
    "feature": feature,
    "band": band,
    "samples": len(group),
    "events": int(group["target"].sum()),
    "minimum": float(group["value"].min()),
    "maximum": float(group["value"].max()),
    "mean_value": float(group["value"].mean()),
    "mean_score": mean_score,
    "event_rate": event_rate,
    "residual": event_rate - mean_score,
  }


def _cohorts(frame: DataFrame, features: tuple[str, ...], min_cohort: int) -> DataFrame:
  if min_cohort <= 0:
    raise ValueError("min_cohort must be positive")
  missing = sorted(set(features) - set(FEATURE_COLUMNS))
  if missing:
    raise ValueError(f"unknown cohort features: {', '.join(missing)}")
  records: list[dict[str, Any]] = []
  for feature in features:
    values = frame[feature].map(_group_value)
    for value, group in frame.assign(_cohort=values).groupby("_cohort", observed=True):
      if len(group) < min_cohort:
        continue
      event_rate = float(group["target"].mean())
      mean_score = float(group["score"].mean())
      records.append(
        {
          "feature": feature,
          "value": str(value),
          "samples": len(group),
          "events": int(group["target"].sum()),
          "mean_score": mean_score,
          "event_rate": event_rate,
          "residual": event_rate - mean_score,
        }
      )
  result = DataFrame(records)
  if result.empty:
    return result
  return result.sort_values("residual", key=abs, ascending=False, ignore_index=True)


def _frontier(frame: DataFrame, shares: tuple[float, ...]) -> DataFrame:
  if not shares or any(not isfinite(share) or not 0 <= share < 1 for share in shares):
    raise ValueError("excluded_shares must contain finite values in [0, 1)")
  if len(set(shares)) != len(shares):
    raise ValueError("excluded_shares must be distinct")
  observed = _observed_exposure(frame)
  total_exposure = float(observed["ending_balance"].sum())
  total_event_exposure = float((observed["target"] * observed["ending_balance"]).sum())
  policies = {
    "GBM score": observed["score"],
    "Lowest credit score": -pd.to_numeric(
      observed["credit_score"], errors="coerce"
    ).fillna(np.inf),
    "Highest current LTV": pd.to_numeric(observed["current_ltv"], errors="coerce").fillna(
      -np.inf
    ),
    "Most delinquent": pd.to_numeric(
      observed["delinquency_days"], errors="coerce"
    ).fillna(-np.inf),
  }
  records = []
  for policy, risk in policies.items():
    ranked = observed.assign(_risk=risk).sort_values(
      ["_risk", "loan_id"], ascending=(False, True), kind="stable"
    )
    cumulative = ranked["ending_balance"].cumsum()
    for target_share in sorted(shares):
      target_exposure = total_exposure * target_share
      excluded = cumulative <= target_exposure
      if target_share > 0 and not excluded.any():
        excluded.iloc[0] = True
      retained = ranked.loc[~excluded]
      excluded_exposure = total_exposure - float(retained["ending_balance"].sum())
      retained_event_exposure = float(
        (retained["target"] * retained["ending_balance"]).sum()
      )
      records.append(
        {
          "policy": policy,
          "target_excluded_share": target_share,
          "excluded_balance_share": excluded_exposure / total_exposure,
          "retained_samples": len(retained),
          "retained_balance": total_exposure - excluded_exposure,
          "weighted_coupon": _weighted(retained, "original_interest_rate"),
          "events": int(retained["target"].sum()),
          "event_rate": float(retained["target"].mean()),
          "observed_event_exposure": retained_event_exposure,
          "event_exposure_avoided": (
            1 - retained_event_exposure / total_event_exposure
            if total_event_exposure
            else float("nan")
          ),
          "expected_event_exposure": float(
            (retained["score"] * retained["ending_balance"]).sum()
          ),
        }
      )
  return DataFrame(records)


def _observed_exposure(frame: DataFrame) -> DataFrame:
  exposure = pd.to_numeric(frame["ending_balance"], errors="coerce")
  if (exposure.dropna() < 0).any():
    raise ValueError("ending_balance cannot be negative")
  observed = frame.loc[exposure.notna() & exposure.gt(0)].copy()
  observed["ending_balance"] = exposure.loc[observed.index]
  if observed.empty:
    raise ValueError("selection requires observed positive ending_balance")
  return observed


def _weighted(frame: DataFrame, column: str) -> float:
  values = pd.to_numeric(frame[column], errors="coerce")
  observed = values.notna()
  exposure = frame.loc[observed, "ending_balance"]
  if exposure.empty or float(exposure.sum()) == 0:
    return float("nan")
  return float((values.loc[observed] * exposure).sum() / exposure.sum())


def _validate_limits(limits: dict[str, float]) -> None:
  missing = sorted(set(limits) - set(FEATURE_COLUMNS))
  if missing:
    raise ValueError(f"unknown concentration fields: {', '.join(missing)}")
  if any(not isfinite(limit) or not 0 < limit <= 1 for limit in limits.values()):
    raise ValueError("concentration limits must be finite values in (0, 1]")


def _allocations(frame: DataFrame, limits: dict[str, float]) -> DataFrame:
  total = float(frame["ending_balance"].sum())
  records = []
  for feature in limits:
    values = frame[feature].map(_group_value)
    grouped = frame.assign(_value=values).groupby("_value", observed=True)
    for value, group in grouped:
      exposure = float(group["ending_balance"].sum())
      records.append(
        {
          "feature": feature,
          "value": str(value),
          "samples": len(group),
          "exposure": exposure,
          "share": exposure / total,
          "limit": limits[feature],
        }
      )
  return DataFrame(records)


def _group_value(value: Any) -> str:
  return "missing" if pd.isna(value) else str(value)
