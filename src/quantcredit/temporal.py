"""Causal loan histories and matched next-report forecasting controls."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date
from math import sqrt
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer

from quantcredit.acquire import DEFAULT_CACHE, verify_asset
from quantcredit.baselines import Baseline, calibrate, fit_baseline, measure
from quantcredit.panel import AssetKey, SnapshotValidator, read_snapshots
from quantcredit.populations import FEATURE_COLUMNS, loan_features, loan_id
from quantcredit.source import SourceManifest
from quantcredit.splits import CausalSplit, causal_split
from quantcredit.targets import LoanState, TargetResult, loan_state, serious_delinquency_target

if TYPE_CHECKING:
  from matplotlib.figure import Figure

TEMPORAL_FEATURES = (
  "ending_balance",
  "current_ltv",
  "remaining_term",
  "scheduled_payment",
  "next_payment_due",
  "delinquency_days",
)


def _lag(feature: str, reports: int) -> str:
  return f"{feature}_lag_{reports}"


def _change(feature: str, reports: int) -> str:
  return f"{feature}_change_{reports}"


HISTORY_COLUMNS = tuple(
  [
    *(_lag(feature, lag) for feature in TEMPORAL_FEATURES for lag in (1, 2)),
    *(_change(feature, lag) for feature in TEMPORAL_FEATURES for lag in (1, 2)),
  ]
)


@dataclass(frozen=True)
class History:
  """Private loan-cutoff histories with aggregate inspection only."""

  audit: DataFrame
  lookback_reports: int
  horizon_reports: int
  _rows: DataFrame = field(repr=False)

  def summary(self) -> DataFrame:
    return self.audit.copy()

  @property
  def features(self) -> DataFrame:
    return DataFrame(
      {
        "feature": TEMPORAL_FEATURES,
        "observations": "t, t-1, t-2",
        "changes": "t-(t-1), (t-1)-(t-2)",
      }
    )

  def plot(self) -> Figure:
    from quantcredit.visuals import plot_history

    return plot_history(self)


@dataclass(frozen=True)
class Forecast:
  """Validation evidence for matched snapshot and history GBMs."""

  results: DataFrame
  calibration: DataFrame
  comparison: DataFrame
  decision: Literal["retain_snapshot", "retain_history"]
  _artifacts: ForecastArtifacts = field(repr=False)

  def summary(self) -> DataFrame:
    return self.results.copy()

  def plot(self) -> Figure:
    from quantcredit.visuals import plot_forecast

    return plot_forecast(self)


@dataclass(frozen=True)
class ForecastEvaluation:
  """One frozen out-of-time reveal of the temporal-information controls."""

  results: DataFrame
  calibration: DataFrame
  validation_decision: str
  decision: str

  def summary(self) -> DataFrame:
    return self.results.copy()

  def plot(self) -> Figure:
    from quantcredit.visuals import plot_forecast_evaluation

    return plot_forecast_evaluation(self)


@dataclass(frozen=True)
class ForecastArtifacts:
  snapshot: Baseline = field(repr=False)
  imputer: SimpleImputer = field(repr=False)
  history: HistGradientBoostingClassifier = field(repr=False)
  shuffled: HistGradientBoostingClassifier = field(repr=False)
  shuffle_seed: int


def materialize_history(
  manifest: SourceManifest,
  split: CausalSplit,
  cache: Path = DEFAULT_CACHE,
  *,
  lookback_reports: int = 3,
) -> History:
  """Build three-report histories while keeping the test outcome held out."""
  folds = {
    **{cutoff: "train" for cutoff in split.train_cutoffs},
    split.validation_cutoff: "validation",
    split.test_cutoff: "test",
  }
  return _materialize_history(
    manifest,
    split,
    cache,
    lookback_reports=lookback_reports,
    folds=folds,
    held_out=frozenset({split.test_cutoff}),
  )


def materialize_test_history(
  manifest: SourceManifest,
  split: CausalSplit,
  cache: Path = DEFAULT_CACHE,
  *,
  lookback_reports: int = 3,
) -> History:
  """Derive the next-report outcome only at the explicit test boundary."""
  return _materialize_history(
    manifest,
    split,
    cache,
    lookback_reports=lookback_reports,
    folds={split.test_cutoff: "test"},
    held_out=frozenset(),
  )


def forecast(
  history: History,
  *,
  depths: tuple[int, ...] = (1, 2, 3, 4),
  learning_rates: tuple[float, ...] = (0.02, 0.05, 0.10),
  estimators: tuple[int, ...] = (60, 120, 240),
  seed: int = 7,
  shuffle_seed: int = 97,
) -> Forecast:
  """Compare one selected snapshot GBM with aligned and shuffled histories."""
  rows = history._rows
  snapshot = fit_baseline(
    rows,
    depths=depths,
    learning_rates=learning_rates,
    estimators=estimators,
    seed=seed,
  )
  train = _fold(rows, "train")
  validation = _fold(rows, "validation")
  train_y = train["target"].to_numpy(dtype=np.int64)
  validation_y = validation["target"].to_numpy(dtype=np.int64)
  train_current = np.asarray(
    snapshot.preprocessor.transform(train[list(FEATURE_COLUMNS)]), dtype=np.float64
  )
  validation_current = np.asarray(
    snapshot.preprocessor.transform(validation[list(FEATURE_COLUMNS)]), dtype=np.float64
  )

  imputer = SimpleImputer(strategy="median", add_indicator=True)
  train_history = np.asarray(imputer.fit_transform(train[list(HISTORY_COLUMNS)]))
  validation_history = np.asarray(imputer.transform(validation[list(HISTORY_COLUMNS)]))
  shuffled_train = _shuffle(train_history, train["cutoff"], shuffle_seed)
  shuffled_validation = _shuffle(
    validation_history, validation["cutoff"], shuffle_seed + 1
  )

  aligned_model = _classifier(snapshot, seed)
  aligned_model.fit(np.concatenate((train_current, train_history), axis=1), train_y)
  shuffled_model = _classifier(snapshot, seed)
  shuffled_model.fit(np.concatenate((train_current, shuffled_train), axis=1), train_y)

  scores = {
    "snapshot_gbm": snapshot.classifier.predict_proba(validation_current)[:, 1],
    "history_gbm": aligned_model.predict_proba(
      np.concatenate((validation_current, validation_history), axis=1)
    )[:, 1],
    "shuffled_history_gbm": shuffled_model.predict_proba(
      np.concatenate((validation_current, shuffled_validation), axis=1)
    )[:, 1],
  }
  fit_scores = {
    "snapshot_gbm": snapshot.classifier.predict_proba(train_current)[:, 1],
    "history_gbm": aligned_model.predict_proba(
      np.concatenate((train_current, train_history), axis=1)
    )[:, 1],
    "shuffled_history_gbm": shuffled_model.predict_proba(
      np.concatenate((train_current, shuffled_train), axis=1)
    )[:, 1],
  }
  results = DataFrame(
    [
      _record(arm, validation_y, values, train_y, fit_scores[arm], snapshot)
      for arm, values in scores.items()
    ]
  )
  calibration = pd.concat(
    [calibrate(validation_y, values).assign(arm=arm) for arm, values in scores.items()],
    ignore_index=True,
  )
  decision: Literal["retain_snapshot", "retain_history"] = (
    "retain_history" if _history_wins(results) else "retain_snapshot"
  )
  return Forecast(
    results,
    calibration,
    _comparisons(validation_y, scores),
    decision,
    ForecastArtifacts(snapshot, imputer, aligned_model, shuffled_model, shuffle_seed),
  )


def reveal(
  study: Forecast,
  history: History,
  manifest: SourceManifest,
  split: CausalSplit,
  cache: Path = DEFAULT_CACHE,
) -> ForecastEvaluation:
  """Reveal one exact held-out test for the validation-frozen history study."""
  observed = materialize_test_history(
    manifest,
    split,
    cache,
    lookback_reports=history.lookback_reports,
  )
  test = _fold(_match_test(history._rows, observed._rows), "test")
  target = test["target"].to_numpy(dtype=np.int64)
  artifacts = study._artifacts
  current = np.asarray(
    artifacts.snapshot.preprocessor.transform(test[list(FEATURE_COLUMNS)]),
    dtype=np.float64,
  )
  values = np.asarray(artifacts.imputer.transform(test[list(HISTORY_COLUMNS)]))
  shuffled = _shuffle(values, test["cutoff"], artifacts.shuffle_seed + 2)
  scores = {
    "snapshot_gbm": artifacts.snapshot.classifier.predict_proba(current)[:, 1],
    "history_gbm": artifacts.history.predict_proba(
      np.concatenate((current, values), axis=1)
    )[:, 1],
    "shuffled_history_gbm": artifacts.shuffled.predict_proba(
      np.concatenate((current, shuffled), axis=1)
    )[:, 1],
  }
  results = DataFrame(
    [_record(arm, target, score, None, None, artifacts.snapshot) for arm, score in scores.items()]
  )
  calibration = pd.concat(
    [calibrate(target, score).assign(arm=arm) for arm, score in scores.items()],
    ignore_index=True,
  )
  decision = (
    "retain_history"
    if study.decision == "retain_history" and _history_wins(results)
    else "retain_snapshot"
  )
  return ForecastEvaluation(results, calibration, study.decision, decision)


def _materialize_history(
  manifest: SourceManifest,
  split: CausalSplit,
  cache: Path,
  *,
  lookback_reports: int,
  folds: dict[date, str],
  held_out: frozenset[date],
) -> History:
  if lookback_reports != 3:
    raise ValueError("the frozen history protocol requires exactly 3 reports")
  expected = causal_split(manifest.report_periods, horizon_reports=split.horizon_reports)
  if split != expected:
    raise ValueError("causal split does not match the source manifest")
  if split.horizon_reports != 1:
    raise ValueError("the frozen history protocol predicts the next report")

  period_index = {period: index for index, period in enumerate(manifest.report_periods)}
  usable_folds = {
    cutoff: fold
    for cutoff, fold in folds.items()
    if period_index[cutoff] >= lookback_reports - 1
  }
  cutoff_indices = {period_index[cutoff] for cutoff in usable_folds}
  history_indices = {
    index - lag
    for index in cutoff_indices
    for lag in range(lookback_reports)
  }
  states: dict[AssetKey, list[LoanState | None]] = {}
  current: dict[tuple[AssetKey, int], dict[str, object]] = {}
  temporal: dict[tuple[AssetKey, int], tuple[object, ...]] = {}
  validator = SnapshotValidator()

  for filing in manifest.filings:
    receipt = verify_asset(filing, cache)
    index = period_index[filing.report_period]
    for snapshot in read_snapshots(receipt.path, manifest, filing):
      validator.observe(snapshot)
      state_history = states.setdefault(
        snapshot.key.asset, [None] * len(manifest.report_periods)
      )
      state_history[index] = loan_state(snapshot)
      if index not in history_indices:
        continue
      features = loan_features(snapshot)
      temporal[snapshot.key.asset, index] = tuple(
        features[feature] for feature in TEMPORAL_FEATURES
      )
      if index in cutoff_indices:
        current[snapshot.key.asset, index] = features
  validator.summary()
  return _assemble(
    manifest,
    split,
    lookback_reports,
    usable_folds,
    held_out,
    states,
    current,
    temporal,
  )


def _assemble(
  manifest: SourceManifest,
  split: CausalSplit,
  lookback_reports: int,
  folds: dict[date, str],
  held_out: frozenset[date],
  states: dict[AssetKey, list[LoanState | None]],
  current: dict[tuple[AssetKey, int], dict[str, object]],
  temporal: dict[tuple[AssetKey, int], tuple[object, ...]],
) -> History:
  audits = {cutoff: Counter[str]() for cutoff in folds}
  records: list[dict[str, object]] = []
  for (asset, index), features in current.items():
    cutoff = manifest.report_periods[index]
    if cutoff not in folds:
      continue
    audit = audits[cutoff]
    audit["reported"] += 1
    sequence = [temporal.get((asset, index - lag)) for lag in range(lookback_reports)]
    if any(values is None for values in sequence):
      audit["incomplete_history"] += 1
      continue
    audit["complete_history"] += 1
    state = states[asset][index]
    if not _performing(state):
      audit["ineligible"] += 1
      continue

    if cutoff in held_out:
      status, target = "held_out", None
    else:
      result = serious_delinquency_target(
        states[asset], index, horizon_reports=split.horizon_reports
      )
      status, target = result.value, _target(result)
    audit[status] += 1
    records.append(
      {
        "loan_id": loan_id(asset),
        "cutoff": cutoff,
        "fold": folds[cutoff],
        "target_status": status,
        "target": target,
        **features,
        **_history_fields(sequence),
      }
    )

  rows = DataFrame(records)
  rows["cutoff"] = pd.to_datetime(rows["cutoff"])
  rows["target"] = rows["target"].astype(pd.Int8Dtype())
  rows = rows.sort_values(["cutoff", "loan_id"], ignore_index=True)
  audit_rows = []
  for cutoff, fold in folds.items():
    counts = audits[cutoff]
    modeled = counts["positive"] + counts["negative"]
    audit_rows.append(
      {
        "cutoff": cutoff,
        "fold": fold,
        "reported": counts["reported"],
        "complete_history": counts["complete_history"],
        "incomplete_history": counts["incomplete_history"],
        "ineligible": counts["ineligible"],
        "modeled": modeled,
        "events": counts["positive"],
        "event_rate": counts["positive"] / modeled if modeled else float("nan"),
        "competing": counts["competing_event"],
        "censored": counts["missing_followup"] + counts["right_censored"],
        "held_out": counts["held_out"],
      }
    )
  return History(DataFrame(audit_rows), lookback_reports, split.horizon_reports, rows)


def _history_fields(sequence: list[tuple[object, ...] | None]) -> dict[str, object]:
  current, lag_1, lag_2 = sequence
  if current is None or lag_1 is None or lag_2 is None:
    raise ValueError("history requires three observed reports")
  fields: dict[str, object] = {}
  for index, feature in enumerate(TEMPORAL_FEATURES):
    fields[_lag(feature, 1)] = lag_1[index]
    fields[_lag(feature, 2)] = lag_2[index]
    fields[_change(feature, 1)] = _difference(current[index], lag_1[index])
    fields[_change(feature, 2)] = _difference(lag_1[index], lag_2[index])
  return fields


def _difference(left: object, right: object) -> float | None:
  if left is None or right is None:
    return None
  return float(cast(Any, left)) - float(cast(Any, right))


def _performing(state: LoanState | None) -> bool:
  return bool(
    state is not None
    and not state.terminal
    and not state.charged_off
    and state.delinquency_days is not None
    and state.delinquency_days < 30
  )


def _target(result: TargetResult) -> int | None:
  if result is TargetResult.POSITIVE:
    return 1
  if result is TargetResult.NEGATIVE:
    return 0
  return None


def _classifier(baseline: Baseline, seed: int) -> HistGradientBoostingClassifier:
  return HistGradientBoostingClassifier(
    max_depth=baseline.selected_depth,
    learning_rate=baseline.selected_learning_rate,
    max_iter=baseline.selected_estimators,
    random_state=seed,
    early_stopping=False,
  )


def _fold(rows: DataFrame, name: str) -> DataFrame:
  frame = rows.loc[(rows["fold"] == name) & rows["target"].notna()].reset_index(drop=True)
  if frame.empty or frame["target"].nunique() != 2:
    raise ValueError(f"{name} requires both target classes")
  return frame


def _shuffle(
  values: NDArray[np.float64], groups: pd.Series[Any], seed: int
) -> NDArray[np.float64]:
  result = values.copy()
  rng = np.random.default_rng(seed)
  for positions in groups.groupby(groups, sort=False).groups.values():
    index = np.asarray(list(positions), dtype=np.int64)
    result[index] = values[rng.permutation(index)]
  return result


def _record(
  arm: str,
  target: NDArray[np.int64],
  scores: NDArray[np.float64],
  fit_target: NDArray[np.int64] | None,
  fit_scores: NDArray[np.float64] | None,
  baseline: Baseline,
) -> dict[str, Any]:
  fit_loss = (
    float("nan")
    if fit_target is None or fit_scores is None
    else measure(fit_target, fit_scores).log_loss
  )
  return {
    "arm": arm,
    "parameters": int(baseline.selected["leaf_budget"]),
    "max_depth": baseline.selected_depth,
    "learning_rate": baseline.selected_learning_rate,
    "n_estimators": baseline.selected_estimators,
    "fit_log_loss": fit_loss,
    **asdict(measure(target, scores)),
  }


def _history_wins(results: DataFrame) -> bool:
  metrics = results.set_index("arm")
  history = cast(Any, metrics.loc["history_gbm"])
  return all(
    float(history["log_loss"])
    < float(cast(Any, metrics.loc[control, "log_loss"]))
    and float(history["average_precision"])
    > float(cast(Any, metrics.loc[control, "average_precision"]))
    for control in ("snapshot_gbm", "shuffled_history_gbm")
  )


def _comparisons(
  target: NDArray[np.int64],
  scores: dict[str, NDArray[np.float64]],
  *,
  bootstrap_samples: int = 200,
  seed: int = 113,
) -> DataFrame:
  reference = scores["snapshot_gbm"]
  reference_losses = _losses(target, reference)
  positive = np.flatnonzero(target == 1)
  negative = np.flatnonzero(target == 0)
  rng = np.random.default_rng(seed)
  records = []
  for arm in ("history_gbm", "shuffled_history_gbm"):
    values = scores[arm]
    difference = _losses(target, values) - reference_losses
    ap_deltas = []
    for _ in range(bootstrap_samples):
      sample = np.concatenate(
        (
          rng.choice(positive, len(positive), replace=True),
          rng.choice(negative, len(negative), replace=True),
        )
      )
      ap_deltas.append(
        measure(target[sample], values[sample]).average_precision
        - measure(target[sample], reference[sample]).average_precision
      )
    records.append(
      {
        "arm": arm,
        "reference": "snapshot_gbm",
        "log_loss_delta": float(difference.mean()),
        "log_loss_delta_se": float(difference.std(ddof=1) / sqrt(len(difference))),
        "average_precision_delta": float(np.mean(ap_deltas)),
        "average_precision_delta_se": float(np.std(ap_deltas, ddof=1)),
      }
    )
  return DataFrame(records)


def _losses(
  target: NDArray[np.int64], scores: NDArray[np.float64]
) -> NDArray[np.float64]:
  probability = np.clip(scores, np.finfo(float).eps, 1 - np.finfo(float).eps)
  return np.asarray(
    -(target * np.log(probability) + (1 - target) * np.log1p(-probability)),
    dtype=np.float64,
  )


def _match_test(expected: DataFrame, observed: DataFrame) -> DataFrame:
  held_out = expected.loc[expected["fold"] == "test"]
  if held_out.empty or held_out["target"].notna().any():
    raise ValueError("history requires a held-out test population")
  columns = ["loan_id", "cutoff", *FEATURE_COLUMNS, *HISTORY_COLUMNS]
  left = held_out[columns].sort_values("loan_id", ignore_index=True)
  right = observed[columns].sort_values("loan_id", ignore_index=True)
  if left.shape != right.shape:
    raise ValueError("revealed history does not match held-out features")
  same = left.eq(right) | (left.isna() & right.isna())
  if not bool(same.to_numpy().all()):
    raise ValueError("revealed history does not match held-out features")
  return observed
