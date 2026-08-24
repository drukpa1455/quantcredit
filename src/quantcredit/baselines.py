"""Train-only preprocessing and validation selection for the shallow GBM baseline."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import sqrt
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
  average_precision_score,
  brier_score_loss,
  log_loss,
  roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from quantcredit.populations import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES

if TYPE_CHECKING:
  from matplotlib.figure import Figure

DEFAULT_DEPTHS = (1, 2, 3, 4)
DEFAULT_LEARNING_RATES = (0.02, 0.05, 0.10)
DEFAULT_ESTIMATORS = (60, 120, 240)
MAX_CANDIDATES = 256


@dataclass(frozen=True)
class Metrics:
  """Declared predictive evidence for one binary fold."""

  samples: int
  events: int
  event_rate: float
  auroc: float
  average_precision: float
  log_loss: float
  brier_score: float


@dataclass(frozen=True)
class Baseline:
  """A validation-selected baseline whose preprocessing was fit on train only."""

  preprocessor: ColumnTransformer
  classifier: HistGradientBoostingClassifier
  candidates: DataFrame
  reference: Metrics
  calibration: DataFrame
  importance: DataFrame

  def __post_init__(self) -> None:
    selected = self.selected
    fitted_parameters = self.classifier.get_params()
    parameters = (
      int(selected["max_depth"]),
      float(selected["learning_rate"]),
      int(selected["n_estimators"]),
    )
    fitted = (
      fitted_parameters["max_depth"],
      fitted_parameters["learning_rate"],
      fitted_parameters["max_iter"],
    )
    if parameters != fitted:
      raise ValueError("selected candidate does not match the fitted classifier")

  @property
  def selected(self) -> pd.Series[Any]:
    """Return the one candidate that owns the frozen model decision."""
    required = {
      "max_depth",
      "learning_rate",
      "n_estimators",
      "samples",
      "events",
      "event_rate",
      "auroc",
      "average_precision",
      "log_loss",
      "brier_score",
      "selected",
    }
    missing = sorted(required - set(self.candidates.columns))
    if missing:
      raise ValueError(f"candidate evidence is missing columns: {', '.join(missing)}")
    selected = self.candidates.loc[self.candidates["selected"]]
    if len(selected) != 1:
      raise ValueError("baseline requires exactly one selected candidate")
    return selected.iloc[0]

  @property
  def selected_depth(self) -> int:
    return int(self.selected["max_depth"])

  @property
  def selected_learning_rate(self) -> float:
    return float(self.selected["learning_rate"])

  @property
  def selected_estimators(self) -> int:
    return int(self.selected["n_estimators"])

  @property
  def validation(self) -> Metrics:
    """Return validation metrics derived from the selected candidate."""
    selected = self.selected
    return Metrics(
      samples=int(selected["samples"]),
      events=int(selected["events"]),
      event_rate=float(selected["event_rate"]),
      auroc=float(selected["auroc"]),
      average_precision=float(selected["average_precision"]),
      log_loss=float(selected["log_loss"]),
      brier_score=float(selected["brier_score"]),
    )

  def plot(self) -> Figure:
    """Render aggregate validation diagnostics through the canonical visual owner."""
    from quantcredit.visuals import plot_baseline

    return plot_baseline(self)

  def surface(self) -> Figure:
    """Render the declared multidimensional validation sensitivity surface."""
    from quantcredit.visuals import plot_sensitivity

    return plot_sensitivity(self)


def fit_baseline(
  examples: DataFrame,
  *,
  depths: tuple[int, ...] = DEFAULT_DEPTHS,
  learning_rates: tuple[float, ...] = DEFAULT_LEARNING_RATES,
  estimators: tuple[int, ...] = DEFAULT_ESTIMATORS,
  seed: int = 7,
) -> Baseline:
  """Fit train-only preprocessing and select a near-best validation candidate."""
  parameters = _parameters(depths, learning_rates, estimators)
  _validate_examples(examples)
  train = _binary_fold(examples, "train")
  validation = _binary_fold(examples, "validation")
  train_x, train_y = _xy(train)
  validation_x, validation_y = _xy(validation)

  preprocessor = _preprocessor()
  transformed_train = preprocessor.fit_transform(train_x)
  transformed_validation = preprocessor.transform(validation_x)

  scores = []
  metrics = []
  for depth, learning_rate, n_estimators in parameters:
    classifier = _classifier(depth, learning_rate, n_estimators, seed)
    classifier.fit(transformed_train, train_y)
    candidate_scores = classifier.predict_proba(transformed_validation)[:, 1]
    scores.append(candidate_scores)
    metrics.append(_metrics(validation_y, candidate_scores))

  candidates, selected_index = _compare(validation_y, parameters, scores, metrics)
  selected = parameters[selected_index]
  selected_scores = scores[selected_index]
  depth, learning_rate, n_estimators = selected
  classifier = _classifier(depth, learning_rate, n_estimators, seed)
  classifier.fit(transformed_train, train_y)
  reference_scores = np.full(len(validation_y), float(train_y.mean()))
  return Baseline(
    preprocessor=preprocessor,
    classifier=classifier,
    candidates=candidates,
    reference=_metrics(validation_y, reference_scores),
    calibration=_calibration(validation_y, selected_scores),
    importance=_importance(
      preprocessor,
      classifier,
      transformed_validation,
      validation_y,
      seed,
    ),
  )


def _parameters(
  depths: tuple[int, ...],
  learning_rates: tuple[float, ...],
  estimators: tuple[int, ...],
) -> tuple[tuple[int, float, int], ...]:
  if not depths or any(depth <= 0 for depth in depths) or len(set(depths)) != len(depths):
    raise ValueError("depths must contain distinct positive integers")
  if (
    not learning_rates
    or any(not 0 < rate <= 1 for rate in learning_rates)
    or len(set(learning_rates)) != len(learning_rates)
  ):
    raise ValueError("learning_rates must contain distinct values in (0, 1]")
  if (
    not estimators
    or any(count <= 0 for count in estimators)
    or len(set(estimators)) != len(estimators)
  ):
    raise ValueError("estimators must contain distinct positive integers")
  count = len(depths) * len(learning_rates) * len(estimators)
  if count > MAX_CANDIDATES:
    raise ValueError(f"sensitivity grid exceeds {MAX_CANDIDATES} candidates")
  return tuple(product(sorted(depths), sorted(learning_rates), sorted(estimators)))


def _validate_examples(examples: DataFrame) -> None:
  required = {"fold", "target", *FEATURE_COLUMNS}
  missing = sorted(required - set(examples.columns))
  if missing:
    raise ValueError(f"examples are missing required columns: {', '.join(missing)}")


def _binary_fold(examples: DataFrame, fold: str) -> DataFrame:
  frame = examples.loc[(examples["fold"] == fold) & examples["target"].notna()]
  if frame.empty:
    raise ValueError(f"{fold} requires binary outcomes")
  if frame["target"].nunique() != 2:
    raise ValueError(f"{fold} requires both target classes")
  return frame


def _xy(frame: DataFrame) -> tuple[DataFrame, NDArray[np.int64]]:
  return frame[list(FEATURE_COLUMNS)].copy(), frame["target"].to_numpy(dtype=np.int64)


def _preprocessor() -> ColumnTransformer:
  return ColumnTransformer(
    (
      ("numeric", SimpleImputer(strategy="median", add_indicator=True), NUMERIC_FEATURES),
      (
        "categorical",
        Pipeline(
          (
            ("missing", SimpleImputer(strategy="constant", fill_value="missing")),
            (
              "one_hot",
              OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=10,
                max_categories=64,
                sparse_output=False,
              ),
            ),
          )
        ),
        CATEGORICAL_FEATURES,
      ),
    ),
    sparse_threshold=0,
    verbose_feature_names_out=False,
  )


def _classifier(
  depth: int,
  learning_rate: float,
  n_estimators: int,
  seed: int,
) -> HistGradientBoostingClassifier:
  return HistGradientBoostingClassifier(
    max_depth=depth,
    learning_rate=learning_rate,
    max_iter=n_estimators,
    random_state=seed,
    early_stopping=False,
  )


def _compare(
  target: NDArray[np.int64],
  parameters: tuple[tuple[int, float, int], ...],
  scores: list[NDArray[np.float64]],
  metrics: list[Metrics],
) -> tuple[DataFrame, int]:
  best_index = min(range(len(metrics)), key=lambda index: metrics[index].log_loss)
  best_losses = _losses(target, scores[best_index])
  records: list[dict[str, int | float | bool]] = []
  for (depth, learning_rate, n_estimators), candidate_scores, candidate_metrics in zip(
    parameters, scores, metrics, strict=True
  ):
    differences = _losses(target, candidate_scores) - best_losses
    delta = float(differences.mean())
    delta_se = float(differences.std(ddof=1) / sqrt(len(differences)))
    records.append(
      {
        "max_depth": depth,
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "leaf_budget": n_estimators * 2**depth,
        **_metric_record(candidate_metrics),
        "log_loss_delta": delta,
        "log_loss_delta_se": delta_se,
        "near_best": delta <= delta_se,
      }
    )

  eligible = [index for index, record in enumerate(records) if record["near_best"]]
  selected_index = min(
    eligible,
    key=lambda index: (
      records[index]["leaf_budget"],
      records[index]["max_depth"],
      records[index]["n_estimators"],
      records[index]["learning_rate"],
    ),
  )
  for index, record in enumerate(records):
    record["selected"] = index == selected_index
  return DataFrame(records), selected_index


def _losses(target: NDArray[np.int64], scores: NDArray[np.float64]) -> NDArray[np.float64]:
  probabilities = np.clip(scores, np.finfo(float).eps, 1 - np.finfo(float).eps)
  losses = -(target * np.log(probabilities) + (1 - target) * np.log1p(-probabilities))
  return np.asarray(losses, dtype=np.float64)


def _metrics(target: NDArray[np.int64], scores: NDArray[np.float64]) -> Metrics:
  events = int(target.sum())
  return Metrics(
    samples=len(target),
    events=events,
    event_rate=events / len(target),
    auroc=float(roc_auc_score(target, scores)),
    average_precision=float(average_precision_score(target, scores)),
    log_loss=float(log_loss(target, scores)),
    brier_score=float(brier_score_loss(target, scores)),
  )


def _metric_record(metrics: Metrics) -> dict[str, int | float]:
  return {
    "samples": metrics.samples,
    "events": metrics.events,
    "event_rate": metrics.event_rate,
    "auroc": metrics.auroc,
    "average_precision": metrics.average_precision,
    "log_loss": metrics.log_loss,
    "brier_score": metrics.brier_score,
  }


def _calibration(
  target: NDArray[np.int64],
  scores: NDArray[np.float64],
  *,
  bands: int = 10,
) -> DataFrame:
  frame = DataFrame({"target": target, "score": scores})
  frame["band"] = pd.qcut(
    frame["score"].rank(method="first"),
    q=min(bands, len(frame)),
    labels=False,
  )
  grouped = frame.groupby("band", observed=True)
  result = grouped.agg(
    samples=("target", "size"),
    events=("target", "sum"),
    mean_score=("score", "mean"),
    event_rate=("target", "mean"),
  )
  result.index = result.index.astype(int) + 1
  result.index.name = "score_band"
  return result.reset_index()


def _importance(
  preprocessor: ColumnTransformer,
  classifier: HistGradientBoostingClassifier,
  features: NDArray[np.float64],
  target: NDArray[np.int64],
  seed: int,
) -> DataFrame:
  names = preprocessor.get_feature_names_out()
  result = permutation_importance(
    classifier,
    features,
    target,
    scoring="neg_log_loss",
    n_repeats=3,
    random_state=seed,
  )
  values = cast(Any, result).importances_mean
  if len(names) != len(values):
    raise RuntimeError("transformed feature names do not align with model importances")
  return (
    DataFrame({"feature": names, "importance": values})
    .sort_values("importance", ascending=False, ignore_index=True)
  )
