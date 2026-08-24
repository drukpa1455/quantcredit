"""Train-only preprocessing and validation selection for the shallow GBM baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
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

DEFAULT_DEPTHS = (2, 3, 4)


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
  classifier: GradientBoostingClassifier
  selected_depth: int
  candidates: DataFrame
  reference: Metrics
  validation: Metrics
  calibration: DataFrame
  importance: DataFrame

  def plot(self) -> Figure:
    """Render aggregate validation diagnostics through the canonical visual owner."""
    from quantcredit.visuals import plot_baseline

    return plot_baseline(self)


def fit_baseline(
  examples: DataFrame,
  *,
  depths: tuple[int, ...] = DEFAULT_DEPTHS,
  n_estimators: int = 120,
  learning_rate: float = 0.05,
  seed: int = 7,
) -> Baseline:
  """Fit preprocessing on train and select tree depth by validation log loss."""
  _validate(examples, depths, n_estimators, learning_rate)
  train = _binary_fold(examples, "train")
  validation = _binary_fold(examples, "validation")
  train_x, train_y = _xy(train)
  validation_x, validation_y = _xy(validation)

  preprocessor = _preprocessor()
  transformed_train = preprocessor.fit_transform(train_x)
  transformed_validation = preprocessor.transform(validation_x)

  candidates: list[
    tuple[
      int,
      GradientBoostingClassifier,
      np.ndarray[Any, np.dtype[np.float64]],
      Metrics,
    ]
  ] = []
  for depth in sorted(set(depths)):
    classifier = GradientBoostingClassifier(
      max_depth=depth,
      n_estimators=n_estimators,
      learning_rate=learning_rate,
      random_state=seed,
    )
    classifier.fit(transformed_train, train_y)
    scores = classifier.predict_proba(transformed_validation)[:, 1]
    candidates.append((depth, classifier, scores, _metrics(validation_y, scores)))

  selected = min(candidates, key=lambda candidate: (candidate[3].log_loss, candidate[0]))
  depth, classifier, scores, metrics = selected
  reference_scores = np.full(len(validation_y), float(train_y.mean()))
  return Baseline(
    preprocessor=preprocessor,
    classifier=classifier,
    selected_depth=depth,
    candidates=DataFrame(
      [
        {"max_depth": candidate_depth, **_metric_record(candidate_metrics)}
        for candidate_depth, _, _, candidate_metrics in candidates
      ]
    ),
    reference=_metrics(validation_y, reference_scores),
    validation=metrics,
    calibration=_calibration(validation_y, scores),
    importance=_importance(preprocessor, classifier),
  )


def _validate(
  examples: DataFrame,
  depths: tuple[int, ...],
  n_estimators: int,
  learning_rate: float,
) -> None:
  required = {"fold", "target", *FEATURE_COLUMNS}
  missing = sorted(required - set(examples.columns))
  if missing:
    raise ValueError(f"examples are missing required columns: {', '.join(missing)}")
  if not depths or any(depth <= 0 for depth in depths):
    raise ValueError("depths must contain positive integers")
  if n_estimators <= 0:
    raise ValueError("n_estimators must be positive")
  if not 0 < learning_rate <= 1:
    raise ValueError("learning_rate must be in (0, 1]")


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


def _metrics(target: np.ndarray[Any, np.dtype[np.int64]], scores: np.ndarray[Any, Any]) -> Metrics:
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
  target: np.ndarray[Any, np.dtype[np.int64]],
  scores: np.ndarray[Any, Any],
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
  classifier: GradientBoostingClassifier,
) -> DataFrame:
  names = preprocessor.get_feature_names_out()
  values = classifier.feature_importances_
  if len(names) != len(values):
    raise RuntimeError("transformed feature names do not align with model importances")
  return (
    DataFrame({"feature": names, "importance": values})
    .sort_values("importance", ascending=False, ignore_index=True)
  )
