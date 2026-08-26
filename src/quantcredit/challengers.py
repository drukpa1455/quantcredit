"""Matched tabular and sparse-graph challengers for the frozen validation fold."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import log
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from tinygrad import Context, Tensor, nn  # type: ignore[attr-defined]
from tinymesh.nn import GINEConv

from quantcredit.acquire import DEFAULT_CACHE
from quantcredit.baselines import Baseline, calibrate, match_test, measure
from quantcredit.populations import FEATURE_COLUMNS, materialize_test_examples
from quantcredit.relations import CohortMap, Cohorts, GraphBatch, cohorts

if TYPE_CHECKING:
  from matplotlib.figure import Figure

  from quantcredit.source import SourceManifest
  from quantcredit.splits import CausalSplit

DEFAULT_SEEDS = (7, 19, 43)


@dataclass(frozen=True)
class GraphStudy:
  """Aggregate validation evidence with fitted artifacts hidden from display."""

  results: DataFrame
  calibration: DataFrame
  topology: DataFrame
  comparison: DataFrame
  decision: Literal["reject_graph", "retain_existing_gine", "investigate_missing_equation"]
  _artifacts: Artifacts = field(repr=False)

  def summary(self) -> DataFrame:
    return self.results.loc[self.results["seed"].isna()].reset_index(drop=True)

  def plot(self) -> Figure:
    from quantcredit.visuals import plot_graph_study

    return plot_graph_study(self)

  def deltas(self) -> DataFrame:
    """Return paired true-GINE deltas against each neural control by seed."""
    seeded = self.results.loc[self.results["seed"].notna()]
    true = seeded.loc[seeded["arm"] == "true_gine"].set_index("seed")
    records = []
    for control in ("node_local", "erased_gine", "false_gine"):
      other = seeded.loc[seeded["arm"] == control].set_index("seed")
      for seed in true.index:
        records.append(
          {
            "control": control,
            "seed": int(seed),
            "log_loss_delta": float(true.loc[seed, "log_loss"] - other.loc[seed, "log_loss"]),
            "average_precision_delta": float(
              true.loc[seed, "average_precision"] - other.loc[seed, "average_precision"]
            ),
          }
        )
    return DataFrame(records)


@dataclass(frozen=True)
class GraphEvaluation:
  """One frozen out-of-time reveal of every validation-selected arm."""

  results: DataFrame
  calibration: DataFrame
  validation_decision: str
  decision: str

  def summary(self) -> DataFrame:
    return self.results.copy()

  def plot(self) -> Figure:
    from quantcredit.visuals import plot_graph_evaluation

    return plot_graph_evaluation(self)


@dataclass(frozen=True)
class Artifacts:
  baseline: Baseline = field(repr=False)
  cohort_map: CohortMap = field(repr=False)
  enriched: HistGradientBoostingClassifier = field(repr=False)
  raw_scaler: StandardScaler = field(repr=False)
  local_scaler: StandardScaler = field(repr=False)
  models: dict[str, tuple[LocalModel | GraphModel, ...]] = field(repr=False)


class LocalModel:
  def __init__(self, features: int, hidden: int, seed: int) -> None:
    Tensor.manual_seed(seed)
    self.hidden = nn.Linear(features, hidden)
    self.middle = nn.Linear(hidden, hidden)
    self.output = nn.Linear(hidden, 2)

  def __call__(self, values: Tensor) -> Tensor:
    return self.output(self.middle(self.hidden(values).relu()).relu())

  @property
  def parameters(self) -> int:
    return _parameters(self)


class GraphModel:
  def __init__(self, features: int, hidden: int, seed: int, kind: str) -> None:
    Tensor.manual_seed(seed)
    self.loan = nn.Linear(features, hidden)
    self.context = nn.Linear(2, hidden)
    self.conv = GINEConv(hidden, 4, hidden)
    self.output = nn.Linear(hidden, 2)
    self.kind = kind

  def __call__(self, raw: Tensor, batch: GraphBatch) -> Tensor:
    context = self.context(Tensor(batch.context_values))
    loan = self.loan(raw)
    nodes = context.cat(loan, dim=0)
    state = self.conv(nodes, Tensor(batch.edge_values), batch.graph)
    return self.output(state[batch.contexts :])

  def predict(self, raw: NDArray[np.float32], facts: Cohorts) -> NDArray[np.float64]:
    batch = facts.graph(self.kind)  # type: ignore[arg-type]
    return _probability(self(Tensor(raw), batch))

  @property
  def parameters(self) -> int:
    return _parameters(self)


def challenge(
  baseline: Baseline,
  examples: DataFrame,
  *,
  seeds: tuple[int, ...] = DEFAULT_SEEDS,
  hidden: int = 16,
  steps: int = 60,
  learning_rate: float = 0.01,
) -> GraphStudy:
  """Run the frozen matched-information validation comparison."""
  _validate_protocol(seeds, hidden, steps, learning_rate)
  train = _fold(examples, "train")
  validation = _fold(examples, "validation")
  train_y = train["target"].to_numpy(dtype=np.int64)
  validation_y = validation["target"].to_numpy(dtype=np.int64)
  train_raw = np.asarray(
    baseline.preprocessor.transform(train[list(FEATURE_COLUMNS)]), dtype=np.float32
  )
  validation_raw = np.asarray(
    baseline.preprocessor.transform(validation[list(FEATURE_COLUMNS)]), dtype=np.float32
  )
  cohort_map, train_facts = cohorts(train)
  validation_facts = cohort_map.transform(validation)

  enriched = HistGradientBoostingClassifier(
    max_depth=baseline.selected_depth,
    learning_rate=baseline.selected_learning_rate,
    max_iter=baseline.selected_estimators,
    random_state=7,
    early_stopping=False,
  )
  enriched.fit(np.concatenate((train_raw, train_facts.flat), axis=1), train_y)
  fixed_scores = {
    "raw_gbm": baseline.classifier.predict_proba(validation_raw)[:, 1],
    "enriched_gbm": enriched.predict_proba(
      np.concatenate((validation_raw, validation_facts.flat), axis=1)
    )[:, 1],
  }
  fixed_train_scores = {
    "raw_gbm": baseline.classifier.predict_proba(train_raw)[:, 1],
    "enriched_gbm": enriched.predict_proba(
      np.concatenate((train_raw, train_facts.flat), axis=1)
    )[:, 1],
  }

  raw_scaler = StandardScaler().fit(train_raw)
  train_raw_scaled = raw_scaler.transform(train_raw).astype(np.float32)
  validation_raw_scaled = raw_scaler.transform(validation_raw).astype(np.float32)
  local_train = np.concatenate((train_raw_scaled, train_facts.flat), axis=1)
  local_scaler = StandardScaler().fit(local_train)
  local_train = local_scaler.transform(local_train).astype(np.float32)
  local_validation = local_scaler.transform(
    np.concatenate((validation_raw_scaled, validation_facts.flat), axis=1)
  ).astype(np.float32)

  records: list[dict[str, Any]] = []
  calibration_records: list[DataFrame] = []
  for arm, scores in fixed_scores.items():
    records.append(
      _record(
        arm,
        None,
        validation_y,
        scores,
        int(baseline.selected["leaf_budget"]),
        validation,
        fit_log_loss=measure(train_y, fixed_train_scores[arm]).log_loss,
      )
    )
    calibration_records.append(_calibration(arm, scores, validation_y))

  models: dict[str, list[LocalModel | GraphModel]] = {
    "node_local": [],
    "true_gine": [],
    "erased_gine": [],
    "false_gine": [],
  }
  score_sets: dict[str, list[NDArray[np.float64]]] = {arm: [] for arm in models}
  for seed in seeds:
    local = LocalModel(local_train.shape[1], hidden, seed)
    _prior(local.output, float(train_y.mean()))
    _train_local(local, local_train, train_y, steps, learning_rate)
    local_fit_scores = _probability(local(Tensor(local_train)))
    local_scores = _probability(local(Tensor(local_validation)))
    models["node_local"].append(local)
    score_sets["node_local"].append(local_scores)
    records.append(
      _record(
        "node_local",
        seed,
        validation_y,
        local_scores,
        local.parameters,
        validation,
        fit_log_loss=measure(train_y, local_fit_scores).log_loss,
      )
    )

    for arm, kind in (("true_gine", "true"), ("erased_gine", "erased"), ("false_gine", "false")):
      model = GraphModel(train_raw_scaled.shape[1], hidden, seed, kind)
      _prior(model.output, float(train_y.mean()))
      _train_graph(model, train_raw_scaled, train_facts, train_y, steps, learning_rate)
      fit_scores = model.predict(train_raw_scaled, train_facts)
      scores = model.predict(validation_raw_scaled, validation_facts)
      models[arm].append(model)
      score_sets[arm].append(scores)
      records.append(
        _record(
          arm,
          seed,
          validation_y,
          scores,
          model.parameters,
          validation,
          fit_log_loss=measure(train_y, fit_scores).log_loss,
        )
      )

  ensemble_scores: dict[str, NDArray[np.float64]] = dict(fixed_scores)
  for arm, arm_scores in score_sets.items():
    ensemble = np.mean(arm_scores, axis=0)
    ensemble_scores[arm] = ensemble
    records.append(
      _record(arm, None, validation_y, ensemble, models[arm][0].parameters, validation)
    )
    calibration_records.append(_calibration(arm, ensemble, validation_y))

  result = DataFrame(records)
  decision = _decision(result)
  topology = _topology(train_facts, validation_facts)
  return GraphStudy(
    result,
    pd.concat(calibration_records, ignore_index=True),
    topology,
    _comparisons(validation_y, ensemble_scores),
    decision,
    Artifacts(
      baseline,
      cohort_map,
      enriched,
      raw_scaler,
      local_scaler,
      {arm: tuple(values) for arm, values in models.items()},
    ),
  )


def confirm(
  study: GraphStudy,
  examples: DataFrame,
  manifest: SourceManifest,
  split: CausalSplit,
  cache: Path = DEFAULT_CACHE,
) -> GraphEvaluation:
  """Reveal one exact held-out test fold without fitting or changing any arm."""
  revealed = materialize_test_examples(manifest, split, cache)
  test = _fold(match_test(examples, revealed), "test")
  target = test["target"].to_numpy(dtype=np.int64)
  artifacts = study._artifacts
  raw = np.asarray(
    artifacts.baseline.preprocessor.transform(test[list(FEATURE_COLUMNS)]),
    dtype=np.float32,
  )
  facts = artifacts.cohort_map.transform(test)
  raw_scaled = artifacts.raw_scaler.transform(raw).astype(np.float32)
  local = artifacts.local_scaler.transform(
    np.concatenate((raw_scaled, facts.flat), axis=1)
  ).astype(np.float32)
  scores: dict[str, NDArray[np.float64]] = {
    "raw_gbm": artifacts.baseline.classifier.predict_proba(raw)[:, 1],
    "enriched_gbm": artifacts.enriched.predict_proba(
      np.concatenate((raw, facts.flat), axis=1)
    )[:, 1],
    "node_local": np.stack(
      [
        _probability(model(Tensor(local)))
        for model in cast(tuple[LocalModel, ...], artifacts.models["node_local"])
      ]
    ).mean(axis=0),
  }
  for arm in ("true_gine", "erased_gine", "false_gine"):
    models = cast(tuple[GraphModel, ...], artifacts.models[arm])
    scores[arm] = np.stack(
      [model.predict(raw_scaled, facts) for model in models]
    ).mean(axis=0)
  parameters = {
    "raw_gbm": int(artifacts.baseline.selected["leaf_budget"]),
    "enriched_gbm": int(artifacts.baseline.selected["leaf_budget"]),
    **{
      arm: artifacts.models[arm][0].parameters
      for arm in ("node_local", "true_gine", "erased_gine", "false_gine")
    },
  }
  results = DataFrame(
    [
      _record(arm, None, target, values, parameters[arm], test)
      for arm, values in scores.items()
    ]
  )
  calibrations = pd.concat(
    [_calibration(arm, values, target) for arm, values in scores.items()],
    ignore_index=True,
  )
  test_passes = _ensemble_passes(results)
  decision = (
    "retain_existing_gine"
    if study.decision == "retain_existing_gine" and test_passes
    else "reject_graph"
  )
  return GraphEvaluation(results, calibrations, study.decision, decision)


def _train_local(
  model: LocalModel,
  features: NDArray[np.float32],
  target: NDArray[np.int64],
  steps: int,
  learning_rate: float,
) -> None:
  x, y = Tensor(features), Tensor(target)
  optimizer = nn.optim.Adam(nn.state.get_parameters(model), lr=learning_rate, fused=False)
  for _ in range(steps):
    with Context(TRAINING=1):  # type: ignore[no-untyped-call]
      optimizer.zero_grad()
      loss = model(x).sparse_categorical_crossentropy(y).backward()
      loss.realize(*optimizer.schedule_step())


def _train_graph(
  model: GraphModel,
  raw: NDArray[np.float32],
  facts: Cohorts,
  target: NDArray[np.int64],
  steps: int,
  learning_rate: float,
) -> None:
  x, y, batch = Tensor(raw), Tensor(target), facts.graph(model.kind)  # type: ignore[arg-type]
  optimizer = nn.optim.Adam(nn.state.get_parameters(model), lr=learning_rate, fused=False)
  for _ in range(steps):
    with Context(TRAINING=1):  # type: ignore[no-untyped-call]
      optimizer.zero_grad()
      loss = model(x, batch).sparse_categorical_crossentropy(y).backward()
      loss.realize(*optimizer.schedule_step())


def _probability(logits: Tensor) -> NDArray[np.float64]:
  return np.asarray(logits.softmax(axis=1)[:, 1].numpy(), dtype=np.float64)


def _parameters(model: object) -> int:
  return sum(int(parameter.numel()) for parameter in nn.state.get_parameters(model))


def _prior(output: nn.Linear, event_rate: float) -> None:
  if output.bias is None:
    raise ValueError("binary output requires a bias")
  output.bias.assign([0.0, log(event_rate / (1 - event_rate))]).realize()


def _fold(examples: DataFrame, name: str) -> DataFrame:
  frame = examples.loc[
    (examples["fold"] == name) & examples["target"].notna()
  ].reset_index(drop=True)
  if frame.empty or frame["target"].nunique() != 2:
    raise ValueError(f"{name} requires both target classes")
  return frame


def _record(
  arm: str,
  seed: int | None,
  target: NDArray[np.int64],
  scores: NDArray[np.float64],
  parameters: int,
  examples: DataFrame,
  *,
  fit_log_loss: float = float("nan"),
) -> dict[str, Any]:
  return {
    "arm": arm,
    "seed": seed,
    "parameters": parameters,
    "fit_log_loss": fit_log_loss,
    "valid": bool(np.ptp(scores) > 1e-12),
    "status": "valid" if np.ptp(scores) > 1e-12 else "invalid_constant_scores",
    **asdict(measure(target, scores)),
    "event_exposure_avoided_10pct": _avoided(target, scores, examples["ending_balance"]),
  }


def _avoided(
  target: NDArray[np.int64], scores: NDArray[np.float64], exposure: pd.Series[Any]
) -> float:
  balance = pd.to_numeric(exposure, errors="coerce").fillna(0).to_numpy(dtype=np.float64)
  order = np.argsort(scores)[::-1]
  cutoff = 0.10 * balance.sum()
  rejected = np.zeros(len(balance), dtype=bool)
  total = 0.0
  for row in order:
    if total >= cutoff:
      break
    rejected[row] = True
    total += balance[row]
  event_exposure = target * balance
  denominator = event_exposure.sum()
  return float(event_exposure[rejected].sum() / denominator) if denominator else float("nan")


def _calibration(arm: str, scores: NDArray[np.float64], target: NDArray[np.int64]) -> DataFrame:
  return calibrate(target, scores).assign(arm=arm)


def _comparisons(
  target: NDArray[np.int64],
  scores: dict[str, NDArray[np.float64]],
  *,
  bootstrap_samples: int = 200,
  seed: int = 113,
) -> DataFrame:
  """Return paired validation effects against enriched GBM without row retention."""
  reference = scores["enriched_gbm"]
  reference_losses = _log_losses(target, reference)
  rng = np.random.default_rng(seed)
  samples = rng.integers(0, len(target), size=(bootstrap_samples, len(target)))
  records = []
  for arm, values in scores.items():
    if arm == "enriched_gbm":
      continue
    difference = _log_losses(target, values) - reference_losses
    ap_deltas = [
      average_precision_score(target[index], values[index])
      - average_precision_score(target[index], reference[index])
      for index in samples
    ]
    records.append(
      {
        "arm": arm,
        "reference": "enriched_gbm",
        "log_loss_delta": float(difference.mean()),
        "log_loss_delta_se": float(difference.std(ddof=1) / np.sqrt(len(difference))),
        "average_precision_delta": float(
          average_precision_score(target, values)
          - average_precision_score(target, reference)
        ),
        "average_precision_delta_se": float(
          np.asarray(ap_deltas, dtype=np.float64).std(ddof=1)
        ),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
      }
    )
  return DataFrame(records)


def _log_losses(
  target: NDArray[np.int64], scores: NDArray[np.float64]
) -> NDArray[np.float64]:
  probability = np.clip(scores, np.finfo(float).eps, 1 - np.finfo(float).eps)
  losses = -(target * np.log(probability) + (1 - target) * np.log1p(-probability))
  return np.asarray(losses, dtype=np.float64)


def _decision(
  results: DataFrame,
) -> Literal["reject_graph", "retain_existing_gine", "investigate_missing_equation"]:
  ensemble = results.loc[results["seed"].isna()].set_index("arm")
  true = results.loc[results["arm"] == "true_gine"]
  controls = results.loc[results["arm"].isin(("node_local", "erased_gine", "false_gine"))]
  per_seed = all(
    bool(row["valid"])
    and bool(controls.loc[controls["seed"] == row["seed"], "valid"].all())
    and float(row["log_loss"])
    < float(controls.loc[controls["seed"] == row["seed"], "log_loss"].min())
    and float(row["average_precision"])
    > float(controls.loc[controls["seed"] == row["seed"], "average_precision"].max())
    for _, row in true.loc[true["seed"].notna()].iterrows()
  )
  passes = (
    len(true.loc[true["seed"].notna()]) >= 3
    and per_seed
    and _ensemble_passes(ensemble.reset_index())
  )
  return "retain_existing_gine" if passes else "reject_graph"


def _validate_protocol(
  seeds: tuple[int, ...], hidden: int, steps: int, learning_rate: float
) -> None:
  if not seeds or len(seeds) != len(set(seeds)):
    raise ValueError("seeds must contain distinct integers")
  if hidden <= 0 or steps <= 0:
    raise ValueError("hidden and steps must be positive")
  if not 0 < learning_rate <= 1:
    raise ValueError("learning_rate must be in (0, 1]")


def _ensemble_passes(results: DataFrame) -> bool:
  ensemble = results.set_index("arm")
  true_loss = float(cast(Any, ensemble.loc["true_gine", "log_loss"]))
  enriched_loss = float(cast(Any, ensemble.loc["enriched_gbm", "log_loss"]))
  true_ap = float(cast(Any, ensemble.loc["true_gine", "average_precision"]))
  enriched_ap = float(cast(Any, ensemble.loc["enriched_gbm", "average_precision"]))
  true_brier = float(cast(Any, ensemble.loc["true_gine", "brier_score"]))
  enriched_brier = float(cast(Any, ensemble.loc["enriched_gbm", "brier_score"]))
  return bool(
    bool(ensemble["valid"].all())
    and true_loss < enriched_loss
    and true_ap > enriched_ap
    and true_brier <= enriched_brier
  )


def _topology(train: Cohorts, validation: Cohorts) -> DataFrame:
  records = []
  for fold, facts in (("train", train), ("validation", validation)):
    true, false = facts.graph(), facts.graph("false")
    records.append(
      {
        "fold": fold,
        "loans": true.loans,
        "contexts": true.contexts,
        "edges": true.graph.edges,
        "edges_per_loan": true.graph.edges / true.loans,
        "false_preserves_source_degree": sorted(true.graph.source) == sorted(false.graph.source),
      }
    )
  return DataFrame(records)
