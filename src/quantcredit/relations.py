"""Leakage-safe cohort facts and their sparse typed graph lowering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame
from sklearn.model_selection import StratifiedGroupKFold
from tinymesh import Graph

from quantcredit.populations import FEATURE_COLUMNS

RELATIONS = ("geography", "vintage", "vehicle", "trust")
CONTEXT_FEATURES = ("smoothed_event_rate", "log1p_sample_count")
CONTEXT_KEYS = {
  "geography": "geography",
  "vintage": "inferred_origination_month",
  "vehicle": "vehicle_type|vehicle_new_used",
  "trust": "all_loans",
}


@dataclass(frozen=True)
class Ontology:
  """The exact node and edge schema tested by the graph challenger."""

  @property
  def nodes(self) -> DataFrame:
    records: list[dict[str, Any]] = [
      {
        "node_type": "loan",
        "identity": "loan_at_cutoff",
        "features": FEATURE_COLUMNS,
      }
    ]
    records.extend(
      {
        "node_type": f"{relation}_context",
        "identity": key,
        "features": CONTEXT_FEATURES,
      }
      for relation, key in CONTEXT_KEYS.items()
    )
    return DataFrame(records)

  @property
  def edges(self) -> DataFrame:
    features = tuple(f"is_{relation}" for relation in RELATIONS)
    return DataFrame(
      [
        {
          "relation": relation,
          "source": f"{relation}_context",
          "target": "loan",
          "direction": "context_to_loan",
          "features": features,
          "active_feature": f"is_{relation}",
          "edges_per_loan": 1,
        }
        for relation in RELATIONS
      ]
    )


ONTOLOGY = Ontology()


@dataclass(frozen=True)
class CohortMap:
  """Train-owned smoothed event facts for the four declared relations."""

  global_rate: float
  smoothing: float
  maps: dict[str, dict[str, tuple[float, float]]] = field(repr=False)

  def transform(self, examples: DataFrame, *, scope: str = "past") -> Cohorts:
    keys = _keys(examples)
    values = np.empty((len(examples), len(RELATIONS), 2), dtype=np.float32)
    contexts: list[tuple[str, str, str]] = []
    for row in range(len(examples)):
      for relation, name in enumerate(RELATIONS):
        key = str(keys[name].iat[row])
        rate, count = self.maps[name].get(key, (self.global_rate, 0.0))
        values[row, relation] = (rate, np.log1p(count))
        contexts.append((name, key, scope))
    return Cohorts(values, tuple(contexts))


@dataclass(frozen=True)
class Cohorts:
  """Two train-owned facts per relation and loan."""

  values: NDArray[np.float32] = field(repr=False)
  contexts: tuple[tuple[str, str, str], ...] = field(repr=False)

  @property
  def flat(self) -> NDArray[np.float32]:
    return self.values.reshape(len(self.values), -1)

  def graph(
    self,
    kind: Literal["true", "erased", "false"] = "true",
    *,
    seed: int = 97,
  ) -> GraphBatch:
    return lower(self, kind=kind, seed=seed)


@dataclass(frozen=True)
class GraphBatch:
  """One sparse context-to-loan quiver with matched node and edge facts."""

  graph: Graph
  context_values: NDArray[np.float32] = field(repr=False)
  edge_values: NDArray[np.float32] = field(repr=False)
  loans: int
  contexts: int


def cohorts(
  examples: DataFrame,
  *,
  folds: int = 5,
  smoothing: float = 20,
  seed: int = 31,
) -> tuple[CohortMap, Cohorts]:
  """Fit all-train maps and return leakage-safe cross-fit training facts."""
  _validate(examples)
  target = examples["target"].to_numpy(dtype=np.int64)
  groups = examples["loan_id"].astype(str).to_numpy()
  global_rate = float(target.mean())
  keys = _keys(examples)
  values = np.empty((len(examples), len(RELATIONS), 2), dtype=np.float32)
  contexts: list[tuple[str, str, str] | None] = [None] * (len(examples) * len(RELATIONS))
  splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
  for fold, (fit_rows, held_rows) in enumerate(splitter.split(keys, target, groups)):
    fold_rate = float(target[fit_rows].mean())
    maps = _fit_maps(keys.iloc[fit_rows], target[fit_rows], smoothing, fold_rate)
    for row in held_rows:
      for relation, name in enumerate(RELATIONS):
        key = str(keys[name].iat[row])
        rate, count = maps[name].get(key, (fold_rate, 0.0))
        values[row, relation] = (rate, np.log1p(count))
        contexts[row * len(RELATIONS) + relation] = (name, key, f"fold-{fold}")
  if any(context is None for context in contexts):
    raise RuntimeError("cross-fit did not assign every training row")
  full = _fit_maps(keys, target, smoothing, global_rate)
  return (
    CohortMap(global_rate, smoothing, full),
    Cohorts(values, tuple(context for context in contexts if context is not None)),
  )


def lower(
  cohorts: Cohorts,
  *,
  kind: Literal["true", "erased", "false"] = "true",
  seed: int = 97,
) -> GraphBatch:
  """Lower cohort facts to four incoming typed edges per loan."""
  if kind not in ("true", "erased", "false"):
    raise ValueError("kind must be true, erased, or false")
  loans, relations, facts = cohorts.values.shape
  if relations != len(RELATIONS) or facts != 2:
    raise ValueError(f"cohort values must have shape [N, {len(RELATIONS)}, 2]")
  if len(cohorts.contexts) != loans * relations:
    raise ValueError("context identities do not align with cohort values")

  identities: dict[tuple[str, str, str], int] = {}
  context_values: list[NDArray[np.float32]] = []
  sources: list[int] = []
  for index, identity in enumerate(cohorts.contexts):
    node = identities.get(identity)
    value = cohorts.values[index // relations, index % relations]
    if node is None:
      node = len(identities)
      identities[identity] = node
      context_values.append(value)
    elif not np.array_equal(context_values[node], value):
      raise ValueError("one context identity has conflicting facts")
    sources.append(node)

  if kind == "false":
    rng = np.random.default_rng(seed)
    for relation in range(relations):
      positions = np.arange(relation, len(sources), relations)
      shuffled = np.asarray(sources, dtype=np.int64)[positions]
      rng.shuffle(shuffled)
      for position, source in zip(positions, shuffled, strict=True):
        sources[int(position)] = int(source)

  targets = [len(identities) + row for row in range(loans) for _ in RELATIONS]
  edge_values = np.tile(np.eye(relations, dtype=np.float32), (loans, 1))
  if kind == "erased":
    edge_values.fill(0)
  values = np.asarray(context_values, dtype=np.float32)
  return GraphBatch(
    Graph(len(identities) + loans, sources, targets),
    values,
    edge_values,
    loans,
    len(identities),
  )


def _fit_maps(
  keys: DataFrame,
  target: NDArray[np.int64],
  smoothing: float,
  global_rate: float,
) -> dict[str, dict[str, tuple[float, float]]]:
  maps: dict[str, dict[str, tuple[float, float]]] = {}
  for name in RELATIONS:
    frame = DataFrame({"key": keys[name], "target": target})
    grouped = frame.groupby("key", observed=True)["target"].agg(["sum", "count"])
    maps[name] = {
      str(key): (
        float((row["sum"] + smoothing * global_rate) / (row["count"] + smoothing)),
        float(row["count"]),
      )
      for key, row in grouped.iterrows()
    }
  return maps


def _keys(examples: DataFrame) -> DataFrame:
  cutoff = pd.to_datetime(examples["cutoff"])
  age = pd.to_numeric(examples["loan_age_months"], errors="coerce")
  month = cutoff.dt.year * 12 + cutoff.dt.month - age.round()
  return DataFrame(
    {
      "geography": _category(examples["geography"]),
      "vintage": month.astype("string").fillna("missing"),
      "vehicle": (
        _category(examples["vehicle_type"])
        + "|"
        + _category(examples["vehicle_new_used"])
      ),
      "trust": "all",
    },
    index=examples.index,
  ).reset_index(drop=True)


def _category(values: pd.Series[Any]) -> pd.Series[str]:
  return values.astype("string").fillna("missing")


def _validate(examples: DataFrame) -> None:
  required = {
    "loan_id",
    "cutoff",
    "target",
    "loan_age_months",
    "geography",
    "vehicle_type",
    "vehicle_new_used",
  }
  missing = sorted(required - set(examples.columns))
  if missing:
    raise ValueError(f"examples are missing cohort columns: {', '.join(missing)}")
  if examples.empty or examples["target"].isna().any():
    raise ValueError("cohort training requires resolved binary outcomes")
  if examples["target"].nunique() != 2:
    raise ValueError("cohort training requires both target classes")
