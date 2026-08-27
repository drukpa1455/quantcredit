from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import date
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pandas as pd

from quantcredit.panel import AssetKey
from quantcredit.populations import CATEGORICAL_FEATURES, FEATURE_COLUMNS
from quantcredit.source import load_manifest
from quantcredit.splits import causal_split
from quantcredit.targets import LoanState
from quantcredit.temporal import (
  HISTORY_COLUMNS,
  TEMPORAL_FEATURES,
  History,
  _assemble,
  _history_fields,
  _match_test,
  _permute_columns,
  _shuffle,
  forecast,
  reveal,
)


class TemporalTests(unittest.TestCase):
  def test_assemble_requires_complete_history_and_a_performing_cutoff(self) -> None:
    manifest = load_manifest()
    split = causal_split(manifest.report_periods, horizon_reports=1)
    cutoff = manifest.report_periods[2]
    states = {}
    current = {}
    temporal = {}
    base = self._row_features()
    for number, kind in enumerate(("positive", "negative", "ineligible", "incomplete")):
      asset = AssetKey("0000000001", f"asset-{number}")
      state = [LoanState(0) for _ in manifest.report_periods]
      if kind == "positive":
        state[3] = LoanState(60)
      if kind == "ineligible":
        state[2] = LoanState(30)
      states[asset] = state
      current[asset, 2] = base
      for index in range(3):
        if not (kind == "incomplete" and index == 0):
          temporal[asset, index] = tuple(base[feature] for feature in TEMPORAL_FEATURES)

    history = _assemble(
      manifest,
      split,
      3,
      {cutoff: "train"},
      frozenset(),
      states,
      current,
      temporal,
    )

    audit = history.summary().iloc[0]
    self.assertEqual(int(audit["reported"]), 4)
    self.assertEqual(int(audit["incomplete_history"]), 1)
    self.assertEqual(int(audit["ineligible"]), 1)
    self.assertEqual(int(audit["modeled"]), 2)
    self.assertEqual(int(audit["events"]), 1)

  def test_history_fields_preserve_lags_and_derive_only_past_changes(self) -> None:
    fields = _history_fields(
      [
        (100.0, 0.5, 10, 20.0, 20.0, 5),
        (110.0, 0.6, 11, 20.0, 20.0, 3),
        (125.0, 0.7, 12, 20.0, 20.0, 0),
      ]
    )

    self.assertEqual(fields["ending_balance_lag_1"], 110.0)
    self.assertEqual(fields["ending_balance_lag_2"], 125.0)
    self.assertEqual(fields["ending_balance_change_1"], -10.0)
    self.assertEqual(fields["ending_balance_change_2"], -15.0)
    self.assertEqual(fields["delinquency_days_change_1"], 2.0)

  def test_shuffle_preserves_each_cutoff_distribution_but_breaks_alignment(self) -> None:
    values = np.arange(24, dtype=np.float64).reshape(8, 3)
    groups = pd.Series(["a"] * 4 + ["b"] * 4)
    shuffled = _shuffle(values, groups, seed=7)

    self.assertFalse(np.array_equal(values, shuffled))
    for start in (0, 4):
      self.assertEqual(
        sorted(map(tuple, values[start : start + 4])),
        sorted(map(tuple, shuffled[start : start + 4])),
      )

  def test_forecast_uses_no_held_out_target_and_detects_aligned_history(self) -> None:
    history = self._history()
    without_test = replace(history, _rows=history._rows.loc[history._rows["fold"] != "test"])

    study = forecast(
      history,
      depths=(2,),
      learning_rates=(0.1,),
      estimators=(30,),
    )
    repeated = forecast(
      without_test,
      depths=(2,),
      learning_rates=(0.1,),
      estimators=(30,),
    )

    pd.testing.assert_frame_equal(study.results, repeated.results)
    pd.testing.assert_frame_equal(study.drivers, repeated.drivers)
    self.assertEqual(
      set(study.results["arm"]),
      {"snapshot_gbm", "history_gbm", "shuffled_history_gbm"},
    )
    self.assertEqual(study.decision, "retain_history")
    self.assertLess(
      float(study.results.set_index("arm").loc["history_gbm", "log_loss"]),
      float(study.results.set_index("arm").loc["snapshot_gbm", "log_loss"]),
    )
    self.assertEqual(tuple(study.drivers["feature"]), TEMPORAL_FEATURES)
    self.assertGreater(float(study.drivers.iloc[0]["log_loss_increase"]), 0)
    self.assertEqual(len(study.plot().axes), 4)

  def test_group_permutation_preserves_joint_values_within_cutoff(self) -> None:
    frame = pd.DataFrame(
      {
        "a": [1, 2, 3, 4, 5, 6],
        "b": [11, 12, 13, 14, 15, 16],
        "untouched": [21, 22, 23, 24, 25, 26],
      }
    )
    groups = pd.Series(["x", "x", "x", "y", "y", "y"])

    permuted = _permute_columns(frame, ("a", "b"), groups, np.random.default_rng(7))

    pd.testing.assert_series_equal(permuted["untouched"], frame["untouched"])
    for positions in ((0, 1, 2), (3, 4, 5)):
      self.assertEqual(
        sorted(map(tuple, permuted.loc[list(positions), ["a", "b"]].to_numpy())),
        sorted(map(tuple, frame.loc[list(positions), ["a", "b"]].to_numpy())),
      )

  def test_reveal_matches_exact_history_before_opening_test(self) -> None:
    history = self._history()
    study = forecast(
      history,
      depths=(2,),
      learning_rates=(0.1,),
      estimators=(30,),
    )
    observed_rows = history._rows.loc[history._rows["fold"] == "test"].copy()
    observed_rows["target"] = (np.arange(len(observed_rows)) % 5 == 0).astype(int)
    observed_rows["target_status"] = np.where(
      observed_rows["target"] == 1, "positive", "negative"
    )
    observed = replace(history, _rows=observed_rows)

    with patch("quantcredit.temporal.materialize_test_history", return_value=observed):
      evaluation = reveal(
        study,
        history,
        cast(Any, SimpleNamespace()),
        cast(Any, SimpleNamespace()),
      )

    self.assertEqual(set(evaluation.results["arm"]), set(study.results["arm"]))
    self.assertEqual(len(evaluation.plot().axes), 2)

    changed = observed_rows.copy()
    changed.loc[changed.index[0], HISTORY_COLUMNS[0]] = -1
    with self.assertRaisesRegex(ValueError, "does not match"):
      _match_test(history._rows, changed)

  def test_history_repr_and_summary_do_not_expose_loan_rows(self) -> None:
    history = self._history()

    self.assertNotIn("loan-0", repr(history))
    self.assertNotIn("loan_id", history.summary().columns)
    self.assertEqual(tuple(history.features["feature"]), TEMPORAL_FEATURES)
    self.assertEqual(len(history.plot().axes), 2)

  @staticmethod
  def _history() -> History:
    records = []
    folds = (("train", 160), ("validation", 80), ("test", 80))
    row = 0
    for fold, count in folds:
      for index in range(count):
        target = index % 5 == 0
        record: dict[str, object] = {
          "loan_id": f"loan-{row}",
          "cutoff": pd.Timestamp(
            date(2025, {"train": 7, "validation": 9, "test": 11}[fold], 30)
          ),
          "fold": fold,
          "target_status": "held_out" if fold == "test" else (
            "positive" if target else "negative"
          ),
          "target": None if fold == "test" else int(target),
        }
        record.update(TemporalTests._row_features())
        for feature in HISTORY_COLUMNS:
          record[feature] = float(target) if feature == "ending_balance_lag_1" else 0.0
        records.append(record)
        row += 1
    rows = pd.DataFrame(records)
    rows["target"] = rows["target"].astype(pd.Int8Dtype())
    audit = pd.DataFrame(
      {
        "cutoff": [date(2025, 7, 31), date(2025, 9, 30), date(2025, 11, 30)],
        "fold": ["train", "validation", "test"],
        "reported": [160, 80, 80],
        "complete_history": [160, 80, 80],
        "modeled": [160, 80, 0],
        "held_out": [0, 0, 80],
        "event_rate": [0.2, 0.2, np.nan],
      }
    )
    return History(audit, 3, 1, rows)

  @staticmethod
  def _row_features() -> dict[str, object]:
    return {
      feature: "same" if feature in CATEGORICAL_FEATURES else 1.0
      for feature in FEATURE_COLUMNS
    }


if __name__ == "__main__":
  unittest.main()
