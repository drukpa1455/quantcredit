from __future__ import annotations

import unittest
from typing import Any, cast

import numpy as np
import pandas as pd

from quantcredit.decisions import analyze_decisions, select_pool
from quantcredit.populations import CATEGORICAL_FEATURES, FEATURE_COLUMNS


class _Identity:
  def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
    return frame


class _Score:
  def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
    credit = pd.to_numeric(frame["credit_score"], errors="coerce").fillna(600)
    risk = np.clip((750 - credit.to_numpy()) / 300, 0.01, 0.99)
    return np.column_stack((1 - risk, risk))


class DecisionTests(unittest.TestCase):
  def setUp(self) -> None:
    self.examples = self._examples()
    self.baseline = cast(Any, type("FrozenModel", (), {
      "preprocessor": _Identity(),
      "classifier": _Score(),
    })())

  def test_explains_only_resolved_validation_rows_without_mutation(self) -> None:
    before = self.examples.copy(deep=True)

    decision = analyze_decisions(
      self.baseline,
      self.examples,
      effects=("credit_score", "current_ltv"),
      cohorts=("geography",),
      bands=3,
      min_cohort=2,
      excluded_shares=(0.0, 0.25),
    )

    self.assertEqual(set(decision.effects["feature"]), {"credit_score", "current_ltv"})
    self.assertEqual(
      decision.effects.groupby("feature")["samples"].sum().to_dict(),
      {"credit_score": 8, "current_ltv": 8},
    )
    self.assertEqual(set(decision.frontier["policy"]), {
      "GBM score", "Lowest credit score", "Highest current LTV", "Most delinquent"
    })
    self.assertEqual(len(decision.frontier), 8)
    self.assertNotIn("loan_id", repr(decision))
    pd.testing.assert_frame_equal(self.examples, before)

  def test_pool_respects_budget_and_concentration_limits(self) -> None:
    pool = select_pool(
      self.baseline,
      self.examples,
      budget=500,
      limits={"geography": 0.6},
    )

    self.assertLessEqual(pool.selected_exposure, 500)
    self.assertGreater(pool.utilization, 0.8)
    self.assertLessEqual(float(pool.allocations["share"].max()), 0.6)
    self.assertEqual(
      pool.summary()["status"],
      "retrospective validation selection, not investment performance",
    )
    self.assertNotIn("loan_id", repr(pool))

  def test_rejects_ambiguous_or_impossible_decision_contracts(self) -> None:
    with self.assertRaisesRegex(ValueError, "unknown effect"):
      analyze_decisions(self.baseline, self.examples, effects=("geography",))
    with self.assertRaisesRegex(ValueError, "excluded_shares"):
      analyze_decisions(self.baseline, self.examples, excluded_shares=(0.1, 0.1))
    with self.assertRaisesRegex(ValueError, "budget"):
      select_pool(self.baseline, self.examples, budget=0)
    with self.assertRaisesRegex(ValueError, "cannot exceed"):
      select_pool(self.baseline, self.examples, budget=10_000)
    with self.assertRaisesRegex(ValueError, "concentration"):
      select_pool(self.baseline, self.examples, budget=500, limits={"geography": 0})
    with self.assertRaisesRegex(ValueError, "compliant"):
      select_pool(self.baseline, self.examples, budget=180, limits={"geography": 0.6})

  @staticmethod
  def _examples() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold, count in (("train", 2), ("validation", 8), ("test", 2)):
      for index in range(count):
        target = int(index in {0, 3})
        row: dict[str, object] = {
          "loan_id": f"{fold}-{index}",
          "fold": fold,
          "target": target,
        }
        for feature_index, feature in enumerate(FEATURE_COLUMNS):
          if feature in CATEGORICAL_FEATURES:
            row[feature] = "east" if index % 2 else "west"
          else:
            row[feature] = float(600 + feature_index + index * 20)
        row["credit_score"] = float(560 + index * 25)
        row["current_ltv"] = float(1.2 - index * 0.05)
        row["delinquency_days"] = float(30 - index)
        row["ending_balance"] = 100.0
        row["original_interest_rate"] = 0.08 + index / 100
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
  unittest.main()
