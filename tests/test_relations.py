from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from quantcredit.relations import RELATIONS, cohorts


class RelationTests(unittest.TestCase):
  def test_cross_fit_excludes_each_loan_and_unknowns_use_past_defaults(self) -> None:
    examples = self._examples()
    mapping, facts = cohorts(examples, folds=3, smoothing=20, seed=31)
    target = examples["target"].to_numpy(dtype=np.int64)
    groups = examples["loan_id"].to_numpy()
    splits = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=31)

    for fold, (fit_rows, held_rows) in enumerate(splits.split(examples, target, groups)):
      expected = float(target[fit_rows].mean())
      for row in held_rows:
        self.assertAlmostEqual(float(facts.values[row, 0, 0]), expected)
        self.assertEqual(float(facts.values[row, 0, 1]), 0)
        self.assertEqual(facts.contexts[row * len(RELATIONS)][2], f"fold-{fold}")

    unknown = examples.iloc[:1].assign(geography="never-seen")
    transformed = mapping.transform(unknown)
    self.assertAlmostEqual(float(transformed.values[0, 0, 0]), mapping.global_rate)
    self.assertEqual(float(transformed.values[0, 0, 1]), 0)

  def test_sparse_lowering_has_four_incoming_edges_and_matched_controls(self) -> None:
    _, facts = cohorts(self._examples(), folds=3)
    true = facts.graph()
    erased = facts.graph("erased")
    false = facts.graph("false")

    self.assertEqual(true.graph.edges, 4 * true.loans)
    self.assertTrue(all(source < true.contexts for source in true.graph.source))
    self.assertTrue(all(target >= true.contexts for target in true.graph.target))
    self.assertTrue(np.array_equal(true.context_values, erased.context_values))
    self.assertTrue(np.array_equal(true.context_values, false.context_values))
    self.assertTrue(np.array_equal(erased.edge_values, np.zeros_like(erased.edge_values)))
    self.assertEqual(sorted(true.graph.source), sorted(false.graph.source))
    self.assertNotEqual(true.graph.source, false.graph.source)
    self.assertNotIn("loan-0", repr(true))

  @staticmethod
  def _examples() -> pd.DataFrame:
    count = 30
    return pd.DataFrame(
      {
        "loan_id": [f"loan-{index}" for index in range(count)],
        "cutoff": pd.Timestamp("2025-01-31"),
        "target": np.asarray(([0, 0, 0, 0, 1] * 6), dtype=np.int64),
        "loan_age_months": np.arange(count) % 12,
        "geography": [f"place-{index}" for index in range(count)],
        "vehicle_type": [f"type-{index % 3}" for index in range(count)],
        "vehicle_new_used": ["new" if index % 2 else "used" for index in range(count)],
      }
    )


if __name__ == "__main__":
  unittest.main()
