from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import date
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pandas as pd

from quantcredit.baselines import _compare, _exposure, evaluate_baseline, fit_baseline, measure
from quantcredit.populations import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES


class BaselineTests(unittest.TestCase):
  def test_paired_uncertainty_can_prefer_a_simpler_near_best_model(self) -> None:
    target = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    simple = np.array(
      [0.12249685, 0.15569417, 0.19820540, 0.12515548, 0.77369409, 0.83662826,
       0.88639665, 0.87206416]
    )
    empirical_best = np.array([0.10, 0.20, 0.12, 0.15, 0.80, 0.90, 0.85, 0.88])

    candidates, selected = _compare(
      target,
      ((1, 0.05, 10), (3, 0.05, 20)),
      [simple, empirical_best],
      [measure(target, simple), measure(target, empirical_best)],
    )

    self.assertGreater(candidates.iloc[0]["log_loss_delta"], 0)
    self.assertLessEqual(
      candidates.iloc[0]["log_loss_delta"], candidates.iloc[0]["log_loss_delta_se"]
    )
    self.assertEqual(selected, 0)
    self.assertTrue(candidates.iloc[0]["selected"])

  def test_maps_the_grid_and_selects_the_simplest_near_best_candidate(self) -> None:
    examples = self._examples()
    before = examples.copy(deep=True)
    depths = (1, 2)
    learning_rates = (0.03, 0.07)
    estimators = (10, 20)

    baseline = fit_baseline(
      examples, depths=depths, learning_rates=learning_rates, estimators=estimators
    )
    without_test = fit_baseline(
      examples.loc[examples["fold"] != "test"],
      depths=depths,
      learning_rates=learning_rates,
      estimators=estimators,
    )

    candidates = baseline.candidates
    selected = candidates.loc[candidates["selected"]].iloc[0]
    self.assertEqual(baseline.selected_depth, selected["max_depth"])
    self.assertEqual(baseline.selected_learning_rate, selected["learning_rate"])
    self.assertEqual(baseline.selected_estimators, selected["n_estimators"])
    self.assertEqual(
      set(candidates[["max_depth", "learning_rate", "n_estimators"]].itertuples(index=False)),
      {
        (depth, rate, trees)
        for depth in depths
        for rate in learning_rates
        for trees in estimators
      },
    )
    near_best = candidates.loc[candidates["near_best"]]
    simplest = near_best.sort_values(
      ["leaf_budget", "max_depth", "n_estimators", "learning_rate"]
    ).iloc[0]
    self.assertEqual(selected["leaf_budget"], simplest["leaf_budget"])
    empirical_best = candidates.loc[candidates["log_loss"].idxmin()]
    self.assertAlmostEqual(empirical_best["log_loss_delta"], 0)
    self.assertAlmostEqual(empirical_best["log_loss_delta_se"], 0)
    self.assertEqual(baseline.validation.samples, 40)
    self.assertEqual(baseline.validation.events, 5)
    self.assertGreater(baseline.reference.log_loss, baseline.validation.log_loss)
    self.assertEqual(int(baseline.calibration["samples"].sum()), 40)
    self.assertEqual(int(baseline.calibration["events"].sum()), 5)
    self.assertEqual(len(candidates), 8)
    self.assertFalse(baseline.importance.empty)
    pd.testing.assert_frame_equal(candidates, without_test.candidates)
    pd.testing.assert_frame_equal(baseline.calibration, without_test.calibration)

    imputer = baseline.preprocessor.named_transformers_["numeric"]
    credit_score_index = NUMERIC_FEATURES.index("credit_score")
    train_median = examples.loc[examples["fold"] == "train", "credit_score"].median()
    self.assertEqual(imputer.statistics_[credit_score_index], train_median)
    pd.testing.assert_frame_equal(examples, before)

    invalid_selections = (
      candidates.assign(selected=False),
      candidates.assign(selected=[True, True, *([False] * (len(candidates) - 2))]),
    )
    for invalid in invalid_selections:
      with self.subTest(selected=int(invalid["selected"].sum())):
        with self.assertRaisesRegex(ValueError, "exactly one selected"):
          replace(baseline, candidates=invalid)

    wrong_classifier = fit_baseline(
      examples,
      depths=(1,),
      learning_rates=(0.03,),
      estimators=(10,),
    ).classifier
    with self.assertRaisesRegex(ValueError, "does not match"):
      replace(baseline, classifier=wrong_classifier)

  def test_rejects_invalid_protocols(self) -> None:
    examples = self._examples()
    with self.assertRaisesRegex(ValueError, "missing required columns"):
      fit_baseline(
        examples.drop(columns="credit_score"),
        depths=(1,),
        learning_rates=(0.05,),
        estimators=(5,),
      )
    with self.assertRaisesRegex(ValueError, "depths"):
      fit_baseline(examples, depths=(), learning_rates=(0.05,), estimators=(5,))
    with self.assertRaisesRegex(ValueError, "learning_rates"):
      fit_baseline(examples, depths=(1,), learning_rates=(0.05, 0.05), estimators=(5,))
    with self.assertRaisesRegex(ValueError, "estimators"):
      fit_baseline(examples, depths=(1,), learning_rates=(0.05,), estimators=(0,))
    with self.assertRaisesRegex(ValueError, "exceeds 256"):
      fit_baseline(
        examples,
        depths=tuple(range(1, 18)),
        learning_rates=(0.01, 0.02, 0.03, 0.04),
        estimators=(1, 2, 3, 4),
      )

    examples.loc[examples["fold"] == "validation", "target"] = 0
    with self.assertRaisesRegex(ValueError, "both target classes"):
      fit_baseline(examples, depths=(1,), learning_rates=(0.05,), estimators=(5,))

  def test_evaluates_the_matched_test_population_without_refitting(self) -> None:
    revealed = self._examples()
    revealed.loc[revealed.index[-1], ["target_status", "target"]] = ["censored", None]
    examples = revealed.copy(deep=True)
    test_rows = examples["fold"] == "test"
    examples.loc[test_rows, "target"] = None
    examples.loc[test_rows, "target_status"] = "held_out"
    examples["target"] = examples["target"].astype(pd.Int8Dtype())
    baseline = fit_baseline(
      examples,
      depths=(1, 2),
      learning_rates=(0.05,),
      estimators=(10,),
    )
    before = baseline.candidates.copy(deep=True)
    split = cast(
      Any,
      SimpleNamespace(
        test_cutoff=date(2025, 9, 30),
        test_labels_observed_through=date(2025, 12, 31),
      ),
    )

    with (
      patch(
        "quantcredit.baselines.materialize_test_examples",
        return_value=revealed.loc[revealed["fold"] == "test"].copy(),
      ),
      patch.object(baseline.classifier, "fit", side_effect=AssertionError("refit")),
    ):
      evaluation = evaluate_baseline(
        baseline,
        examples,
        cast(Any, None),
        split,
      )

    self.assertEqual(evaluation.metrics.samples, 19)
    self.assertEqual(evaluation.metrics.events, 3)
    self.assertEqual(evaluation.reference.samples, 19)
    self.assertEqual(evaluation.reference.events, 3)
    self.assertGreater(evaluation.reference.log_loss, 0)
    self.assertEqual(int(evaluation.calibration["samples"].sum()), 19)
    self.assertEqual(evaluation.exposure.samples, 19)
    self.assertEqual(evaluation.exposure.exposure_samples, 19)
    self.assertGreater(evaluation.exposure.total_exposure, 0)
    self.assertAlmostEqual(
      evaluation.exposure.scenario(0.6)["expected_loss"],
      evaluation.exposure.expected_event_exposure * 0.6,
    )
    self.assertNotIn("loan_id", repr(evaluation))
    pd.testing.assert_frame_equal(baseline.candidates, before)

    changed = examples.copy(deep=True)
    changed.loc[test_rows, "credit_score"] += 1
    with (
      patch(
        "quantcredit.baselines.materialize_test_examples",
        return_value=revealed.loc[revealed["fold"] == "test"].copy(),
      ),
      self.assertRaisesRegex(ValueError, "does not match"),
    ):
      evaluate_baseline(baseline, changed, cast(Any, None), split)

  def test_exposure_uses_observed_balances_and_explicit_lgd(self) -> None:
    target = np.array([0, 1, 0, 1], dtype=np.int64)
    scores = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    result = _exposure(target, scores, pd.Series([100.0, 200.0, None, 400.0]), bands=2)

    self.assertEqual(result.samples, 4)
    self.assertEqual(result.exposure_samples, 3)
    self.assertAlmostEqual(result.coverage, 0.75)
    self.assertAlmostEqual(result.total_exposure, 700.0)
    self.assertAlmostEqual(result.expected_event_exposure, 210.0)
    self.assertAlmostEqual(result.observed_event_exposure, 600.0)
    self.assertAlmostEqual(result.weighted_pd, 0.3)
    self.assertEqual(int(result.bands["samples"].sum()), 4)
    self.assertEqual(int(result.bands["exposure_samples"].sum()), 3)
    self.assertEqual(
      result.scenario(0.5),
      {
        "assumed_lgd": 0.5,
        "expected_loss": 105.0,
        "expected_loss_rate": 0.15,
        "status": "scenario, not an estimated ultimate net loss",
      },
    )
    for lgd in (-0.1, 1.1, float("nan")):
      with self.subTest(lgd=lgd), self.assertRaisesRegex(ValueError, "lgd"):
        result.scenario(lgd)

    with self.assertRaisesRegex(ValueError, "cannot be negative"):
      _exposure(target, scores, pd.Series([100.0, -1.0, 200.0, 300.0]))

  @staticmethod
  def _examples() -> pd.DataFrame:
    rows = []
    for fold, count, shift in (("train", 80, 0), ("validation", 40, 3), ("test", 20, 6)):
      for index in range(count):
        target = int(index % 8 == 0)
        row: dict[str, object] = {
          "loan_id": f"{fold}-{index}",
          "cutoff": pd.Timestamp("2025-01-31") + pd.DateOffset(months=shift),
          "fold": fold,
          "target_status": "positive" if target else "negative",
          "target": target,
        }
        for feature_index, feature in enumerate(FEATURE_COLUMNS):
          if feature in CATEGORICAL_FEATURES:
            row[feature] = f"group-{(index + feature_index) % 3}"
          else:
            row[feature] = float(600 + feature_index + index + shift - 50 * target)
        if index % 13 == 0:
          row["payment_to_income"] = None
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.loc[frame["fold"] == "test", "credit_score"] = 1_000_000.0
    frame.loc[frame["fold"] == "validation", "geography"] = "unseen"
    frame["target"] = frame["target"].astype(pd.Int8Dtype())
    return frame


if __name__ == "__main__":
  unittest.main()
