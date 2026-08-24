from __future__ import annotations

import unittest
from datetime import date
from typing import Any, cast

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from quantcredit.audits import Audit
from quantcredit.baselines import Baseline, Metrics
from quantcredit.populations import FEATURE_COLUMNS
from quantcredit.splits import causal_split
from quantcredit.visuals import (
  plot_audit,
  plot_baseline,
  plot_examples,
  plot_sensitivity,
  plot_split,
  sapphire,
)


class VisualTests(unittest.TestCase):
  def tearDown(self) -> None:
    plt.close("all")

  def test_sapphire_is_scoped(self) -> None:
    before = mpl.rcParams["figure.facecolor"]

    with sapphire():
      self.assertEqual(mpl.rcParams["figure.facecolor"], "#212c2a")

    self.assertEqual(mpl.rcParams["figure.facecolor"], before)

  def test_audit_figure_exposes_all_four_aggregate_questions(self) -> None:
    audit = self._audit()
    figure = audit.plot()
    self.assertEqual(len(plot_audit(audit).axes), 4)

    self.assertEqual(
      {axis.get_title() for axis in figure.axes},
      {
        "Reported loan population",
        "Observed state share · log scale",
        "Next-report transition rate",
        "Three-report target disposition · log scale",
      },
    )
    self.assertEqual(figure.axes[1].get_xscale(), "log")
    self.assertEqual(figure.axes[3].get_xscale(), "log")
    self.assertEqual(len(figure.axes[0].lines), 1)
    self.assertGreater(len(figure.axes[2].texts), 0)
    self.assertIn(
      "Zero balance 1+99",
      {tick.get_text() for tick in figure.axes[1].get_yticklabels()},
    )
    self.assertIn(
      "Zero balance 1+99",
      {tick.get_text() for tick in figure.axes[2].get_xticklabels()},
    )

  def test_split_figure_preserves_cutoffs_and_maturity(self) -> None:
    periods = tuple(date(2025, month, 1) for month in range(1, 13))
    split = causal_split(periods)

    figure = split.plot()
    self.assertEqual(len(plot_split(split).axes), 1)
    axis = figure.axes[0]

    self.assertEqual(axis.get_yticklabels()[0].get_text(), "Test")
    self.assertEqual(axis.get_yticklabels()[2].get_text(), "Train")
    self.assertEqual(len(axis.collections), 3)
    self.assertEqual(len(axis.lines), 6)
    self.assertIn("3-report label horizon", axis.get_title())
    self.assertEqual(axis.get_xlabel(), "2025 report period")

  def test_example_figure_exposes_population_missingness_and_drift(self) -> None:
    examples = self._examples()
    before = examples.copy(deep=True)

    figure = plot_examples(examples)

    self.assertEqual(
      {axis.get_title() for axis in figure.axes},
      {
        "Fold outcome composition · log scale",
        "Binary event-rate drift",
        "Feature missingness by fold",
        "Median shift · train IQR units",
      },
    )
    self.assertEqual(figure.axes[0].get_yscale(), "log")
    self.assertIn(
      "Payment To Income",
      {tick.get_text() for tick in figure.axes[2].get_yticklabels()},
    )
    self.assertGreater(len(figure.axes[3].get_yticklabels()), 0)
    self.assertGreaterEqual(figure.get_size_inches()[1], 9)
    pd.testing.assert_frame_equal(examples, before)

  def test_example_figure_rejects_an_incomplete_frame(self) -> None:
    with self.assertRaisesRegex(ValueError, "missing required columns"):
      plot_examples(pd.DataFrame({"fold": ["train"]}))

    with self.assertRaisesRegex(ValueError, "at least one eligible cutoff"):
      plot_examples(self._examples().iloc[0:0])

  def test_example_figure_keeps_additional_fold_and_disposition_labels(self) -> None:
    examples = self._examples()
    extra = examples.iloc[[0]].copy()
    extra["fold"] = "monitor"
    extra["target_status"] = "manual_review"
    extra["target"] = None

    figure = plot_examples(pd.concat([examples, extra], ignore_index=True))

    self.assertIn("Monitor", {tick.get_text() for tick in figure.axes[0].get_xticklabels()})
    legend = figure.axes[0].get_legend()
    self.assertIsNotNone(legend)
    assert legend is not None
    self.assertIn("Manual Review", {text.get_text() for text in legend.texts})

  def test_baseline_figure_exposes_validation_evidence(self) -> None:
    baseline = Baseline(
      preprocessor=cast(Any, None),
      classifier=cast(Any, None),
      selected_depth=2,
      selected_learning_rate=0.05,
      selected_estimators=60,
      candidates=pd.DataFrame(
        {
          "max_depth": [1, 1, 1, 1, 2, 2, 2, 2],
          "learning_rate": [0.02, 0.02, 0.05, 0.05] * 2,
          "n_estimators": [60, 120, 60, 120] * 2,
          "auroc": [0.68, 0.69, 0.70, 0.71, 0.70, 0.71, 0.72, 0.73],
          "average_precision": [0.08, 0.09, 0.10, 0.11, 0.09, 0.10, 0.12, 0.11],
          "log_loss": [0.095, 0.09, 0.088, 0.085, 0.087, 0.084, 0.08, 0.082],
          "near_best": [False, False, False, False, False, True, True, True],
          "selected": [False, False, False, False, False, False, True, False],
        }
      ),
      reference=Metrics(100, 10, 0.1, 0.5, 0.1, 0.12, 0.09),
      validation=Metrics(100, 10, 0.1, 0.72, 0.12, 0.08, 0.07),
      calibration=pd.DataFrame(
        {
          "score_band": [1, 2, 3],
          "samples": [34, 33, 33],
          "events": [1, 3, 6],
          "mean_score": [0.03, 0.09, 0.18],
          "event_rate": [0.03, 0.09, 0.18],
        }
      ),
      importance=pd.DataFrame(
        {"feature": ["credit_score", "current_ltv"], "importance": [0.7, 0.3]}
      ),
    )

    figure = plot_baseline(baseline)

    self.assertEqual(
      {axis.get_title() for axis in figure.axes},
      {
        "Selected d2 · η 0.05 · 60 trees",
        "Ranking-metric tradeoff",
        "Score-band calibration · Brier 0.0700",
        "Selected model · permutation importance",
      },
    )
    self.assertIn("validation only", figure.get_suptitle())

    surface = plot_sensitivity(baseline)
    self.assertEqual([axis.get_title() for axis in surface.axes], ["Depth 1", "Depth 2"])
    self.assertIn("log-loss sensitivity", surface.get_suptitle())
    self.assertTrue(any("★" in text.get_text() for text in surface.axes[1].texts))

  @staticmethod
  def _audit() -> Audit:
    return Audit(
      source={},
      panel={},
      fields={},
      continuity=(
        {"report_period": "2025-01-31", "reported": 100},
        {"report_period": "2025-02-28", "reported": 94},
        {"report_period": "2025-03-31", "reported": 90},
      ),
      states={
        "delinquency:current": 250,
        "delinquency:1-29": 20,
        "delinquency:30-59": 8,
        "delinquency:60-89": 4,
        "delinquency:90+": 2,
        "zero_balance:1": 10,
        "zero_balance:4": 1,
        "zero_balance:1+99": 1,
      },
      transitions={
        "delinquency:current -> delinquency:current": 180,
        "delinquency:current -> delinquency:1-29": 12,
        "delinquency:1-29 -> delinquency:current": 8,
        "delinquency:1-29 -> delinquency:30-59": 4,
        "delinquency:30-59 -> delinquency:60-89": 2,
        "delinquency:60-89 -> zero_balance:4": 1,
        "zero_balance:1+99 -> zero_balance:1+99": 1,
      },
      targets=(
        {
          "status": "derived",
          "counts": {
            "positive": 12,
            "negative": 220,
            "competing_event": 14,
            "right_censored": 30,
            "ineligible_at_cutoff": 20,
            "missing_followup": 0,
          },
        },
      ),
    )

  @staticmethod
  def _examples() -> pd.DataFrame:
    categories = {
      "credit_score_status",
      "geography",
      "vehicle_new_used",
      "vehicle_type",
      "credit_score_type",
      "income_verification",
      "employment_verification",
      "payment_type",
    }
    rows = []
    for fold_index, fold in enumerate(("train", "validation", "test")):
      for row_index in range(4):
        status = ("positive", "negative", "negative", "competing_event")[row_index]
        row: dict[str, object] = {
          "fold": fold,
          "target_status": status,
          "target": 1 if status == "positive" else 0 if status == "negative" else None,
        }
        for feature_index, feature in enumerate(FEATURE_COLUMNS):
          if feature in categories:
            row[feature] = f"group-{row_index % 2}"
          else:
            row[feature] = float(feature_index + row_index + fold_index)
        if row_index < fold_index + 1:
          row["payment_to_income"] = None
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["target"] = frame["target"].astype(pd.Int8Dtype())
    return frame


if __name__ == "__main__":
  unittest.main()
