from __future__ import annotations

import unittest
from datetime import date
from typing import Any, cast

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from quantcredit.audits import Audit
from quantcredit.baselines import Baseline, Evaluation, Exposure, Metrics
from quantcredit.cashflows import Tranche, project_collateral, run_waterfall
from quantcredit.challengers import Artifacts, GraphEvaluation, GraphStudy
from quantcredit.decisions import Decision
from quantcredit.populations import FEATURE_COLUMNS
from quantcredit.splits import causal_split
from quantcredit.visuals import (
  plot_audit,
  plot_baseline,
  plot_deal,
  plot_decision,
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

  def test_graph_figures_exclude_invalid_validation_controls(self) -> None:
    results = pd.DataFrame(
      [
        {
          "arm": arm,
          "seed": None,
          "valid": valid,
          "log_loss": loss,
          "average_precision": precision,
        }
        for arm, valid, loss, precision in (
          ("raw_gbm", True, 0.04, 0.35),
          ("enriched_gbm", True, 0.041, 0.34),
          ("true_gine", True, 0.05, 0.16),
          ("node_local", False, 0.40, 0.01),
        )
      ]
    )
    calibration = pd.DataFrame(
      {
        "arm": results["arm"],
        "mean_score": [0.01] * len(results),
        "event_rate": [0.012] * len(results),
      }
    )
    study = GraphStudy(
      results,
      calibration,
      pd.DataFrame(),
      pd.DataFrame(),
      "no_value_from_current_ontology",
      cast(Artifacts, None),
    )
    evaluation = GraphEvaluation(
      results,
      calibration,
      "no_value_from_current_ontology",
      "no_value_from_current_ontology",
    )

    self.assertEqual(len(study.plot().axes), 2)
    self.assertIn("1 invalid constant control", study.plot().get_suptitle())
    self.assertEqual(len(evaluation.plot().axes), 2)

  def test_decision_figure_exposes_effects_residuals_and_frontier(self) -> None:
    effects = pd.DataFrame(
      [
        {
          "feature": feature,
          "band": str(band),
          "mean_score": 0.01 * band,
          "event_rate": 0.012 * band,
        }
        for feature in ("credit_score", "current_ltv")
        for band in range(1, 4)
      ]
    )
    cohorts = pd.DataFrame(
      {
        "feature": ["geography", "geography"],
        "value": ["east", "west"],
        "residual": [0.01, -0.005],
      }
    )
    frontier = pd.DataFrame(
      [
        {
          "policy": policy,
          "target_excluded_share": share,
          "excluded_balance_share": share,
          "event_exposure_avoided": share * lift,
        }
        for policy, lift in (("GBM score", 2.0), ("Lowest credit score", 1.3))
        for share in (0.0, 0.1, 0.2)
      ]
    )
    decision = Decision(effects, cohorts, frontier)

    figure = decision.plot()

    self.assertEqual(len(plot_decision(decision).axes), 4)
    self.assertIn("GBM avoids 20.0%", figure.get_suptitle())
    self.assertIn("Matched-balance", figure.axes[3].get_title())

  def test_deal_figure_exposes_cash_timing_and_tranche_runoff(self) -> None:
    collateral = project_collateral(
      balance=100,
      annual_rate=0.08,
      months=12,
      annual_default_rate=0.04,
      annual_prepayment_rate=0.10,
      recovery_rate=0.40,
    )
    deal = run_waterfall(
      collateral,
      (Tranche("Senior", 80, 0.04), Tranche("Equity", 20, 0)),
    )

    figure = deal.plot()

    self.assertEqual(len(plot_deal(deal).axes), 2)
    self.assertEqual(
      {axis.get_title() for axis in figure.axes},
      {
        "Collateral cash and loss timing",
        "Tranche principal runoff · senior paid first",
      },
    )

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
    self.assertIn("Population contracts 10.0%", figure.get_suptitle())
    self.assertIn(
      "Zero balance 1+99",
      {tick.get_text() for tick in figure.axes[1].get_yticklabels()},
    )
    self.assertIn(
      "Zero bal. 1+99",
      {tick.get_text() for tick in figure.axes[2].get_xticklabels()},
    )
    self.assertTrue(all(not tick.get_rotation() for tick in figure.axes[2].get_xticklabels()))

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
    self.assertIn("Labels mature before the next fold", axis.get_title())
    self.assertIn("3-report horizon", axis.get_title())
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
    self.assertIn("Held out", {text.get_text() for text in figure.axes[1].texts})
    self.assertGreater(len({patch.get_hatch() for patch in figure.axes[0].patches}), 1)
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
      classifier=HistGradientBoostingClassifier(
        max_depth=2,
        learning_rate=0.05,
        max_iter=60,
      ),
      candidates=pd.DataFrame(
        {
          "max_depth": [1, 1, 1, 1, 2, 2, 2, 2],
          "learning_rate": [0.02, 0.02, 0.05, 0.05] * 2,
          "n_estimators": [60, 120, 60, 120] * 2,
          "auroc": [0.68, 0.69, 0.70, 0.71, 0.70, 0.71, 0.72, 0.73],
          "average_precision": [0.08, 0.09, 0.10, 0.11, 0.09, 0.10, 0.12, 0.11],
          "log_loss": [0.095, 0.09, 0.088, 0.085, 0.087, 0.084, 0.08, 0.082],
          "samples": [100] * 8,
          "events": [10] * 8,
          "event_rate": [0.1] * 8,
          "brier_score": [0.07] * 8,
          "near_best": [False, False, False, False, False, True, True, True],
          "selected": [False, False, False, False, False, False, True, False],
        }
      ),
      reference_probability=0.1,
      reference=Metrics(100, 10, 0.1, 0.5, 0.1, 0.12, 0.09),
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
    self.assertIn("selected within validation uncertainty", figure.get_suptitle())
    self.assertIsNone(figure.axes[0].get_legend())
    self.assertIsNone(figure.axes[1].get_legend())
    self.assertIsNone(figure.axes[2].get_legend())
    self.assertGreaterEqual(len(figure.axes[1].texts), 3)
    self.assertEqual(len(figure.axes[2].lines), 2)

    surface = plot_sensitivity(baseline)
    self.assertEqual([axis.get_title() for axis in surface.axes], ["Depth 1", "Depth 2"])
    self.assertIn("Near-best validation region favors depth 2", surface.get_suptitle())
    self.assertTrue(any("★" in text.get_text() for text in surface.axes[1].texts))

    evaluation = Evaluation(
      baseline=baseline,
      cutoff=date(2025, 9, 30),
      labels_observed_through=date(2025, 12, 31),
      metrics=Metrics(90, 12, 12 / 90, 0.70, 0.15, 0.09, 0.08),
      reference=Metrics(90, 12, 12 / 90, 0.5, 12 / 90, 0.14, 0.10),
      calibration=baseline.calibration,
      exposure=Exposure(
        samples=90,
        exposure_samples=90,
        total_exposure=900_000,
        expected_event_exposure=90_000,
        observed_event_exposure=100_000,
        bands=pd.DataFrame(
          {
            "score_band": [1, 2],
            "samples": [45, 45],
            "events": [2, 10],
            "exposure_samples": [45, 45],
            "total_exposure": [400_000, 500_000],
            "mean_pd": [0.02, 0.18],
            "exposure_weighted_pd": [0.02, 0.164],
            "expected_event_exposure": [8_000, 82_000],
            "observed_event_exposure": [15_000, 85_000],
          }
        ),
      ),
    )
    evaluation_figure = evaluation.plot()
    self.assertEqual(
      {axis.get_title() for axis in evaluation_figure.axes},
      {
        "Log loss · lower is better",
        "Brier score · lower is better",
        "Ranking stability",
        "Test score-band calibration · Brier 0.0800",
      },
    )
    self.assertIn("Frozen test AUROC 0.700 vs 0.720 validation", evaluation_figure.get_suptitle())
    self.assertTrue(all(axis.get_legend() is None for axis in evaluation_figure.axes))
    self.assertEqual(evaluation_figure.get_size_inches()[1], 8)

    exposure_figure = evaluation.exposure.plot()
    self.assertEqual(
      {axis.get_title() for axis in exposure_figure.axes},
      {
        "Outstanding balance by risk band",
        "Predicted versus observed event exposure",
      },
    )
    self.assertIn("modeled event exposure—not ultimate loss", exposure_figure.get_suptitle())
    self.assertTrue(all(axis.get_legend() is None for axis in exposure_figure.axes))
    self.assertTrue(all(not axis.spines["top"].get_visible() for axis in figure.axes))
    self.assertTrue(all(not axis.spines["right"].get_visible() for axis in figure.axes))

  def test_audit_figure_renders_a_rejected_target_decision(self) -> None:
    audit = self._audit()
    rejected = {
      **audit.targets[0],
      "status": "rejected",
      "counts": {
        "positive": 0,
        "negative": 1,
        "competing_event": 0,
        "right_censored": 0,
        "ineligible_at_cutoff": 0,
        "missing_followup": 0,
      },
    }

    figure = plot_audit(
      Audit(
        source=audit.source,
        panel=audit.panel,
        fields=audit.fields,
        continuity=audit.continuity,
        states=audit.states,
        transitions=audit.transitions,
        targets=(rejected,),
      )
    )

    self.assertIn("rejected", figure.axes[3].get_title())

    missing = {**rejected, "name": "another_target"}
    with self.assertRaisesRegex(ValueError, "missing the serious-delinquency"):
      plot_audit(
        Audit(
          source=audit.source,
          panel=audit.panel,
          fields=audit.fields,
          continuity=audit.continuity,
          states=audit.states,
          transitions=audit.transitions,
          targets=(missing,),
        )
      )

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
          "name": "serious_delinquency_or_chargeoff",
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
        status = (
          "held_out"
          if fold == "test"
          else ("positive", "negative", "negative", "competing_event")[row_index]
        )
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
