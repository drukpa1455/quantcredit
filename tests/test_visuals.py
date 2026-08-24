from __future__ import annotations

import unittest
from datetime import date

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt

from quantcredit.splits import causal_split
from quantcredit.visuals import plot_audit, plot_split, sapphire


class VisualTests(unittest.TestCase):
  def tearDown(self) -> None:
    plt.close("all")

  def test_sapphire_is_scoped(self) -> None:
    before = mpl.rcParams["figure.facecolor"]

    with sapphire():
      self.assertEqual(mpl.rcParams["figure.facecolor"], "#212c2a")

    self.assertEqual(mpl.rcParams["figure.facecolor"], before)

  def test_audit_figure_exposes_all_four_aggregate_questions(self) -> None:
    figure = plot_audit(self._audit())

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

    figure = plot_split(split)
    axis = figure.axes[0]

    self.assertEqual(axis.get_yticklabels()[0].get_text(), "Test")
    self.assertEqual(axis.get_yticklabels()[2].get_text(), "Train")
    self.assertEqual(len(axis.collections), 3)
    self.assertEqual(len(axis.lines), 6)
    self.assertIn("3-report label horizon", axis.get_title())
    self.assertEqual(axis.get_xlabel(), "2025 report period")

  @staticmethod
  def _audit() -> dict[str, object]:
    return {
      "continuity": [
        {"report_period": "2025-01-31", "reported": 100},
        {"report_period": "2025-02-28", "reported": 94},
        {"report_period": "2025-03-31", "reported": 90},
      ],
      "states": {
        "delinquency:current": 250,
        "delinquency:1-29": 20,
        "delinquency:30-59": 8,
        "delinquency:60-89": 4,
        "delinquency:90+": 2,
        "zero_balance:1": 10,
        "zero_balance:4": 1,
        "zero_balance:1+99": 1,
      },
      "transitions": {
        "delinquency:current -> delinquency:current": 180,
        "delinquency:current -> delinquency:1-29": 12,
        "delinquency:1-29 -> delinquency:current": 8,
        "delinquency:1-29 -> delinquency:30-59": 4,
        "delinquency:30-59 -> delinquency:60-89": 2,
        "delinquency:60-89 -> zero_balance:4": 1,
        "zero_balance:1+99 -> zero_balance:1+99": 1,
      },
      "targets": [
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
        }
      ],
    }


if __name__ == "__main__":
  unittest.main()
