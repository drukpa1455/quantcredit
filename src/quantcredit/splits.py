"""Causal chronological cutoffs for fixed-report-horizon outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from itertools import pairwise


@dataclass(frozen=True)
class CausalSplit:
  train_cutoffs: tuple[date, ...]
  validation_cutoff: date
  test_cutoff: date
  train_labels_observed_through: date
  validation_labels_observed_through: date
  test_labels_observed_through: date
  horizon_reports: int

  def summary(self) -> dict[str, str | int | list[str]]:
    return {
      "horizon_reports": self.horizon_reports,
      "train_cutoffs": [cutoff.isoformat() for cutoff in self.train_cutoffs],
      "train_labels_observed_through": self.train_labels_observed_through.isoformat(),
      "validation_cutoff": self.validation_cutoff.isoformat(),
      "validation_labels_observed_through": (
        self.validation_labels_observed_through.isoformat()
      ),
      "test_cutoff": self.test_cutoff.isoformat(),
      "test_labels_observed_through": self.test_labels_observed_through.isoformat(),
    }


def causal_split(report_periods: tuple[date, ...], *, horizon_reports: int = 3) -> CausalSplit:
  """Choose train, validation, and test cutoffs with fully matured prior labels."""
  if horizon_reports <= 0:
    raise ValueError("label horizon must be positive")
  if report_periods != tuple(sorted(set(report_periods))):
    raise ValueError("report periods must be unique and increasing")
  if not _consecutive_months(report_periods):
    raise ValueError("report periods must be consecutive months")

  required = 3 * horizon_reports + 3
  if len(report_periods) < required:
    raise ValueError(
      f"causal train/validation/test split requires at least {required} report periods"
    )

  test_index = len(report_periods) - horizon_reports - 1
  validation_index = test_index - horizon_reports - 1
  train_end = validation_index - horizon_reports
  train_cutoffs = report_periods[:train_end]
  return CausalSplit(
    train_cutoffs=train_cutoffs,
    validation_cutoff=report_periods[validation_index],
    test_cutoff=report_periods[test_index],
    train_labels_observed_through=report_periods[train_end - 1 + horizon_reports],
    validation_labels_observed_through=report_periods[validation_index + horizon_reports],
    test_labels_observed_through=report_periods[test_index + horizon_reports],
    horizon_reports=horizon_reports,
  )


def _consecutive_months(periods: tuple[date, ...]) -> bool:
  months = tuple(period.year * 12 + period.month for period in periods)
  return all(right == left + 1 for left, right in pairwise(months))
