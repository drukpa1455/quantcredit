from __future__ import annotations

import unittest
from datetime import date

from quantcredit.splits import chronological_split


class SplitTests(unittest.TestCase):
  def test_keeps_each_prior_label_horizon_before_the_next_cutoff(self) -> None:
    periods = tuple(date(2025, month, 1) for month in range(1, 13))

    split = chronological_split(periods, horizon_reports=3)

    self.assertEqual(split.train_cutoffs, (date(2025, 1, 1),))
    self.assertEqual(split.train_labels_observed_through, date(2025, 4, 1))
    self.assertEqual(split.validation_cutoff, date(2025, 5, 1))
    self.assertEqual(split.validation_labels_observed_through, date(2025, 8, 1))
    self.assertEqual(split.test_cutoff, date(2025, 9, 1))
    self.assertEqual(split.test_labels_observed_through, date(2025, 12, 1))

  def test_uses_all_earlier_matured_cutoffs_when_history_is_longer(self) -> None:
    periods = tuple(
      date(year, month, 1) for year in (2024, 2025) for month in range(1, 13)
    )

    split = chronological_split(periods, horizon_reports=3)

    self.assertEqual(split.train_cutoffs[0], date(2024, 1, 1))
    self.assertEqual(split.train_cutoffs[-1], date(2025, 1, 1))
    self.assertEqual(split.validation_cutoff, date(2025, 5, 1))
    self.assertEqual(split.test_cutoff, date(2025, 9, 1))

  def test_rejects_noncausal_or_ambiguous_period_sequences(self) -> None:
    twelve = tuple(date(2025, month, 1) for month in range(1, 13))
    cases = (
      (twelve[:11], "at least 12"),
      (twelve[:5] + twelve[6:], "consecutive"),
      ((twelve[1], twelve[0], *twelve[2:]), "unique and increasing"),
      ((twelve[0], twelve[0], *twelve[2:]), "unique and increasing"),
    )
    for periods, message in cases:
      with self.subTest(message=message):
        with self.assertRaisesRegex(ValueError, message):
          chronological_split(periods, horizon_reports=3)

    with self.assertRaisesRegex(ValueError, "positive"):
      chronological_split(twelve, horizon_reports=0)


if __name__ == "__main__":
  unittest.main()
