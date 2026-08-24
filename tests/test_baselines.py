from __future__ import annotations

import unittest

import pandas as pd

from quantcredit.baselines import fit_baseline
from quantcredit.populations import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES


class BaselineTests(unittest.TestCase):
  def test_fits_preprocessing_on_train_and_selects_validation_log_loss(self) -> None:
    examples = self._examples()
    before = examples.copy(deep=True)

    baseline = fit_baseline(examples, depths=(1, 2), n_estimators=20)
    without_test = fit_baseline(
      examples.loc[examples["fold"] != "test"], depths=(1, 2), n_estimators=20
    )

    selected = baseline.candidates.loc[baseline.candidates["log_loss"].idxmin()]
    self.assertEqual(baseline.selected_depth, selected["max_depth"])
    self.assertEqual(baseline.validation.samples, 40)
    self.assertEqual(baseline.validation.events, 5)
    self.assertGreater(baseline.reference.log_loss, baseline.validation.log_loss)
    self.assertEqual(int(baseline.calibration["samples"].sum()), 40)
    self.assertEqual(int(baseline.calibration["events"].sum()), 5)
    self.assertEqual(len(baseline.candidates), 2)
    self.assertFalse(baseline.importance.empty)
    pd.testing.assert_frame_equal(baseline.candidates, without_test.candidates)
    pd.testing.assert_frame_equal(baseline.calibration, without_test.calibration)

    imputer = baseline.preprocessor.named_transformers_["numeric"]
    credit_score_index = NUMERIC_FEATURES.index("credit_score")
    train_median = examples.loc[examples["fold"] == "train", "credit_score"].median()
    self.assertEqual(imputer.statistics_[credit_score_index], train_median)
    pd.testing.assert_frame_equal(examples, before)

  def test_rejects_invalid_protocols(self) -> None:
    examples = self._examples()
    with self.assertRaisesRegex(ValueError, "missing required columns"):
      fit_baseline(examples.drop(columns="credit_score"), n_estimators=5)
    with self.assertRaisesRegex(ValueError, "depths"):
      fit_baseline(examples, depths=(), n_estimators=5)

    examples.loc[examples["fold"] == "validation", "target"] = 0
    with self.assertRaisesRegex(ValueError, "both target classes"):
      fit_baseline(examples, n_estimators=5)

  @staticmethod
  def _examples() -> pd.DataFrame:
    rows = []
    for fold, count, shift in (("train", 80, 0), ("validation", 40, 3), ("test", 20, 6)):
      for index in range(count):
        target = int(index % 8 == 0)
        row: dict[str, object] = {"fold": fold, "target": target}
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
