from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from quantcredit.challengers import _comparisons, _decision, _record, _validate_protocol


class ChallengerTests(unittest.TestCase):
  def test_constant_scores_are_explicitly_invalid(self) -> None:
    target = np.array([0, 0, 1, 1], dtype=np.int64)
    examples = pd.DataFrame({"ending_balance": [100, 100, 100, 100]})

    result = _record(
      "node_local",
      7,
      target,
      np.full(4, 0.5),
      10,
      examples,
    )

    self.assertFalse(result["valid"])
    self.assertEqual(result["status"], "invalid_constant_scores")

  def test_structural_claim_requires_three_valid_seed_wins(self) -> None:
    records = [
      self._result("raw_gbm", None, 0.20, 0.40, 0.10),
      self._result("enriched_gbm", None, 0.20, 0.40, 0.10),
      self._result("true_gine", None, 0.18, 0.45, 0.09),
      self._result("node_local", None, 0.21, 0.39, 0.11),
      self._result("erased_gine", None, 0.21, 0.39, 0.11),
      self._result("false_gine", None, 0.21, 0.39, 0.11),
    ]
    for seed in (7, 19, 43):
      records.extend(
        [
          self._result("true_gine", seed, 0.18, 0.45, 0.09),
          self._result("node_local", seed, 0.21, 0.39, 0.11),
          self._result("erased_gine", seed, 0.21, 0.39, 0.11),
          self._result("false_gine", seed, 0.21, 0.39, 0.11),
        ]
      )

    self.assertEqual(_decision(pd.DataFrame(records)), "retain_existing_gine")
    records[-1]["valid"] = False
    self.assertEqual(_decision(pd.DataFrame(records)), "reject_graph")
    self.assertEqual(_decision(pd.DataFrame(records[:-4])), "reject_graph")

  def test_paired_comparison_reports_effect_and_uncertainty(self) -> None:
    target = np.asarray(([0, 0, 0, 1] * 25), dtype=np.int64)
    reference = np.linspace(0.01, 0.6, len(target))
    challenger = np.clip(reference + np.where(target == 1, 0.05, -0.01), 0.001, 0.999)

    result = _comparisons(
      target,
      {"enriched_gbm": reference, "true_gine": challenger},
      bootstrap_samples=20,
      seed=3,
    ).iloc[0]

    self.assertEqual(result["reference"], "enriched_gbm")
    self.assertEqual(result["bootstrap_samples"], 20)
    self.assertEqual(result["bootstrap_seed"], 3)
    self.assertLess(result["log_loss_delta"], 0)
    self.assertGreater(result["average_precision_delta"], 0)
    self.assertGreater(result["average_precision_delta_se"], 0)

  def test_protocol_rejects_ambiguous_or_unbounded_values(self) -> None:
    invalid = (
      ((7, 7), 16, 60, 0.01),
      ((7,), 0, 60, 0.01),
      ((7,), 16, 0, 0.01),
      ((7,), 16, 60, 0.0),
    )
    for arguments in invalid:
      with self.subTest(arguments=arguments), self.assertRaises(ValueError):
        _validate_protocol(*arguments)

  @staticmethod
  def _result(
    arm: str,
    seed: int | None,
    log_loss: float,
    average_precision: float,
    brier_score: float,
  ) -> dict[str, object]:
    return {
      "arm": arm,
      "seed": seed,
      "valid": True,
      "log_loss": log_loss,
      "average_precision": average_precision,
      "brier_score": brier_score,
    }


if __name__ == "__main__":
  unittest.main()
