from __future__ import annotations

import unittest
from decimal import Decimal

from quantcredit.panel import ZeroBalanceCode
from quantcredit.targets import LoanState, TargetResult, serious_delinquency_target

CURRENT = LoanState(0)


class TargetTests(unittest.TestCase):
  def test_classifies_event_negative_competing_and_censoring(self) -> None:
    cases = (
      ([CURRENT, LoanState(60), CURRENT, CURRENT], 0, TargetResult.POSITIVE),
      (
        [
          CURRENT,
          LoanState(None, (ZeroBalanceCode.CHARGED_OFF,), Decimal("10")),
          CURRENT,
          CURRENT,
        ],
        0,
        TargetResult.POSITIVE,
      ),
      ([CURRENT, CURRENT, CURRENT, CURRENT], 0, TargetResult.NEGATIVE),
      (
        [CURRENT, LoanState(0, (ZeroBalanceCode.PREPAID_OR_MATURED,)), CURRENT, CURRENT],
        0,
        TargetResult.COMPETING,
      ),
      (
        [CURRENT, LoanState(60, (ZeroBalanceCode.PREPAID_OR_MATURED,)), CURRENT, CURRENT],
        0,
        TargetResult.COMPETING,
      ),
      ([CURRENT, None, CURRENT, CURRENT], 0, TargetResult.CENSORED),
      ([CURRENT, CURRENT, CURRENT], 0, TargetResult.RIGHT_CENSORED),
      ([LoanState(60), CURRENT, CURRENT, CURRENT], 0, TargetResult.INELIGIBLE),
    )
    for history, cutoff, expected in cases:
      with self.subTest(expected=expected):
        self.assertEqual(
          serious_delinquency_target(history, cutoff, horizon_reports=3), expected
        )

  def test_rejects_nonpositive_horizon(self) -> None:
    with self.assertRaisesRegex(ValueError, "horizon"):
      serious_delinquency_target([CURRENT], 0, horizon_reports=0)

  def test_rejects_cutoff_outside_history(self) -> None:
    with self.assertRaisesRegex(ValueError, "cutoff"):
      serious_delinquency_target([CURRENT], 1)


if __name__ == "__main__":
  unittest.main()
