from __future__ import annotations

import unittest
from datetime import date
from decimal import Decimal

from quantcredit.panel import AssetKey, LoanSnapshot, SnapshotKey, SourceField, ZeroBalanceCode
from quantcredit.targets import (
  LoanState,
  TargetResult,
  eligible_at_cutoff,
  loan_state,
  serious_delinquency_target,
)

CURRENT = LoanState(0)


class TargetTests(unittest.TestCase):
  def test_derives_state_from_one_snapshot(self) -> None:
    snapshot = LoanSnapshot(
      SnapshotKey(AssetKey("0000000000", "PRIVATE"), date(2025, 1, 31)),
      "accession",
      "auto",
      date(2025, 1, 1),
      (
        SourceField("currentDelinquencyStatus", ("60",)),
        SourceField("zeroBalanceCode", ("4",)),
        SourceField("chargedoffPrincipalAmount", ("100",)),
      ),
    )

    self.assertEqual(
      loan_state(snapshot),
      LoanState(60, (ZeroBalanceCode.CHARGED_OFF,), Decimal("100"), None),
    )

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

  def test_cutoff_eligibility_depends_only_on_the_observed_state(self) -> None:
    self.assertTrue(eligible_at_cutoff(CURRENT))
    self.assertFalse(eligible_at_cutoff(None))
    self.assertFalse(eligible_at_cutoff(LoanState(None)))
    self.assertFalse(eligible_at_cutoff(LoanState(60)))
    self.assertFalse(
      eligible_at_cutoff(LoanState(0, (ZeroBalanceCode.PREPAID_OR_MATURED,)))
    )
    self.assertFalse(eligible_at_cutoff(LoanState(0, charged_off_principal=Decimal("1"))))

  def test_rejects_nonpositive_horizon(self) -> None:
    with self.assertRaisesRegex(ValueError, "horizon"):
      serious_delinquency_target([CURRENT], 0, horizon_reports=0)

  def test_rejects_cutoff_outside_history(self) -> None:
    with self.assertRaisesRegex(ValueError, "cutoff"):
      serious_delinquency_target([CURRENT], 1)


if __name__ == "__main__":
  unittest.main()
