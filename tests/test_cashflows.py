from __future__ import annotations

import unittest

import pandas as pd

from quantcredit.cashflows import Tranche, project_collateral, run_waterfall


class CashflowTests(unittest.TestCase):
  def test_projection_reconciles_principal_default_recovery_and_loss(self) -> None:
    collateral = project_collateral(
      balance=1_000,
      annual_rate=0.08,
      months=12,
      annual_default_rate=0.12,
      annual_prepayment_rate=0.10,
      recovery_rate=0.40,
      recovery_lag=2,
    )

    self.assertAlmostEqual(float(collateral.iloc[-1]["ending_balance"]), 0)
    self.assertAlmostEqual(
      float(collateral[["scheduled_principal", "prepayment", "default"]].sum().sum()),
      1_000,
    )
    self.assertAlmostEqual(
      float(collateral["recovery"].sum()), float(collateral["default"].sum()) * 0.40
    )
    self.assertAlmostEqual(
      float(collateral["loss"].sum()), float(collateral["default"].sum()) * 0.60
    )
    self.assertEqual(len(collateral), 14)

  def test_waterfall_pays_senior_first_and_allocates_loss_junior_first(self) -> None:
    collateral = pd.DataFrame([{
      "month": 1,
      "beginning_balance": 100.0,
      "interest": 1.0,
      "scheduled_principal": 5.0,
      "prepayment": 0.0,
      "default": 15.0,
      "recovery": 0.0,
      "loss": 15.0,
      "ending_balance": 80.0,
    }])
    tranches = (
      Tranche("Senior", 70, 0),
      Tranche("Mezzanine", 20, 0),
      Tranche("Equity", 10, 0),
    )

    deal = run_waterfall(collateral, tranches)
    cashflows = deal.cashflows.set_index("tranche")

    self.assertEqual(cashflows.loc["Senior", "principal"], 5)
    self.assertEqual(cashflows.loc["Senior", "ending_balance"], 65)
    self.assertEqual(cashflows.loc["Mezzanine", "loss"], 5)
    self.assertEqual(cashflows.loc["Equity", "loss"], 10)
    self.assertEqual(cashflows.loc["Equity", "residual"], 1)
    self.assertEqual(set(deal.summary()["status"]), {"illustrative scenario, not a deal valuation"})

  def test_zero_hazard_projection_returns_principal_and_par_yield(self) -> None:
    collateral = project_collateral(
      balance=120,
      annual_rate=0,
      months=12,
      annual_default_rate=0,
      annual_prepayment_rate=0,
      recovery_rate=0,
    )
    deal = run_waterfall(collateral, (Tranche("Senior", 120, 0),))

    self.assertAlmostEqual(float(collateral["scheduled_principal"].sum()), 120)
    self.assertAlmostEqual(float(deal.summary().iloc[0]["scenario_yield"]), 0)
    self.assertAlmostEqual(float(deal.summary().iloc[0]["cash_multiple"]), 1)

  def test_zero_lag_recovery_is_paid_in_the_default_month(self) -> None:
    collateral = project_collateral(
      balance=100,
      annual_rate=0,
      months=2,
      annual_default_rate=0.12,
      annual_prepayment_rate=0,
      recovery_rate=0.40,
      recovery_lag=0,
    )

    self.assertAlmostEqual(
      float(collateral["recovery"].sum()), float(collateral["default"].sum()) * 0.40
    )
    self.assertAlmostEqual(
      float(collateral["default"].sum()),
      float(collateral["recovery"].sum() + collateral["loss"].sum()),
    )

  def test_rejects_invalid_assumptions_and_waterfalls(self) -> None:
    arguments = {
      "balance": 100,
      "annual_rate": 0.05,
      "months": 12,
      "annual_default_rate": 0.05,
      "annual_prepayment_rate": 0.10,
      "recovery_rate": 0.40,
    }
    with self.assertRaisesRegex(ValueError, "months"):
      project_collateral(**(arguments | {"months": 0}))  # type: ignore[arg-type]
    with self.assertRaisesRegex(ValueError, "annual_default_rate"):
      project_collateral(**(arguments | {"annual_default_rate": 2}))  # type: ignore[arg-type]

    collateral = project_collateral(**arguments)
    with self.assertRaisesRegex(ValueError, "unique"):
      run_waterfall(collateral, (Tranche("A", 40, 0), Tranche("A", 40, 0)))
    with self.assertRaisesRegex(ValueError, "must equal"):
      run_waterfall(collateral, (Tranche("A", 101, 0),))
    invalid = collateral.copy()
    invalid.loc[0, "loss"] = -1
    with self.assertRaisesRegex(ValueError, "cannot be negative"):
      run_waterfall(invalid, (Tranche("A", 100, 0),))


if __name__ == "__main__":
  unittest.main()
