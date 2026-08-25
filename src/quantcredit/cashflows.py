"""Explicit collateral scenarios and a simple sequential tranche waterfall."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
from pandas import DataFrame

if TYPE_CHECKING:
  from matplotlib.figure import Figure

_COLLATERAL_COLUMNS = {
  "month",
  "beginning_balance",
  "interest",
  "scheduled_principal",
  "prepayment",
  "default",
  "recovery",
  "loss",
  "ending_balance",
}


@dataclass(frozen=True)
class Tranche:
  """One claim in senior-to-junior waterfall order."""

  name: str
  balance: float
  annual_rate: float
  price: float | None = None

  def __post_init__(self) -> None:
    if not self.name.strip():
      raise ValueError("tranche name cannot be empty")
    _positive(self.balance, "tranche balance")
    _rate(self.annual_rate, "tranche annual_rate")
    if self.price is not None:
      _positive(self.price, "tranche price")

  @property
  def purchase_price(self) -> float:
    return self.balance if self.price is None else self.price


@dataclass(frozen=True)
class Deal:
  """Aggregate output of one declared collateral and waterfall scenario."""

  collateral: DataFrame
  cashflows: DataFrame
  tranches: tuple[Tranche, ...]
  annual_fee_rate: float

  def summary(self) -> DataFrame:
    records = []
    for tranche in self.tranches:
      frame = self.cashflows.loc[self.cashflows["tranche"] == tranche.name]
      total_cash = float(frame["total_cash"].sum())
      records.append(
        {
          "tranche": tranche.name,
          "original_balance": tranche.balance,
          "purchase_price": tranche.purchase_price,
          "interest": float(frame["interest"].sum()),
          "principal": float(frame["principal"].sum()),
          "residual": float(frame["residual"].sum()),
          "loss": float(frame["loss"].sum()),
          "interest_shortfall": float(frame.iloc[-1]["interest_shortfall"]),
          "ending_balance": float(frame.iloc[-1]["ending_balance"]),
          "total_cash": total_cash,
          "cash_multiple": total_cash / tranche.purchase_price,
          "scenario_yield": _annual_yield(tranche.purchase_price, frame["total_cash"]),
          "status": "illustrative scenario, not a deal valuation",
        }
      )
    return DataFrame(records)

  def plot(self) -> Figure:
    """Render collateral cash generation and tranche balance runoff."""
    from quantcredit.visuals import plot_deal

    return plot_deal(self)


def project_collateral(
  *,
  balance: float,
  annual_rate: float,
  months: int,
  annual_default_rate: float,
  annual_prepayment_rate: float,
  recovery_rate: float,
  recovery_lag: int = 3,
) -> DataFrame:
  """Project one aggregate amortizing pool from explicit constant-rate assumptions."""
  _positive(balance, "balance")
  _rate(annual_rate, "annual_rate")
  _rate(annual_default_rate, "annual_default_rate")
  _rate(annual_prepayment_rate, "annual_prepayment_rate")
  _rate(recovery_rate, "recovery_rate")
  if months <= 0:
    raise ValueError("months must be positive")
  if recovery_lag < 0:
    raise ValueError("recovery_lag cannot be negative")

  monthly_rate = annual_rate / 12
  monthly_default = _monthly(annual_default_rate)
  monthly_prepayment = _monthly(annual_prepayment_rate)
  payment = _payment(balance, monthly_rate, months)
  recoveries = [0.0] * (months + recovery_lag + 1)
  outstanding = balance
  records = []

  for month in range(1, months + recovery_lag + 1):
    beginning = outstanding
    interest = beginning * monthly_rate if month <= months else 0.0
    scheduled = (
      min(max(payment - interest, 0.0), beginning) if month <= months else 0.0
    )
    after_scheduled = beginning - scheduled
    default = after_scheduled * monthly_default if month <= months else 0.0
    after_default = after_scheduled - default
    prepayment = after_default * monthly_prepayment if month <= months else 0.0
    if month == months:
      scheduled += after_default - prepayment
      after_default = prepayment
    outstanding = max(after_default - prepayment, 0.0)
    recovery = recoveries[month]
    if default:
      expected_recovery = default * recovery_rate
      if recovery_lag == 0:
        recovery += expected_recovery
      else:
        recoveries[month + recovery_lag] += expected_recovery
    records.append(
      {
        "month": month,
        "beginning_balance": beginning,
        "interest": interest,
        "scheduled_principal": scheduled,
        "prepayment": prepayment,
        "default": default,
        "recovery": recovery,
        "loss": default * (1 - recovery_rate),
        "ending_balance": outstanding,
      }
    )
  return DataFrame(records)


def run_waterfall(
  collateral: DataFrame,
  tranches: tuple[Tranche, ...],
  *,
  annual_fee_rate: float = 0.0,
) -> Deal:
  """Pay interest and principal senior-first and allocate loss junior-first."""
  missing = sorted(_COLLATERAL_COLUMNS - set(collateral.columns))
  if missing:
    raise ValueError(f"collateral is missing required columns: {', '.join(missing)}")
  if collateral.empty:
    raise ValueError("collateral must contain at least one month")
  if not tranches:
    raise ValueError("waterfall requires at least one tranche")
  names = [tranche.name for tranche in tranches]
  if len(set(names)) != len(names):
    raise ValueError("tranche names must be unique")
  _rate(annual_fee_rate, "annual_fee_rate")
  _validate_collateral(collateral)
  collateral_balance = float(collateral.iloc[0]["beginning_balance"])
  if not isclose(sum(tranche.balance for tranche in tranches), collateral_balance):
    raise ValueError("tranche balance must equal initial collateral balance")

  balances = [tranche.balance for tranche in tranches]
  shortfalls = [0.0] * len(tranches)
  records: list[dict[str, Any]] = []
  for raw_row in collateral.itertuples(index=False):
    collateral_row = cast(Any, raw_row)
    opening = balances.copy()
    asset_interest = float(collateral_row.interest)
    asset_principal = float(
      collateral_row.scheduled_principal
      + collateral_row.prepayment
      + collateral_row.recovery
    )
    asset_loss = float(collateral_row.loss)
    fee = min(
      asset_interest,
      float(collateral_row.beginning_balance) * annual_fee_rate / 12,
    )
    available_interest = asset_interest - fee
    paid_interest = [0.0] * len(tranches)
    paid_principal = [0.0] * len(tranches)
    allocated_loss = [0.0] * len(tranches)
    residual = [0.0] * len(tranches)

    for index, tranche in enumerate(tranches):
      due = opening[index] * tranche.annual_rate / 12 + shortfalls[index]
      paid_interest[index] = min(available_interest, due)
      available_interest -= paid_interest[index]
      shortfalls[index] = due - paid_interest[index]

    if available_interest:
      residual[-1] += available_interest

    remaining_loss = asset_loss
    for index in reversed(range(len(tranches))):
      allocated_loss[index] = min(balances[index], remaining_loss)
      balances[index] -= allocated_loss[index]
      remaining_loss -= allocated_loss[index]
    if remaining_loss > 1e-6:
      raise ValueError("collateral loss exceeds remaining tranche balance")

    remaining_principal = asset_principal
    for index in range(len(tranches)):
      paid_principal[index] = min(balances[index], remaining_principal)
      balances[index] -= paid_principal[index]
      remaining_principal -= paid_principal[index]

    if remaining_principal:
      residual[-1] += remaining_principal

    for index, tranche in enumerate(tranches):
      total_cash = paid_interest[index] + paid_principal[index] + residual[index]
      records.append(
        {
          "month": int(collateral_row.month),
          "tranche": tranche.name,
          "opening_balance": opening[index],
          "interest": paid_interest[index],
          "principal": paid_principal[index],
          "residual": residual[index],
          "loss": allocated_loss[index],
          "interest_shortfall": shortfalls[index],
          "ending_balance": balances[index],
          "total_cash": total_cash,
          "fee": fee if index == 0 else 0.0,
        }
      )

  return Deal(
    collateral=collateral.copy(),
    cashflows=DataFrame(records),
    tranches=tranches,
    annual_fee_rate=annual_fee_rate,
  )


def _monthly(annual_rate: float) -> float:
  return float(1 - (1 - annual_rate) ** (1 / 12))


def _payment(balance: float, monthly_rate: float, months: int) -> float:
  if monthly_rate == 0:
    return balance / months
  return balance * monthly_rate / (1 - (1 + monthly_rate) ** -months)


def _annual_yield(price: float, cashflows: pd.Series[Any]) -> float:
  values = [-price, *[float(value) for value in cashflows]]

  def npv(rate: float) -> float:
    return sum(value / (1 + rate) ** month for month, value in enumerate(values))

  low, high = -0.999, 10.0
  low_value, high_value = npv(low), npv(high)
  if low_value == 0:
    return (1 + low) ** 12 - 1
  if high_value == 0:
    return (1 + high) ** 12 - 1
  if low_value * high_value > 0:
    return float("nan")
  for _ in range(100):
    middle = (low + high) / 2
    value = npv(middle)
    if abs(value) < 1e-10:
      break
    if low_value * value <= 0:
      high = middle
    else:
      low = middle
      low_value = value
  return (1 + (low + high) / 2) ** 12 - 1


def _validate_collateral(collateral: DataFrame) -> None:
  numeric = collateral[list(_COLLATERAL_COLUMNS)].apply(pd.to_numeric, errors="coerce")
  if numeric.isna().any().any() or not numeric.map(isfinite).all().all():
    raise ValueError("collateral values must be finite numbers")
  if (numeric < 0).any().any():
    raise ValueError("collateral values cannot be negative")
  months = numeric["month"]
  if (months % 1 != 0).any() or not months.is_monotonic_increasing or months.duplicated().any():
    raise ValueError("collateral months must be unique and increasing")
  rollforward = (
    numeric["beginning_balance"]
    - numeric["scheduled_principal"]
    - numeric["prepayment"]
    - numeric["default"]
  )
  if not _close(rollforward, numeric["ending_balance"]):
    raise ValueError("collateral balance roll-forward does not reconcile")
  if len(numeric) > 1 and not _close(
    numeric["ending_balance"].iloc[:-1].reset_index(drop=True),
    numeric["beginning_balance"].iloc[1:].reset_index(drop=True),
  ):
    raise ValueError("collateral beginning balance does not follow prior ending balance")
  if not isclose(
    float(numeric["default"].sum()),
    float(numeric["recovery"].sum() + numeric["loss"].sum()),
    rel_tol=1e-9,
    abs_tol=1e-6,
  ):
    raise ValueError("collateral default must reconcile to recovery plus loss")


def _close(left: pd.Series[Any], right: pd.Series[Any]) -> bool:
  difference = (left - right).abs()
  scale = pd.concat((left.abs(), right.abs()), axis=1).max(axis=1)
  return bool((difference <= 1e-6 + 1e-9 * scale).all())


def _positive(value: float, name: str) -> None:
  if not isfinite(value) or value <= 0:
    raise ValueError(f"{name} must be a finite positive value")


def _rate(value: float, name: str) -> None:
  if not isfinite(value) or not 0 <= value <= 1:
    raise ValueError(f"{name} must be a finite value in [0, 1]")
