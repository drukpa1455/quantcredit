"""Explicit fixed-horizon credit outcomes and censoring."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from quantcredit.panel import ZeroBalanceCode


@dataclass(frozen=True)
class LoanState:
  delinquency_days: int | None
  zero_balance_codes: tuple[ZeroBalanceCode, ...] = ()
  charged_off_principal: Decimal | None = None
  recovered_amount: Decimal | None = None

  @property
  def charged_off(self) -> bool:
    return (
      ZeroBalanceCode.CHARGED_OFF in self.zero_balance_codes
      or (self.charged_off_principal is not None and self.charged_off_principal > 0)
    )

  @property
  def terminal(self) -> bool:
    return bool(self.zero_balance_codes)


class TargetResult(StrEnum):
  POSITIVE = "positive"
  NEGATIVE = "negative"
  COMPETING = "competing_event"
  CENSORED = "missing_followup"
  RIGHT_CENSORED = "right_censored"
  INELIGIBLE = "ineligible_at_cutoff"


def serious_delinquency_target(
  history: Sequence[LoanState | None],
  cutoff: int,
  *,
  horizon_reports: int = 3,
) -> TargetResult:
  """Classify first 60+ delinquency or charge-off after one eligible cutoff."""
  if horizon_reports <= 0:
    raise ValueError("target horizon must be positive")
  if cutoff < 0 or cutoff >= len(history):
    raise ValueError("target cutoff is outside the loan history")
  state = history[cutoff]
  if (
    state is None
    or state.terminal
    or state.delinquency_days is None
    or state.delinquency_days >= 60
    or state.charged_off
  ):
    return TargetResult.INELIGIBLE
  if cutoff + horizon_reports >= len(history):
    return TargetResult.RIGHT_CENSORED

  for future in history[cutoff + 1 : cutoff + horizon_reports + 1]:
    if future is None:
      return TargetResult.CENSORED
    if future.charged_off:
      return TargetResult.POSITIVE
    if future.terminal:
      return TargetResult.COMPETING
    if future.delinquency_days is not None and future.delinquency_days >= 60:
      return TargetResult.POSITIVE
    if future.delinquency_days is None:
      return TargetResult.CENSORED
  return TargetResult.NEGATIVE


def target_counts(
  histories: Sequence[Sequence[LoanState | None]], *, horizon_reports: int = 3
) -> dict[str, int]:
  """Count target classifications across every loan-cutoff pair."""
  counts: Counter[str] = Counter()
  for history in histories:
    for cutoff in range(len(history)):
      counts[serious_delinquency_target(history, cutoff, horizon_reports=horizon_reports)] += 1
  return {result.value: counts[result] for result in TargetResult}
