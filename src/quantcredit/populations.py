"""Materialize causal loan-cutoff populations from verified panel snapshots."""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from quantcredit.acquire import DEFAULT_CACHE, verify_asset
from quantcredit.panel import AssetKey, LoanSnapshot, SnapshotValidator, read_snapshots
from quantcredit.source import SourceManifest
from quantcredit.splits import CausalSplit, causal_split
from quantcredit.targets import (
  LoanState,
  TargetResult,
  eligible_at_cutoff,
  loan_state,
  serious_delinquency_target,
)

FEATURE_LINEAGE = {
  "credit_score": ("obligorCreditScore",),
  "credit_score_status": ("obligorCreditScore",),
  "original_loan_amount": ("originalLoanAmount",),
  "original_loan_term": ("originalLoanTerm",),
  "original_interest_rate": ("originalInterestRatePercentage",),
  "payment_to_income": ("paymentToIncomePercentage",),
  "vehicle_value": ("vehicleValueAmount",),
  "original_ltv": ("originalLoanAmount", "vehicleValueAmount"),
  "remaining_term": ("remainingTermToMaturityNumber",),
  "beginning_balance": ("reportingPeriodBeginningLoanBalanceAmount",),
  "ending_balance": ("reportingPeriodActualEndBalanceAmount",),
  "current_ltv": ("reportingPeriodActualEndBalanceAmount", "vehicleValueAmount"),
  "scheduled_payment": ("reportingPeriodScheduledPaymentAmount",),
  "next_payment_due": ("nextReportingPeriodPaymentAmountDue",),
  "delinquency_days": ("currentDelinquencyStatus",),
  "loan_age_months": ("originationDate", "reportingPeriodEndingDate"),
  "vehicle_age": ("vehicleModelYear", "reportingPeriodEndingDate"),
  "geography": ("obligorGeographicLocation",),
  "vehicle_new_used": ("vehicleNewUsedCode",),
  "vehicle_type": ("vehicleTypeCode",),
  "credit_score_type": ("obligorCreditScoreType",),
  "income_verification": ("obligorIncomeVerificationLevelCode",),
  "employment_verification": ("obligorEmploymentVerificationCode",),
  "payment_type": ("paymentTypeCode",),
}
FEATURE_COLUMNS = tuple(FEATURE_LINEAGE)
CATEGORICAL_FEATURES = (
  "credit_score_status",
  "geography",
  "vehicle_new_used",
  "vehicle_type",
  "credit_score_type",
  "income_verification",
  "employment_verification",
  "payment_type",
)
NUMERIC_FEATURES = tuple(
  feature for feature in FEATURE_COLUMNS if feature not in CATEGORICAL_FEATURES
)
LEAKAGE_FIELDS = {
  "assetNumber": "identity, not a predictive feature",
  "reportingPeriodBeginningDate": "time key",
  "reportingPeriodEndingDate": "prediction cutoff",
  "zeroBalanceCode": "terminal outcome state",
  "zeroBalanceEffectiveDate": "terminal outcome timing",
  "chargedoffPrincipalAmount": "target definition",
  "recoveredAmount": "post-charge-off outcome",
  "repossessedProceedsAmount": "post-default realization",
}
_COLUMNS = ("loan_id", "cutoff", "fold", "target_status", "target", *FEATURE_COLUMNS)


def materialize_examples(
  manifest: SourceManifest,
  split: CausalSplit,
  cache: Path = DEFAULT_CACHE,
) -> DataFrame:
  """Return causal rows with test outcomes held out."""
  folds = {
    **{cutoff: "train" for cutoff in split.train_cutoffs},
    split.validation_cutoff: "validation",
    split.test_cutoff: "test",
  }
  return _materialize_examples(
    manifest,
    split,
    cache,
    folds=folds,
    held_out=frozenset({split.test_cutoff}),
  )


def materialize_test_examples(
  manifest: SourceManifest,
  split: CausalSplit,
  cache: Path = DEFAULT_CACHE,
) -> DataFrame:
  """Derive outcomes only for the explicit test evaluation boundary."""
  return _materialize_examples(
    manifest,
    split,
    cache,
    folds={split.test_cutoff: "test"},
    held_out=frozenset(),
  )


def _materialize_examples(
  manifest: SourceManifest,
  split: CausalSplit,
  cache: Path,
  *,
  folds: dict[date, str],
  held_out: frozenset[date],
) -> DataFrame:
  expected = causal_split(manifest.report_periods, horizon_reports=split.horizon_reports)
  if split != expected:
    raise ValueError("causal split does not match the source manifest")

  period_index = {period: index for index, period in enumerate(manifest.report_periods)}
  histories: dict[AssetKey, list[LoanState | None]] = {}
  cutoffs: dict[tuple[AssetKey, date], LoanSnapshot] = {}
  validator = SnapshotValidator()

  for filing in manifest.filings:
    receipt = verify_asset(filing, cache)
    index = period_index[filing.report_period]
    for snapshot in read_snapshots(receipt.path, manifest, filing):
      validator.observe(snapshot)
      history = histories.setdefault(snapshot.key.asset, [None] * len(manifest.filings))
      history[index] = loan_state(snapshot)
      if filing.report_period in folds:
        cutoffs[snapshot.key.asset, filing.report_period] = snapshot
  validator.summary()

  records: list[dict[str, object]] = []
  for (asset, cutoff), snapshot in cutoffs.items():
    history = histories[asset]
    index = period_index[cutoff]
    if cutoff in held_out:
      if not eligible_at_cutoff(history[index]):
        continue
      target_status = "held_out"
      target = None
    else:
      result = serious_delinquency_target(
        history,
        index,
        horizon_reports=split.horizon_reports,
      )
      if result is TargetResult.INELIGIBLE:
        continue
      target_status = result.value
      target = _target(result)
    records.append(
      {
        "loan_id": _loan_id(asset),
        "cutoff": cutoff,
        "fold": folds[cutoff],
        "target_status": target_status,
        "target": target,
        **_features(snapshot),
      }
    )

  frame = DataFrame.from_records(records, columns=_COLUMNS)
  frame["cutoff"] = pd.to_datetime(frame["cutoff"])
  frame["target"] = frame["target"].astype(pd.Int8Dtype())
  return frame.sort_values(["cutoff", "loan_id"], ignore_index=True)


def _features(snapshot: LoanSnapshot) -> dict[str, object]:
  cutoff = snapshot.key.report_period
  original_amount = _number(snapshot, "originalLoanAmount")
  vehicle_value = _number(snapshot, "vehicleValueAmount")
  ending_balance = _number(snapshot, "reportingPeriodActualEndBalanceAmount")
  origination = _month(snapshot, "originationDate")
  model_year = snapshot.integer("vehicleModelYear")
  credit_score, credit_score_status = _credit_score(snapshot)
  if origination is not None and origination > cutoff:
    raise ValueError(f"{snapshot.accession}: originationDate follows prediction cutoff")
  return {
    "credit_score": credit_score,
    "credit_score_status": credit_score_status,
    "original_loan_amount": original_amount,
    "original_loan_term": snapshot.integer("originalLoanTerm"),
    "original_interest_rate": _number(snapshot, "originalInterestRatePercentage"),
    "payment_to_income": _number(snapshot, "paymentToIncomePercentage"),
    "vehicle_value": vehicle_value,
    "original_ltv": _ratio(original_amount, vehicle_value),
    "remaining_term": snapshot.integer("remainingTermToMaturityNumber"),
    "beginning_balance": _number(snapshot, "reportingPeriodBeginningLoanBalanceAmount"),
    "ending_balance": ending_balance,
    "current_ltv": _ratio(ending_balance, vehicle_value),
    "scheduled_payment": _number(snapshot, "reportingPeriodScheduledPaymentAmount"),
    "next_payment_due": _number(snapshot, "nextReportingPeriodPaymentAmountDue"),
    "delinquency_days": snapshot.current_delinquency_status,
    "loan_age_months": None if origination is None else _months(origination, cutoff),
    "vehicle_age": None if model_year is None else cutoff.year - model_year,
    "geography": snapshot.value("obligorGeographicLocation"),
    "vehicle_new_used": snapshot.value("vehicleNewUsedCode"),
    "vehicle_type": snapshot.value("vehicleTypeCode"),
    "credit_score_type": snapshot.value("obligorCreditScoreType"),
    "income_verification": snapshot.value("obligorIncomeVerificationLevelCode"),
    "employment_verification": snapshot.value("obligorEmploymentVerificationCode"),
    "payment_type": snapshot.value("paymentTypeCode"),
  }


def _number(snapshot: LoanSnapshot, field: str) -> float | None:
  value = snapshot.decimal(field)
  return None if value is None else float(value)


def _credit_score(snapshot: LoanSnapshot) -> tuple[int | None, str]:
  value = snapshot.value("obligorCreditScore")
  if value is None:
    return None, "missing"
  if value == "No Score":
    return None, "no_score"
  try:
    return int(value), "reported"
  except ValueError as error:
    raise ValueError(f"{snapshot.accession}: invalid obligorCreditScore") from error


def _month(snapshot: LoanSnapshot, field: str) -> date | None:
  value = snapshot.value(field)
  if value is None:
    return None
  try:
    return datetime.strptime(value, "%m/%Y").date()
  except ValueError as error:
    raise ValueError(f"{snapshot.accession}: invalid month element {field}") from error


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
  if numerator is None or denominator is None or denominator == 0:
    return None
  return numerator / denominator


def _months(start: date, end: date) -> int:
  return (end.year - start.year) * 12 + end.month - start.month


def _target(result: TargetResult) -> int | None:
  if result is TargetResult.POSITIVE:
    return 1
  if result is TargetResult.NEGATIVE:
    return 0
  return None


def _loan_id(asset: AssetKey) -> str:
  identity = f"{asset.cik}:{asset.asset_number}".encode()
  return hashlib.sha256(identity).hexdigest()
