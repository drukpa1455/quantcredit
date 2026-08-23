"""Stream typed auto-loan snapshots without retaining source rows."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from xml.etree.ElementTree import Element, ParseError, iterparse

from quantcredit.source import Filing, SourceManifest

SCHEMA_VERSION = "3.1"
NAMESPACE = "http://www.sec.gov/edgar/document/absee/autoloan/assetdata"
ROOT_TAG = f"{{{NAMESPACE}}}assetData"
ASSET_TAG = f"{{{NAMESPACE}}}assets"
REQUIRED_FIELDS = (
  "assetTypeNumber",
  "assetNumber",
  "reportingPeriodBeginningDate",
  "reportingPeriodEndingDate",
)
REPEATED_FIELDS = frozenset(
  {"subvented", "zeroBalanceCode", "repurchaseReplacementReasonCode", "modificationTypeCode"}
)
IMMUTABLE_FIELDS = (
  "originatorName",
  "originationDate",
  "originalLoanAmount",
  "originalLoanTerm",
  "originalInterestRatePercentage",
  "originalInterestRateTypeCode",
  "originalFirstPaymentDate",
)
DECIMAL_FIELDS = frozenset(
  {
    "originalLoanAmount",
    "originalInterestRatePercentage",
    "chargedoffPrincipalAmount",
    "recoveredAmount",
  }
)
INTEGER_FIELDS = frozenset({"originalLoanTerm", "currentDelinquencyStatus"})


class ZeroBalanceCode(StrEnum):
  PREPAID_OR_MATURED = "1"
  THIRD_PARTY_SALE = "2"
  REPURCHASED_OR_REPLACED = "3"
  CHARGED_OFF = "4"
  SERVICING_TRANSFER = "5"
  UNAVAILABLE = "99"


@dataclass(frozen=True)
class AssetKey:
  cik: str
  asset_number: str = field(repr=False)


@dataclass(frozen=True)
class SnapshotKey:
  asset: AssetKey
  report_period: date


@dataclass(frozen=True)
class SourceField:
  name: str
  values: tuple[str, ...] = field(repr=False)

  @property
  def missing(self) -> bool:
    return not self.values


@dataclass(frozen=True)
class LoanSnapshot:
  key: SnapshotKey
  accession: str
  asset_type: str
  reporting_period_begin: date
  fields: tuple[SourceField, ...] = field(repr=False)

  def field(self, name: str) -> SourceField:
    return next((item for item in self.fields if item.name == name), SourceField(name, ()))

  def decimal(self, name: str) -> Decimal | None:
    value = self._one(name)
    if value is None:
      return None
    try:
      number = Decimal(value)
    except InvalidOperation as error:
      raise ValueError(f"{self.accession}: invalid decimal element {name}") from error
    if not number.is_finite():
      raise ValueError(f"{self.accession}: non-finite decimal element {name}")
    return number

  def integer(self, name: str) -> int | None:
    value = self._one(name)
    if value is None:
      return None
    try:
      return int(value)
    except ValueError as error:
      raise ValueError(f"{self.accession}: invalid integer element {name}") from error

  @property
  def zero_balance_codes(self) -> tuple[ZeroBalanceCode, ...]:
    try:
      return tuple(ZeroBalanceCode(value) for value in self.field("zeroBalanceCode").values)
    except ValueError as error:
      raise ValueError(f"{self.accession}: invalid zeroBalanceCode") from error

  @property
  def current_delinquency_status(self) -> int | None:
    status = self.integer("currentDelinquencyStatus")
    if status is None:
      return None
    if status < 0:
      raise ValueError(f"{self.accession}: negative element currentDelinquencyStatus")
    return status

  def _one(self, name: str) -> str | None:
    values = self.field(name).values
    if not values:
      return None
    if len(values) != 1:
      raise ValueError(f"{self.accession}: repeated singleton element {name}")
    return values[0]


@dataclass(frozen=True)
class PanelSummary:
  snapshots: int
  loans: int
  periods: int


def read_snapshots(
  path: Path,
  manifest: SourceManifest,
  filing: Filing,
) -> Iterator[LoanSnapshot]:
  """Yield one validated snapshot at a time from a declared auto-loan filing."""
  if manifest.abs_schema_version != SCHEMA_VERSION:
    raise ValueError(f"unsupported ABS schema version: {manifest.abs_schema_version}")
  if filing not in manifest.filings:
    raise ValueError(f"filing is not declared by manifest: {filing.accession}")

  seen: set[str] = set()
  count = 0
  try:
    events = iterparse(path, events=("start", "end"))
    try:
      _, root = next(events)
    except StopIteration as error:
      raise ValueError(f"{filing.accession}: empty XML document") from error
    if root.tag != ROOT_TAG:
      raise ValueError(f"{filing.accession}: root element does not match ABS {SCHEMA_VERSION}")

    for event, element in events:
      if event != "end" or _local_name(element.tag) != "assets":
        continue
      if element.tag != ASSET_TAG:
        raise ValueError(f"{filing.accession}: assets element uses an invalid namespace")
      snapshot = _snapshot(element, manifest.cik, filing)
      asset_number = snapshot.key.asset.asset_number
      if asset_number in seen:
        raise ValueError(f"{filing.accession}: duplicate element assetNumber")
      seen.add(asset_number)
      count += 1
      root.clear()
      yield snapshot
  except (OSError, ParseError) as error:
    raise ValueError(f"{filing.accession}: cannot parse XML document {path}: {error}") from error
  if count == 0:
    raise ValueError(f"{filing.accession}: XML document contains no assets")


def validate_snapshots(snapshots: Iterable[LoanSnapshot]) -> PanelSummary:
  """Validate ordered snapshot identity and immutable origination facts."""
  keys: set[SnapshotKey] = set()
  loans: set[AssetKey] = set()
  periods: set[date] = set()
  immutable: dict[tuple[AssetKey, str], str | int | Decimal] = {}
  last_period: date | None = None
  count = 0

  for snapshot in snapshots:
    if last_period is not None and snapshot.key.report_period < last_period:
      raise ValueError(f"{snapshot.accession}: report periods are not monotone")
    last_period = snapshot.key.report_period
    if snapshot.key in keys:
      raise ValueError(f"{snapshot.accession}: duplicate snapshot key")
    keys.add(snapshot.key)
    loans.add(snapshot.key.asset)
    periods.add(snapshot.key.report_period)
    count += 1

    for name in IMMUTABLE_FIELDS:
      values = snapshot.field(name).values
      if not values:
        continue
      if len(values) != 1:
        raise ValueError(f"{snapshot.accession}: repeated immutable element {name}")
      key = (snapshot.key.asset, name)
      value = _immutable_value(snapshot, name, values[0])
      previous = immutable.setdefault(key, value)
      if previous != value:
        raise ValueError(f"{snapshot.accession}: contradictory immutable element {name}")

  if count == 0:
    raise ValueError("panel contains no snapshots")
  return PanelSummary(count, len(loans), len(periods))


def _snapshot(element: Element, cik: str, filing: Filing) -> LoanSnapshot:
  if element.attrib:
    raise ValueError(f"{filing.accession}: assets element must not have attributes")
  raw: dict[str, list[str]] = {}
  for child in element:
    name = _local_name(child.tag)
    if child.tag != f"{{{NAMESPACE}}}{name}":
      raise ValueError(f"{filing.accession}: element {name} uses an invalid namespace")
    if child.attrib:
      raise ValueError(f"{filing.accession}: element {name} must not have attributes")
    if list(child):
      raise ValueError(f"{filing.accession}: element {name} must contain text only")
    value = (child.text or "").strip()
    if not value:
      raise ValueError(f"{filing.accession}: element {name} is empty")
    values = raw.setdefault(name, [])
    if values and name not in REPEATED_FIELDS:
      raise ValueError(f"{filing.accession}: repeated singleton element {name}")
    values.append(value)

  for name in REQUIRED_FIELDS:
    if name not in raw:
      raise ValueError(f"{filing.accession}: missing required element {name}")

  asset_number = raw["assetNumber"][0]
  if len(asset_number) > 25:
    raise ValueError(f"{filing.accession}: element assetNumber exceeds schema length")
  period_begin = _date(
    raw["reportingPeriodBeginningDate"][0],
    filing,
    "reportingPeriodBeginningDate",
  )
  period_end = _date(raw["reportingPeriodEndingDate"][0], filing, "reportingPeriodEndingDate")
  if period_begin > period_end:
    raise ValueError(f"{filing.accession}: reporting period begins after it ends")
  if period_end != filing.report_period:
    raise ValueError(f"{filing.accession}: element reportingPeriodEndingDate contradicts manifest")

  snapshot = LoanSnapshot(
    SnapshotKey(AssetKey(cik, asset_number), period_end),
    filing.accession,
    raw["assetTypeNumber"][0],
    period_begin,
    tuple(SourceField(name, tuple(values)) for name, values in raw.items()),
  )
  for name in DECIMAL_FIELDS:
    snapshot.decimal(name)
  for name in INTEGER_FIELDS:
    snapshot.integer(name)
  snapshot.current_delinquency_status
  snapshot.zero_balance_codes
  return snapshot


def _date(value: str, filing: Filing, element: str) -> date:
  try:
    return datetime.strptime(value, "%m-%d-%Y").date()
  except ValueError as error:
    raise ValueError(f"{filing.accession}: invalid date element {element}") from error


def _immutable_value(snapshot: LoanSnapshot, name: str, value: str) -> str | int | Decimal:
  if name in DECIMAL_FIELDS:
    decimal_value = snapshot.decimal(name)
    assert decimal_value is not None
    return decimal_value
  if name in INTEGER_FIELDS:
    integer_value = snapshot.integer(name)
    assert integer_value is not None
    return integer_value
  return value


def _local_name(tag: str) -> str:
  return tag.rsplit("}", 1)[-1]
