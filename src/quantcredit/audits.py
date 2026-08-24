"""Emit a checksum-bound, aggregate-only audit of the declared loan panel."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from quantcredit.acquire import DEFAULT_ASSET_LIMIT, DEFAULT_CACHE
from quantcredit.fetch import verify
from quantcredit.panel import AssetKey, LoanSnapshot, SnapshotValidator, read_snapshots
from quantcredit.source import DEFAULT_MANIFEST, SourceManifest, load_manifest
from quantcredit.targets import LoanState, target_counts

DEFAULT_HORIZON_REPORTS = 3

if TYPE_CHECKING:
  from matplotlib.figure import Figure


@dataclass(frozen=True)
class Audit:
  """Aggregate evidence produced by one verified panel scan."""

  source: dict[str, Any]
  panel: dict[str, Any]
  fields: dict[str, dict[str, int]]
  states: dict[str, int]
  continuity: tuple[dict[str, str | int], ...]
  transitions: dict[str, int]
  targets: tuple[dict[str, Any], ...]

  def plot(self) -> Figure:
    """Render the canonical aggregate evidence figure."""
    from quantcredit.visuals import plot_audit

    return plot_audit(self)

  def to_dict(self) -> dict[str, Any]:
    """Return a JSON-serializable representation."""
    return asdict(self)


def audit_sources(
  manifest: SourceManifest,
  cache: Path = DEFAULT_CACHE,
  *,
  horizon_reports: int = DEFAULT_HORIZON_REPORTS,
) -> Audit:
  """Validate all pinned bytes and return no consumer-level observations."""
  if horizon_reports <= 0:
    raise ValueError("target horizon must be positive")

  validator = SnapshotValidator()
  field_presence: Counter[str] = Counter()
  states: Counter[str] = Counter()
  transitions: Counter[str] = Counter()
  histories: dict[AssetKey, list[LoanState | None]] = {}
  previous: dict[AssetKey, str] = {}
  ever_seen: set[AssetKey] = set()
  continuity: list[dict[str, str | int]] = []
  coverage: list[dict[str, str | int]] = []

  for period_index, filing in enumerate(manifest.filings):
    if not filing.pinned:
      raise ValueError(f"unpinned EX-102 source: {filing.accession}")
    assert filing.ex102_url is not None
    assert filing.bytes is not None
    assert filing.sha256 is not None
    filename = Path(urlparse(filing.ex102_url).path).name
    path = cache / filing.accession / filename
    receipt = verify(
      path,
      expected_bytes=filing.bytes,
      expected_sha256=filing.sha256,
      max_bytes=DEFAULT_ASSET_LIMIT,
    )

    current: dict[AssetKey, str] = {}
    snapshot_count = 0
    for snapshot in read_snapshots(path, manifest, filing):
      validator.observe(snapshot)
      snapshot_count += 1
      for source_field in snapshot.fields:
        field_presence[source_field.name] += 1
      state = _loan_state(snapshot)
      label = _state_label(state)
      states[label] += 1
      current[snapshot.key.asset] = label
      history = histories.setdefault(snapshot.key.asset, [None] * len(manifest.filings))
      history[period_index] = state

    current_assets = set(current)
    previous_assets = set(previous)
    reappeared = (current_assets & ever_seen) - previous_assets
    continuity.append(
      {
        "report_period": filing.report_period.isoformat(),
        "reported": len(current_assets),
        "new": len(current_assets - ever_seen),
        "continued": len(current_assets & previous_assets),
        "disappeared": len(previous_assets - current_assets),
        "reappeared": len(reappeared),
      }
    )
    for asset in current_assets & previous_assets:
      transitions[f"{previous[asset]} -> {current[asset]}"] += 1
    ever_seen.update(current_assets)
    previous = current
    coverage.append(
      {
        "cik": manifest.cik,
        "accession": filing.accession,
        "report_period": filing.report_period.isoformat(),
        "ex102_url": filing.ex102_url,
        "bytes": receipt.bytes,
        "sha256": receipt.sha256,
        "abs_schema_version": manifest.abs_schema_version,
        "snapshots": snapshot_count,
      }
    )

  summary = validator.summary()
  counts = target_counts(list(histories.values()), horizon_reports=horizon_reports)
  return Audit(
    source={
      **manifest.summary(),
      "documents": coverage,
    },
    panel={
      **asdict(summary),
      "duplicate_snapshot_keys": 0,
      "immutable_contradictions": 0,
    },
    fields={
      name: {"reported": count, "missing": summary.snapshots - count}
      for name, count in sorted(field_presence.items())
    },
    states=dict(sorted(states.items())),
    continuity=tuple(continuity),
    transitions=dict(sorted(transitions.items())),
    targets=tuple(_target_decisions(counts, horizon_reports)),
  )


def _loan_state(snapshot: LoanSnapshot) -> LoanState:
  return LoanState(
    snapshot.current_delinquency_status,
    snapshot.zero_balance_codes,
    snapshot.decimal("chargedoffPrincipalAmount"),
    snapshot.decimal("recoveredAmount"),
  )


def _state_label(state: LoanState) -> str:
  if state.zero_balance_codes:
    return "zero_balance:" + "+".join(code.value for code in state.zero_balance_codes)
  days = state.delinquency_days
  if days is None:
    return "delinquency:missing"
  if days == 0:
    return "delinquency:current"
  if days < 30:
    return "delinquency:1-29"
  if days < 60:
    return "delinquency:30-59"
  if days < 90:
    return "delinquency:60-89"
  return "delinquency:90+"


def _target_decisions(counts: dict[str, int], horizon: int) -> list[dict[str, Any]]:
  derived = counts["positive"] > 0 and counts["negative"] > 0
  return [
    {
      "name": "serious_delinquency_or_chargeoff",
      "status": "derived" if derived else "rejected",
      "field_lineage": [
        "currentDelinquencyStatus",
        "zeroBalanceCode=4",
        "chargedoffPrincipalAmount>0",
      ],
      "horizon": f"next {horizon} reports",
      "event": "first 60+ days delinquent or charge-off",
      "competing_event": "any other reported zero-balance code",
      "censoring": "missing follow-up state or insufficient future reports",
      "counts": counts,
    },
    {
      "name": "prepayment",
      "status": "rejected",
      "field_lineage": ["zeroBalanceCode=1"],
      "reason": "Schedule AL code 1 combines prepaid and matured loans",
    },
    {
      "name": "ultimate_net_loss",
      "status": "rejected",
      "field_lineage": ["chargedoffPrincipalAmount", "recoveredAmount"],
      "reason": "the bounded panel does not observe a complete post-charge-off recovery horizon",
    },
  ]


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
  parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
  parser.add_argument("--horizon-reports", type=int, default=DEFAULT_HORIZON_REPORTS)
  args = parser.parse_args(argv)
  try:
    result = audit_sources(
      load_manifest(args.manifest), args.cache, horizon_reports=args.horizon_reports
    )
  except ValueError as error:
    parser.error(str(error))
  print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
