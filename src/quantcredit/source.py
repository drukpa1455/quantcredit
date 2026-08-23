"""Validate the tracked SEC source declaration without exposing loan rows."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ACCESSION = re.compile(r"^\d{10}-\d{2}-\d{6}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_MANIFEST = Path("sources/ford-credit-auto-owner-trust-2024-a.json")


@dataclass(frozen=True)
class Filing:
  report_period: date
  accession: str
  index_url: str
  ex102_url: str | None
  bytes: int | None
  sha256: str | None

  @property
  def pinned(self) -> bool:
    return self.ex102_url is not None and self.bytes is not None and self.sha256 is not None


@dataclass(frozen=True)
class SourceManifest:
  dataset: str
  issuer: str
  cik: str
  abs_schema_version: str
  filings: tuple[Filing, ...]

  def summary(self) -> dict[str, str | int]:
    return {
      "dataset": self.dataset,
      "issuer": self.issuer,
      "cik": self.cik,
      "abs_schema_version": self.abs_schema_version,
      "filings": len(self.filings),
      "first_report_period": self.filings[0].report_period.isoformat(),
      "last_report_period": self.filings[-1].report_period.isoformat(),
      "pinned_ex102_documents": sum(filing.pinned for filing in self.filings),
    }


def load_manifest(path: Path = DEFAULT_MANIFEST) -> SourceManifest:
  """Load and validate one tracked public-source declaration."""
  try:
    raw = json.loads(path.read_text())
  except (OSError, json.JSONDecodeError) as error:
    raise ValueError(f"cannot read source manifest {path}: {error}") from error
  if not isinstance(raw, dict):
    raise ValueError("source manifest must be a JSON object")

  schema = _field(raw, "schema", int)
  if schema != 1:
    raise ValueError(f"unsupported source manifest schema: {schema}")
  cik = _field(raw, "cik", str)
  if not re.fullmatch(r"\d{10}", cik):
    raise ValueError("cik must contain exactly ten digits")

  items = _field(raw, "filings", list)
  filings = tuple(_filing(item, cik) for item in items)
  if len(filings) == 0:
    raise ValueError("source manifest must declare at least one filing")
  periods = tuple(filing.report_period for filing in filings)
  if periods != tuple(sorted(periods)) or len(set(periods)) != len(periods):
    raise ValueError("report periods must be unique and increasing")
  accessions = tuple(filing.accession for filing in filings)
  if len(set(accessions)) != len(accessions):
    raise ValueError("accessions must be unique")

  return SourceManifest(
    dataset=_field(raw, "dataset", str),
    issuer=_field(raw, "issuer", str),
    cik=cik,
    abs_schema_version=_field(raw, "abs_schema_version", str),
    filings=filings,
  )


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
  args = parser.parse_args(argv)
  print(json.dumps(load_manifest(args.manifest).summary(), indent=2, sort_keys=True))


def _filing(raw: Any, cik: str) -> Filing:
  if not isinstance(raw, dict):
    raise ValueError("each filing must be a JSON object")
  accession = _field(raw, "accession", str)
  if ACCESSION.fullmatch(accession) is None:
    raise ValueError(f"invalid accession: {accession}")
  index_url = _field(raw, "index_url", str)
  _validate_sec_url(index_url, cik, accession, suffix=f"/{accession}-index.htm")

  ex102_url = _optional(raw, "ex102_url", str)
  byte_count = _optional(raw, "bytes", int)
  digest = _optional(raw, "sha256", str)
  present = (ex102_url is not None, byte_count is not None, digest is not None)
  if any(present) and not all(present):
    raise ValueError(f"EX-102 pin must be complete for {accession}")
  if ex102_url is not None:
    _validate_sec_url(ex102_url, cik, accession, suffix=".xml")
    if byte_count is None or byte_count <= 0:
      raise ValueError(f"bytes must be positive for {accession}")
    if digest is None or SHA256.fullmatch(digest) is None:
      raise ValueError(f"invalid SHA-256 for {accession}")

  try:
    report_period = date.fromisoformat(_field(raw, "report_period", str))
  except ValueError as error:
    raise ValueError(f"invalid report period for {accession}") from error
  return Filing(report_period, accession, index_url, ex102_url, byte_count, digest)


def _validate_sec_url(url: str, cik: str, accession: str, *, suffix: str) -> None:
  parsed = urlparse(url)
  accession_path = accession.replace("-", "")
  prefix = f"/Archives/edgar/data/{int(cik)}/{accession_path}/"
  if parsed.scheme != "https" or parsed.netloc != "www.sec.gov":
    raise ValueError(f"SEC URL must use https://www.sec.gov: {url}")
  if not parsed.path.startswith(prefix) or not parsed.path.endswith(suffix):
    raise ValueError(f"SEC URL does not match accession {accession}: {url}")
  if parsed.params or parsed.query or parsed.fragment:
    raise ValueError(f"SEC URL must not contain parameters: {url}")


def _field(raw: dict[str, Any], name: str, expected: type[Any]) -> Any:
  value = raw.get(name)
  if not isinstance(value, expected) or isinstance(value, bool):
    raise ValueError(f"{name} must be {expected.__name__}")
  return value


def _optional(raw: dict[str, Any], name: str, expected: type[Any]) -> Any | None:
  value = raw.get(name)
  if value is None:
    return None
  if not isinstance(value, expected) or isinstance(value, bool):
    raise ValueError(f"{name} must be null or {expected.__name__}")
  return value


if __name__ == "__main__":
  main()
