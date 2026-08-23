from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from decimal import Decimal
from itertools import chain
from pathlib import Path

from quantcredit.panel import PanelSummary, ZeroBalanceCode, read_snapshots, validate_snapshots
from quantcredit.source import Filing, SourceManifest, load_manifest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "sources/ford-credit-auto-owner-trust-2024-a.json"
NS = "http://www.sec.gov/edgar/document/absee/autoloan/assetdata"


class PanelTests(unittest.TestCase):
  def setUp(self) -> None:
    self.manifest = load_manifest(MANIFEST)
    self.first = self.manifest.filings[0]
    self.second = self.manifest.filings[1]

  def test_streams_typed_states_and_valid_continuity(self) -> None:
    first = self._write(
      self._xml(
        self._asset(
          self.first,
          "PRIVATE-ASSET-1",
          ("originalLoanAmount", "100.00"),
          ("chargedoffPrincipalAmount", "0.00"),
          ("currentDelinquencyStatus", "0"),
        ),
        self._asset(
          self.first,
          "PRIVATE-ASSET-2",
          ("zeroBalanceCode", "4"),
          ("chargedoffPrincipalAmount", "20.00"),
          ("recoveredAmount", "5.00"),
          ("currentDelinquencyStatus", "60"),
        ),
      )
    )
    second = self._write(
      self._xml(
        self._asset(
          self.second,
          "PRIVATE-ASSET-1",
          ("originalLoanAmount", "100.00000000"),
          ("loanMaturityDate", "02-01-2029"),
          ("zeroBalanceCode", "1"),
          ("recoveredAmount", "0.00"),
        )
      )
    )

    first_rows = list(read_snapshots(first, self.manifest, self.first))
    second_rows = list(read_snapshots(second, self.manifest, self.second))
    snapshot = first_rows[0]

    self.assertEqual(snapshot.decimal("chargedoffPrincipalAmount"), Decimal("0.00"))
    self.assertTrue(snapshot.field("recoveredAmount").missing)
    self.assertEqual(first_rows[1].zero_balance_codes, (ZeroBalanceCode.CHARGED_OFF,))
    self.assertEqual(first_rows[1].decimal("recoveredAmount"), Decimal("5.00"))
    self.assertEqual(first_rows[1].current_delinquency_status, 60)
    self.assertEqual(second_rows[0].zero_balance_codes, (ZeroBalanceCode.PREPAID_OR_MATURED,))
    self.assertNotIn("PRIVATE-ASSET-1", repr(snapshot))
    self.assertEqual(
      validate_snapshots(chain(first_rows, second_rows)),
      PanelSummary(snapshots=3, loans=2, periods=2),
    )

  def test_allows_schema_defined_repeated_fields(self) -> None:
    path = self._write(
      self._xml(
        self._asset(
          self.first,
          "A",
          ("subvented", "0"),
          ("subvented", "1"),
          ("modificationTypeCode", "1"),
          ("modificationTypeCode", "4"),
        )
      )
    )

    snapshot = next(read_snapshots(path, self.manifest, self.first))

    self.assertEqual(snapshot.field("subvented").values, ("0", "1"))
    self.assertEqual(snapshot.field("modificationTypeCode").values, ("1", "4"))

  def test_rejects_schema_root_required_fields_nil_and_duplicate_identity(self) -> None:
    cases = (
      (self._xml(self._asset(self.first, "A"), namespace="https://example.com"), "root element"),
      (self._xml(self._asset(self.first, "A", omit="assetNumber")), "assetNumber"),
      (
        self._xml(
          self._asset(
            self.first,
            "A",
            (
              "recoveredAmount",
              '<recoveredAmount xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
              'xsi:nil="true"/>',
            ),
            raw=True,
          )
        ),
        "recoveredAmount must not have attributes",
      ),
      (self._xml(self._asset(self.first, "A"), self._asset(self.first, "A")), "duplicate element"),
      (
        self._xml(
          self._asset(
            self.first,
            "A",
            ("originalLoanAmount", "100"),
            ("originalLoanAmount", "100"),
          )
        ),
        "repeated singleton element originalLoanAmount",
      ),
    )
    for xml, message in cases:
      with self.subTest(message=message):
        with self.assertRaisesRegex(ValueError, message):
          list(read_snapshots(self._write(xml), self.manifest, self.first))

  def test_rejects_invalid_schema_version_period_and_source_domains(self) -> None:
    wrong_schema = replace(self.manifest, abs_schema_version="3.0")
    cases: tuple[tuple[SourceManifest, Filing, str, str], ...] = (
      (wrong_schema, self.first, self._xml(self._asset(self.first, "A")), "schema version"),
      (
        self.manifest,
        self.first,
        self._xml(self._asset(self.second, "A")),
        "reportingPeriodEndingDate",
      ),
      (
        self.manifest,
        self.first,
        self._xml(self._asset(self.first, "A", ("zeroBalanceCode", "7"))),
        "zeroBalanceCode",
      ),
      (
        self.manifest,
        self.first,
        self._xml(self._asset(self.first, "A", ("currentDelinquencyStatus", "-1"))),
        "currentDelinquencyStatus",
      ),
    )
    for manifest, filing, xml, message in cases:
      with self.subTest(message=message):
        with self.assertRaisesRegex(ValueError, message):
          list(read_snapshots(self._write(xml), manifest, filing))

  def test_rejects_changed_immutable_with_safe_context(self) -> None:
    first = self._write(
      self._xml(self._asset(self.first, "PRIVATE-ASSET-1", ("originalLoanAmount", "100")))
    )
    second = self._write(
      self._xml(self._asset(self.second, "PRIVATE-ASSET-1", ("originalLoanAmount", "999")))
    )
    rows = chain(
      read_snapshots(first, self.manifest, self.first),
      read_snapshots(second, self.manifest, self.second),
    )

    with self.assertRaisesRegex(
      ValueError,
      rf"{self.second.accession}: contradictory immutable element originalLoanAmount",
    ) as raised:
      validate_snapshots(rows)
    self.assertNotIn("PRIVATE-ASSET-1", str(raised.exception))
    self.assertNotIn("999", str(raised.exception))

  def _write(self, xml: str) -> Path:
    directory = tempfile.TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    path = Path(directory.name) / "assets.xml"
    path.write_text(xml)
    return path

  @staticmethod
  def _xml(*assets: str, namespace: str = NS) -> str:
    return f'<assetData xmlns="{namespace}">{"".join(assets)}</assetData>'

  @staticmethod
  def _asset(
    filing: Filing,
    asset_number: str,
    *extra: tuple[str, str],
    omit: str | None = None,
    raw: bool = False,
  ) -> str:
    fields = [
      ("assetTypeNumber", "auto"),
      ("assetNumber", asset_number),
      ("reportingPeriodBeginningDate", "01-01-2025"),
      ("reportingPeriodEndingDate", filing.report_period.strftime("%m-%d-%Y")),
    ]
    fields = [item for item in fields if item[0] != omit]
    body = "".join(
      value if raw and value.startswith("<") else f"<{name}>{value}</{name}>"
      for name, value in (*fields, *extra)
    )
    return f"<assets>{body}</assets>"

if __name__ == "__main__":
  unittest.main()
