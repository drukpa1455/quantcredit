from __future__ import annotations

import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from quantcredit.audit import audit_sources
from quantcredit.source import Filing, load_manifest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "sources/ford-credit-auto-owner-trust-2024-a.json"
NS = "http://www.sec.gov/edgar/document/absee/autoloan/assetdata"


class AuditTests(unittest.TestCase):
  def setUp(self) -> None:
    self.directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    self.cache = Path(self.directory.name)
    source = load_manifest(MANIFEST)
    first, second = source.filings[:2]
    first_xml = self._xml(
      self._asset(first, "PRIVATE-A", ("currentDelinquencyStatus", "0")),
      self._asset(first, "PRIVATE-B", ("currentDelinquencyStatus", "0")),
      self._asset(first, "PRIVATE-C", ("currentDelinquencyStatus", "0")),
      self._asset(first, "PRIVATE-D", ("currentDelinquencyStatus", "0")),
    )
    second_xml = self._xml(
      self._asset(second, "PRIVATE-A", ("currentDelinquencyStatus", "60")),
      self._asset(
        second,
        "PRIVATE-B",
        ("zeroBalanceCode", "1"),
        ("currentDelinquencyStatus", "0"),
      ),
      self._asset(second, "PRIVATE-C", ("currentDelinquencyStatus", "0")),
      self._asset(second, "PRIVATE-E", ("currentDelinquencyStatus", "0")),
    )
    filings = (
      self._pin(first, "first.xml", first_xml),
      self._pin(second, "second.xml", second_xml),
    )
    self.manifest = replace(source, filings=filings)

  def test_emits_aggregate_source_state_continuity_and_target_evidence(self) -> None:
    result = audit_sources(self.manifest, self.cache, horizon_reports=1)

    self.assertEqual(result["panel"]["snapshots"], 8)
    self.assertEqual(result["panel"]["loans"], 5)
    self.assertEqual(result["continuity"][1]["disappeared"], 1)
    self.assertEqual(result["continuity"][1]["new"], 1)
    self.assertEqual(result["transitions"]["delinquency:current -> delinquency:60-89"], 1)
    decision = result["targets"][0]
    self.assertEqual(decision["status"], "derived")
    self.assertEqual(decision["counts"]["positive"], 1)
    self.assertEqual(decision["counts"]["competing_event"], 1)
    self.assertEqual(decision["counts"]["missing_followup"], 1)
    self.assertNotIn("PRIVATE", repr(result))

  def test_fails_on_source_drift_without_returning_partial_evidence(self) -> None:
    filing = self.manifest.filings[1]
    path = self.cache / filing.accession / "second.xml"
    path.write_text("changed")

    with self.assertRaisesRegex(ValueError, "byte count mismatch"):
      audit_sources(self.manifest, self.cache, horizon_reports=1)

  def _pin(self, filing: Filing, name: str, xml: str) -> Filing:
    payload = xml.encode()
    path = self.cache / filing.accession / name
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    source_url = filing.ex102_url or filing.index_url
    base = source_url.rsplit("/", 1)[0]
    return replace(
      filing,
      ex102_url=f"{base}/{name}",
      bytes=len(payload),
      sha256=hashlib.sha256(payload).hexdigest(),
    )

  @staticmethod
  def _xml(*assets: str) -> str:
    return f'<assetData xmlns="{NS}">{"".join(assets)}</assetData>'

  @staticmethod
  def _asset(filing: Filing, asset: str, *extra: tuple[str, str]) -> str:
    fields = (
      ("assetTypeNumber", "auto"),
      ("assetNumber", asset),
      ("reportingPeriodBeginningDate", "01-01-2025"),
      ("reportingPeriodEndingDate", filing.report_period.strftime("%m-%d-%Y")),
      *extra,
    )
    return "<assets>" + "".join(f"<{name}>{value}</{name}>" for name, value in fields) + "</assets>"


if __name__ == "__main__":
  unittest.main()
