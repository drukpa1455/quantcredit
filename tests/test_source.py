from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from quantcredit.source import load_manifest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "sources/ford-credit-auto-owner-trust-2024-a.json"


class SourceManifestTests(unittest.TestCase):
  def test_declares_one_ordered_trust_year(self) -> None:
    manifest = load_manifest(MANIFEST)

    self.assertEqual(manifest.cik, "0002014176")
    self.assertEqual(len(manifest.filings), 12)
    self.assertEqual(manifest.filings[0].report_period.isoformat(), "2025-01-31")
    self.assertEqual(manifest.filings[-1].report_period.isoformat(), "2025-12-31")
    self.assertEqual(manifest.summary()["pinned_ex102_documents"], 12)
    self.assertEqual(manifest.access_policy.maximum_requests_per_second, 10)

  def test_rejects_duplicate_report_period(self) -> None:
    raw = json.loads(MANIFEST.read_text())
    raw["filings"][1]["report_period"] = raw["filings"][0]["report_period"]

    with self.assertRaisesRegex(ValueError, "report periods must be unique and increasing"):
      load_manifest(self._write(raw))

  def test_rejects_non_sec_url(self) -> None:
    raw = json.loads(MANIFEST.read_text())
    raw["filings"][0]["index_url"] = "https://example.com/filing.htm"

    with self.assertRaisesRegex(ValueError, "SEC URL must use"):
      load_manifest(self._write(raw))

  def test_rejects_partial_ex102_pin(self) -> None:
    raw = json.loads(MANIFEST.read_text())
    raw["filings"][0]["ex102_url"] = None
    raw["filings"][0]["sha256"] = None
    raw["filings"][0]["bytes"] = 10

    with self.assertRaisesRegex(ValueError, "EX-102 pin must be complete"):
      load_manifest(self._write(raw))

  def _write(self, raw: object) -> Path:
    directory = tempfile.TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    path = Path(directory.name) / "manifest.json"
    path.write_text(json.dumps(raw))
    return path


if __name__ == "__main__":
  unittest.main()
