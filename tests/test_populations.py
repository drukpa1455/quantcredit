from __future__ import annotations

import hashlib
import tempfile
import unittest
from dataclasses import replace
from itertools import chain
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

import quantcredit as qc
from quantcredit.populations import FEATURE_COLUMNS, FEATURE_LINEAGE, LEAKAGE_FIELDS
from quantcredit.source import Filing, load_manifest
from quantcredit.splits import causal_split

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "sources/ford-credit-auto-owner-trust-2024-a.json"
NS = "http://www.sec.gov/edgar/document/absee/autoloan/assetdata"


class PopulationTests(unittest.TestCase):
  def setUp(self) -> None:
    self.directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    self.cache = Path(self.directory.name)
    source = load_manifest(MANIFEST)
    filings = tuple(self._filing(source.filings[index], index) for index in range(12))
    self.manifest = replace(source, filings=filings)
    self.split = causal_split(self.manifest.report_periods)

  def test_materializes_only_eligible_cutoffs_with_auditable_dispositions(self) -> None:
    examples = qc.examples(self.manifest, self.split, self.cache)

    counts = examples.groupby(["fold", "target_status"]).size().to_dict()
    self.assertEqual(
      counts,
      {
        ("train", "positive"): 1,
        ("train", "negative"): 3,
        ("train", "competing_event"): 1,
        ("train", "missing_followup"): 1,
        ("validation", "positive"): 1,
        ("validation", "negative"): 3,
        ("test", "positive"): 1,
        ("test", "negative"): 3,
      },
    )
    self.assertEqual(len(examples), 14)
    self.assertEqual(examples["target"].dtype, pd.Int8Dtype())
    self.assertEqual(examples["target"].notna().sum(), 12)
    self.assertEqual(set(FEATURE_COLUMNS), set(examples.columns[5:]))
    self.assertTrue(set(LEAKAGE_FIELDS).isdisjoint(examples.columns))
    unsafe = set(LEAKAGE_FIELDS) - {"reportingPeriodEndingDate"}
    self.assertTrue(unsafe.isdisjoint(chain.from_iterable(FEATURE_LINEAGE.values())))
    self.assertEqual(examples["original_ltv"].dropna().unique().tolist(), [0.8])
    self.assertEqual(examples["credit_score_status"].value_counts()["no_score"], 1)
    excluded = hashlib.sha256(f"{self.manifest.cik}:PRIVATE-G".encode()).hexdigest()
    self.assertNotIn(excluded, set(examples["loan_id"]))
    self.assertNotIn("PRIVATE", repr(examples))

  def test_rejects_a_split_from_a_different_panel(self) -> None:
    wrong = replace(self.split, test_cutoff=self.manifest.report_periods[7])

    with self.assertRaisesRegex(ValueError, "does not match"):
      qc.examples(self.manifest, wrong, self.cache)

  def test_fails_before_returning_a_frame_when_a_source_pin_drifts(self) -> None:
    filing = self.manifest.filings[-1]
    assert filing.ex102_url is not None
    path = self.cache / filing.accession / Path(urlparse(filing.ex102_url).path).name
    path.write_text("changed")

    with self.assertRaisesRegex(ValueError, "byte count mismatch"):
      qc.examples(self.manifest, self.split, self.cache)

  def _filing(self, filing: Filing, index: int) -> Filing:
    assets = [self._asset(filing, "PRIVATE-A", 0)]
    for name, event_index in (("PRIVATE-B", 1), ("PRIVATE-C", 5), ("PRIVATE-D", 9)):
      status = 60 if index == event_index else 0
      assets.append(self._asset(filing, name, status))
    if index <= 1:
      assets.append(
        self._asset(filing, "PRIVATE-E", 0, terminal=index == 1)
      )
    if index == 0:
      assets.append(self._asset(filing, "PRIVATE-F", 0, score="No Score"))
      assets.append(self._asset(filing, "PRIVATE-G", 60))
    xml = f'<assetData xmlns="{NS}">{"".join(assets)}</assetData>'
    payload = xml.encode()
    name = f"period-{index}.xml"
    path = self.cache / filing.accession / name
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    return replace(
      filing,
      ex102_url=f"{filing.index_url.rsplit('/', 1)[0]}/{name}",
      bytes=len(payload),
      sha256=hashlib.sha256(payload).hexdigest(),
    )

  @staticmethod
  def _asset(
    filing: Filing,
    asset: str,
    status: int,
    *,
    terminal: bool = False,
    score: str = "720",
  ) -> str:
    fields = {
      "assetTypeNumber": "auto",
      "assetNumber": asset,
      "reportingPeriodBeginningDate": "01-01-2025",
      "reportingPeriodEndingDate": filing.report_period.strftime("%m-%d-%Y"),
      "currentDelinquencyStatus": str(status),
      "obligorCreditScore": score,
      "originalLoanAmount": "20000",
      "originalLoanTerm": "60",
      "originalInterestRatePercentage": "0.08",
      "paymentToIncomePercentage": "0.10",
      "vehicleValueAmount": "25000",
      "remainingTermToMaturityNumber": "48",
      "reportingPeriodBeginningLoanBalanceAmount": "17000",
      "reportingPeriodActualEndBalanceAmount": "16800",
      "reportingPeriodScheduledPaymentAmount": "400",
      "nextReportingPeriodPaymentAmountDue": "400",
      "originationDate": "01/2024",
      "vehicleModelYear": "2023",
      "obligorGeographicLocation": "CA",
      "vehicleNewUsedCode": "1",
      "vehicleTypeCode": "1",
      "obligorCreditScoreType": "1",
      "obligorIncomeVerificationLevelCode": "2",
      "obligorEmploymentVerificationCode": "2",
      "paymentTypeCode": "1",
    }
    if terminal:
      fields["zeroBalanceCode"] = "1"
    return "<assets>" + "".join(
      f"<{name}>{value}</{name}>" for name, value in fields.items()
    ) + "</assets>"


if __name__ == "__main__":
  unittest.main()
