from __future__ import annotations

import unittest

import quantcredit as qc
from quantcredit.audit import Audit, audit_sources
from quantcredit.splits import CausalSplit, causal_split


class ApiTests(unittest.TestCase):
  def test_exposes_memorable_research_entrypoints(self) -> None:
    self.assertIs(qc.audit, audit_sources)
    self.assertIs(qc.split, causal_split)
    self.assertIs(qc.Audit, Audit)
    self.assertIs(qc.CausalSplit, CausalSplit)


if __name__ == "__main__":
  unittest.main()
