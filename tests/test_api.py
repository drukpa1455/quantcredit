from __future__ import annotations

import subprocess
import sys
import unittest

import quantcredit as qc


class ApiTests(unittest.TestCase):
  def test_exposes_memorable_research_entrypoints(self) -> None:
    self.assertEqual(qc.audit.__module__, "quantcredit.api")
    self.assertEqual(qc.examples.__module__, "quantcredit.api")
    self.assertEqual(qc.split.__module__, "quantcredit.api")

  def test_package_import_does_not_preload_the_audit_cli(self) -> None:
    result = subprocess.run(
      [sys.executable, "-m", "quantcredit.audits", "--help"],
      check=False,
      capture_output=True,
      text=True,
    )

    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertNotIn("RuntimeWarning", result.stderr)


if __name__ == "__main__":
  unittest.main()
