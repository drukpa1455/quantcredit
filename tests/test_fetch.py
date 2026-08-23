from __future__ import annotations

import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from quantcredit.fetch import fetch


class FetchTests(unittest.TestCase):
  def test_fetches_non_sec_dataset_and_accepts_https_redirect(self) -> None:
    data = b"loan_id,balance\n1,100\n"
    url = "https://data.example/loans.csv"
    final_url = "https://cdn.example/loans.csv"
    directory = tempfile.TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    destination = Path(directory.name) / "loans.csv"

    with patch("quantcredit.fetch._open", return_value=_Response(data, final_url)):
      receipt = fetch(url, destination)

    self.assertEqual(receipt.bytes, len(data))
    self.assertEqual(receipt.sha256, hashlib.sha256(data).hexdigest())
    self.assertEqual(destination.read_bytes(), data)

  def test_reuses_only_a_verified_cache(self) -> None:
    data = b"dataset"
    digest = hashlib.sha256(data).hexdigest()
    url = "https://data.example/dataset.csv"
    directory = tempfile.TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    destination = Path(directory.name) / "dataset.csv"
    destination.write_bytes(data)

    with patch("quantcredit.fetch._open", side_effect=AssertionError("network used")):
      receipt = fetch(
        url,
        destination,
        timeout_seconds=1,
        max_bytes=100,
        expected_bytes=len(data),
        expected_sha256=digest,
      )

    self.assertTrue(receipt.cache_hit)


class _Response:
  def __init__(self, data: bytes, url: str) -> None:
    self._data = io.BytesIO(data)
    self._url = url
    self.headers = {"Content-Length": str(len(data))}

  def read(self, size: int = -1) -> bytes:
    return self._data.read(size)

  def geturl(self) -> str:
    return self._url

  def __enter__(self) -> _Response:
    return self

  def __exit__(self, *args: object) -> None:
    return None


if __name__ == "__main__":
  unittest.main()
