from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from quantcredit.acquire import acquire_manifest, discover_ex102, download
from quantcredit.source import load_manifest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "sources/ford-credit-auto-owner-trust-2024-a.json"


class AcquisitionTests(unittest.TestCase):
  def test_discovers_unique_ex102_document(self) -> None:
    manifest = load_manifest(MANIFEST)
    filing = manifest.filings[0]

    asset = discover_ex102(self._index("asset.xml", 7), filing, manifest.cik)

    self.assertEqual(
      asset.url,
      "https://www.sec.gov/Archives/edgar/data/2014176/000201417625000006/asset.xml",
    )
    self.assertEqual(asset.declared_bytes, 7)

  def test_rejects_missing_ex102_document(self) -> None:
    manifest = load_manifest(MANIFEST)
    filing = manifest.filings[0]

    with self.assertRaisesRegex(ValueError, "expected one EX-102 XML document"):
      discover_ex102(b"<html><table></table></html>", filing, manifest.cik)

  def test_download_is_atomic_bounded_and_reuses_verified_cache(self) -> None:
    data = b"<asset/>"
    digest = hashlib.sha256(data).hexdigest()
    url = self._asset_url()
    directory = tempfile.TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    destination = Path(directory.name) / "asset.xml"

    with patch("quantcredit.acquire._open", return_value=_Response(data, url)):
      receipt = download(
        url,
        destination,
        user_agent="research contact@example.invalid",
        timeout_seconds=1,
        max_bytes=100,
        expected_bytes=len(data),
        expected_sha256=digest,
      )
    self.assertFalse(receipt.cache_hit)
    self.assertEqual(destination.read_bytes(), data)
    self.assertFalse(destination.with_name("asset.xml.part").exists())

    with patch("quantcredit.acquire._open", side_effect=AssertionError("network used")):
      cached = download(
        url,
        destination,
        user_agent="research contact@example.invalid",
        timeout_seconds=1,
        max_bytes=100,
        expected_bytes=len(data),
        expected_sha256=digest,
      )
    self.assertTrue(cached.cache_hit)

  def test_download_rejects_excess_bytes_and_cleans_partial_file(self) -> None:
    data = b"too-large"
    url = self._asset_url()
    directory = tempfile.TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    destination = Path(directory.name) / "asset.xml"

    with (
      patch("quantcredit.acquire._open", return_value=_Response(data, url, content_length=-1)),
      self.assertRaisesRegex(ValueError, "exceeds byte ceiling"),
    ):
      download(
        url,
        destination,
        user_agent="research contact@example.invalid",
        timeout_seconds=1,
        max_bytes=4,
      )
    self.assertFalse(destination.exists())
    self.assertFalse(destination.with_name("asset.xml.part").exists())

  def test_download_rejects_external_redirect(self) -> None:
    url = self._asset_url()
    directory = tempfile.TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    destination = Path(directory.name) / "asset.xml"

    with (
      patch("quantcredit.acquire._open", return_value=_Response(b"x", "https://example.com/x")),
      self.assertRaisesRegex(ValueError, "SEC URL must use"),
    ):
      download(
        url,
        destination,
        user_agent="research contact@example.invalid",
        timeout_seconds=1,
        max_bytes=10,
      )
    self.assertFalse(destination.exists())

  def test_download_rejects_partial_response(self) -> None:
    data = b"short"
    url = self._asset_url()
    directory = tempfile.TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    destination = Path(directory.name) / "asset.xml"

    with (
      patch("quantcredit.acquire._open", return_value=_Response(data, url)),
      self.assertRaisesRegex(ValueError, "byte count mismatch"),
    ):
      download(
        url,
        destination,
        user_agent="research contact@example.invalid",
        timeout_seconds=1,
        max_bytes=100,
        expected_bytes=len(data) + 1,
      )
    self.assertFalse(destination.exists())
    self.assertFalse(destination.with_name("asset.xml.part").exists())

  def test_download_rejects_checksum_mismatch(self) -> None:
    data = b"changed"
    url = self._asset_url()
    directory = tempfile.TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    destination = Path(directory.name) / "asset.xml"

    with (
      patch("quantcredit.acquire._open", return_value=_Response(data, url)),
      self.assertRaisesRegex(ValueError, "checksum mismatch"),
    ):
      download(
        url,
        destination,
        user_agent="research contact@example.invalid",
        timeout_seconds=1,
        max_bytes=100,
        expected_sha256="0" * 64,
      )
    self.assertFalse(destination.exists())
    self.assertFalse(destination.with_name("asset.xml.part").exists())

  def test_acquisition_records_pin_and_then_uses_verified_cache(self) -> None:
    raw = json.loads(MANIFEST.read_text())
    raw["filings"] = raw["filings"][:1]
    raw["filings"][0]["ex102_url"] = None
    raw["filings"][0]["bytes"] = None
    raw["filings"][0]["sha256"] = None
    manifest_path = self._write(raw)
    filing = load_manifest(manifest_path).filings[0]
    data = b"<asset/>"
    asset_url = self._asset_url()
    index = self._index("asset.xml", len(data))
    cache = manifest_path.parent / "cache"

    with patch(
      "quantcredit.acquire._open",
      side_effect=[_Response(index, filing.index_url), _Response(data, asset_url)],
    ):
      acquired = acquire_manifest(
        manifest_path,
        cache,
        user_agent="research contact@example.invalid",
        timeout_seconds=1,
        max_index_bytes=10_000,
        max_asset_bytes=100,
      )

    self.assertEqual(acquired.summary()["pinned_ex102_documents"], 1)
    self.assertEqual(acquired.filings[0].bytes, len(data))
    self.assertEqual(acquired.filings[0].sha256, hashlib.sha256(data).hexdigest())

    with patch("quantcredit.acquire._open", side_effect=AssertionError("network used")):
      repeated = acquire_manifest(
        manifest_path,
        cache,
        user_agent="research contact@example.invalid",
        timeout_seconds=1,
        max_index_bytes=10_000,
        max_asset_bytes=100,
      )
    self.assertEqual(repeated, acquired)

  def _write(self, raw: object) -> Path:
    directory = tempfile.TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    path = Path(directory.name) / "manifest.json"
    path.write_text(json.dumps(raw))
    return path

  @staticmethod
  def _index(filename: str, byte_count: int) -> bytes:
    return f"""
      <html><table><tr>
        <td>2</td><td>EX-102</td>
        <td><a href="{filename}">{filename}</a></td>
        <td>EX-102</td><td>{byte_count}</td>
      </tr></table></html>
    """.encode()

  @staticmethod
  def _asset_url() -> str:
    return "https://www.sec.gov/Archives/edgar/data/2014176/000201417625000006/asset.xml"


class _Response:
  def __init__(self, data: bytes, url: str, *, content_length: int | None = None) -> None:
    self._data = io.BytesIO(data)
    self._url = url
    length = len(data) if content_length is None else content_length
    self.headers = {"Content-Length": str(length)} if length >= 0 else {}

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
