"""Acquire and checksum-pin the EX-102 documents declared by a source manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, BinaryIO
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from quantcredit.source import (
  DEFAULT_MANIFEST,
  SHA256,
  Filing,
  SourceManifest,
  load_manifest,
  record_pin,
  validate_sec_host,
  validate_sec_url,
)

DEFAULT_CACHE = Path("data/sec")
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_INDEX_LIMIT = 5_000_000
DEFAULT_ASSET_LIMIT = 650_000_000
CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class Asset:
  url: str
  declared_bytes: int | None


@dataclass(frozen=True)
class Receipt:
  path: Path
  bytes: int
  sha256: str
  cache_hit: bool


def discover_ex102(index_html: bytes, filing: Filing, cik: str) -> Asset:
  """Return the unique EX-102 XML link declared by one filing index."""
  parser = _FilingIndexParser()
  try:
    parser.feed(index_html.decode("utf-8"))
    parser.close()
  except UnicodeDecodeError as error:
    raise ValueError(f"filing index is not UTF-8 for {filing.accession}") from error

  matches: dict[str, int | None] = {}
  for cells, links in parser.rows:
    if "EX-102" not in {cell.upper() for cell in cells}:
      continue
    declared_bytes = _last_integer(cells)
    for link in links:
      url = urljoin(filing.index_url, link)
      if not urlparse(url).path.lower().endswith(".xml"):
        continue
      validate_sec_url(url, cik, filing.accession, suffix=".xml")
      previous = matches.setdefault(url, declared_bytes)
      if previous != declared_bytes:
        raise ValueError(f"contradictory EX-102 sizes for {filing.accession}")

  if len(matches) != 1:
    raise ValueError(
      f"expected one EX-102 XML document for {filing.accession}, found {len(matches)}"
    )
  url, declared_bytes = next(iter(matches.items()))
  return Asset(url, declared_bytes)


def acquire_manifest(
  manifest_path: Path = DEFAULT_MANIFEST,
  cache: Path = DEFAULT_CACHE,
  *,
  user_agent: str,
  timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
  max_index_bytes: int = DEFAULT_INDEX_LIMIT,
  max_asset_bytes: int = DEFAULT_ASSET_LIMIT,
) -> SourceManifest:
  """Acquire every declared EX-102 and atomically record its exact pin."""
  if not user_agent.strip():
    raise ValueError("SEC user agent must be non-empty")
  if timeout_seconds <= 0 or max_index_bytes <= 0 or max_asset_bytes <= 0:
    raise ValueError("timeout and byte ceilings must be positive")

  manifest = load_manifest(manifest_path)
  for filing in manifest.filings:
    if filing.pinned:
      assert filing.ex102_url is not None
      asset = Asset(filing.ex102_url, filing.bytes)
    else:
      index_html = _fetch(
        filing.index_url,
        user_agent=user_agent,
        timeout_seconds=timeout_seconds,
        max_bytes=max_index_bytes,
      )
      asset = discover_ex102(index_html, filing, manifest.cik)

    filename = Path(urlparse(asset.url).path).name
    destination = cache / filing.accession / filename
    receipt = download(
      asset.url,
      destination,
      user_agent=user_agent,
      timeout_seconds=timeout_seconds,
      max_bytes=max_asset_bytes,
      expected_bytes=filing.bytes or asset.declared_bytes,
      expected_sha256=filing.sha256,
    )
    if not filing.pinned:
      record_pin(manifest_path, filing.accession, asset.url, receipt.bytes, receipt.sha256)

  return load_manifest(manifest_path)


def download(
  url: str,
  destination: Path,
  *,
  user_agent: str,
  timeout_seconds: float,
  max_bytes: int,
  expected_bytes: int | None = None,
  expected_sha256: str | None = None,
) -> Receipt:
  """Download one SEC document atomically or reuse an exactly verified cache."""
  validate_sec_host(url)
  if timeout_seconds <= 0 or max_bytes <= 0:
    raise ValueError("timeout and byte ceiling must be positive")
  if expected_bytes is not None and expected_bytes <= 0:
    raise ValueError("expected bytes must be positive")
  if expected_sha256 is not None and SHA256.fullmatch(expected_sha256) is None:
    raise ValueError("expected SHA-256 is invalid")

  if destination.exists():
    if expected_bytes is None or expected_sha256 is None:
      raise ValueError(f"unverified cache file exists: {destination}")
    receipt = _hash_file(destination, max_bytes=max_bytes, cache_hit=True)
    _verify(receipt, expected_bytes, expected_sha256)
    return receipt

  destination.parent.mkdir(parents=True, exist_ok=True)
  temporary = destination.with_name(f"{destination.name}.part")
  temporary.unlink(missing_ok=True)
  request = Request(url, headers={"Accept": "application/xml,text/html", "User-Agent": user_agent})
  try:
    with _open(request, timeout_seconds) as response:
      final_url = response.geturl()
      validate_sec_host(final_url)
      if final_url != url:
        raise ValueError(f"unexpected redirect for SEC document: {url} -> {final_url}")
      declared = response.headers.get("Content-Length")
      if declared is not None and int(declared) > max_bytes:
        raise ValueError(f"SEC document exceeds byte ceiling: {url}")
      with temporary.open("xb") as output:
        receipt = _copy(response, output, temporary, max_bytes=max_bytes)
    if expected_bytes is not None or expected_sha256 is not None:
      _verify(receipt, expected_bytes, expected_sha256)
    os.replace(temporary, destination)
    return Receipt(destination, receipt.bytes, receipt.sha256, cache_hit=False)
  except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
    temporary.unlink(missing_ok=True)
    if isinstance(error, ValueError):
      raise
    raise ValueError(f"cannot acquire SEC document {url}: {error}") from error


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
  parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
  parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
  parser.add_argument("--max-index-bytes", type=int, default=DEFAULT_INDEX_LIMIT)
  parser.add_argument("--max-asset-bytes", type=int, default=DEFAULT_ASSET_LIMIT)
  args = parser.parse_args(argv)
  declared = load_manifest(args.manifest)
  user_agent_env = declared.access_policy.user_agent_env
  user_agent = os.environ.get(user_agent_env)
  if user_agent is None:
    parser.error(f"acquisition requires {user_agent_env}")
  try:
    manifest = acquire_manifest(
      args.manifest,
      args.cache,
      user_agent=user_agent,
      timeout_seconds=args.timeout_seconds,
      max_index_bytes=args.max_index_bytes,
      max_asset_bytes=args.max_asset_bytes,
    )
  except ValueError as error:
    parser.error(str(error))
  print(json.dumps(manifest.summary(), indent=2, sort_keys=True))


class _FilingIndexParser(HTMLParser):
  def __init__(self) -> None:
    super().__init__(convert_charrefs=True)
    self.rows: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    self._cells: list[str] | None = None
    self._links: list[str] = []
    self._text: list[str] | None = None

  def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
    if tag == "tr":
      self._cells, self._links = [], []
    elif tag in {"td", "th"} and self._cells is not None:
      self._text = []
    elif tag == "a" and self._cells is not None:
      href = dict(attrs).get("href")
      if href is not None:
        self._links.append(href)

  def handle_data(self, data: str) -> None:
    if self._text is not None:
      self._text.append(data)

  def handle_endtag(self, tag: str) -> None:
    if tag in {"td", "th"} and self._cells is not None and self._text is not None:
      self._cells.append(" ".join(" ".join(self._text).split()))
      self._text = None
    elif tag == "tr" and self._cells is not None:
      if self._cells:
        self.rows.append((tuple(self._cells), tuple(self._links)))
      self._cells, self._links, self._text = None, [], None


def _fetch(url: str, *, user_agent: str, timeout_seconds: float, max_bytes: int) -> bytes:
  validate_sec_host(url)
  request = Request(url, headers={"Accept": "text/html", "User-Agent": user_agent})
  try:
    with _open(request, timeout_seconds) as response:
      final_url = response.geturl()
      validate_sec_host(final_url)
      if final_url != url:
        raise ValueError(f"unexpected redirect for SEC index: {url} -> {final_url}")
      declared = response.headers.get("Content-Length")
      if declared is not None and int(declared) > max_bytes:
        raise ValueError(f"SEC index exceeds byte ceiling: {url}")
      content = bytearray()
      while chunk := response.read(min(CHUNK_BYTES, max_bytes + 1 - len(content))):
        content.extend(chunk)
        if len(content) > max_bytes:
          raise ValueError(f"SEC index exceeds byte ceiling: {url}")
      return bytes(content)
  except (HTTPError, URLError, TimeoutError, OSError) as error:
    raise ValueError(f"cannot acquire SEC index {url}: {error}") from error


def _copy(source: BinaryIO, output: BinaryIO, path: Path, *, max_bytes: int) -> Receipt:
  digest = hashlib.sha256()
  byte_count = 0
  while chunk := source.read(CHUNK_BYTES):
    byte_count += len(chunk)
    if byte_count > max_bytes:
      raise ValueError(f"SEC document exceeds byte ceiling: {path}")
    output.write(chunk)
    digest.update(chunk)
  output.flush()
  os.fsync(output.fileno())
  return Receipt(path, byte_count, digest.hexdigest(), cache_hit=False)


def _hash_file(path: Path, *, max_bytes: int, cache_hit: bool) -> Receipt:
  digest = hashlib.sha256()
  byte_count = 0
  with path.open("rb") as source:
    while chunk := source.read(CHUNK_BYTES):
      byte_count += len(chunk)
      if byte_count > max_bytes:
        raise ValueError(f"cached SEC document exceeds byte ceiling: {path}")
      digest.update(chunk)
  return Receipt(path, byte_count, digest.hexdigest(), cache_hit)


def _verify(receipt: Receipt, expected_bytes: int | None, expected_sha256: str | None) -> None:
  if expected_bytes is not None and receipt.bytes != expected_bytes:
    raise ValueError(f"SEC document byte count mismatch: {receipt.path}")
  if expected_sha256 is not None and receipt.sha256 != expected_sha256:
    raise ValueError(f"SEC document checksum mismatch: {receipt.path}")


def _last_integer(cells: tuple[str, ...]) -> int | None:
  for cell in reversed(cells):
    number = cell.replace(",", "")
    if number.isdigit():
      return int(number)
  return None


def _open(request: Request, timeout_seconds: float) -> Any:
  return urlopen(request, timeout=timeout_seconds)


if __name__ == "__main__":
  main()
