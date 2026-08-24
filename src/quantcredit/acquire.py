"""Acquire and checksum-pin the EX-102 documents declared by a source manifest."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from quantcredit.fetch import Receipt, fetch, verify
from quantcredit.source import (
  DEFAULT_MANIFEST,
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


def verify_asset(
  filing: Filing,
  cache: Path = DEFAULT_CACHE,
  *,
  max_bytes: int = DEFAULT_ASSET_LIMIT,
) -> Receipt:
  """Resolve and verify one pinned EX-102 cache asset."""
  if not filing.pinned:
    raise ValueError(f"unpinned EX-102 source: {filing.accession}")
  assert filing.ex102_url is not None
  assert filing.bytes is not None
  assert filing.sha256 is not None
  filename = Path(urlparse(filing.ex102_url).path).name
  return verify(
    cache / filing.accession / filename,
    expected_bytes=filing.bytes,
    expected_sha256=filing.sha256,
    max_bytes=max_bytes,
  )


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
  return fetch(
    url,
    destination,
    headers={"Accept": "application/xml,text/html", "User-Agent": user_agent},
    timeout_seconds=timeout_seconds,
    max_bytes=max_bytes,
    expected_bytes=expected_bytes,
    expected_sha256=expected_sha256,
    allow_redirects=False,
  )


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
