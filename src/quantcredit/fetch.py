"""Download one HTTPS resource into a bounded, verified local cache."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

CHUNK_BYTES = 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_BYTES = 500_000_000


@dataclass(frozen=True)
class Receipt:
  path: Path
  bytes: int
  sha256: str
  cache_hit: bool


def fetch(
  url: str,
  destination: Path,
  *,
  timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
  max_bytes: int = DEFAULT_MAX_BYTES,
  expected_bytes: int | None = None,
  expected_sha256: str | None = None,
  headers: Mapping[str, str] | None = None,
  allow_redirects: bool = True,
) -> Receipt:
  """Atomically download one HTTPS resource or reuse its verified cache."""
  _validate_https(url)
  if timeout_seconds <= 0 or max_bytes <= 0:
    raise ValueError("timeout and byte ceiling must be positive")
  if expected_bytes is not None and expected_bytes <= 0:
    raise ValueError("expected bytes must be positive")
  if expected_sha256 is not None and not _is_sha256(expected_sha256):
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
  request = Request(url, headers=dict(headers or {}))
  try:
    with _open(request, timeout_seconds) as response:
      final_url = response.geturl()
      _validate_https(final_url)
      if not allow_redirects and final_url != url:
        raise ValueError(f"unexpected redirect: {url} -> {final_url}")
      declared = response.headers.get("Content-Length")
      if declared is not None and int(declared) > max_bytes:
        raise ValueError(f"download exceeds byte ceiling: {url}")
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
    raise ValueError(f"cannot download {url}: {error}") from error


def verify(
  path: Path,
  *,
  expected_bytes: int,
  expected_sha256: str,
  max_bytes: int = DEFAULT_MAX_BYTES,
) -> Receipt:
  """Verify one existing cache file without performing network I/O."""
  if expected_bytes <= 0 or max_bytes <= 0:
    raise ValueError("expected bytes and byte ceiling must be positive")
  if not _is_sha256(expected_sha256):
    raise ValueError("expected SHA-256 is invalid")
  try:
    receipt = _hash_file(path, max_bytes=max_bytes, cache_hit=True)
  except OSError as error:
    raise ValueError(f"cannot verify cached download {path}: {error}") from error
  _verify(receipt, expected_bytes, expected_sha256)
  return receipt


def _validate_https(url: str) -> None:
  parsed = urlparse(url)
  if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
    raise ValueError(f"download URL must be public HTTPS: {url}")


def _is_sha256(value: str) -> bool:
  try:
    return (
      value == value.lower()
      and len(value) == hashlib.sha256().digest_size * 2
      and len(bytes.fromhex(value)) == 32
    )
  except ValueError:
    return False


def _copy(source: BinaryIO, output: BinaryIO, path: Path, *, max_bytes: int) -> Receipt:
  digest = hashlib.sha256()
  byte_count = 0
  while chunk := source.read(CHUNK_BYTES):
    byte_count += len(chunk)
    if byte_count > max_bytes:
      raise ValueError(f"download exceeds byte ceiling: {path}")
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
        raise ValueError(f"cached download exceeds byte ceiling: {path}")
      digest.update(chunk)
  return Receipt(path, byte_count, digest.hexdigest(), cache_hit)


def _verify(receipt: Receipt, expected_bytes: int | None, expected_sha256: str | None) -> None:
  if expected_bytes is not None and receipt.bytes != expected_bytes:
    raise ValueError(f"download byte count mismatch: {receipt.path}")
  if expected_sha256 is not None and receipt.sha256 != expected_sha256:
    raise ValueError(f"download checksum mismatch: {receipt.path}")


def _open(request: Request, timeout_seconds: float) -> Any:
  return urlopen(request, timeout=timeout_seconds)
