"""Executable consumer-credit research."""

from quantcredit.api import (
  audit,
  decide,
  evaluate,
  examples,
  fit,
  project,
  select,
  split,
  waterfall,
)
from quantcredit.cashflows import Tranche

__all__ = [
  "Tranche",
  "audit",
  "decide",
  "evaluate",
  "examples",
  "fit",
  "project",
  "select",
  "split",
  "waterfall",
]

__version__ = "0.1.0"
