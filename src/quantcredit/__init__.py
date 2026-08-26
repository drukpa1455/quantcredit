"""Executable consumer-credit research."""

from quantcredit.api import (
  audit,
  challenge,
  confirm,
  decide,
  evaluate,
  examples,
  fit,
  forecast,
  history,
  project,
  reveal,
  select,
  split,
  waterfall,
)
from quantcredit.cashflows import Tranche

__all__ = [
  "Tranche",
  "audit",
  "challenge",
  "confirm",
  "decide",
  "evaluate",
  "examples",
  "fit",
  "forecast",
  "history",
  "project",
  "reveal",
  "select",
  "split",
  "waterfall",
]

__version__ = "0.1.0"
