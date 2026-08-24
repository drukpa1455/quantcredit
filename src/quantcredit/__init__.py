"""Executable consumer-credit research."""

from quantcredit.audit import Audit
from quantcredit.audit import audit_sources as audit
from quantcredit.splits import CausalSplit
from quantcredit.splits import causal_split as split

__all__ = ["Audit", "CausalSplit", "audit", "split"]

__version__ = "0.1.0"
