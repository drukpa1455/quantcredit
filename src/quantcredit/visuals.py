"""Sapphire figures for aggregate credit evidence and causal time."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, timedelta
from typing import Any, cast

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import StrMethodFormatter

from quantcredit.splits import CausalSplit

# Ported from Reia Sapphire at revision 0ad104c; quantcredit owns this small snapshot.
_COLORS = {
  "background": "#212c2a",
  "surface": "#1e2826",
  "deep": "#171f1d",
  "foreground": "#f8f8f2",
  "muted": "#aeb4af",
  "accent": "#a7ffa0",
  "border": "#324d30",
  "cyan": "#80ffea",
  "green": "#8aff80",
  "orange": "#ffca80",
  "purple": "#9580ff",
  "pink": "#ff80bf",
  "yellow": "#ffff80",
  "red": "#ff9580",
  "positive": "#7de972",
  "negative": "#e67070",
}

_STYLE: dict[str, Any] = {
  "figure.facecolor": _COLORS["background"],
  "figure.dpi": 120,
  "savefig.facecolor": _COLORS["background"],
  "axes.facecolor": _COLORS["surface"],
  "axes.edgecolor": _COLORS["border"],
  "axes.labelcolor": _COLORS["foreground"],
  "axes.titlecolor": _COLORS["foreground"],
  "axes.titleweight": "bold",
  "axes.grid": True,
  "axes.axisbelow": True,
  "axes.prop_cycle": cycler(
    color=[
      _COLORS["cyan"],
      _COLORS["green"],
      _COLORS["orange"],
      _COLORS["purple"],
      _COLORS["pink"],
      _COLORS["yellow"],
      _COLORS["red"],
    ]
  ),
  "font.family": "monospace",
  "font.monospace": ["SFMono-Regular", "Menlo", "Monaco", "DejaVu Sans Mono"],
  "font.size": 10,
  "grid.color": _COLORS["foreground"],
  "grid.alpha": 0.12,
  "grid.linewidth": 0.6,
  "legend.frameon": False,
  "legend.labelcolor": _COLORS["foreground"],
  "text.color": _COLORS["foreground"],
  "xtick.color": _COLORS["muted"],
  "ytick.color": _COLORS["muted"],
}

_STATE_ORDER = (
  "delinquency:current",
  "delinquency:1-29",
  "delinquency:30-59",
  "delinquency:60-89",
  "delinquency:90+",
  "delinquency:missing",
  "zero_balance:1",
  "zero_balance:3",
  "zero_balance:4",
)
_STATE_LABEL = {
  "delinquency:current": "Current",
  "delinquency:1-29": "1-29",
  "delinquency:30-59": "30-59",
  "delinquency:60-89": "60-89",
  "delinquency:90+": "90+",
  "delinquency:missing": "Missing",
  "zero_balance:1": "Paid/matured",
  "zero_balance:3": "Repurchased",
  "zero_balance:4": "Charged off",
}
_STATE_COLOR = {
  "delinquency:current": _COLORS["cyan"],
  "delinquency:1-29": _COLORS["green"],
  "delinquency:30-59": _COLORS["yellow"],
  "delinquency:60-89": _COLORS["orange"],
  "delinquency:90+": _COLORS["pink"],
  "delinquency:missing": _COLORS["purple"],
  "zero_balance:1": _COLORS["muted"],
  "zero_balance:3": _COLORS["purple"],
  "zero_balance:4": _COLORS["negative"],
}


@contextmanager
def sapphire() -> Iterator[None]:
  """Apply the repository's scoped Matplotlib theme."""
  with mpl.rc_context(cast(Any, _STYLE)):
    yield


def plot_audit(audit: dict[str, Any]) -> Figure:
  """Show population, state, transition, and target evidence without source rows."""
  with sapphire():
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    figure.suptitle("Consumer credit data audit", fontsize=16, fontweight="bold")
    _plot_population(axes[0, 0], audit)
    _plot_states(axes[0, 1], audit)
    _plot_transitions(axes[1, 0], audit)
    _plot_targets(axes[1, 1], audit)
    return figure


def plot_split(split: CausalSplit) -> Figure:
  """Show when features are measured and when each label horizon matures."""
  with sapphire():
    figure, raw_axis = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    axis = cast(Any, raw_axis)
    lanes = (
      (
        "Train",
        split.train_cutoffs[0],
        split.train_labels_observed_through,
        _COLORS["cyan"],
      ),
      (
        "Validation",
        split.validation_cutoff,
        split.validation_labels_observed_through,
        _COLORS["orange"],
      ),
      ("Test", split.test_cutoff, split.test_labels_observed_through, _COLORS["pink"]),
    )
    for lane, (label, cutoff, maturity, color) in enumerate(reversed(lanes)):
      axis.hlines(lane, cutoff, maturity, color=color, linewidth=12, alpha=0.28)
      axis.plot(cutoff, lane, "o", color=color, markersize=8)
      axis.plot(maturity, lane, "|", color=color, markersize=18, markeredgewidth=2)
      axis.annotate(
        cutoff.strftime("%b %d\nfeatures"),
        (cutoff, lane),
        xytext=(0, 14),
        textcoords="offset points",
        ha="center",
        color=color,
        fontsize=9,
      )
      axis.annotate(
        maturity.strftime("%b %d\nlabels mature"),
        (maturity, lane),
        xytext=(0, -30),
        textcoords="offset points",
        ha="center",
        color=_COLORS["muted"],
        fontsize=9,
      )

    start = split.train_cutoffs[0] - timedelta(days=12)
    end = split.test_labels_observed_through + timedelta(days=12)
    axis.set(
      title=f"Causal split · {split.horizon_reports}-report label horizon",
      xlabel=_period_label([split.train_cutoffs[0], split.test_labels_observed_through]),
      xlim=(start, end),
      ylim=(-0.6, 2.6),
      yticks=range(3),
      yticklabels=("Test", "Validation", "Train"),
    )
    axis.xaxis.set_major_locator(
      mdates.MonthLocator(bymonth=(2, 4, 6, 8, 10, 12))  # type: ignore[no-untyped-call]
    )
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b"))  # type: ignore[no-untyped-call]
    axis.grid(axis="y", visible=False)
    return figure


def _plot_population(axis: Any, audit: dict[str, Any]) -> None:
  continuity = cast(list[dict[str, Any]], audit["continuity"])
  periods = [date.fromisoformat(str(row["report_period"])) for row in continuity]
  reported = [int(row["reported"]) for row in continuity]
  axis.plot(periods, reported, color=_COLORS["cyan"], linewidth=2.2, marker="o", markersize=4)
  axis.fill_between(periods, reported, color=_COLORS["cyan"], alpha=0.08)
  axis.annotate(
    f"{reported[0]:,}",
    (periods[0], reported[0]),
    xytext=(8, 8),
    textcoords="offset points",
    color=_COLORS["foreground"],
  )
  axis.annotate(
    f"{reported[-1]:,}",
    (periods[-1], reported[-1]),
    xytext=(-8, 8),
    textcoords="offset points",
    ha="right",
    color=_COLORS["foreground"],
  )
  axis.set(title="Reported loan population", xlabel=_period_label(periods), ylabel="Loans")
  axis.xaxis.set_major_locator(
    mdates.MonthLocator(bymonth=(2, 4, 6, 8, 10, 12))  # type: ignore[no-untyped-call]
  )
  axis.xaxis.set_major_formatter(mdates.DateFormatter("%b"))  # type: ignore[no-untyped-call]
  axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))


def _plot_states(axis: Any, audit: dict[str, Any]) -> None:
  observed = cast(dict[str, int], audit["states"])
  states = _ordered_states(observed)
  counts = [observed[state] for state in states]
  total = sum(counts)
  positions = range(len(states))
  axis.barh(
    positions,
    counts,
    color=[_STATE_COLOR.get(state, _COLORS["accent"]) for state in states],
    alpha=0.82,
  )
  for position, count in zip(positions, counts, strict=True):
    axis.text(
      count * 1.15,
      position,
      f"{count / total:.2%} · {count:,}",
      va="center",
      color=_COLORS["foreground"],
      fontsize=8,
    )
  axis.set(
    title="Observed state share · log scale",
    xlabel="Snapshots",
    yticks=positions,
    yticklabels=[_state_label(state) for state in states],
    xscale="log",
  )
  axis.set_xlim(left=max(1, min(counts) / 2), right=max(counts) * 12)
  axis.invert_yaxis()
  axis.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))


def _period_label(periods: list[date]) -> str:
  years = sorted({period.year for period in periods})
  span = str(years[0]) if len(years) == 1 else f"{years[0]}-{years[-1]}"
  return f"{span} report period"


def _ordered_states(observed: dict[str, int]) -> list[str]:
  known = [state for state in _STATE_ORDER if observed.get(state, 0) > 0]
  unknown = sorted(
    state for state, count in observed.items() if count > 0 and state not in _STATE_ORDER
  )
  return known + unknown


def _state_label(state: str) -> str:
  return _STATE_LABEL.get(state, state.replace("zero_balance:", "Zero balance "))


def _plot_transitions(axis: Any, audit: dict[str, Any]) -> None:
  observed_states = cast(dict[str, int], audit["states"])
  states = _ordered_states(observed_states)
  index = {state: position for position, state in enumerate(states)}
  matrix = [[0 for _ in states] for _ in states]
  transitions = cast(dict[str, int], audit["transitions"])
  for transition, count in transitions.items():
    source, destination = transition.split(" -> ", 1)
    if source in index and destination in index:
      matrix[index[source]][index[destination]] = count
  rates = [
    [count / total if total else 0 for count in row]
    for row in matrix
    for total in [sum(row)]
  ]
  annotations = [
    ["" if rate == 0 else f"{rate:.0%}" if rate >= 0.01 else "<1%" for rate in row]
    for row in rates
  ]
  colors = LinearSegmentedColormap.from_list(
    "sapphire-transition", (_COLORS["deep"], _COLORS["cyan"])
  )
  labels = [_state_label(state) for state in states]
  sns.heatmap(
    rates,
    ax=axis,
    annot=annotations,
    fmt="",
    cmap=colors,
    vmin=0,
    vmax=1,
    cbar=False,
    linewidths=0.5,
    linecolor=_COLORS["border"],
    xticklabels=labels,
    yticklabels=labels,
    annot_kws={"fontsize": 7},
  )
  axis.set(title="Next-report transition rate", xlabel="To state", ylabel="From state")
  axis.tick_params(axis="x", labelrotation=42, labelsize=8)
  axis.tick_params(axis="y", labelrotation=0, labelsize=8)


def _plot_targets(axis: Any, audit: dict[str, Any]) -> None:
  targets = cast(list[dict[str, Any]], audit["targets"])
  decision = next(target for target in targets if target["status"] == "derived")
  counts = cast(dict[str, int], decision["counts"])
  categories = (
    ("positive", "Event", _COLORS["negative"]),
    ("negative", "No event", _COLORS["cyan"]),
    ("competing_event", "Competing", _COLORS["orange"]),
    ("right_censored", "Right-censored", _COLORS["purple"]),
    ("ineligible_at_cutoff", "Ineligible", _COLORS["muted"]),
    ("missing_followup", "Missing follow-up", _COLORS["yellow"]),
  )
  positions = range(len(categories))
  values = [counts[key] for key, _, _ in categories]
  for position, (value, (_, _, color)) in enumerate(zip(values, categories, strict=True)):
    if value:
      axis.barh(position, value, color=color, alpha=0.82)
    else:
      axis.plot(1, position, "|", color=color, markersize=12, markeredgewidth=2)
    axis.text(
      max(value, 1) * 1.2,
      position,
      f"{value:,}",
      va="center",
      color=_COLORS["foreground"],
      fontsize=8,
    )
  axis.set(
    title="Three-report target disposition · log scale",
    xlabel="Loan-cutoff positions",
    yticks=positions,
    yticklabels=[label for _, label, _ in categories],
    xscale="log",
    xlim=(0.7, max(values) * 8),
  )
  axis.invert_yaxis()
  axis.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
