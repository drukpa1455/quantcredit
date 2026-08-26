"""Sapphire figures for aggregate credit evidence, causal time, and populations."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, timedelta
from math import ceil
from typing import TYPE_CHECKING, Any, cast

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter, ScalarFormatter, StrMethodFormatter
from pandas import DataFrame
from pandas.api.types import is_numeric_dtype

from quantcredit.audits import Audit
from quantcredit.populations import FEATURE_COLUMNS
from quantcredit.splits import CausalSplit

if TYPE_CHECKING:
  from quantcredit.baselines import Baseline, Evaluation, Exposure
  from quantcredit.cashflows import Deal
  from quantcredit.challengers import GraphEvaluation, GraphStudy
  from quantcredit.decisions import Decision

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
  "axes.spines.top": False,
  "axes.spines.right": False,
  "axes.grid": True,
  "axes.grid.axis": "y",
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
_STATE_COMPACT_LABEL = {
  **_STATE_LABEL,
  "zero_balance:1": "Paid",
  "zero_balance:3": "Repurch.",
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
_FOLD_ORDER = ("train", "validation", "test")
_STATUS_ORDER = (
  "positive",
  "negative",
  "competing_event",
  "missing_followup",
  "right_censored",
  "held_out",
)
_STATUS_LABEL = {
  "positive": "Event",
  "negative": "No event",
  "competing_event": "Competing",
  "missing_followup": "Missing follow-up",
  "right_censored": "Right-censored",
  "held_out": "Held out",
}
_STATUS_COLOR = {
  "positive": _COLORS["negative"],
  "negative": _COLORS["cyan"],
  "competing_event": _COLORS["orange"],
  "missing_followup": _COLORS["yellow"],
  "right_censored": _COLORS["purple"],
  "held_out": _COLORS["muted"],
}
_STATUS_HATCH = ("", "//", "xx", "..", "\\\\", "++")


@contextmanager
def sapphire() -> Iterator[None]:
  """Apply the repository's scoped Matplotlib theme."""
  with mpl.rc_context(cast(Any, _STYLE)):
    yield


def plot_graph_study(study: GraphStudy) -> Figure:
  """Compare ensemble discrimination and calibration for matched challengers."""
  all_results = study.summary()
  summary = all_results.loc[all_results["valid"]].sort_values("log_loss")
  valid_arms = set(summary["arm"])
  calibration = study.calibration.loc[study.calibration["arm"].isin(valid_arms)]
  labels = summary["arm"].str.replace("_", " ")
  with sapphire():
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].barh(labels, summary["log_loss"], color=_COLORS["cyan"], alpha=0.82)
    axes[0].invert_yaxis()
    axes[0].set(title="Validation log loss · lower is better", xlabel="Log loss")
    for arm, group in calibration.groupby("arm", observed=True):
      axes[1].plot(group["mean_score"], group["event_rate"], marker="o", label=arm)
    axes[1].plot((0, 1), (0, 1), linestyle="--", color=_COLORS["muted"], alpha=0.6)
    axes[1].set(
      title="Ensemble score-band calibration",
      xlabel="Mean predicted probability",
      ylabel="Observed event rate",
    )
    axes[1].legend(fontsize=7)
    invalid = int((~all_results["valid"]).sum())
    suffix = "" if invalid == 0 else f" · {invalid} invalid constant control"
    figure.suptitle(
      f"Matched graph decision: {study.decision.replace('_', ' ')}{suffix}"
    )
  return figure


def plot_graph_evaluation(evaluation: GraphEvaluation) -> Figure:
  """Compare frozen test loss and discrimination without implying reselection."""
  results = evaluation.results.sort_values("log_loss")
  labels = results["arm"].str.replace("_", " ")
  with sapphire():
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].barh(labels, results["log_loss"], color=_COLORS["cyan"], alpha=0.82)
    axes[0].invert_yaxis()
    axes[0].set(title="Frozen test log loss", xlabel="Log loss")
    axes[1].barh(
      labels,
      results["average_precision"],
      color=_COLORS["accent"],
      alpha=0.82,
    )
    axes[1].invert_yaxis()
    axes[1].set(title="Frozen test average precision", xlabel="Average precision")
    figure.suptitle(
      f"Validation {evaluation.validation_decision.replace('_', ' ')} → "
      f"{evaluation.decision.replace('_', ' ')}"
    )
  return figure


def plot_audit(audit: Audit) -> Figure:
  """Show population, state, transition, and target evidence without source rows."""
  reported = [int(row["reported"]) for row in audit.continuity]
  if reported[0] == 0:
    population_finding = f"Population reaches {reported[-1]:,}"
  else:
    change = reported[-1] / reported[0] - 1
    population_finding = (
      "Population is unchanged"
      if change == 0
      else f"Population {'contracts' if change < 0 else 'expands'} {abs(change):.1%}"
    )
  with sapphire():
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    figure.suptitle(
      f"{population_finding} across {len(reported)} reports",
      fontsize=16,
      fontweight="bold",
    )
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
      title=f"Labels mature before the next fold · {split.horizon_reports}-report horizon",
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


def plot_examples(examples: DataFrame) -> Figure:
  """Show fold composition, event rate, missingness, and robust feature drift."""
  required = {"fold", "target_status", "target", *FEATURE_COLUMNS}
  missing = sorted(required - set(examples.columns))
  if missing:
    raise ValueError(f"examples are missing required columns: {', '.join(missing)}")
  if examples.empty:
    raise ValueError("examples must contain at least one eligible cutoff")

  folds = _ordered(examples["fold"], _FOLD_ORDER)
  if not folds or examples["target_status"].dropna().empty:
    raise ValueError("examples require reported folds and target dispositions")
  missing_rows = _missingness(examples, folds)
  drift_rows = _drift(examples, folds)
  largest_shift = max(
    (abs(value) for _, shifts in drift_rows for value in shifts if value == value),
    default=0.0,
  )
  height = max(9.0, 5.4 + 0.24 * max(len(missing_rows), len(drift_rows)))
  with sapphire():
    figure, axes = plt.subplots(2, 2, figsize=(15, height), constrained_layout=True)
    held_out = "test outcomes held out · " if "held_out" in set(examples["target_status"]) else ""
    figure.suptitle(
      f"Causal population · {held_out}largest median shift {largest_shift:.2f} train IQR",
      fontsize=16,
      fontweight="bold",
    )
    _plot_fold_composition(axes[0, 0], examples, folds)
    _plot_event_rate(axes[0, 1], examples, folds)
    _plot_missingness(axes[1, 0], missing_rows, folds)
    _plot_drift(axes[1, 1], drift_rows, folds)
    return figure


def plot_baseline(baseline: Baseline) -> Figure:
  """Show validation-only model selection, ranking, calibration, and importance."""
  with sapphire():
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    figure.suptitle(
      f"Depth {baseline.selected_depth} selected within validation uncertainty",
      fontsize=16,
      fontweight="bold",
    )
    _plot_candidate_loss(axes[0, 0], baseline)
    _plot_candidate_ranking(axes[0, 1], baseline)
    _plot_calibration(
      axes[1, 0],
      baseline.calibration,
      baseline.validation.brier_score,
      "Score-band calibration",
      "Validation",
    )
    _plot_importance(axes[1, 1], baseline)
    return figure


def plot_evaluation(evaluation: Evaluation) -> Figure:
  """Show aggregate validation-to-test evidence for one frozen model."""
  validation = evaluation.baseline.validation
  with sapphire():
    figure, axes = plt.subplots(
      2,
      2,
      figsize=(14, 8),
      constrained_layout=True,
      gridspec_kw={"height_ratios": (0.7, 1.3)},
    )
    figure.suptitle(
      (
        f"Frozen test AUROC {evaluation.metrics.auroc:.3f} vs "
        f"{validation.auroc:.3f} validation"
      ),
      fontsize=16,
      fontweight="bold",
    )
    _plot_score_comparison(
      axes[0, 0],
      "Log loss · lower is better",
      validation.log_loss,
      evaluation.metrics.log_loss,
      evaluation.baseline.reference.log_loss,
      evaluation.reference.log_loss,
    )
    _plot_score_comparison(
      axes[0, 1],
      "Brier score · lower is better",
      validation.brier_score,
      evaluation.metrics.brier_score,
      evaluation.baseline.reference.brier_score,
      evaluation.reference.brier_score,
    )
    _plot_ranking_comparison(axes[1, 0], evaluation)
    _plot_calibration(
      axes[1, 1],
      evaluation.calibration,
      evaluation.metrics.brier_score,
      "Test score-band calibration",
      "Test",
    )
    return figure


def plot_exposure(exposure: Exposure) -> Figure:
  """Show where balance and predicted versus observed event exposure concentrate."""
  bands = exposure.bands
  positions = list(range(len(bands)))
  top_share = float(bands.iloc[-1]["total_exposure"] / exposure.total_exposure)
  top_expected_share = float(
    bands.iloc[-1]["expected_event_exposure"] / exposure.expected_event_exposure
  )
  with sapphire():
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    figure.suptitle(
      (
        f"Top risk band holds {top_expected_share:.0%} of modeled event exposure"
        "—not ultimate loss"
      ),
      fontsize=16,
      fontweight="bold",
    )

    bars = axes[0].bar(
      positions,
      bands["total_exposure"],
      color=_COLORS["cyan"],
      alpha=0.82,
    )
    bars[-1].set_color(_COLORS["accent"])
    axes[0].annotate(
      f"{top_share:.1%} of exposure",
      (positions[-1], float(bands.iloc[-1]["total_exposure"])),
      xytext=(0, 7),
      textcoords="offset points",
      ha="center",
      color=_COLORS["accent"],
      fontsize=8,
    )
    axes[0].set(
      title="Outstanding balance by risk band",
      xlabel="Test score band · low to high risk",
      ylabel="Cutoff balance ($)",
      xticks=positions,
      xticklabels=bands["score_band"],
      ylim=(0, float(bands["total_exposure"].max()) * 1.18),
    )
    axes[0].yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))

    predicted = axes[1].plot(
      positions,
      bands["expected_event_exposure"],
      color=_COLORS["cyan"],
      linewidth=2,
      marker="o",
      markersize=4,
    )[0]
    observed = axes[1].plot(
      positions,
      bands["observed_event_exposure"],
      color=_COLORS["orange"],
      linewidth=1.6,
      linestyle="--",
      marker="x",
      markersize=5,
    )[0]
    for line, label, offset in (
      (predicted, "PD x EAD", 8),
      (observed, "Event-loan EAD", -10),
    ):
      axes[1].annotate(
        f"{label} ${line.get_ydata()[-1]:,.0f}",
        (positions[-1], line.get_ydata()[-1]),
        xytext=(7, offset),
        textcoords="offset points",
        color=line.get_color(),
        fontsize=7,
        va="center",
      )
    axes[1].set(
      title="Predicted versus observed event exposure",
      xlabel="Test score band · low to high risk",
      ylabel="Event-associated cutoff balance ($)",
      xticks=positions,
      xticklabels=bands["score_band"],
      xlim=(-0.4, len(positions) - 0.2),
      ylim=(
        0,
        max(
          float(bands["expected_event_exposure"].max()),
          float(bands["observed_event_exposure"].max()),
        )
        * 1.25,
      ),
    )
    axes[1].yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
    return figure


def plot_decision(decision: Decision) -> Figure:
  """Show validation feature shape, residual cohorts, and selection efficiency."""
  features = list(dict.fromkeys(decision.effects["feature"]))
  if len(features) < 2:
    raise ValueError("decision figure requires at least two effect features")
  target = decision.frontier.loc[
    decision.frontier["target_excluded_share"].sub(0.10).abs().idxmin(),
    "target_excluded_share",
  ]
  comparison = decision.frontier.loc[decision.frontier["target_excluded_share"] == target]
  model = float(
    comparison.loc[comparison["policy"] == "GBM score", "event_exposure_avoided"].iloc[0]
  )
  simple = float(
    comparison.loc[comparison["policy"] != "GBM score", "event_exposure_avoided"].max()
  )
  with sapphire():
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    figure.suptitle(
      (
        f"At {float(cast(Any, target)):.0%} excluded balance, "
        f"GBM avoids {model:.1%} event exposure"
        f" vs {simple:.1%} best simple rule"
      ),
      fontsize=16,
      fontweight="bold",
    )
    _plot_effect(axes[0, 0], decision.effects, features[0])
    _plot_effect(axes[0, 1], decision.effects, features[1])
    _plot_cohort_residuals(axes[1, 0], decision.cohorts)
    _plot_frontier(axes[1, 1], decision.frontier)
    return figure


def plot_deal(deal: Deal) -> Figure:
  """Show collateral cash generation and tranche balance runoff for one scenario."""
  collateral = deal.collateral.copy()
  collateral["principal_cash"] = (
    collateral["scheduled_principal"]
    + collateral["prepayment"]
    + collateral["recovery"]
  )
  summary = deal.summary()
  impaired = summary.loc[summary["loss"] > 0, "tranche"].tolist()
  finding = "No tranche principal loss" if not impaired else f"Loss reaches {', '.join(impaired)}"
  with sapphire():
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    figure.suptitle(
      f"{finding} in the declared scenario—not a deal valuation",
      fontsize=16,
      fontweight="bold",
    )
    axes[0].plot(
      collateral["month"],
      collateral["interest"],
      color=_COLORS["cyan"],
      linewidth=1.8,
      label="Interest",
    )
    axes[0].plot(
      collateral["month"],
      collateral["principal_cash"],
      color=_COLORS["green"],
      linewidth=1.8,
      label="Principal + recovery",
    )
    axes[0].plot(
      collateral["month"],
      collateral["loss"],
      color=_COLORS["negative"],
      linewidth=1.5,
      linestyle="--",
      label="Net loss",
    )
    axes[0].set(
      title="Collateral cash and loss timing",
      xlabel="Scenario month",
      ylabel="Amount ($)",
      xlim=(1, int(collateral["month"].max())),
      ylim=(
        0,
        max(
          float(collateral["interest"].max()),
          float(collateral["principal_cash"].max()),
          float(collateral["loss"].max()),
        )
        * 1.12,
      ),
    )
    axes[0].yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
    axes[0].legend(fontsize=8)

    colors = (_COLORS["cyan"], _COLORS["orange"], _COLORS["pink"], _COLORS["purple"])
    for index, tranche in enumerate(deal.tranches):
      frame = deal.cashflows.loc[deal.cashflows["tranche"] == tranche.name]
      axes[1].plot(
        frame["month"],
        frame["ending_balance"],
        color=colors[index % len(colors)],
        linewidth=2,
        label=tranche.name,
      )
    axes[1].set(
      title="Tranche principal runoff · senior paid first",
      xlabel="Scenario month",
      ylabel="Ending tranche balance ($)",
      xlim=(1, int(collateral["month"].max())),
      ylim=(0, max(tranche.balance for tranche in deal.tranches) * 1.05),
    )
    axes[1].yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
    axes[1].legend(fontsize=8)
    return figure


def _plot_effect(axis: Any, effects: DataFrame, feature: str) -> None:
  frame = effects.loc[(effects["feature"] == feature) & (effects["band"] != "missing")]
  positions = list(range(len(frame)))
  axis.plot(
    positions,
    frame["mean_score"],
    color=_COLORS["cyan"],
    linewidth=1.8,
    marker="o",
    markersize=4,
    label="Predicted",
  )
  axis.plot(
    positions,
    frame["event_rate"],
    color=_COLORS["orange"],
    linewidth=1.5,
    linestyle="--",
    marker="x",
    markersize=5,
    label="Observed",
  )
  axis.set(
    title=feature.replace("_", " ").title(),
    xlabel="Validation feature band · low to high",
    ylabel="Event probability",
    xticks=positions,
    xticklabels=frame["band"],
    ylim=(
      0,
      max(float(frame["event_rate"].max()), float(frame["mean_score"].max()), 0.01)
      * 1.25,
    ),
  )
  axis.yaxis.set_major_formatter(PercentFormatter(1.0))
  axis.legend(fontsize=8)


def _plot_cohort_residuals(axis: Any, cohorts: DataFrame) -> None:
  if cohorts.empty:
    _empty_axis(axis, "Largest cohort calibration residuals", "No cohort meets the minimum")
    return
  frame = cohorts.head(10).iloc[::-1]
  labels = [f"{row.feature}: {row.value}" for row in frame.itertuples(index=False)]
  colors = [
    _COLORS["negative"] if residual > 0 else _COLORS["cyan"]
    for residual in frame["residual"]
  ]
  axis.barh(labels, frame["residual"], color=colors, alpha=0.82)
  axis.axvline(0, color=_COLORS["foreground"], linewidth=0.8, alpha=0.5)
  axis.set(
    title="Largest cohort calibration residuals",
    xlabel="Observed event rate - mean prediction",
    ylabel="Past-only cohort",
  )
  axis.xaxis.set_major_formatter(PercentFormatter(1.0))
  axis.grid(False)
  axis.tick_params(axis="y", labelsize=7)


def _plot_frontier(axis: Any, frontier: DataFrame) -> None:
  colors = (_COLORS["accent"], _COLORS["cyan"], _COLORS["orange"], _COLORS["purple"])
  markers = ("o", "s", "^", "x")
  for index, (policy, frame) in enumerate(frontier.groupby("policy", sort=False)):
    axis.plot(
      frame["excluded_balance_share"],
      frame["event_exposure_avoided"],
      color=colors[index % len(colors)],
      marker=markers[index % len(markers)],
      linewidth=2 if policy == "GBM score" else 1.3,
      markersize=5,
      label=policy,
    )
  axis.set(
    title="Matched-balance selection frontier",
    xlabel="Excluded validation balance",
    ylabel="Observed event exposure avoided",
    xlim=(0, max(float(frontier["excluded_balance_share"].max()) * 1.05, 0.01)),
    ylim=(0, max(float(frontier["event_exposure_avoided"].max()) * 1.12, 0.01)),
  )
  axis.xaxis.set_major_formatter(PercentFormatter(1.0))
  axis.yaxis.set_major_formatter(PercentFormatter(1.0))
  axis.legend(fontsize=7)


def plot_sensitivity(baseline: Baseline) -> Figure:
  """Show the common-scale validation log-loss surface for every declared depth."""
  depths = sorted(int(depth) for depth in baseline.candidates["max_depth"].unique())
  columns = min(2, len(depths))
  rows = ceil(len(depths) / columns)
  with sapphire():
    figure, axes = plt.subplots(
      rows,
      columns,
      figsize=(7 * columns, 4.8 * rows),
      constrained_layout=True,
      squeeze=False,
    )
    figure.suptitle(
      f"Near-best validation region favors depth {baseline.selected_depth} · ★ selected",
      fontsize=16,
      fontweight="bold",
    )
    limits = (
      float(baseline.candidates["log_loss"].min()),
      float(baseline.candidates["log_loss"].max()),
    )
    colors = LinearSegmentedColormap.from_list(
      "sapphire-loss", (_COLORS["accent"], _COLORS["surface"], _COLORS["pink"])
    )
    for axis, depth in zip(axes.flat, depths, strict=False):
      _plot_sensitivity_depth(axis, baseline.candidates, depth, colors, limits)
    for axis in list(axes.flat)[len(depths) :]:
      axis.set_axis_off()
    return figure


def _plot_sensitivity_depth(
  axis: Any,
  candidates: DataFrame,
  depth: int,
  colors: LinearSegmentedColormap,
  limits: tuple[float, float],
) -> None:
  subset = candidates.loc[candidates["max_depth"] == depth]
  loss = subset.pivot(index="learning_rate", columns="n_estimators", values="log_loss")
  annotations = loss.map(lambda value: f"{value:.4f}")
  selected = subset.loc[subset["selected"]]
  if not selected.empty:
    row = selected.iloc[0]
    annotations.loc[row["learning_rate"], row["n_estimators"]] += "★"
  sns.heatmap(
    loss,
    ax=axis,
    annot=annotations,
    fmt="",
    cmap=colors,
    vmin=limits[0],
    vmax=limits[1],
    cbar=False,
    linewidths=0.5,
    linecolor=_COLORS["border"],
    annot_kws={"fontsize": 8},
  )
  row_positions = {float(value): index for index, value in enumerate(loss.index)}
  column_positions = {int(value): index for index, value in enumerate(loss.columns)}
  for candidate in subset.loc[subset["near_best"]].itertuples(index=False):
    axis.add_patch(
      Rectangle(
        (
          column_positions[int(cast(Any, candidate.n_estimators))],
          row_positions[float(cast(Any, candidate.learning_rate))],
        ),
        1,
        1,
        fill=False,
        edgecolor=_COLORS["accent"] if candidate.selected else _COLORS["cyan"],
        linewidth=3 if candidate.selected else 1.5,
      )
    )
  axis.set(
    title=f"Depth {depth}",
    xlabel="Boosting trees",
    ylabel="Learning rate",
  )
  axis.tick_params(axis="x", labelrotation=0)
  axis.tick_params(axis="y", labelrotation=0)


def _plot_candidate_loss(axis: Any, baseline: Baseline) -> None:
  candidates = baseline.candidates
  best_by_depth = candidates.groupby("max_depth", observed=True)["log_loss"].min()
  labels = [f"depth {depth}" for depth in best_by_depth.index]
  colors = [
    _COLORS["accent"] if depth == baseline.selected_depth else _COLORS["muted"]
    for depth in best_by_depth.index
  ]
  bars = axis.bar(labels, best_by_depth, color=colors, alpha=0.82)
  axis.bar_label(bars, fmt="%.4f", padding=3, color=_COLORS["foreground"], fontsize=8)
  axis.axhline(
    baseline.reference.log_loss,
    color=_COLORS["orange"],
    linestyle="--",
    linewidth=1.4,
  )
  axis.annotate(
    "Train-rate reference",
    (0.98, baseline.reference.log_loss),
    xycoords=axis.get_yaxis_transform(),
    xytext=(-4, 4),
    textcoords="offset points",
    ha="right",
    color=_COLORS["orange"],
    fontsize=8,
  )
  axis.set(
    title=(
      f"Selected d{baseline.selected_depth} · η {baseline.selected_learning_rate:g} · "
      f"{baseline.selected_estimators} trees"
    ),
    xlabel="Best candidate at each depth",
    ylabel="Validation log loss · lower is better",
    ylim=(0, max(float(best_by_depth.max()), baseline.reference.log_loss) * 1.2),
  )


def _plot_candidate_ranking(axis: Any, baseline: Baseline) -> None:
  candidates = baseline.candidates
  palette = (_COLORS["cyan"], _COLORS["green"], _COLORS["orange"], _COLORS["purple"])
  for index, (depth, group) in enumerate(candidates.groupby("max_depth", observed=True)):
    color = palette[index % len(palette)]
    axis.scatter(
      group["auroc"],
      group["average_precision"],
      color=color,
      alpha=0.72,
    )
    endpoint = group.loc[group["auroc"].idxmax()]
    axis.annotate(
      f"depth {depth}",
      (endpoint["auroc"], endpoint["average_precision"]),
      xytext=(5, 0),
      textcoords="offset points",
      va="center",
      color=color,
      fontsize=7,
    )
  selected = candidates.loc[candidates["selected"]].iloc[0]
  axis.scatter(
    selected["auroc"],
    selected["average_precision"],
    marker="*",
    s=180,
    color=_COLORS["accent"],
    edgecolor=_COLORS["deep"],
    linewidth=0.8,
    zorder=4,
  )
  axis.annotate(
    "selected",
    (selected["auroc"], selected["average_precision"]),
    xytext=(0, 9),
    textcoords="offset points",
    ha="center",
    color=_COLORS["accent"],
    fontsize=7,
  )
  axis.set(
    title="Ranking-metric tradeoff",
    xlabel="AUROC",
    ylabel="Average precision",
  )


def _plot_calibration(
  axis: Any,
  calibration: DataFrame,
  brier_score: float,
  title: str,
  fold: str,
) -> None:
  positions = list(range(len(calibration)))
  predicted = axis.plot(
    positions,
    calibration["mean_score"],
    color=_COLORS["cyan"],
    linewidth=1.8,
    marker="o",
    markersize=4,
  )[0]
  observed = axis.plot(
    positions,
    calibration["event_rate"],
    color=_COLORS["orange"],
    linewidth=1.5,
    linestyle="--",
    marker="x",
    markersize=5,
  )[0]
  for line, label, offset in ((predicted, "Predicted", 7), (observed, "Observed", -9)):
    axis.annotate(
      f"{label} {line.get_ydata()[-1]:.1%}",
      (positions[-1], line.get_ydata()[-1]),
      xytext=(7, offset),
      textcoords="offset points",
      color=line.get_color(),
      fontsize=7,
      va="center",
    )
  upper = max(float(calibration["event_rate"].max()), 0.01) * 1.32
  axis.set(
    title=f"{title} · Brier {brier_score:.4f}",
    xlabel=f"{fold} score band · low to high risk",
    ylabel="Event probability",
    xticks=positions,
    xticklabels=calibration["score_band"],
    xlim=(-0.4, len(positions) - 0.25),
    ylim=(0, upper),
  )
  axis.yaxis.set_major_formatter(PercentFormatter(1.0))


def _plot_score_comparison(
  axis: Any,
  title: str,
  validation: float,
  test: float,
  validation_reference: float,
  test_reference: float,
) -> None:
  positions = (0, 1)
  model = (validation, test)
  reference = (validation_reference, test_reference)
  axis.hlines(positions, model, reference, color=_COLORS["border"], linewidth=2)
  axis.scatter(model, positions, color=_COLORS["cyan"], s=45, zorder=3)
  axis.scatter(
    reference,
    positions,
    facecolors="none",
    edgecolors=_COLORS["orange"],
    s=45,
    linewidth=1.5,
    zorder=3,
  )
  for position, model_score, reference_score in zip(positions, model, reference, strict=True):
    axis.annotate(
      f"model {model_score:.4f}",
      (model_score, position),
      xytext=(0, -10),
      textcoords="offset points",
      ha="center",
      color=_COLORS["cyan"],
      fontsize=7,
    )
    axis.annotate(
      f"reference {reference_score:.4f}",
      (reference_score, position),
      xytext=(0, 8),
      textcoords="offset points",
      ha="center",
      color=_COLORS["orange"],
      fontsize=7,
    )
  axis.set(
    title=title,
    xlabel="Score · filled model, open reference",
    ylabel="Prediction fold",
    yticks=positions,
    yticklabels=("Validation", "Test"),
    xlim=(0, max(validation, test, validation_reference, test_reference) * 1.2),
  )
  axis.invert_yaxis()
  axis.grid(False)


def _plot_ranking_comparison(axis: Any, evaluation: Evaluation) -> None:
  positions = (0, 1)
  validation = evaluation.baseline.validation
  validation_scores = (validation.auroc, validation.average_precision)
  test_scores = (evaluation.metrics.auroc, evaluation.metrics.average_precision)
  axis.hlines(positions, validation_scores, test_scores, color=_COLORS["border"], linewidth=2)
  axis.scatter(validation_scores, positions, color=_COLORS["purple"], s=45, zorder=3)
  axis.scatter(test_scores, positions, color=_COLORS["accent"], marker="D", s=38, zorder=3)
  for position, validation_score, test_score in zip(
    positions, validation_scores, test_scores, strict=True
  ):
    axis.annotate(
      f"validation {validation_score:.3f}",
      (validation_score, position),
      xytext=(0, -10),
      textcoords="offset points",
      ha="center",
      color=_COLORS["purple"],
      fontsize=7,
    )
    axis.annotate(
      f"test {test_score:.3f}",
      (test_score, position),
      xytext=(0, 8),
      textcoords="offset points",
      ha="center",
      color=_COLORS["accent"],
      fontsize=7,
    )
  axis.set(
    title="Ranking stability",
    xlabel="Score · circle validation, diamond test",
    ylabel="Metric",
    yticks=positions,
    yticklabels=("AUROC", "Average precision"),
    xlim=(0, 1),
  )
  axis.invert_yaxis()
  axis.grid(False)


def _plot_importance(axis: Any, baseline: Baseline) -> None:
  importance = baseline.importance.head(12).iloc[::-1]
  labels = [str(feature).replace("_", " ") for feature in importance["feature"]]
  colors = [_COLORS["muted"]] * max(0, len(importance) - 1) + [_COLORS["accent"]]
  axis.barh(labels, importance["importance"], color=colors, alpha=0.82)
  axis.set(
    title="Selected model · permutation importance",
    xlabel="Validation log-loss increase when permuted",
    ylabel="Transformed feature",
  )
  axis.tick_params(axis="y", labelsize=7)
  axis.grid(False)


def _ordered(values: Any, preferred: tuple[str, ...]) -> list[str]:
  observed = {str(value) for value in values.dropna().unique()}
  return [value for value in preferred if value in observed] + sorted(observed - set(preferred))


def _plot_fold_composition(axis: Any, examples: DataFrame, folds: list[str]) -> None:
  statuses = _ordered(examples["target_status"], _STATUS_ORDER)
  width = 0.8 / len(statuses)
  centers = list(range(len(folds)))
  maximum = 1
  for status_index, status in enumerate(statuses):
    counts = [
      int(((examples["fold"] == fold) & (examples["target_status"] == status)).sum())
      for fold in folds
    ]
    maximum = max(maximum, *counts)
    offset = (status_index - (len(statuses) - 1) / 2) * width
    bars = axis.bar(
      [center + offset for center in centers],
      counts,
      width=width,
      label=_STATUS_LABEL.get(status, status.replace("_", " ").title()),
      color=_STATUS_COLOR.get(status, _COLORS["accent"]),
      hatch=_STATUS_HATCH[status_index % len(_STATUS_HATCH)],
      alpha=0.82,
    )
    axis.bar_label(
      bars,
      labels=[f"{count:,}" if count else "" for count in counts],
      padding=2,
      fontsize=7,
      color=_COLORS["foreground"],
    )
  axis.set(
    title="Fold outcome composition · log scale",
    xlabel="Prediction fold",
    ylabel="Eligible loan-cutoff rows",
    xticks=centers,
    xticklabels=[fold.title() for fold in folds],
    yscale="log",
    ylim=(0.8, maximum * 10),
  )
  count_formatter = ScalarFormatter()
  count_formatter.set_scientific(False)
  axis.yaxis.set_major_formatter(count_formatter)
  axis.legend(ncols=2, fontsize=8)


def _plot_event_rate(axis: Any, examples: DataFrame, folds: list[str]) -> None:
  rates: list[float] = []
  labels: list[str] = []
  for fold in folds:
    target = examples.loc[
      (examples["fold"] == fold) & examples["target"].notna(), "target"
    ]
    if target.empty:
      rates.append(float("nan"))
      statuses = examples.loc[examples["fold"] == fold, "target_status"]
      labels.append("Held out" if set(statuses) == {"held_out"} else "No binary outcomes")
      continue
    events = int(target.sum())
    rates.append(float(target.mean()))
    labels.append(f"{events:,} / {len(target):,}")
  positions = list(range(len(folds)))
  axis.plot(
    positions,
    rates,
    color=_COLORS["pink"],
    linewidth=2.2,
    marker="o",
    markersize=7,
  )
  for position, rate, label in zip(positions, rates, labels, strict=True):
    if rate == rate:
      axis.annotate(
        f"{rate:.2%}\n{label}",
        (position, rate),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        color=_COLORS["foreground"],
        fontsize=8,
      )
    else:
      axis.annotate(
        label,
        (position, 0),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        color=_COLORS["muted"],
        fontsize=8,
      )
  observed = [rate for rate in rates if rate == rate]
  upper = max(observed, default=0.01) * 1.4
  axis.set(
    title="Binary event-rate drift",
    xlabel="Prediction fold",
    ylabel="Observed event rate",
    xticks=positions,
    xticklabels=[fold.title() for fold in folds],
    ylim=(0, upper),
  )
  axis.yaxis.set_major_formatter(PercentFormatter(1.0))


def _missingness(examples: DataFrame, folds: list[str]) -> list[tuple[str, list[float]]]:
  rows = []
  for feature in FEATURE_COLUMNS:
    rates = [
      float(examples.loc[examples["fold"] == fold, feature].isna().mean()) for fold in folds
    ]
    if any(rate > 0 for rate in rates):
      rows.append((feature, rates))
  return sorted(rows, key=lambda row: max(row[1]), reverse=True)


def _plot_missingness(
  axis: Any,
  rows: list[tuple[str, list[float]]],
  folds: list[str],
) -> None:
  if not rows:
    _empty_axis(axis, "Feature missingness by fold", "No selected feature is missing")
    return
  values = [rates for _, rates in rows]
  annotations = [
    ["<0.1%" if 0 < rate < 0.001 else f"{rate:.1%}" for rate in rates]
    for rates in values
  ]
  colors = LinearSegmentedColormap.from_list(
    "sapphire-missingness", (_COLORS["deep"], _COLORS["orange"])
  )
  sns.heatmap(
    values,
    ax=axis,
    annot=annotations,
    fmt="",
    cmap=colors,
    vmin=0,
    vmax=max(max(rates) for rates in values),
    cbar=False,
    linewidths=0.5,
    linecolor=_COLORS["border"],
    xticklabels=[fold.title() for fold in folds],
    yticklabels=[feature.replace("_", " ").title() for feature, _ in rows],
    annot_kws={"fontsize": 8},
  )
  axis.set(title="Feature missingness by fold", xlabel="Prediction fold", ylabel="Feature")
  axis.tick_params(axis="x", labelrotation=0)
  axis.tick_params(axis="y", labelrotation=0, labelsize=8)


def _drift(examples: DataFrame, folds: list[str]) -> list[tuple[str, list[float]]]:
  baseline = "train" if "train" in folds else folds[0]
  rows = []
  for feature in FEATURE_COLUMNS:
    if not is_numeric_dtype(examples[feature].dtype):
      continue
    train = examples.loc[examples["fold"] == baseline, feature].dropna()
    if train.empty:
      continue
    scale = float(train.quantile(0.75) - train.quantile(0.25))
    if scale <= 0:
      continue
    center = float(train.median())
    shifts = []
    for fold in folds:
      observed = examples.loc[examples["fold"] == fold, feature].dropna()
      shifts.append(float("nan") if observed.empty else (float(observed.median()) - center) / scale)
    rows.append((feature, shifts))
  return sorted(
    rows,
    key=lambda row: max((abs(value) for value in row[1] if value == value), default=0),
    reverse=True,
  )


def _plot_drift(
  axis: Any,
  rows: list[tuple[str, list[float]]],
  folds: list[str],
) -> None:
  baseline = "train" if "train" in folds else folds[0]
  if not rows:
    _empty_axis(axis, "Robust numeric drift", "No varying numeric feature is available")
    return
  values = [shifts for _, shifts in rows]
  observed = [abs(value) for shifts in values for value in shifts if value == value]
  limit = max(0.1, max(observed, default=0.1))
  annotations = [
    ["" if value != value else f"{value:+.2f}" for value in shifts] for shifts in values
  ]
  colors = LinearSegmentedColormap.from_list(
    "sapphire-drift", (_COLORS["pink"], _COLORS["surface"], _COLORS["cyan"])
  )
  sns.heatmap(
    values,
    ax=axis,
    annot=annotations,
    fmt="",
    cmap=colors,
    center=0,
    vmin=-limit,
    vmax=limit,
    cbar=False,
    linewidths=0.5,
    linecolor=_COLORS["border"],
    xticklabels=[fold.title() for fold in folds],
    yticklabels=[feature.replace("_", " ").title() for feature, _ in rows],
    annot_kws={"fontsize": 8},
  )
  axis.set(
    title=f"Median shift · {baseline} IQR units",
    xlabel="Prediction fold",
    ylabel="Numeric feature",
  )
  axis.tick_params(axis="x", labelrotation=0)
  axis.tick_params(axis="y", labelrotation=0, labelsize=8)


def _empty_axis(axis: Any, title: str, message: str) -> None:
  axis.set_title(title)
  axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
  axis.set_axis_off()


def _plot_population(axis: Any, audit: Audit) -> None:
  continuity = audit.continuity
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


def _plot_states(axis: Any, audit: Audit) -> None:
  observed = audit.states
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
  axis.grid(False)


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


def _state_label(state: str, *, compact: bool = False) -> str:
  labels = _STATE_COMPACT_LABEL if compact else _STATE_LABEL
  fallback = state.replace("zero_balance:", "Zero bal. " if compact else "Zero balance ")
  return labels.get(state, fallback)


def _plot_transitions(axis: Any, audit: Audit) -> None:
  observed_states = audit.states
  states = _ordered_states(observed_states)
  index = {state: position for position, state in enumerate(states)}
  matrix = [[0 for _ in states] for _ in states]
  transitions = audit.transitions
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
  compact_labels = [_state_label(state, compact=True) for state in states]
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
    xticklabels=compact_labels,
    yticklabels=labels,
    annot_kws={"fontsize": 7},
  )
  axis.set(title="Next-report transition rate", xlabel="To state", ylabel="From state")
  axis.tick_params(axis="x", labelrotation=0, labelsize=7)
  axis.tick_params(axis="y", labelrotation=0, labelsize=8)


def _plot_targets(axis: Any, audit: Audit) -> None:
  targets = audit.targets
  decision = next(
    (
      target
      for target in targets
      if target.get("name") == "serious_delinquency_or_chargeoff"
    ),
    None,
  )
  if decision is None:
    raise ValueError("audit is missing the serious-delinquency target decision")
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
  status = str(decision["status"])
  title = "Three-report target disposition · log scale"
  if status != "derived":
    title += f" · {status.replace('_', ' ')}"
  axis.set(
    title=title,
    xlabel="Loan-cutoff positions",
    yticks=positions,
    yticklabels=[label for _, label, _ in categories],
    xscale="log",
    xlim=(0.7, max(max(values), 1) * 8),
  )
  axis.invert_yaxis()
  axis.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
  axis.grid(False)
