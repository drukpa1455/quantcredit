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
  from quantcredit.baselines import Baseline, Evaluation

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


@contextmanager
def sapphire() -> Iterator[None]:
  """Apply the repository's scoped Matplotlib theme."""
  with mpl.rc_context(cast(Any, _STYLE)):
    yield


def plot_audit(audit: Audit) -> Figure:
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
  height = max(9.0, 5.4 + 0.24 * max(len(missing_rows), len(drift_rows)))
  with sapphire():
    figure, axes = plt.subplots(2, 2, figsize=(15, height), constrained_layout=True)
    figure.suptitle("Causal modeling population", fontsize=16, fontweight="bold")
    _plot_fold_composition(axes[0, 0], examples, folds)
    _plot_event_rate(axes[0, 1], examples, folds)
    _plot_missingness(axes[1, 0], missing_rows, folds)
    _plot_drift(axes[1, 1], drift_rows, folds)
    return figure


def plot_baseline(baseline: Baseline) -> Figure:
  """Show validation-only model selection, ranking, calibration, and importance."""
  with sapphire():
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    figure.suptitle("Shallow GBM · validation only", fontsize=16, fontweight="bold")
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
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    figure.suptitle(
      f"Frozen GBM · out-of-time test · {evaluation.cutoff.isoformat()}",
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
      "Validation log-loss sensitivity · outline near-best · ★ selected",
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
    _COLORS["accent"] if depth == baseline.selected_depth else _COLORS["cyan"]
    for depth in best_by_depth.index
  ]
  bars = axis.bar(labels, best_by_depth, color=colors, alpha=0.82)
  axis.bar_label(bars, fmt="%.4f", padding=3, color=_COLORS["foreground"], fontsize=8)
  axis.axhline(
    baseline.reference.log_loss,
    color=_COLORS["orange"],
    linestyle="--",
    linewidth=1.4,
    label="Train-rate reference",
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
  axis.legend(fontsize=8)


def _plot_candidate_ranking(axis: Any, baseline: Baseline) -> None:
  candidates = baseline.candidates
  palette = (_COLORS["cyan"], _COLORS["green"], _COLORS["orange"], _COLORS["purple"])
  for index, (depth, group) in enumerate(candidates.groupby("max_depth", observed=True)):
    axis.scatter(
      group["auroc"],
      group["average_precision"],
      color=palette[index % len(palette)],
      alpha=0.72,
      label=f"Depth {depth}",
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
    label="Selected",
    zorder=4,
  )
  axis.set(
    title="Ranking-metric tradeoff",
    xlabel="AUROC",
    ylabel="Average precision",
  )
  axis.legend(fontsize=7)


def _plot_calibration(
  axis: Any,
  calibration: DataFrame,
  brier_score: float,
  title: str,
  fold: str,
) -> None:
  positions = list(range(len(calibration)))
  width = 0.38
  predicted = axis.bar(
    [position - width / 2 for position in positions],
    calibration["mean_score"],
    width,
    color=_COLORS["cyan"],
    alpha=0.82,
    label="Predicted",
  )
  observed = axis.bar(
    [position + width / 2 for position in positions],
    calibration["event_rate"],
    width,
    color=_COLORS["orange"],
    alpha=0.82,
    label="Observed",
  )
  axis.bar_label(
    predicted,
    labels=[f"{value:.1%}" for value in calibration["mean_score"]],
    padding=2,
    color=_COLORS["foreground"],
    fontsize=6,
    rotation=90,
  )
  axis.bar_label(
    observed,
    labels=[f"{value:.1%}" for value in calibration["event_rate"]],
    padding=2,
    color=_COLORS["foreground"],
    fontsize=6,
    rotation=90,
  )
  upper = max(float(calibration["event_rate"].max()), 0.01) * 1.32
  axis.set(
    title=f"{title} · Brier {brier_score:.4f}",
    xlabel=f"{fold} score band · low to high risk",
    ylabel="Event probability",
    xticks=positions,
    xticklabels=calibration["score_band"],
    ylim=(0, upper),
  )
  axis.yaxis.set_major_formatter(PercentFormatter(1.0))
  axis.legend(fontsize=8)


def _plot_score_comparison(
  axis: Any,
  title: str,
  validation: float,
  test: float,
  validation_reference: float,
  test_reference: float,
) -> None:
  positions = (0, 1)
  width = 0.36
  model = axis.bar(
    [position - width / 2 for position in positions],
    (validation, test),
    width,
    color=_COLORS["cyan"],
    alpha=0.82,
    label="Frozen model",
  )
  reference = axis.bar(
    [position + width / 2 for position in positions],
    (validation_reference, test_reference),
    width,
    color=_COLORS["orange"],
    alpha=0.82,
    label="Train-rate reference",
  )
  for bars in (model, reference):
    axis.bar_label(bars, fmt="%.4f", padding=3, fontsize=8, color=_COLORS["foreground"])
  axis.set(
    title=title,
    xlabel="Prediction fold",
    ylabel="Score",
    xticks=positions,
    xticklabels=("Validation", "Test"),
    ylim=(0, max(validation, test, validation_reference, test_reference) * 1.25),
  )
  axis.legend(fontsize=8)


def _plot_ranking_comparison(axis: Any, evaluation: Evaluation) -> None:
  positions = (0, 1)
  width = 0.36
  validation = evaluation.baseline.validation
  validation_bars = axis.bar(
    [position - width / 2 for position in positions],
    (validation.auroc, validation.average_precision),
    width,
    color=_COLORS["purple"],
    alpha=0.82,
    label="Validation",
  )
  test_bars = axis.bar(
    [position + width / 2 for position in positions],
    (evaluation.metrics.auroc, evaluation.metrics.average_precision),
    width,
    color=_COLORS["accent"],
    alpha=0.82,
    label="Test",
  )
  for bars in (validation_bars, test_bars):
    axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8, color=_COLORS["foreground"])
  axis.set(
    title="Ranking stability",
    xlabel="Metric",
    ylabel="Score",
    xticks=positions,
    xticklabels=("AUROC", "Average precision"),
    ylim=(0, 1),
  )
  axis.legend(fontsize=8)


def _plot_importance(axis: Any, baseline: Baseline) -> None:
  importance = baseline.importance.head(12).iloc[::-1]
  labels = [str(feature).replace("_", " ") for feature in importance["feature"]]
  axis.barh(labels, importance["importance"], color=_COLORS["purple"], alpha=0.82)
  axis.set(
    title="Selected model · permutation importance",
    xlabel="Validation log-loss increase when permuted",
    ylabel="Transformed feature",
  )
  axis.tick_params(axis="y", labelsize=7)


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
