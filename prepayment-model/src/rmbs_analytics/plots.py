"""Charts for the README and results write-up (static PNGs via matplotlib).

Ordered groups (vintage, FICO, LTV) use one blue ramp, light to dark, so order reads
as order. Actual vs predicted uses two distinct categorical hues. Every multi-series
chart has a legend; with four or fewer series, lines are also labelled at their end.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = {"actual": "#2a78d6", "predicted": "#eb6834"}  # categorical slots 1 and 2
BLUE_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
             "#184f95", "#104281", "#0d366b"]


def _ramp(n: int) -> list[str]:
    """``n`` ordered blues from the ramp, spread from light to dark."""
    if n == 1:
        return [BLUE_RAMP[4]]
    idx = np.linspace(0, len(BLUE_RAMP) - 1, n).round().astype(int)
    return [BLUE_RAMP[i] for i in idx]


def _figure(title: str, subtitle: str | None = None):
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    fig.suptitle(title, x=0.06, y=0.97, ha="left", fontsize=13, color=INK, fontweight="bold")
    if subtitle:
        fig.text(0.06, 0.905, subtitle, ha="left", fontsize=9.5, color=INK_SECONDARY)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def _finish(fig, ax, path: Path, ylabel: str, xlabel: str, labels: list | None = None) -> Path:
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9.5)
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9.5)
    ax.set_ylim(bottom=0)
    fig.tight_layout(rect=(0, 0, 0.97, 0.86))
    if labels:
        _direct_labels(ax, labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def _last_point(x, y) -> tuple | None:
    ok = ~pd.isna(np.asarray(y, dtype="float64"))
    if not ok.any():
        return None
    return np.asarray(x)[ok][-1], np.asarray(y, dtype="float64")[ok][-1]


def _direct_labels(ax, items: list[tuple], min_gap_pt: float = 11.0) -> None:
    """Label each line at its last point, nudging labels apart so they never overlap."""
    items = [it for it in items if it[0] is not None]
    if not items:
        return
    ax.figure.canvas.draw()  # fix the axis limits so data -> points is final
    to_pt = 72.0 / ax.figure.dpi
    ys = [ax.transData.transform((0.0, y))[1] * to_pt for (_, y), _ in items]
    order = np.argsort(ys)
    placed = np.array(ys, dtype="float64")
    for prev, cur in zip(order[:-1], order[1:], strict=True):
        placed[cur] = max(placed[cur], placed[prev] + min_gap_pt)
    for i, ((x, y), text) in enumerate(items):
        ax.annotate(text, (x, y), xytext=(5, placed[i] - ys[i]), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK_SECONDARY, annotation_clip=False)


def _legend(ax, n: int, title: str | None = None) -> None:
    """Legend in one row above the plot, clear of the data."""
    leg = ax.legend(title=title, frameon=False, fontsize=8.5, title_fontsize=9,
                    labelcolor=INK_SECONDARY, loc="lower left", bbox_to_anchor=(0, 1.0),
                    ncol=min(n, 8), alignment="left")
    if leg.get_title():
        leg.get_title().set_color(INK_SECONDARY)


def _date_axis(ax, x: pd.Series) -> None:
    """Month-based date ticks (never weekly or daily) for monthly data."""
    if pd.api.types.is_datetime64_any_dtype(x) and x.notna().any():
        months = (x.max() - x.min()).days / 30.4
        if months <= 18:
            locator = mdates.MonthLocator()
        elif months <= 60:
            locator = mdates.MonthLocator(bymonth=[1, 7])
        else:
            locator = mdates.YearLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def lines_by_group(table: pd.DataFrame, x: str, group: str, value: str, path: Path,
                   title: str, subtitle: str | None = None, xlabel: str = "",
                   ylabel: str = "CPR") -> Path:
    """One line per group (ordered), e.g. CPR by loan age per vintage."""
    fig, ax = _figure(title, subtitle)
    groups = [g for g in pd.unique(table[group].dropna())]
    try:
        groups = sorted(groups)
    except TypeError:
        pass
    if isinstance(table[group].dtype, pd.CategoricalDtype):
        groups = [g for g in table[group].cat.categories if g in set(groups)]
    labels = []
    for g, color in zip(groups, _ramp(len(groups)), strict=True):
        sub = table[table[group] == g].sort_values(x)
        ax.plot(sub[x], sub[value], color=color, linewidth=2, label=str(g))
        labels.append((_last_point(sub[x], sub[value]), str(g)))
    _date_axis(ax, table[x])
    _legend(ax, len(groups), group.replace("_", " "))
    return _finish(fig, ax, path, ylabel, xlabel, labels if len(groups) <= 4 else None)


def single_line(table: pd.DataFrame, x: str, value: str, path: Path, title: str,
                subtitle: str | None = None, xlabel: str = "", ylabel: str = "CPR",
                categorical_x: bool = False) -> Path:
    """One series, e.g. the seasoning ramp or the S-curve."""
    fig, ax = _figure(title, subtitle)
    xs = np.arange(len(table)) if categorical_x else table[x]
    ax.plot(xs, table[value], color=BLUE_RAMP[4], linewidth=2,
            marker="o" if categorical_x else None, markersize=5)
    if categorical_x:
        ax.set_xticks(xs, table[x].astype(str), rotation=45, ha="right")
    else:
        _date_axis(ax, table[x])
    return _finish(fig, ax, path, ylabel, xlabel)


def actual_vs_predicted(table: pd.DataFrame, x: str, path: Path, title: str,
                        subtitle: str | None = None, xlabel: str = "",
                        categorical_x: bool = False) -> Path:
    """Actual vs model-predicted CPR (columns ``actual_cpr`` and ``predicted_cpr``)."""
    fig, ax = _figure(title, subtitle)
    xs = np.arange(len(table)) if categorical_x else table[x]
    labels = []
    for col, label, style in (("actual_cpr", "Actual", "-"), ("predicted_cpr", "Predicted", "--")):
        ax.plot(xs, table[col], color=SERIES[label.lower()], linewidth=2, linestyle=style,
                label=label, marker="o" if categorical_x else None, markersize=5)
        labels.append((_last_point(xs, table[col]), label))
    if categorical_x:
        ax.set_xticks(xs, table[x].astype(str), rotation=45, ha="right")
    else:
        _date_axis(ax, table[x])
    _legend(ax, 2)
    return _finish(fig, ax, path, "CPR", xlabel, labels)


def make_cohort_charts(tables: dict[str, pd.DataFrame], dest: Path) -> list[Path]:
    """The milestone-4/5 charts: CPR by vintage, FICO, LTV, the S-curve and seasoning ramp."""
    out = []
    if "vintage_by_loan_age" in tables:
        out.append(lines_by_group(
            tables["vintage_by_loan_age"], "loan_age", "vintage", "cpr",
            dest / "cpr_by_vintage.png", "Prepayment speed by vintage",
            "CPR by loan age, balance-weighted; small cohorts suppressed",
            xlabel="Loan age (months)"))
        out.append(lines_by_group(
            tables["vintage_by_loan_age"], "loan_age", "vintage", "cdr",
            dest / "cdr_by_vintage.png", "Default rate by vintage",
            "CDR by loan age (default = first month 180+ days delinquent or credit exit)",
            xlabel="Loan age (months)", ylabel="CDR"))
    for dim, name in (("fico_band", "FICO"), ("ltv_band", "LTV")):
        key = f"{dim}_by_period"
        if key in tables:
            out.append(lines_by_group(
                tables[key], "period", dim, "cpr", dest / f"cpr_by_{dim.split('_')[0]}.png",
                f"Prepayment speed by {name} band", "CPR by reporting month, balance-weighted",
                xlabel="Reporting month"))
    if "incentive_bucket" in tables:
        s = tables["incentive_bucket"].dropna(subset=["cpr"])
        out.append(single_line(
            s, "incentive_bucket", "cpr", dest / "s_curve.png", "The refinancing S-curve",
            "CPR by refinancing incentive (note rate minus PMMS 30y, bp), all months pooled",
            xlabel="Incentive bucket (bp)", categorical_x=True))
    if "overall_by_loan_age" in tables:
        out.append(single_line(
            tables["overall_by_loan_age"], "loan_age", "cpr", dest / "seasoning_ramp.png",
            "Seasoning ramp", "CPR by loan age, all vintages pooled",
            xlabel="Loan age (months)"))
    return out


def make_model_charts(evaluation: dict, response: pd.DataFrame, dest: Path) -> list[Path]:
    """Actual vs predicted CPR by month and by incentive, plus the fitted S-curve."""
    return [
        actual_vs_predicted(evaluation["by_month"], "period",
                            dest / "model_actual_vs_predicted.png",
                            "Out-of-time test: actual vs predicted CPR",
                            "Logistic hazard model, test period after the training cut-off",
                            xlabel="Reporting month"),
        actual_vs_predicted(evaluation["by_incentive"].dropna(subset=["actual_cpr"]),
                            "incentive_bucket", dest / "model_by_incentive.png",
                            "Out-of-time test: CPR by incentive bucket",
                            "Does the model reproduce the S-curve shape?",
                            xlabel="Incentive bucket (bp)", categorical_x=True),
        single_line(response, "incentive_bp", "cpr", dest / "model_incentive_response.png",
                    "Model-implied S-curve",
                    "Predicted CPR vs incentive, other features at typical values",
                    xlabel="Incentive (bp)"),
    ]
