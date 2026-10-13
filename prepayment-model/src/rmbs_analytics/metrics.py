"""Balance-weighted prepayment and default rates and cohort tables.

For a group of loan-months (all rates are decimals, not percent):

    SMM = sum(start_upb * prepaid)   / sum(start_upb * active_start)     monthly
    MDR = sum(start_upb * defaulted) / sum(start_upb * active_start)     monthly
    CPR = 1 - (1 - SMM)^12,  CDR = 1 - (1 - MDR)^12                      annualised
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rmbs_analytics.config import Config
from rmbs_analytics.panel import default_period

COHORT_DIMENSIONS = ("vintage", "fico_band", "ltv_band", "incentive_bucket")
TIME_AXES = ("period", "loan_age")


def smm_to_cpr(smm):
    """Annualise a monthly rate: ``1 - (1 - smm)**12``.

    Works for SMM -> CPR and MDR -> CDR. Accepts a float, array or Series of decimals
    (0.01 = 1% per month) and returns the same shape, as decimals per year.
    """
    return 1.0 - (1.0 - smm) ** 12


def cpr_to_smm(cpr):
    """Inverse of :func:`smm_to_cpr`: ``1 - (1 - cpr)**(1/12)`` (decimals)."""
    return 1.0 - (1.0 - cpr) ** (1.0 / 12.0)


SUM_COLUMNS = ["loans", "start_upb", "prepaid_loans", "prepaid_upb", "defaulted_loans",
               "defaulted_upb"]


def cohort_sums(panel: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Additive building blocks of the rates, per group.

    Sums are additive across panels that hold disjoint loans, so tables can be built
    one vintage file at a time and combined with :func:`combine_sums`.
    """
    active = panel.loc[panel["active_start"], [*by, "start_upb", "prepaid", "defaulted"]]
    work = active.assign(
        prepaid_upb=active["start_upb"] * active["prepaid"],
        defaulted_upb=active["start_upb"] * active["defaulted"],
        prepaid_loans=active["prepaid"].astype("int64"),
        defaulted_loans=active["defaulted"].astype("int64"),
        loans=1,
    )
    if by:
        return work.groupby(by, observed=True, sort=True)[SUM_COLUMNS].sum().reset_index()
    return work[SUM_COLUMNS].sum().to_frame().T


def add_rates(sums: pd.DataFrame, by: list[str], min_active_loans: int = 0) -> pd.DataFrame:
    """Turn :func:`cohort_sums` output into SMM, CPR, MDR, CDR (decimals)."""
    out = sums.copy()
    upb = out["start_upb"].where(out["start_upb"] > 0)
    out["smm"] = out["prepaid_upb"] / upb
    out["mdr"] = out["defaulted_upb"] / upb
    out["suppressed"] = out["loans"] < min_active_loans
    out.loc[out["suppressed"], ["smm", "mdr"]] = np.nan
    out["cpr"] = smm_to_cpr(out["smm"])
    out["cdr"] = smm_to_cpr(out["mdr"])
    return out[[*by, *SUM_COLUMNS, "smm", "cpr", "mdr", "cdr", "suppressed"]]


def cohort_rates(panel: pd.DataFrame, by: list[str], min_active_loans: int = 0) -> pd.DataFrame:
    """SMM, CPR, MDR and CDR per group, balance-weighted by ``start_upb``.

    Parameters
    ----------
    panel : loan-month panel with ``start_upb``, ``active_start``, ``prepaid``,
        ``defaulted`` and the ``by`` columns.
    by    : grouping columns, e.g. ``["vintage", "loan_age"]``. Empty list = one row.
    min_active_loans : groups with fewer active loan-months get null rates and
        ``suppressed = True`` (counts and balances are still reported).

    Returns
    -------
    One row per group with ``loans`` (active loan-months), ``start_upb`` (dollars),
    ``prepaid_loans``, ``prepaid_upb``, ``defaulted_loans``, ``defaulted_upb``,
    ``smm``, ``cpr``, ``mdr``, ``cdr`` (decimals) and ``suppressed``.
    """
    return add_rates(cohort_sums(panel, by), by, min_active_loans)


def _table_specs(columns) -> list[tuple[str, list[str], bool]]:
    """(table name, grouping columns, restrict to loan ages 1..max) for every cut."""
    specs = [("overall_by_period", ["period"], False),
             ("overall_by_loan_age", ["loan_age"], True)]
    for dim in COHORT_DIMENSIONS:
        if dim in columns:
            specs += [(dim, [dim], False), (f"{dim}_by_period", [dim, "period"], False),
                      (f"{dim}_by_loan_age", [dim, "loan_age"], True)]
    return specs


def all_cohort_sums(panel: pd.DataFrame, cfg: Config) -> dict[str, pd.DataFrame]:
    """:func:`cohort_sums` for every cut in :func:`all_cohort_tables`."""
    aged = panel[(panel["loan_age"] >= 1) & (panel["loan_age"] <= cfg.max_loan_age)]
    return {name: cohort_sums(aged if age_cut else panel, by)
            for name, by, age_cut in _table_specs(panel.columns)}


def combine_sums(parts: list[dict[str, pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    """Add up per-file :func:`all_cohort_sums` results table by table."""
    out = {}
    for name in parts[0]:
        frames = [p[name] for p in parts if not p[name].empty]
        stacked = pd.concat(frames, ignore_index=True)
        by = [c for c in stacked.columns if c not in SUM_COLUMNS]
        out[name] = (stacked.groupby(by, observed=True, sort=True)[SUM_COLUMNS].sum()
                     .reset_index() if by else stacked[SUM_COLUMNS].sum().to_frame().T)
    return out


def tables_from_sums(sums: dict[str, pd.DataFrame], cfg: Config) -> dict[str, pd.DataFrame]:
    """Rates for every combined table, with small cells suppressed."""
    return {name: add_rates(df, [c for c in df.columns if c not in SUM_COLUMNS],
                            cfg.min_active_loans)
            for name, df in sums.items()}


def all_cohort_tables(panel: pd.DataFrame, cfg: Config) -> dict[str, pd.DataFrame]:
    """Every cohort cut in the spec, keyed by table name.

    * ``overall_by_period`` and ``overall_by_loan_age`` (the seasoning ramp)
    * ``<dimension>`` pooled over time, plus ``<dimension>_by_period`` and
      ``<dimension>_by_loan_age`` for vintage, FICO, LTV and incentive buckets.

    Loan age is limited to ``cfg.max_loan_age`` months; small cells are suppressed.
    """
    return tables_from_sums(all_cohort_sums(panel, cfg), cfg)


def liquidations(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """One row per liquidated loan: default year, UPB at liquidation, actual loss.

    Liquidated = exits with a credit-related zero balance code and a reported loss.
    The UPB is the zero balance removal UPB (falling back to ``start_upb``).
    """
    liq = panel[
        panel["zero_balance_code"].isin(cfg.schema.credit_zero_balance).fillna(False)
        & panel["actual_loss"].notna()
    ]
    defaults = default_period(panel)
    return pd.DataFrame({
        "default_year": liq["loan_id"].map(defaults).dt.year.to_numpy(),
        "upb_at_default": liq["zero_balance_removal_upb"].fillna(liq["start_upb"]).to_numpy(),
        "actual_loss": liq["actual_loss"].to_numpy(),
    })


def severity_from_liquidations(liq: pd.DataFrame) -> pd.DataFrame:
    """Severity = sum(actual_loss) / sum(UPB at liquidation), by default year."""
    out = liq.groupby("default_year").agg(
        resolved_loans=("actual_loss", "size"),
        upb_at_default=("upb_at_default", "sum"),
        actual_loss=("actual_loss", "sum"),
    ).reset_index()
    out["severity"] = out["actual_loss"] / out["upb_at_default"]
    return out


def severity_by_default_year(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Loss severity by the year each loan defaulted, for liquidated loans only.

    Severity = sum(actual_loss) / sum(UPB at liquidation), where ``actual_loss`` is
    positive for a loss (Release 47 convention). Only loans with a completed liquidation
    have a final loss, so ``resolved_loans`` is shown beside the rate.
    """
    return severity_from_liquidations(liquidations(panel, cfg))


def write_tables(tables: dict[str, pd.DataFrame], dest: Path) -> list[Path]:
    """Write each table to ``dest/<name>.csv``; returns the paths written."""
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, df in tables.items():
        path = dest / f"{name}.csv"
        df.to_csv(path, index=False, float_format="%.6g")
        paths.append(path)
    return paths
