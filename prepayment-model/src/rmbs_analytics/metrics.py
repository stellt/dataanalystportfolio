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
    active = panel.loc[panel["active_start"], [*by, "start_upb", "prepaid", "defaulted"]]
    work = active.assign(
        prepaid_upb=active["start_upb"] * active["prepaid"],
        defaulted_upb=active["start_upb"] * active["defaulted"],
        prepaid_loans=active["prepaid"].astype("int64"),
        defaulted_loans=active["defaulted"].astype("int64"),
        loans=1,
    )
    sums = ["loans", "start_upb", "prepaid_loans", "prepaid_upb", "defaulted_loans",
            "defaulted_upb"]
    if by:
        out = work.groupby(by, observed=True, sort=True)[sums].sum().reset_index()
    else:
        out = work[sums].sum().to_frame().T
    upb = out["start_upb"].where(out["start_upb"] > 0)
    out["smm"] = out["prepaid_upb"] / upb
    out["mdr"] = out["defaulted_upb"] / upb
    out["suppressed"] = out["loans"] < min_active_loans
    out.loc[out["suppressed"], ["smm", "mdr"]] = np.nan
    out["cpr"] = smm_to_cpr(out["smm"])
    out["cdr"] = smm_to_cpr(out["mdr"])
    cols = [*by, "loans", "start_upb", "prepaid_loans", "prepaid_upb", "defaulted_loans",
            "defaulted_upb", "smm", "cpr", "mdr", "cdr", "suppressed"]
    return out[cols]


def all_cohort_tables(panel: pd.DataFrame, cfg: Config) -> dict[str, pd.DataFrame]:
    """Every cohort cut in the spec, keyed by table name.

    * ``overall_by_period`` and ``overall_by_loan_age`` (the seasoning ramp)
    * ``<dimension>`` pooled over time, plus ``<dimension>_by_period`` and
      ``<dimension>_by_loan_age`` for vintage, FICO, LTV and incentive buckets.

    Loan age is limited to ``cfg.max_loan_age`` months; small cells are suppressed.
    """
    n = cfg.min_active_loans
    aged = panel[(panel["loan_age"] >= 1) & (panel["loan_age"] <= cfg.max_loan_age)]
    tables = {
        "overall_by_period": cohort_rates(panel, ["period"], n),
        "overall_by_loan_age": cohort_rates(aged, ["loan_age"], n),
    }
    for dim in COHORT_DIMENSIONS:
        if dim not in panel:
            continue
        tables[dim] = cohort_rates(panel, [dim], n)
        tables[f"{dim}_by_period"] = cohort_rates(panel, [dim, "period"], n)
        tables[f"{dim}_by_loan_age"] = cohort_rates(aged, [dim, "loan_age"], n)
    return tables


def severity_by_default_year(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Loss severity by the year each loan defaulted, for liquidated loans only.

    Severity = -sum(actual_loss) / sum(UPB at liquidation), where ``actual_loss`` is
    negative for a loss and the UPB is the zero balance removal UPB. Only loans that
    exit with a credit-related zero balance code and a reported loss are counted, so
    ``resolved_loans`` is shown beside the rate.
    """
    liquidated = panel[
        panel["zero_balance_code"].isin(cfg.schema.credit_zero_balance).fillna(False)
        & panel["actual_loss"].notna()
    ][["loan_id", "actual_loss", "zero_balance_removal_upb", "current_upb", "start_upb"]]
    if liquidated.empty:
        return pd.DataFrame(columns=["default_year", "resolved_loans", "upb_at_default",
                                     "actual_loss", "severity"])
    upb = (liquidated["zero_balance_removal_upb"]
           .fillna(liquidated["start_upb"]))
    defaults = default_period(panel)
    years = liquidated["loan_id"].map(defaults).dt.year
    work = pd.DataFrame({"default_year": years, "upb_at_default": upb,
                         "actual_loss": liquidated["actual_loss"]})
    out = work.groupby("default_year").agg(
        resolved_loans=("actual_loss", "size"),
        upb_at_default=("upb_at_default", "sum"),
        actual_loss=("actual_loss", "sum"),
    ).reset_index()
    out["severity"] = -out["actual_loss"] / out["upb_at_default"]
    return out


def write_tables(tables: dict[str, pd.DataFrame], dest: Path) -> list[Path]:
    """Write each table to ``dest/<name>.csv``; returns the paths written."""
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, df in tables.items():
        path = dest / f"{name}.csv"
        df.to_csv(path, index=False, float_format="%.6g")
        paths.append(path)
    return paths
