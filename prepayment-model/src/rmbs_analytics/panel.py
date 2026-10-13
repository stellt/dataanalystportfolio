"""Build the loan-month panel and its event flags.

One row per loan per reporting month, with origination attributes joined on and:

* ``active_start`` - loan is outstanding at the start of the month (denominator rows)
* ``start_upb``    - prior month's current actual UPB (first month: original UPB), dollars
* ``prepaid``      - voluntary payoff this month (excludes payoffs at scheduled maturity)
* ``matured``      - payoff at scheduled maturity (not a prepayment)
* ``defaulted``    - first month the loan is 180+ days delinquent / REO, or leaves with a
  credit-related zero balance code; at most once per loan

``loan_age`` on a termination row is corrected to one more than the prior month's age,
because Freddie Mac sometimes repeats the prior age on the payoff record.

Curtailments (partial prepayments) are not counted as prepayments in version 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rmbs_analytics.config import Config

ORIG_COLUMNS = [
    "loan_id", "credit_score", "orig_ltv", "orig_cltv", "orig_dti", "orig_upb", "orig_rate",
    "orig_term", "first_payment_date", "maturity_date", "loan_purpose", "occupancy",
    "property_state", "property_type", "channel", "num_borrowers", "first_time_homebuyer",
]
PERF_COLUMNS = [
    "loan_id", "period", "current_upb", "delinquency_status", "loan_age", "remaining_months",
    "zero_balance_code", "zero_balance_date", "current_rate", "actual_loss",
    "zero_balance_removal_upb",
]


def vintage_from_loan_id(loan_id: pd.Series) -> pd.Series:
    """Origination year from a loan identifier ``PYYQnXXXXXXX`` (e.g. F21Q3... -> 2021)."""
    return (2000 + pd.to_numeric(loan_id.str.slice(1, 3), errors="coerce")).astype("Int16")


def delinquency_months(status: pd.Series, cfg: Config) -> pd.Series:
    """Months delinquent from the status code: ``00`` -> 0, ``03`` -> 3.

    The REO code (``RA``) maps to ``cfg.default_dq_months`` so it counts as 180+ days;
    not-available codes (``XX``) map to null.
    """
    s = status.astype("string").str.strip()
    months = pd.to_numeric(s.where(s.str.fullmatch(r"\d+", na=False)), errors="coerce")
    months = months.where(s != cfg.schema.reo_status, cfg.default_dq_months)
    return months.astype("Int16")


def _bucket(values: pd.Series, edges: tuple[float, ...], labels: tuple[str, ...],
            right: bool) -> pd.Series:
    bins = [-np.inf, *edges, np.inf]
    return pd.cut(values.astype("float64"), bins=bins, labels=list(labels), right=right)


def fico_band(score: pd.Series, cfg: Config) -> pd.Series:
    """FICO bucket; edges are lower-inclusive (680 falls in 680-719)."""
    return _bucket(score, cfg.fico_edges, cfg.fico_labels, right=False)


def ltv_band(ltv: pd.Series, cfg: Config) -> pd.Series:
    """LTV bucket; edges are upper-inclusive (80 falls in 60-80)."""
    return _bucket(ltv, cfg.ltv_edges, cfg.ltv_labels, right=True)


def build_panel(orig: pd.DataFrame, perf: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Join origination onto performance and add event flags and ``start_upb``.

    Parameters
    ----------
    orig : typed origination table (one row per loan), as from ``load_origination``.
    perf : typed performance table (one row per loan-month), as from ``load_performance``.
    cfg  : analysis config; supplies code lists, the default threshold and vintages.

    Returns
    -------
    DataFrame sorted by ``loan_id, period`` with the performance and origination
    columns listed in ``PERF_COLUMNS``/``ORIG_COLUMNS`` plus ``vintage``,
    ``fico_band``, ``ltv_band``, ``dq_months``, ``start_upb`` (dollars),
    ``active_start``, ``prepaid``, ``matured``, ``defaulted`` (booleans) and
    ``note_rate`` (percent; the current rate, falling back to the original rate).
    """
    perf = perf[PERF_COLUMNS].copy()
    perf["vintage"] = vintage_from_loan_id(perf["loan_id"])
    if cfg.vintages is not None:
        perf = perf[perf["vintage"].isin(cfg.vintages)]

    panel = perf.merge(orig[ORIG_COLUMNS], on="loan_id", how="inner", validate="many_to_one")
    panel = (panel.sort_values(["loan_id", "period"], kind="stable")
                  .drop_duplicates(["loan_id", "period"], keep="last")
                  .reset_index(drop=True))
    if panel.empty:
        return panel.assign(start_upb=pd.Series(dtype="float64"), active_start=False,
                            prepaid=False, matured=False, defaulted=False)

    loan = panel["loan_id"]
    first_row = loan.ne(loan.shift())
    zb = panel["zero_balance_code"].astype("string").str.strip().str.zfill(2)
    panel["zero_balance_code"] = zb

    # Termination rows sometimes repeat last month's loan age (or report 0); a payoff
    # month is always one month older than the month before it.
    prior_age = panel["loan_age"].shift().where(~first_row)
    fix_age = zb.notna() & prior_age.notna() & (panel["loan_age"] <= prior_age)
    panel["loan_age"] = panel["loan_age"].mask(fix_age, prior_age + 1)

    # Balance at the start of the month: last month's ending balance.
    prior_upb = panel["current_upb"].shift()
    panel["start_upb"] = prior_upb.where(~first_row, panel["orig_upb"]).astype("float64")

    # A loan is active at the start of the month unless it already exited.
    exit_here = zb.notna().to_numpy()
    exits_so_far = pd.Series(exit_here, index=panel.index).groupby(loan, sort=False).cumsum()
    exited_before = (exits_so_far - exit_here) > 0
    panel["active_start"] = ~exited_before & (panel["start_upb"] > 0)

    # Voluntary payoff vs payoff at scheduled maturity.
    payoff = zb.isin(cfg.schema.prepay_zero_balance).fillna(False)
    at_maturity = (
        (panel["remaining_months"].astype("float64") <= cfg.maturity_tolerance_months)
        | (panel["period"] >= panel["maturity_date"])
    ).fillna(False)
    panel["matured"] = panel["active_start"] & payoff & at_maturity
    panel["prepaid"] = panel["active_start"] & payoff & ~at_maturity

    # Default: first month 180+ days delinquent / REO, or a credit-related exit.
    panel["dq_months"] = delinquency_months(panel["delinquency_status"], cfg)
    default_event = (
        (panel["dq_months"] >= cfg.default_dq_months).fillna(False)
        | zb.isin(cfg.schema.credit_zero_balance).fillna(False)
    ) & panel["active_start"]
    first_default = default_event.astype("int8").groupby(loan, sort=False).cumsum() == 1
    panel["defaulted"] = (default_event & first_default).astype(bool)

    panel["note_rate"] = panel["current_rate"].fillna(panel["orig_rate"])
    panel["fico_band"] = fico_band(panel["credit_score"], cfg)
    panel["ltv_band"] = ltv_band(panel["orig_ltv"], cfg)
    for col in ("active_start", "prepaid", "matured"):
        panel[col] = panel[col].astype(bool)
    return panel


def default_period(panel: pd.DataFrame) -> pd.Series:
    """Reporting month of each loan's default event, indexed by ``loan_id``."""
    return panel.loc[panel["defaulted"], ["loan_id", "period"]].set_index("loan_id")["period"]
