"""Market mortgage rate (Freddie Mac PMMS 30-year), refinancing incentive and burnout.

The PMMS series is FRED ``MORTGAGE30US`` (weekly, percent). It is averaged to
calendar months and compared with each loan's note rate:

    incentive_bp = (note rate - PMMS 30y rate) * 100

Positive means the borrower is in the money to refinance. ``months_itm`` counts the
months *before* the current one in which the loan was in the money by more than
``itm_threshold_bp``: a burnout proxy for borrowers who have ignored the incentive.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from rmbs_analytics.config import Config

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"


def download_pmms(dest: Path, url: str = FRED_URL) -> Path:
    """Download the weekly PMMS 30-year rate CSV from FRED to ``dest``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (fixed https URL)
        dest.write_bytes(resp.read())
    return dest


def load_pmms(path: Path | str) -> pd.DataFrame:
    """Read the FRED ``MORTGAGE30US`` CSV and average it to calendar months.

    Returns a DataFrame with ``period`` (month-start timestamp) and ``pmms_rate``
    (percent, monthly mean of the weekly survey rates).
    """
    raw = pd.read_csv(path, na_values=["."])
    date_col = next(c for c in raw.columns if c.lower() in ("observation_date", "date"))
    rate_col = next(c for c in raw.columns if c != date_col)
    weekly = pd.DataFrame({
        "period": pd.to_datetime(raw[date_col]).dt.to_period("M").dt.to_timestamp(),
        "pmms_rate": pd.to_numeric(raw[rate_col], errors="coerce"),
    }).dropna()
    monthly = weekly.groupby("period", as_index=False)["pmms_rate"].mean()
    monthly["period"] = monthly["period"].astype("datetime64[ms]")
    return monthly


def _incentive_edges(cfg: Config) -> list[int]:
    step = cfg.incentive_step_bp
    return list(range(cfg.incentive_min_bp, cfg.incentive_max_bp + step, step))


def incentive_labels(cfg: Config) -> list[str]:
    """Bucket labels in order: ``<=-100``, ``-100 to -75``, ..., ``125 to 150``, ``>150``."""
    edges = _incentive_edges(cfg)
    middle = [f"{a} to {b}" for a, b in zip(edges[:-1], edges[1:], strict=True)]
    return [f"<={edges[0]}", *middle, f">{edges[-1]}"]


def incentive_buckets(incentive_bp: pd.Series, cfg: Config) -> pd.Series:
    """Bucket incentive (bp) in ``step`` bp bins with open-ended tails (upper-inclusive)."""
    return pd.cut(incentive_bp, bins=[-np.inf, *_incentive_edges(cfg), np.inf],
                  labels=incentive_labels(cfg), right=True)


def add_incentive(panel: pd.DataFrame, pmms: pd.DataFrame, cfg: Config | None = None,
                  lag_months: int | None = None,
                  itm_threshold_bp: float | None = None) -> pd.DataFrame:
    """Add ``pmms_rate``, ``incentive_bp``, ``incentive_bucket`` and ``months_itm``.

    Parameters
    ----------
    panel : loan-month panel from ``build_panel`` (needs ``loan_id``, ``period``,
        ``note_rate``), sorted by ``loan_id, period``.
    pmms  : monthly PMMS from :func:`load_pmms` (``period``, ``pmms_rate`` in percent).
    cfg   : supplies the lag, the in-the-money threshold and incentive buckets.
    lag_months : compare the note rate with the PMMS rate this many months earlier
        (refinancing applications lock before the loan pays off). Overrides ``cfg``.
    itm_threshold_bp : in-the-money threshold for ``months_itm``. Overrides ``cfg``.

    Returns
    -------
    The panel with ``incentive_bp`` in basis points (positive = in the money) and
    ``months_itm`` as the count of earlier months with incentive above the threshold.
    Months with no PMMS data get a null incentive.
    """
    if cfg is None:
        from rmbs_analytics.config import load_config

        cfg = load_config()
    lag = cfg.incentive_lag_months if lag_months is None else lag_months
    threshold = cfg.itm_threshold_bp if itm_threshold_bp is None else itm_threshold_bp

    rates = pmms[["period", "pmms_rate"]].copy()
    rates["period"] = (rates["period"].dt.to_period("M") + lag).dt.to_timestamp()
    rates["period"] = rates["period"].astype(panel["period"].dtype)
    out = panel.drop(columns=["pmms_rate"], errors="ignore").merge(rates, on="period",
                                                                     how="left")
    out["incentive_bp"] = (out["note_rate"] - out["pmms_rate"]) * 100.0
    out["incentive_bucket"] = incentive_buckets(out["incentive_bp"], cfg)

    itm = (out["incentive_bp"] > threshold).astype("int32")
    out["months_itm"] = (itm.groupby(out["loan_id"], sort=False).cumsum() - itm).astype("int32")
    return out
