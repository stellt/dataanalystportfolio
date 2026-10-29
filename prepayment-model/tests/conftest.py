"""Shared test helpers: config and tiny in-memory loan histories."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rmbs_analytics.config import load_config

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def cfg():
    return load_config()


def orig_row(loan_id: str, **overrides) -> dict:
    """One origination record with sensible defaults (vintage 2015)."""
    row = {
        "loan_id": loan_id, "credit_score": 740, "orig_ltv": 80, "orig_cltv": 80,
        "orig_dti": 35, "orig_upb": 100_000.0, "orig_rate": 4.0, "orig_term": 360,
        "first_payment_date": pd.Timestamp("2016-01-01"),
        "maturity_date": pd.Timestamp("2045-12-01"), "loan_purpose": "P", "occupancy": "P",
        "property_state": "VA", "property_type": "SF", "channel": "R", "num_borrowers": 2,
        "first_time_homebuyer": "N",
    }
    row.update(overrides)
    return row


def perf_rows(loan_id: str, upbs: list[float], start: str = "2016-01",
              dq: list[str] | None = None, zb: dict[int, str] | None = None,
              remaining: list[int] | None = None, rate: float = 4.0) -> list[dict]:
    """Monthly records for one loan. ``zb`` maps a row index to a zero balance code."""
    periods = pd.period_range(start, periods=len(upbs), freq="M").to_timestamp()
    dq = dq or ["00"] * len(upbs)
    zb = zb or {}
    remaining = remaining or [359 - i for i in range(len(upbs))]
    return [{
        "loan_id": loan_id, "period": p, "current_upb": float(u), "delinquency_status": d,
        "loan_age": i + 1, "remaining_months": r, "zero_balance_code": zb.get(i, pd.NA),
        "zero_balance_date": p if i in zb else pd.NaT, "current_rate": rate,
        "actual_loss": np.nan, "zero_balance_removal_upb": np.nan,
    } for i, (p, u, d, r) in enumerate(zip(periods, upbs, dq, remaining, strict=True))]


def frames(origs: list[dict], perfs: list[list[dict]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    orig = pd.DataFrame(origs)
    perf = pd.DataFrame([r for rows in perfs for r in rows])
    perf["zero_balance_code"] = perf["zero_balance_code"].astype("string")
    perf["delinquency_status"] = perf["delinquency_status"].astype("string")
    return orig, perf


@pytest.fixture
def four_loan_panel(cfg):
    """Spec hand case: 4 loans of $100k, flat balances, loan A prepays in month 3."""
    from rmbs_analytics.panel import build_panel

    ids = ["F15Q10000001", "F15Q10000002", "F15Q10000003", "F15Q10000004"]
    perfs = [perf_rows(ids[0], [100_000, 100_000, 0], zb={2: "01"})]
    perfs += [perf_rows(i, [100_000] * 4) for i in ids[1:]]
    orig, perf = frames([orig_row(i) for i in ids], perfs)
    return build_panel(orig, perf, cfg)
