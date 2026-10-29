"""Synthetic demo data in the Freddie Mac Release 47 file layout.

``rmbs demo`` writes seeded, made-up origination and performance files plus a
PMMS-style rate file, then runs the real pipeline on them. Nothing here is Freddie
Mac data: the rate path is a stylised version of 2015-2026 (refinancing wave in
2020-21, rate rise in 2022) and loan behaviour follows a simple hazard model with an
S-curve, a seasoning ramp, burnout, a COVID forbearance spike and post-2022 lock-in.
It exists so anyone can run the pipeline end to end without registering for the data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rmbs_analytics.config import Schema

START, END = pd.Period("2015-01", "M"), pd.Period("2026-03", "M")
VINTAGES = tuple(range(2015, 2023))
RATE_KNOTS = {  # stylised 30-year mortgage rate path, percent
    "2015-01": 3.8, "2016-07": 3.5, "2018-11": 4.9, "2019-09": 3.6, "2020-12": 2.7,
    "2021-12": 3.1, "2022-06": 5.6, "2022-10": 6.9, "2023-10": 7.6, "2024-09": 6.1,
    "2025-01": 7.0, "2026-03": 6.2,
}
STATES = ("CA", "TX", "FL", "NY", "IL", "VA", "WA", "GA", "OH", "AZ")


def demo_rates() -> pd.Series:
    """Monthly demo mortgage rate (percent), indexed by monthly ``Period``."""
    months = pd.period_range(START - 24, END, freq="M")
    knots = pd.Series(RATE_KNOTS)
    knots.index = pd.PeriodIndex(knots.index, freq="M")
    x = np.array([(p - START).n for p in knots.index], dtype=float)
    grid = np.array([(p - START).n for p in months], dtype=float)
    return pd.Series(np.interp(grid, x, knots.to_numpy()), index=months)


def write_rates_csv(path: Path, rates: pd.Series) -> None:
    """Write the rate path as a FRED-style weekly CSV (four readings a month)."""
    rows = ["observation_date,MORTGAGE30US"]
    for period, rate in rates.items():
        for day in (1, 8, 15, 22):
            rows.append(f"{period.start_time.replace(day=day):%Y-%m-%d},{rate:.2f}")
    path.write_text("\n".join(rows) + "\n")


def _ym(p: pd.Period) -> str:
    return f"{p.year}{p.month:02d}"


def _line(values: dict, columns) -> str:
    return "|".join(str(values.get(c.name, "")) for c in columns)


def _simulate_loan(rng, seq: int, vintage: int, rates: pd.Series, orig_cols, perf_cols):
    first_pay = pd.Period(year=vintage, month=int(rng.integers(1, 13)), freq="M")
    term = 180 if rng.random() < 0.1 else 360
    note = round((rates[first_pay - 2] + rng.normal(0.35, 0.3)) * 8) / 8
    fico = int(np.clip(rng.normal(748, 42), 600, 832))
    ltv = int(rng.choice([45, 60, 70, 75, 80, 85, 90, 95, 97],
                         p=[.06, .1, .1, .12, .3, .07, .08, .12, .05]))
    upb = float(np.clip(round(rng.lognormal(np.log(280_000), 0.45), -3), 50_000, 1_200_000))
    purpose = rng.choice(["P", "N", "C"], p=[.55, .3, .15])
    occupancy = rng.choice(["P", "I", "S"], p=[.9, .07, .03])
    loan_id = f"F{vintage % 100:02d}Q{(first_pay.month - 1) // 3 + 1}{seq:07d}"

    orig = {
        "credit_score": fico, "first_payment_date": _ym(first_pay), "first_time_homebuyer": "N",
        "maturity_date": _ym(first_pay + term - 1), "mi_pct": 25 if ltv > 80 else 0,
        "num_units": 1, "occupancy": occupancy, "orig_cltv": ltv,
        "orig_dti": int(rng.integers(18, 50)), "orig_upb": int(upb), "orig_ltv": ltv,
        "orig_rate": f"{note:.3f}", "channel": "R", "ppm_flag": "N", "amortization_type": "FRM",
        "property_state": rng.choice(STATES), "property_type": "SF", "postal_code": "000",
        "loan_id": loan_id, "loan_purpose": purpose, "orig_term": term,
        "num_borrowers": int(rng.integers(1, 3)), "seller_name": "OTHER",
        "super_conforming": "N", "relief_refi": "N", "valuation_method": 2,
        "io_indicator": "N", "vantage_score": 9999,
    }

    r = note / 1200
    payment = upb * r / (1 - (1 + r) ** -term)
    bal, dq, months_itm, forbearance = upb, 0, 0, False
    default_base = 0.00025 * np.exp(0.03 * (ltv - 80) - 0.015 * (fico - 748))
    lines = []
    for age, period in enumerate(pd.period_range(first_pay, END, freq="M"), start=1):
        start_bal = bal
        row = {"loan_id": loan_id, "period": _ym(period), "loan_age": age,
               "remaining_months": term - age, "current_rate": f"{note:.3f}",
               "non_interest_upb": "0.00", "cumulative_mod_cost": "0.00", "mi_cancel": "7",
               "servicer_name": "OTHER", "cramdown_costs": "0.00"}
        covid = pd.Period("2020-04", "M") <= period <= pd.Period("2020-09", "M")

        if dq == 0:
            inc = (note - rates[period - 1]) * 100
            z = (-6.3 + 3.3 / (1 + np.exp(-(inc - 60) / 25)) + 0.05 * min(age, 30)
                 - 0.35 * np.log1p(months_itm) + 0.3 * np.log(upb / 250_000)
                 + 0.25 * (fico >= 760) - 0.25 * (fico < 680) + 0.3 * (occupancy == "P")
                 + 0.25 * np.sin(2 * np.pi * (period.month - 3) / 12)
                 - 0.006 * min(max(-inc - 100, 0), 150))  # lock-in when deep out of the money
            months_itm += inc > 50
            if rng.random() < 1 / (1 + np.exp(-z)) or age == term:
                row.update(current_upb="0.00", delinquency_status="00", zero_balance_code="01",
                           zero_balance_date=_ym(period), interest_bearing_upb="0.00")
                lines.append(_line(row, perf_cols))
                break
            if rng.random() < default_base * (25 if covid else 1):
                dq, forbearance = 1, covid
        else:
            dq += 1
            if forbearance and dq >= 6 and rng.random() < 0.35:
                dq = 0  # forbearance ends and the loan cures (after counting as 180+ days)
        if dq == 0:
            interest = bal * r
            bal = max(bal - (payment - interest), 0.0)
        if dq >= 10:  # liquidation: REO disposition with a loss
            loss = start_bal * float(np.clip(rng.normal(0.15, 0.05), 0.02, 0.4))
            row.update(current_upb="0.00", delinquency_status=f"{dq:02d}",
                       zero_balance_code="09", zero_balance_date=_ym(period),
                       actual_loss=f"{loss:.2f}", zero_balance_removal_upb=f"{start_bal:.2f}",
                       interest_bearing_upb="0.00")
            lines.append(_line(row, perf_cols))
            break
        shown = round(bal, -3) if age <= 6 else round(bal, 2)
        row.update(current_upb=f"{shown:.2f}", delinquency_status=f"{dq:02d}",
                   interest_bearing_upb=f"{shown:.2f}")
        lines.append(_line(row, perf_cols))
    return _line(orig, orig_cols), lines


def write_demo_files(raw_dir: Path, schema: Schema, loans_per_vintage: int = 400,
                     seed: int = 7) -> dict[str, int]:
    """Write ``demo_orig_<year>.txt``, ``demo_perf_<year>.txt`` and the rate CSV.

    Returns row counts per file kind. Output is fully determined by ``seed``.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    layout = next(lay for lay in schema.layouts if lay.name == "release_47")
    rates = demo_rates()
    write_rates_csv(raw_dir / "MORTGAGE30US.csv", rates)
    rng = np.random.default_rng(seed)
    counts = {"loans": 0, "loan_months": 0}
    for vintage in VINTAGES:
        origs, perfs = [], []
        for i in range(loans_per_vintage):
            o, p = _simulate_loan(rng, i + 1, vintage, rates, layout.origination,
                                  layout.performance)
            origs.append(o)
            perfs.extend(p)
        (raw_dir / f"demo_orig_{vintage}.txt").write_text("\n".join(origs) + "\n")
        (raw_dir / f"demo_perf_{vintage}.txt").write_text("\n".join(perfs) + "\n")
        counts["loans"] += len(origs)
        counts["loan_months"] += len(perfs)
    return counts
