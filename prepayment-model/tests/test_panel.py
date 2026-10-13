import pandas as pd
import pytest

from rmbs_analytics.panel import build_panel, delinquency_months

from .conftest import frames, orig_row, perf_rows


def _loan(panel: pd.DataFrame, loan_id: str) -> pd.DataFrame:
    return panel[panel["loan_id"] == loan_id].reset_index(drop=True)


def test_start_upb(cfg):
    orig, perf = frames([orig_row("F15Q10000001", orig_upb=100_500.0)],
                        [perf_rows("F15Q10000001", [100_000, 99_000, 98_000])])
    panel = build_panel(orig, perf, cfg)
    assert panel["start_upb"].tolist() == [100_500.0, 100_000.0, 99_000.0]


def test_prepaid_flag(cfg):
    orig, perf = frames(
        [orig_row("F15Q10000001"),
         orig_row("F15Q10000002", maturity_date=pd.Timestamp("2016-03-01"))],
        [perf_rows("F15Q10000001", [100_000, 100_000, 0], zb={2: "01"}),
         perf_rows("F15Q10000002", [2_000, 1_000, 0], zb={2: "01"}, remaining=[2, 1, 0])],
    )
    panel = build_panel(orig, perf, cfg)
    payoff = _loan(panel, "F15Q10000001")
    assert payoff["prepaid"].tolist() == [False, False, True]
    maturity = _loan(panel, "F15Q10000002")
    assert not maturity["prepaid"].any()
    assert maturity["matured"].tolist() == [False, False, True]


def test_default_once(cfg):
    dq = ["00", "04", "05", "06", "07", "08", "RA"]
    orig, perf = frames([orig_row("F15Q10000001")],
                        [perf_rows("F15Q10000001", [100_000] * 7, dq=dq)])
    panel = build_panel(orig, perf, cfg)
    assert panel["defaulted"].tolist() == [False, False, False, True, False, False, False]


def test_default_on_credit_exit_after_delinquency_counts_once(cfg):
    orig, perf = frames(
        [orig_row("F15Q10000001")],
        [perf_rows("F15Q10000001", [100_000] * 4 + [0], dq=["00", "06", "07", "RA", "RA"],
                   zb={4: "09"})],
    )
    panel = build_panel(orig, perf, cfg)
    assert panel["defaulted"].sum() == 1
    assert panel.loc[panel["defaulted"], "loan_age"].item() == 2


def test_short_sale_without_delinquency_is_a_default(cfg):
    orig, perf = frames([orig_row("F15Q10000001")],
                        [perf_rows("F15Q10000001", [100_000, 0], zb={1: "03"})])
    panel = build_panel(orig, perf, cfg)
    assert panel["defaulted"].tolist() == [False, True]
    assert not panel["prepaid"].any()


def test_active_denominator(cfg):
    # A defect record (zero balance 96) reported after the payoff must not re-enter.
    orig, perf = frames(
        [orig_row("F15Q10000001")],
        [perf_rows("F15Q10000001", [100_000, 0, 0, 0], zb={1: "01", 3: "96"})],
    )
    panel = build_panel(orig, perf, cfg)
    assert panel["active_start"].tolist() == [True, True, False, False]


def test_every_loan_has_at_most_one_exit(cfg, four_loan_panel):
    exits = four_loan_panel.groupby("loan_id")[["prepaid", "matured"]].sum().sum(axis=1)
    assert (exits <= 1).all()


def test_vintage_filter(cfg):
    from dataclasses import replace

    orig, perf = frames(
        [orig_row("F15Q10000001"), orig_row("F23Q10000001")],
        [perf_rows("F15Q10000001", [1, 1]), perf_rows("F23Q10000001", [1, 1])],
    )
    panel = build_panel(orig, perf, replace(cfg, vintages=(2015,)))
    assert panel["loan_id"].unique().tolist() == ["F15Q10000001"]
    assert panel["vintage"].unique().tolist() == [2015]


@pytest.mark.parametrize(("status", "months"), [("00", 0), ("0", 0), ("03", 3), ("12", 12),
                                                 ("RA", 6)])
def test_delinquency_months(cfg, status, months):
    assert delinquency_months(pd.Series([status]), cfg).iloc[0] == months


def test_unknown_delinquency_is_null(cfg):
    assert pd.isna(delinquency_months(pd.Series(["XX"]), cfg).iloc[0])


def test_buckets(cfg):
    orig, perf = frames(
        [orig_row("F15Q10000001", credit_score=679, orig_ltv=60),
         orig_row("F15Q10000002", credit_score=680, orig_ltv=61),
         orig_row("F15Q10000003", credit_score=760, orig_ltv=95)],
        [perf_rows(f"F15Q1000000{i}", [1]) for i in (1, 2, 3)],
    )
    panel = build_panel(orig, perf, cfg)
    assert panel["fico_band"].astype(str).tolist() == ["<680", "680-719", "760+"]
    assert panel["ltv_band"].astype(str).tolist() == ["<=60", "60-80", "90+"]


def test_payoff_row_loan_age_is_one_past_prior_month(cfg):
    rows = perf_rows("F15Q10000001", [100_000, 100_000, 0], zb={2: "01"})
    rows[2]["loan_age"] = 2  # payoff record repeats last month's age
    orig, perf = frames([orig_row("F15Q10000001")], [rows])
    assert build_panel(orig, perf, cfg)["loan_age"].tolist() == [1, 2, 3]
