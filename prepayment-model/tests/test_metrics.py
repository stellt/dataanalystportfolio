import numpy as np
import pandas as pd
import pytest

from rmbs_analytics.metrics import (
    all_cohort_sums,
    all_cohort_tables,
    cohort_rates,
    combine_sums,
    cpr_to_smm,
    severity_by_default_year,
    smm_to_cpr,
    tables_from_sums,
)
from rmbs_analytics.panel import build_panel

from .conftest import frames, orig_row, perf_rows


def test_smm_to_cpr():
    assert smm_to_cpr(0.01) == pytest.approx(0.113615, abs=1e-6)
    assert smm_to_cpr(0.0) == 0.0
    assert cpr_to_smm(smm_to_cpr(0.0123)) == pytest.approx(0.0123)


def test_cohort_cpr_hand_case(four_loan_panel):
    by_month = cohort_rates(four_loan_panel, ["period"]).set_index("period")
    month3 = by_month.loc[pd.Timestamp("2016-03-01")]
    assert month3["loans"] == 4
    assert month3["start_upb"] == 400_000
    assert month3["smm"] == pytest.approx(0.25)
    assert month3["cpr"] == pytest.approx(1 - 0.75**12)
    assert month3["cpr"] == pytest.approx(0.968, abs=1e-3)
    month4 = by_month.loc[pd.Timestamp("2016-04-01")]
    assert month4["loans"] == 3 and month4["smm"] == 0.0


def test_active_denominator_in_rates(four_loan_panel):
    month4 = cohort_rates(four_loan_panel, ["period"]).iloc[-1]
    assert month4["start_upb"] == 300_000


def test_cdr(cfg):
    ids = [f"F15Q1000000{i}" for i in range(1, 5)]
    perfs = [perf_rows(ids[0], [100_000] * 2, dq=["00", "06"])]
    perfs += [perf_rows(i, [100_000] * 2) for i in ids[1:]]
    orig, perf = frames([orig_row(i) for i in ids], perfs)
    rates = cohort_rates(build_panel(orig, perf, cfg), ["period"])
    assert rates["mdr"].tolist() == [0.0, 0.25]
    assert rates["cdr"].iloc[1] == pytest.approx(1 - 0.75**12)


def test_balance_weighting(cfg):
    # A $300k loan prepaying among $100k loans moves SMM by its balance, not its count.
    orig, perf = frames(
        [orig_row("F15Q10000001", orig_upb=300_000.0), orig_row("F15Q10000002")],
        [perf_rows("F15Q10000001", [0], zb={0: "01"}), perf_rows("F15Q10000002", [100_000])],
    )
    rates = cohort_rates(build_panel(orig, perf, cfg), [])
    assert rates["smm"].item() == pytest.approx(0.75)


def test_small_cells_suppressed(four_loan_panel):
    out = cohort_rates(four_loan_panel, ["period"], min_active_loans=4)
    assert out["suppressed"].tolist() == [False, False, False, True]
    assert np.isnan(out["cpr"].iloc[-1])
    assert out["loans"].iloc[-1] == 3


def test_severity_by_default_year(cfg):
    rows = perf_rows("F15Q10000001", [100_000, 0], zb={1: "09"})
    rows[1].update(actual_loss=25_000.0, zero_balance_removal_upb=100_000.0)
    orig, perf = frames([orig_row("F15Q10000001")], [rows])
    sev = severity_by_default_year(build_panel(orig, perf, cfg), cfg)
    assert sev["resolved_loans"].tolist() == [1]
    assert sev["severity"].item() == pytest.approx(0.25)


def test_tables_built_per_file_match_whole_panel(cfg, four_loan_panel):
    from dataclasses import replace

    cfg = replace(cfg, min_active_loans=1)
    halves = [four_loan_panel[four_loan_panel["loan_id"].isin(ids)]
              for ids in (["F15Q10000001", "F15Q10000002"], ["F15Q10000003", "F15Q10000004"])]
    combined = tables_from_sums(combine_sums([all_cohort_sums(h, cfg) for h in halves]), cfg)
    whole = all_cohort_tables(four_loan_panel, cfg)
    assert combined.keys() == whole.keys()
    for name in whole:
        pd.testing.assert_frame_equal(combined[name], whole[name], check_dtype=False)
