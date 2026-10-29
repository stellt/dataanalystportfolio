import pandas as pd
import pytest

from rmbs_analytics.panel import build_panel
from rmbs_analytics.rates import add_incentive, incentive_buckets, load_pmms

from .conftest import FIXTURES, frames, orig_row, perf_rows


def _pmms(rates: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame({"period": pd.to_datetime(list(rates)), "pmms_rate": list(rates.values())})


def _panel(cfg, months: int, rate: float = 6.5):
    orig, perf = frames([orig_row("F15Q10000001", orig_rate=rate)],
                        [perf_rows("F15Q10000001", [100_000] * months, rate=rate)])
    return build_panel(orig, perf, cfg)


def test_incentive(cfg):
    pmms = _pmms({"2015-12-01": 5.0, "2016-01-01": 5.0})
    out = add_incentive(_panel(cfg, 1), pmms, cfg, lag_months=0)
    assert out["incentive_bp"].item() == pytest.approx(150.0)


def test_incentive_uses_lagged_rate(cfg):
    pmms = _pmms({"2015-12-01": 5.0, "2016-01-01": 6.0})
    out = add_incentive(_panel(cfg, 1), pmms, cfg, lag_months=1)
    assert out["incentive_bp"].item() == pytest.approx(150.0)


def test_missing_pmms_month_gives_null_incentive(cfg):
    out = add_incentive(_panel(cfg, 2), _pmms({"2016-01-01": 5.0}), cfg, lag_months=0)
    assert out["incentive_bp"].isna().tolist() == [False, True]


def test_months_itm_counts_prior_months_only(cfg):
    # In the money by 150, 25, 150, 150 bp; threshold 50 bp.
    pmms = _pmms({"2016-01-01": 5.0, "2016-02-01": 6.25, "2016-03-01": 5.0,
                  "2016-04-01": 5.0})
    out = add_incentive(_panel(cfg, 4), pmms, cfg, lag_months=0)
    assert out["months_itm"].tolist() == [0, 1, 1, 2]


def test_incentive_buckets(cfg):
    b = incentive_buckets(pd.Series([-150.0, -100.0, -99.0, 0.0, 150.0, 151.0]), cfg)
    assert b.astype(str).tolist() == ["<=-100", "<=-100", "-100 to -75", "-25 to 0",
                                      "125 to 150", ">150"]


def test_load_pmms_monthly_average():
    pmms = load_pmms(FIXTURES / "raw" / "MORTGAGE30US.csv").set_index("period")
    assert pmms.loc[pd.Timestamp("2015-01-01"), "pmms_rate"] == pytest.approx(3.80)
    assert pmms.loc[pd.Timestamp("2015-03-01"), "pmms_rate"] == pytest.approx(3.70)
    assert len(pmms) == 12
