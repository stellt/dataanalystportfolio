from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from rmbs_analytics.model import (
    evaluate,
    fit_prepay_model,
    incentive_response,
    predicted_cpr,
    prepare_features,
    split_out_of_time,
)
from rmbs_analytics.rates import incentive_buckets


@pytest.fixture
def synthetic_panel(cfg):
    """Seeded loan-month panel whose true prepay hazard rises with incentive.

    2,000 loans observed for 36 months (2019-01 .. 2021-12). The PMMS path falls in
    2020, so later months are deeper in the money, as in the real refinancing wave.
    """
    rng = np.random.default_rng(7)
    n_loans, n_months = 2_000, 36
    periods = pd.period_range("2019-01", periods=n_months, freq="M").to_timestamp()
    pmms = np.r_[np.full(12, 4.5), np.linspace(4.5, 2.8, 12), np.full(12, 3.0)]
    note = rng.uniform(3.0, 5.5, n_loans)
    rows = []
    for i in range(n_loans):
        months_itm = 0
        upb = rng.uniform(80_000, 600_000)
        for t in range(n_months):
            inc = (note[i] - pmms[t]) * 100
            z = -5.2 + 3.0 / (1 + np.exp(-(inc - 60) / 25)) + 0.04 * min(t + 1, 30) \
                - 0.25 * np.log1p(months_itm)
            prepaid = rng.random() < 1 / (1 + np.exp(-z))
            rows.append((f"F19Q1{i:07d}", periods[t], t + 1, upb, inc, months_itm, prepaid))
            months_itm += inc > 50
            if prepaid:
                break
    df = pd.DataFrame(rows, columns=["loan_id", "period", "loan_age", "start_upb",
                                     "incentive_bp", "months_itm", "prepaid"])
    df["active_start"] = True
    df["orig_upb"] = df["start_upb"]
    df["fico_band"] = rng.choice(["<680", "680-719", "720-759", "760+"], len(df))
    df["ltv_band"] = rng.choice(["<=60", "60-80", "80-90"], len(df))
    df["loan_purpose"] = rng.choice(["P", "N", "C"], len(df))
    df["occupancy"] = rng.choice(["P", "I"], len(df), p=[0.9, 0.1])
    df["incentive_bucket"] = incentive_buckets(df["incentive_bp"], cfg)
    return df


def test_model_smoke(cfg, synthetic_panel):
    features = prepare_features(synthetic_panel, cfg)
    train, test = split_out_of_time(features, cfg)
    assert train["period"].max() < test["period"].min() == pd.Timestamp("2021-01-01")

    model = fit_prepay_model(train, cfg)
    p = model.predict(test)
    assert np.all((p >= 0) & (p <= 1))

    by_month = predicted_cpr(model, test)
    assert by_month[["actual_cpr", "predicted_cpr"]].notna().all().all()
    evaluation = evaluate(model, test)
    assert 0.5 < evaluation["stats"]["auc"] <= 1
    buckets = evaluation["by_incentive"]["incentive_bucket"].astype(str).tolist()
    order = list(test["incentive_bucket"].cat.categories)
    assert buckets == sorted(buckets, key=order.index)  # kept in bucket order, not A-Z


def test_intercept_correction_keeps_predictions_calibrated(cfg, synthetic_panel):
    features = prepare_features(synthetic_panel, cfg)
    model = fit_prepay_model(features, replace(cfg, nonevent_sample_frac=0.2))
    predicted = model.predict(features).sum()
    actual = features["prepaid"].sum()
    assert predicted == pytest.approx(actual, rel=0.15)


def test_s_curve_and_burnout_signs(cfg, synthetic_panel):
    features = prepare_features(synthetic_panel, cfg)
    model = fit_prepay_model(features, cfg)
    response = incentive_response(model, features, np.arange(-100, 201, 50))
    assert response["cpr"].iloc[-1] > 3 * response["cpr"].iloc[0]
    assert model.coefficients().set_index("term").loc["log_months_itm", "coef"] < 0


def test_unseen_category_in_test_does_not_crash(cfg, synthetic_panel):
    features = prepare_features(synthetic_panel, cfg)
    train, test = split_out_of_time(features, cfg)
    model = fit_prepay_model(train, cfg)
    test = test.assign(occupancy="S")  # second homes never seen in training
    assert np.isfinite(model.predict(test)).all()
