"""Loan-month logistic prepayment model (discrete-time hazard).

    P(prepaid_it = 1 | active) = 1 / (1 + exp(-(b0 + b'x_it)))

Fitted with ``statsmodels`` so coefficients carry standard errors. Training uses
reporting months up to ``cfg.train_end``; testing uses later months (out-of-time),
because a random split would leak future rate conditions into training.

Non-prepay rows are downsampled in training (keep share ``f``). Sampling only the
non-events multiplies the odds by ``1/f``, so predictions add ``log(f)`` back to the
linear predictor before the logistic transform.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import log_loss, roc_auc_score

from rmbs_analytics.config import Config
from rmbs_analytics.metrics import smm_to_cpr

CATEGORICAL_FEATURES = ("fico_band", "ltv_band", "loan_purpose", "occupancy", "month_of_year")
SPLINE_FEATURES = ("incentive_clipped",)


def prepare_features(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Model rows: active loan-months with every feature present.

    Adds ``incentive_clipped`` (bp), ``age_capped`` (months, capped at ``cfg.age_cap``),
    ``log_months_itm`` (log(1 + months in the money)), ``log_orig_upb`` and
    ``month_of_year`` (1-12). Categorical features become plain strings.
    """
    df = panel.loc[panel["active_start"] & panel["incentive_bp"].notna()].copy()
    lo, hi = cfg.incentive_clip_bp
    df["incentive_clipped"] = df["incentive_bp"].clip(lo, hi)
    df["age_capped"] = df["loan_age"].astype("float64").clip(0, cfg.age_cap)
    df["log_months_itm"] = np.log1p(df["months_itm"].astype("float64"))
    df["log_orig_upb"] = np.log(df["orig_upb"].astype("float64"))
    df["month_of_year"] = df["period"].dt.month
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("string").astype(object)
    df["prepaid"] = df["prepaid"].astype("int8")
    needed = ["incentive_clipped", "age_capped", "log_months_itm", "log_orig_upb",
              *CATEGORICAL_FEATURES]
    return df.dropna(subset=needed)


def split_out_of_time(features: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on reporting months <= ``cfg.train_end``; test on later months."""
    cutoff = pd.Period(cfg.train_end, freq="M").to_timestamp(how="end")
    train = features[features["period"] <= cutoff]
    test = features[features["period"] > cutoff]
    return train, test


def downsample_nonevents(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    """Keep every prepay row and a random ``frac`` of the non-prepay rows."""
    if frac >= 1:
        return df
    events = df[df["prepaid"] == 1]
    nonevents = df[df["prepaid"] == 0].sample(frac=frac, random_state=seed)
    return pd.concat([events, nonevents]).sort_index()


@dataclass
class PrepayModel:
    """A fitted model plus what is needed to score new rows correctly."""

    results: object  # statsmodels BinaryResultsWrapper
    sample_frac: float
    levels: dict[str, list] = field(default_factory=dict)
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    def _align(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit new rows to the training domain before scoring.

        Categorical levels unseen in training map to the most common training level,
        and spline inputs are clamped to the training range (no extrapolation).
        """
        out = df.copy()
        for col, levels in self.levels.items():
            out[col] = out[col].where(out[col].isin(levels), levels[0])
        for col, (lo, hi) in self.bounds.items():
            out[col] = out[col].clip(lo, hi)
        return out

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Monthly prepayment probability (SMM per loan), corrected for downsampling."""
        linear = np.asarray(self.results.predict(self._align(df), which="linear"))
        linear = linear + np.log(self.sample_frac)
        return 1.0 / (1.0 + np.exp(-linear))

    def coefficients(self) -> pd.DataFrame:
        """Coefficient table: estimate, standard error, z and p-value per term."""
        r = self.results
        table = pd.DataFrame({"coef": r.params, "std_err": r.bse, "z": r.tvalues,
                              "p_value": r.pvalues})
        table.loc["Intercept", "coef"] += np.log(self.sample_frac)
        return table.rename_axis("term").reset_index()


def fit_prepay_model(train: pd.DataFrame, cfg: Config, formula: str | None = None,
                     presampled: bool = False) -> PrepayModel:
    """Fit the logistic prepayment model on training rows from :func:`prepare_features`.

    Non-events are downsampled by ``cfg.nonevent_sample_frac`` (skipped when
    ``presampled``, i.e. the caller already applied :func:`downsample_nonevents`) and
    the intercept is corrected on prediction. Returns a :class:`PrepayModel`; the
    statsmodels result (with standard errors) is ``.results``.
    """
    if train["prepaid"].sum() == 0:
        raise ValueError("Training data has no prepayments; check the split date and data.")
    sample = train if presampled else downsample_nonevents(
        train, cfg.nonevent_sample_frac, cfg.random_seed)
    levels = {col: sample[col].value_counts().index.tolist() for col in CATEGORICAL_FEATURES}
    bounds = {col: (float(sample[col].min()), float(sample[col].max()))
              for col in SPLINE_FEATURES}
    results = smf.logit(formula or cfg.model_formula, data=sample).fit(disp=0, maxiter=200)
    return PrepayModel(results=results, sample_frac=cfg.nonevent_sample_frac, levels=levels,
                       bounds=bounds)


def score_sums(test: pd.DataFrame, p: np.ndarray, by: str) -> pd.DataFrame:
    """Per-group UPB, actual prepaid UPB and predicted prepaid UPB (additive across files)."""
    w = test["start_upb"].to_numpy()
    work = pd.DataFrame({by: test[by].reset_index(drop=True), "upb": w,
                         "actual": w * test["prepaid"].to_numpy(), "predicted": w * p})
    return work.groupby(by, observed=True).agg(
        loans=("upb", "size"), start_upb=("upb", "sum"), actual=("actual", "sum"),
        predicted=("predicted", "sum")).reset_index()


def finish_scores(sums: pd.DataFrame, by: str) -> pd.DataFrame:
    """Combine :func:`score_sums` parts and convert to actual vs predicted SMM and CPR."""
    out = sums.groupby(by, observed=True, sort=True)[["loans", "start_upb", "actual",
                                                      "predicted"]].sum().reset_index()
    out["actual_smm"] = out["actual"] / out["start_upb"]
    out["predicted_smm"] = out["predicted"] / out["start_upb"]
    out["actual_cpr"] = smm_to_cpr(out["actual_smm"])
    out["predicted_cpr"] = smm_to_cpr(out["predicted_smm"])
    return out.drop(columns=["actual", "predicted"])


def _actual_vs_predicted(test: pd.DataFrame, p: np.ndarray, by: str) -> pd.DataFrame:
    return finish_scores(score_sums(test, p, by), by)


def classification_stats(y: np.ndarray, p: np.ndarray) -> dict:
    """Row counts, AUC and log-loss for test labels ``y`` and probabilities ``p``."""
    stats = {"test_rows": int(len(y)), "test_prepays": int(y.sum())}
    if 0 < y.sum() < len(y):
        stats["auc"] = float(roc_auc_score(y, p))
        stats["log_loss"] = float(log_loss(y, np.clip(p, 1e-12, 1 - 1e-12)))
    return stats


def predicted_cpr(model: PrepayModel, test: pd.DataFrame, by: str = "period") -> pd.DataFrame:
    """Actual vs predicted CPR per ``by`` group (default: reporting month).

    Predicted SMM = sum(p * start_upb) / sum(start_upb), then annualised; CPRs are decimals.
    """
    return _actual_vs_predicted(test, model.predict(test), by)


def evaluate(model: PrepayModel, test: pd.DataFrame) -> dict:
    """Test-period diagnostics: CPR tables by month and incentive bucket, AUC, log-loss."""
    p = model.predict(test)
    stats = classification_stats(test["prepaid"].to_numpy(), p)
    return {
        "by_month": _actual_vs_predicted(test, p, "period"),
        "by_incentive": _actual_vs_predicted(test, p, "incentive_bucket"),
        "stats": stats,
    }


def incentive_response(model: PrepayModel, reference: pd.DataFrame,
                       grid_bp: np.ndarray | None = None) -> pd.DataFrame:
    """Model-implied CPR across incentive, all other features held at typical values.

    Uses the most common category and the median numeric value from ``reference``.
    Used to check the S-curve shape (CPR should rise with incentive).
    """
    grid = np.arange(-150, 301, 25) if grid_bp is None else grid_bp
    row = {}
    for col in ("age_capped", "log_months_itm", "log_orig_upb"):
        row[col] = float(reference[col].median())
    for col in CATEGORICAL_FEATURES:
        row[col] = reference[col].mode().iloc[0]
    df = pd.DataFrame([row] * len(grid))
    df["incentive_clipped"] = grid
    return pd.DataFrame({"incentive_bp": grid, "cpr": smm_to_cpr(model.predict(df))})
