"""Command-line entry point: ``rmbs ingest | panel | metrics | model | all``."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import typer

from rmbs_analytics import ingest, metrics, model, panel, plots, rates
from rmbs_analytics.config import DEFAULT_ANALYSIS, Config, load_config

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="RMBS prepayment and default analytics on Freddie Mac loan-level data.")
log = logging.getLogger("rmbs")

ConfigOption = typer.Option(DEFAULT_ANALYSIS, "--config", "-c", help="Path to analysis.yaml")

PANEL_READ_COLUMNS = [
    "loan_id", "period", "loan_age", "vintage", "fico_band", "ltv_band", "incentive_bucket",
    "incentive_bp", "months_itm", "start_upb", "current_upb", "active_start", "prepaid",
    "defaulted", "zero_balance_code", "actual_loss", "zero_balance_removal_upb", "orig_upb",
    "loan_purpose", "occupancy",
]


def _setup(config: Path) -> Config:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    return load_config(config)


def _panel_dir(cfg: Config) -> Path:
    return cfg.processed_dir / "panel"


def read_panel(cfg: Config, columns: list[str] | None = None) -> pd.DataFrame:
    """Load the processed loan-month panel, restoring bucket order from the config."""
    files = sorted(_panel_dir(cfg).glob("panel_*.parquet"))
    if not files:
        raise typer.BadParameter("No panel found; run `rmbs panel` first.")
    df = pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)
    orders = {"fico_band": list(cfg.fico_labels), "ltv_band": list(cfg.ltv_labels),
              "incentive_bucket": rates.incentive_labels(cfg)}
    for col, cats in orders.items():
        if col in df:
            df[col] = pd.Categorical(df[col].astype("string").astype(object), categories=cats,
                                     ordered=True)
    return df


def run_ingest(cfg: Config, force: bool = False) -> None:
    pairs = ingest.ingest_all(cfg, force=force)
    log.info("Ingested %d file pair(s): %s", len(pairs), ", ".join(p.tag for p in pairs))


def run_panel(cfg: Config) -> None:
    if not cfg.pmms_path.exists():
        raise typer.BadParameter(f"{cfg.pmms_path} not found; run `rmbs fetch-pmms` first.")
    pmms = rates.load_pmms(cfg.pmms_path)
    origs = sorted(cfg.interim_dir.glob("orig_*.parquet"))
    if not origs:
        raise typer.BadParameter("No interim files; run `rmbs ingest` first.")
    out_dir = _panel_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("panel_*.parquet"):
        stale.unlink()
    total = 0
    for orig_path in origs:
        tag = orig_path.stem.removeprefix("orig_")
        perf_path = cfg.interim_dir / f"perf_{tag}.parquet"
        orig = pd.read_parquet(orig_path, columns=panel.ORIG_COLUMNS)
        perf = pd.read_parquet(perf_path, columns=panel.PERF_COLUMNS)
        p = panel.build_panel(orig, perf, cfg)
        if p.empty:
            log.info("%s: no loans in configured vintages; skipped", tag)
            continue
        p = rates.add_incentive(p, pmms, cfg)
        p.to_parquet(out_dir / f"panel_{tag}.parquet", index=False)
        total += len(p)
        log.info("%s: %s loan-months, %s prepaid, %s defaulted", tag, f"{len(p):,}",
                 f"{int(p['prepaid'].sum()):,}", f"{int(p['defaulted'].sum()):,}")
    if total == 0:
        raise typer.BadParameter(
            f"No loan-months left after filtering to vintages {cfg.vintages}. "
            "Check analysis.yaml `vintages` against the files in data/raw.")
    log.info("Panel: %s loan-months in %s", f"{total:,}", out_dir)


def run_metrics(cfg: Config) -> None:
    df = read_panel(cfg, PANEL_READ_COLUMNS)
    tables = metrics.all_cohort_tables(df, cfg)
    tables["severity_by_default_year"] = metrics.severity_by_default_year(df, cfg)
    metrics.write_tables(tables, cfg.processed_dir / "cohorts")
    summary = {k: tables[k] for k in ("vintage", "fico_band", "ltv_band", "incentive_bucket",
                                      "overall_by_loan_age", "severity_by_default_year")
               if k in tables}
    metrics.write_tables(summary, cfg.tables_dir)
    charts = plots.make_cohort_charts(tables, cfg.figures_dir)
    log.info("Wrote %d tables and %d charts", len(tables), len(charts))


def run_model(cfg: Config) -> None:
    df = read_panel(cfg, PANEL_READ_COLUMNS)
    features = model.prepare_features(df, cfg)
    train, test = model.split_out_of_time(features, cfg)
    if train.empty or test.empty:
        span = f"{features['period'].min():%Y-%m} to {features['period'].max():%Y-%m}"
        raise typer.BadParameter(
            f"Out-of-time split at {cfg.train_end} leaves train={len(train):,} and "
            f"test={len(test):,} rows (data covers {span}). Adjust model.train_end.")
    log.info("Fitting on %s train rows (%s prepays); testing on %s rows", f"{len(train):,}",
             f"{int(train['prepaid'].sum()):,}", f"{len(test):,}")
    fitted = model.fit_prepay_model(train, cfg)
    evaluation = model.evaluate(fitted, test)
    response = model.incentive_response(fitted, train)

    cfg.tables_dir.mkdir(parents=True, exist_ok=True)
    coefs = fitted.coefficients()
    coefs.to_csv(cfg.tables_dir / "model_coefficients.csv", index=False, float_format="%.6g")
    evaluation["by_month"].to_csv(cfg.tables_dir / "model_actual_vs_predicted_by_month.csv",
                                  index=False, float_format="%.6g")
    evaluation["by_incentive"].to_csv(cfg.tables_dir / "model_by_incentive.csv", index=False,
                                      float_format="%.6g")
    burnout = coefs.set_index("term")["coef"].get("log_months_itm")
    stats = {**evaluation["stats"], "train_rows": len(train),
             "train_prepays": int(train["prepaid"].sum()), "train_end": cfg.train_end,
             "nonevent_sample_frac": cfg.nonevent_sample_frac,
             "converged": bool(fitted.results.mle_retvals.get("converged", False)),
             "burnout_coef": None if burnout is None else float(burnout),
             "s_curve_increasing": bool(response["cpr"].is_monotonic_increasing)}
    (cfg.tables_dir / "model_stats.json").write_text(json.dumps(stats, indent=2))
    (cfg.tables_dir / "model_summary.txt").write_text(str(fitted.results.summary()))
    plots.make_model_charts(evaluation, response, cfg.figures_dir)
    if not stats["converged"]:
        log.warning("Model did not converge; see %s", cfg.tables_dir / "model_summary.txt")
    log.info("Model stats: %s", json.dumps(stats))


@app.command("ingest")
def ingest_cmd(config: Path = ConfigOption,
               force: bool = typer.Option(False, help="Re-convert files already in interim/")):
    """Convert raw Freddie Mac text files to typed Parquet."""
    run_ingest(_setup(config), force)


@app.command("fetch-pmms")
def fetch_pmms_cmd(config: Path = ConfigOption):
    """Download the PMMS 30-year rate (FRED MORTGAGE30US) to data/raw."""
    cfg = _setup(config)
    rates.download_pmms(cfg.pmms_path)
    log.info("Saved %s", cfg.pmms_path)


@app.command("panel")
def panel_cmd(config: Path = ConfigOption):
    """Build the loan-month panel with event flags, start_upb and incentive."""
    run_panel(_setup(config))


@app.command("metrics")
def metrics_cmd(config: Path = ConfigOption):
    """Compute cohort tables (SMM/CPR/MDR/CDR) and charts."""
    run_metrics(_setup(config))


@app.command("model")
def model_cmd(config: Path = ConfigOption):
    """Fit the logistic prepayment model and evaluate it out of time."""
    run_model(_setup(config))


@app.command("all")
def all_cmd(config: Path = ConfigOption,
            force: bool = typer.Option(False, help="Re-convert files already in interim/")):
    """Run ingest, panel, metrics and model in order."""
    cfg = _setup(config)
    if not cfg.pmms_path.exists():
        rates.download_pmms(cfg.pmms_path)
    run_ingest(cfg, force)
    run_panel(cfg)
    run_metrics(cfg)
    run_model(cfg)


if __name__ == "__main__":
    app()
