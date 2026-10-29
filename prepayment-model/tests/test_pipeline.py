"""End-to-end: raw fixture files -> parquet -> panel -> cohort tables and charts."""

import shutil
from dataclasses import replace

import pandas as pd
import pytest

from rmbs_analytics.cli import read_panel, run_ingest, run_metrics, run_panel

from .conftest import FIXTURES


@pytest.fixture
def project(cfg, tmp_path):
    raw = tmp_path / "data" / "raw"
    shutil.copytree(FIXTURES / "raw", raw)
    return replace(
        cfg, root=tmp_path, raw_dir=raw, interim_dir=tmp_path / "data" / "interim",
        processed_dir=tmp_path / "data" / "processed", figures_dir=tmp_path / "figures",
        tables_dir=tmp_path / "tables", pmms_path=raw / "MORTGAGE30US.csv",
        min_active_loans=1,
    )


def test_pipeline_end_to_end(project):
    run_ingest(project)
    assert (project.interim_dir / "perf_2015.parquet").exists()
    run_panel(project)
    panel = read_panel(project).set_index(["loan_id", "loan_age"])

    assert panel.loc[("F15Q10000001", 3), "prepaid"]
    assert not panel.loc["F15Q10000004", "prepaid"].any()          # paid at maturity
    assert panel.loc[("F15Q10000004", 8), "matured"]
    assert panel.loc[("F15Q10000003", 8), "defaulted"]             # 180+ days in month 8
    assert panel.loc[("F15Q10000006", 5), "defaulted"]             # short sale
    assert panel["defaulted"].sum() == 2
    # Note rate 4.00% vs March 2015 PMMS 3.70% (lagged from Feb: 3.80%) -> +20 bp.
    assert panel.loc[("F15Q10000001", 1), "incentive_bp"] == pytest.approx(20.0)

    run_metrics(project)
    vintage = pd.read_csv(project.tables_dir / "vintage.csv")
    assert vintage["loans"].item() == 40
    severity = pd.read_csv(project.tables_dir / "severity_by_default_year.csv")
    assert severity["severity"].item() == pytest.approx(0.25)
    for chart in ("cpr_by_vintage", "cpr_by_fico", "cpr_by_ltv", "s_curve", "seasoning_ramp"):
        assert (project.figures_dir / f"{chart}.png").exists()
