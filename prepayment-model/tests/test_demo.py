"""The synthetic demo: deterministic files in the Release 47 layout, full pipeline run."""

import json

from rmbs_analytics.cli import run_demo
from rmbs_analytics.demo import write_demo_files
from rmbs_analytics.ingest import count_fields


def test_demo_files_are_deterministic_and_release_47(cfg, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_demo_files(a, cfg.schema, loans_per_vintage=5, seed=1)
    write_demo_files(b, cfg.schema, loans_per_vintage=5, seed=1)
    for name in ("demo_orig_2015.txt", "demo_perf_2015.txt", "MORTGAGE30US.csv"):
        assert (a / name).read_bytes() == (b / name).read_bytes()
    assert count_fields(a / "demo_orig_2015.txt") == 31
    assert count_fields(a / "demo_perf_2015.txt") == 35


def test_demo_runs_whole_pipeline(cfg, tmp_path):
    out = run_demo(cfg, tmp_path / "demo", loans_per_vintage=80, seed=3)
    stats = json.loads((out.tables_dir / "model_stats.json").read_text())
    assert stats["train_prepays"] > 0 and stats["test_prepays"] > 0
    for chart in ("cpr_by_vintage", "s_curve", "model_actual_vs_predicted"):
        assert (out.figures_dir / f"{chart}.png").exists()
