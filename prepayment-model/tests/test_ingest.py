import pandas as pd
import pytest

from rmbs_analytics.config import load_schema
from rmbs_analytics.ingest import (
    count_fields,
    discover_raw_files,
    ingest_file,
    load_origination,
    load_performance,
)

from .conftest import FIXTURES

RAW = FIXTURES / "raw"
LEGACY = FIXTURES / "legacy"
PANDAS_KIND = {"int": "Int32", "float": "float64", "str": "string", "yyyymm": "datetime64[ms]"}


def _check_against_schema(df: pd.DataFrame, layout_name: str, kind: str) -> None:
    schema = load_schema()
    layout = next(lay for lay in schema.layouts if lay.name == layout_name)
    expected = layout.columns(kind)
    assert list(df.columns) == [c.name for c in expected]
    for c in expected:
        assert str(df[c.name].dtype) == PANDAS_KIND[c.dtype], c.name


@pytest.mark.parametrize(
    ("path", "kind", "layout"),
    [
        (RAW / "sample_orig_2015.txt", "origination", "release_47"),
        (RAW / "sample_svcg_2015.txt", "performance", "release_47"),
        (LEGACY / "historical_data_2019Q1.txt", "origination", "legacy"),
        (LEGACY / "historical_data_time_2019Q1.txt", "performance", "legacy"),
    ],
)
def test_schema(path, kind, layout):
    loader = load_origination if kind == "origination" else load_performance
    _check_against_schema(loader(path), layout, kind)


def test_row_counts_match_raw_files():
    for path, loader in ((RAW / "sample_orig_2015.txt", load_origination),
                         (RAW / "sample_svcg_2015.txt", load_performance)):
        n_lines = sum(1 for _ in open(path))
        assert len(loader(path)) == n_lines


def test_sentinels_become_null_and_dates_parse():
    orig = load_origination(RAW / "sample_orig_2015.txt").set_index("loan_id")
    assert pd.isna(orig.loc["F15Q10000005", "credit_score"])
    assert pd.isna(orig.loc["F15Q10000005", "orig_ltv"])
    assert orig.loc["F15Q10000001", "credit_score"] == 760
    assert orig.loc["F15Q10000001", "first_payment_date"] == pd.Timestamp("2015-03-01")
    assert orig.loc["F15Q10000001", "orig_rate"] == pytest.approx(4.0)


def test_zero_balance_code_keeps_leading_zero():
    perf = load_performance(RAW / "sample_svcg_2015.txt")
    assert set(perf["zero_balance_code"].dropna()) == {"01", "03"}


def test_unknown_layout_is_rejected(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text("a|b|c\n")
    assert count_fields(bad) == 3
    with pytest.raises(ValueError, match="layout"):
        load_origination(bad)


def test_ingest_file_roundtrip(tmp_path):
    schema = load_schema()
    dest = tmp_path / "perf.parquet"
    rows = ingest_file(RAW / "sample_svcg_2015.txt", dest, "performance", schema)
    back = pd.read_parquet(dest)
    assert rows == len(back) == 40
    assert back["period"].min() == pd.Timestamp("2015-03-01")


def test_discover_pairs_skips_performance_files(cfg, tmp_path):
    from dataclasses import replace

    for f in LEGACY.iterdir():
        (tmp_path / f.name).write_bytes(f.read_bytes())
    pairs = discover_raw_files(replace(cfg, raw_dir=tmp_path))
    assert [p.tag for p in pairs] == ["2019Q1"]
    assert pairs[0].perf.name == "historical_data_time_2019Q1.txt"
