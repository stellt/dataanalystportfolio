"""Raw Freddie Mac text files -> typed DataFrames and Parquet.

Raw files are pipe-delimited with no header. The column layout is chosen from
``schema.yaml`` by counting the fields in the first row, so pre- and post-July 2026
(Release 47) downloads both load. Performance files are streamed in blocks and
written to Parquet incrementally, so a multi-gigabyte file never sits in memory.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from rmbs_analytics.config import Column, Config, Schema, load_schema

log = logging.getLogger(__name__)

_ARROW_READ_TYPES = {"str": pa.string(), "int": pa.int64(), "float": pa.float64(),
                     "yyyymm": pa.int64()}
_ARROW_OUT_TYPES = {"str": pa.string(), "int": pa.int32(), "float": pa.float64(),
                    "yyyymm": pa.timestamp("ms")}
_PANDAS_TYPES = {"str": "string", "int": "Int32", "float": "float64"}
_BLOCK_BYTES = 64 << 20


@dataclass(frozen=True)
class RawPair:
    """One origination file and its matching performance file."""

    tag: str
    orig: Path
    perf: Path


def count_fields(path: Path | str) -> int:
    """Number of pipe-delimited fields in the first row of ``path``."""
    with open(path, encoding="latin-1") as fh:
        first = fh.readline().rstrip("\r\n")
    if not first:
        raise ValueError(f"{path} is empty")
    return first.count("|") + 1


def arrow_schema(columns: tuple[Column, ...]) -> pa.Schema:
    """Arrow schema of the typed (post-ingest) table for a layout."""
    return pa.schema([(c.name, _ARROW_OUT_TYPES[c.dtype]) for c in columns])


def _yyyymm_to_date(s: pd.Series) -> pd.Series:
    """Map integer YYYYMM codes to month-start timestamps (invalid -> NaT)."""
    uniq = s.dropna().unique()
    parsed = pd.to_datetime(pd.Series(uniq).astype("int64").astype(str), format="%Y%m",
                            errors="coerce")
    return s.map(dict(zip(uniq, parsed, strict=True))).astype("datetime64[ms]")


def _type_frame(df: pd.DataFrame, columns: tuple[Column, ...]) -> pd.DataFrame:
    """Apply schema dtypes and not-available sentinels to a freshly read block."""
    out = {}
    for c in columns:
        s = df[c.name]
        if c.na_values:
            s = s.mask(s.isin(c.na_values))
        if c.dtype == "yyyymm":
            out[c.name] = _yyyymm_to_date(s)
        elif c.dtype == "str":
            out[c.name] = s.astype("string").str.strip().replace("", pd.NA)
        else:
            out[c.name] = s.astype(_PANDAS_TYPES[c.dtype])
        if c.negate:
            out[c.name] = -out[c.name]
    return pd.DataFrame(out)


def _iter_blocks(path: Path, kind: str, schema: Schema) -> Iterator[tuple[pd.DataFrame,
                                                                          tuple[Column, ...]]]:
    layout = schema.layout_for(kind, count_fields(path))
    columns = layout.columns(kind)
    log.info("%s: %s layout (%d fields)", path.name, layout.name, len(columns))
    reader = pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(column_names=[c.name for c in columns],
                                       block_size=_BLOCK_BYTES, encoding="latin-1"),
        parse_options=pacsv.ParseOptions(delimiter="|", quote_char=False),
        convert_options=pacsv.ConvertOptions(
            column_types={c.name: _ARROW_READ_TYPES[c.dtype] for c in columns},
            null_values=[""],
            strings_can_be_null=True,
        ),
    )
    for batch in reader:
        if batch.num_rows:
            yield _type_frame(batch.to_pandas(), columns), columns


def _load(path: Path | str, kind: str, schema: Schema | None) -> pd.DataFrame:
    schema = schema or load_schema()
    frames = [df for df, _ in _iter_blocks(Path(path), kind, schema)]
    if not frames:
        raise ValueError(f"{path} has no rows")
    return pd.concat(frames, ignore_index=True)


def load_origination(path: Path | str, schema: Schema | None = None) -> pd.DataFrame:
    """Read a raw origination file into a typed loan-level DataFrame.

    Returns one row per loan with the columns named in ``schema.yaml``. Rates are in
    percent (6.125 means 6.125%), balances in dollars, LTV/CLTV/DTI in whole percent,
    and YYYYMM dates as month-start timestamps. Not-available sentinels become null.
    Money fields use the Release 47 sign convention (losses and expenses positive).
    """
    return _load(path, "origination", schema)


def load_performance(path: Path | str, schema: Schema | None = None) -> pd.DataFrame:
    """Read a raw monthly performance file into a typed loan-month DataFrame.

    Returns one row per loan per reporting month, typed as in :func:`load_origination`.
    ``delinquency_status`` stays a string (``00``, ``01``, ..., ``RA``, ``XX``) and
    ``zero_balance_code`` a two-character string (``01``, ``09``, ...).
    """
    return _load(path, "performance", schema)


def ingest_file(src: Path, dest: Path, kind: str, schema: Schema) -> int:
    """Stream one raw file into a typed Parquet file. Returns the row count."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".parquet.tmp")
    writer, rows = None, 0
    try:
        for df, columns in _iter_blocks(src, kind, schema):
            table = pa.Table.from_pandas(df, schema=arrow_schema(columns), preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
            writer.write_table(table)
            rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError(f"{src} has no rows")
    tmp.replace(dest)
    return rows


def discover_raw_files(cfg: Config) -> list[RawPair]:
    """Find origination/performance pairs in ``cfg.raw_dir`` using ``cfg.raw_files``."""
    pairs: dict[str, RawPair] = {}
    perf_paths = set()
    candidates = []
    for orig_glob, perf_template in cfg.raw_files:
        prefix, suffix = orig_glob.split("*")
        for orig in sorted(cfg.raw_dir.glob(orig_glob)):
            tag = orig.name[len(prefix):len(orig.name) - len(suffix)]
            perf = cfg.raw_dir / perf_template.format(tag=tag)
            candidates.append((tag, orig, perf))
            if perf.exists():
                perf_paths.add(perf)
    unpaired = []
    for tag, orig, perf in candidates:
        if orig in perf_paths:  # e.g. historical_data_time_* also matches historical_data_*
            continue
        if perf.exists():
            pairs.setdefault(tag, RawPair(tag, orig, perf))
        else:
            unpaired.append(orig)
    for orig in dict.fromkeys(unpaired):
        if all(p.orig != orig for p in pairs.values()):
            log.warning("No performance file found for %s; skipping", orig.name)
    return sorted(pairs.values(), key=lambda p: p.tag)


def ingest_all(cfg: Config, force: bool = False) -> list[RawPair]:
    """Convert every raw pair to ``interim/orig_<tag>.parquet`` and ``perf_<tag>.parquet``.

    Files already converted are skipped unless ``force``. Returns the pairs found.
    """
    pairs = discover_raw_files(cfg)
    if not pairs:
        raise FileNotFoundError(
            f"No raw Freddie Mac files found in {cfg.raw_dir}. See README 'Get the data'."
        )
    for pair in pairs:
        for kind, src, prefix in (("origination", pair.orig, "orig"),
                                  ("performance", pair.perf, "perf")):
            dest = cfg.interim_dir / f"{prefix}_{pair.tag}.parquet"
            if dest.exists() and not force:
                log.info("%s exists; skipping", dest.name)
                continue
            rows = ingest_file(src, dest, kind, cfg.schema)
            log.info("%s -> %s (%s rows)", src.name, dest.name, f"{rows:,}")
    return pairs
