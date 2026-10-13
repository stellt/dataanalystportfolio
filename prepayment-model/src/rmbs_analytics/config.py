"""Load and validate the two YAML config files.

``config/schema.yaml`` describes the raw Freddie Mac file layouts and code meanings;
``config/analysis.yaml`` holds every analysis choice (vintages, buckets, split date,
thresholds). Nothing in the pipeline hard-codes a column position or a code value.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

SOURCE_ROOT = Path(__file__).resolve().parents[2]


def _default_config_file(name: str) -> Path:
    """``config/<name>`` in the working directory, else in the source checkout."""
    for root in (Path.cwd(), SOURCE_ROOT):
        if (root / "config" / name).exists():
            return root / "config" / name
    return Path.cwd() / "config" / name


DEFAULT_ANALYSIS = _default_config_file("analysis.yaml")
DEFAULT_SCHEMA = _default_config_file("schema.yaml")

DTYPES = {"str", "int", "float", "yyyymm"}
FILE_KINDS = ("origination", "performance")


class ConfigError(ValueError):
    """Raised when a config file is missing a key or holds an invalid value."""


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str
    na_values: tuple[int, ...] = ()


@dataclass(frozen=True)
class Layout:
    name: str
    origination: tuple[Column, ...]
    performance: tuple[Column, ...]

    def columns(self, kind: str) -> tuple[Column, ...]:
        return getattr(self, kind)


@dataclass(frozen=True)
class Schema:
    layouts: tuple[Layout, ...]
    zero_balance_meanings: dict[str, str]
    prepay_zero_balance: tuple[str, ...]
    credit_zero_balance: tuple[str, ...]
    reo_status: str
    unknown_status: tuple[str, ...]

    def layout_for(self, kind: str, n_fields: int) -> Layout:
        """Return the layout whose ``kind`` file has ``n_fields`` columns."""
        matches = [lay for lay in self.layouts if len(lay.columns(kind)) == n_fields]
        if len(matches) != 1:
            known = {lay.name: len(lay.columns(kind)) for lay in self.layouts}
            raise ConfigError(
                f"No unique {kind} layout with {n_fields} fields; known layouts: {known}"
            )
        return matches[0]


@dataclass(frozen=True)
class Config:
    """Every analysis setting, with paths resolved against the project root."""

    root: Path
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    figures_dir: Path
    tables_dir: Path
    pmms_path: Path
    raw_files: tuple[tuple[str, str], ...]
    vintages: tuple[int, ...] | None
    default_dq_months: int
    maturity_tolerance_months: int
    fico_edges: tuple[float, ...]
    fico_labels: tuple[str, ...]
    ltv_edges: tuple[float, ...]
    ltv_labels: tuple[str, ...]
    incentive_min_bp: int
    incentive_max_bp: int
    incentive_step_bp: int
    max_loan_age: int
    itm_threshold_bp: float
    incentive_lag_months: int
    min_active_loans: int
    train_end: str
    nonevent_sample_frac: float
    random_seed: int
    incentive_clip_bp: tuple[float, float]
    age_cap: int
    model_formula: str
    schema: Schema


def _require(d: dict[str, Any], key: str, where: str) -> Any:
    if key not in d:
        raise ConfigError(f"Missing key '{key}' in {where}")
    return d[key]


def _check_edges(edges: list, labels: list, name: str) -> None:
    if list(edges) != sorted(edges) or len(set(edges)) != len(edges):
        raise ConfigError(f"{name} edges must be strictly increasing: {edges}")
    if len(labels) != len(edges) + 1:
        raise ConfigError(f"{name} needs {len(edges) + 1} labels for {len(edges)} edges")


def load_schema(path: Path | str = DEFAULT_SCHEMA) -> Schema:
    """Read ``schema.yaml`` and validate its layouts and code lists."""
    raw = yaml.safe_load(Path(path).read_text())
    layouts = []
    for name, spec in _require(raw, "layouts", str(path)).items():
        parts = {}
        for kind in FILE_KINDS:
            cols = []
            for c in _require(spec, kind, f"layout {name}"):
                if c["dtype"] not in DTYPES:
                    raise ConfigError(f"{name}.{kind}.{c['name']}: bad dtype {c['dtype']}")
                cols.append(Column(c["name"], c["dtype"], tuple(c.get("na_values", ()))))
            names = [c.name for c in cols]
            if len(set(names)) != len(names):
                raise ConfigError(f"Duplicate column names in {name}.{kind}")
            parts[kind] = tuple(cols)
        layouts.append(Layout(name=name, **parts))

    codes = _require(raw, "codes", str(path))
    dq = _require(codes, "delinquency", "codes")
    return Schema(
        layouts=tuple(layouts),
        zero_balance_meanings={str(k): v for k, v in codes["zero_balance"].items()},
        prepay_zero_balance=tuple(str(c) for c in codes["prepay_zero_balance"]),
        credit_zero_balance=tuple(str(c) for c in codes["credit_zero_balance"]),
        reo_status=str(dq["reo_acquisition"]),
        unknown_status=tuple(str(c) for c in dq["not_available"]),
    )


def load_config(
    path: Path | str = DEFAULT_ANALYSIS, schema_path: Path | str | None = None
) -> Config:
    """Read ``analysis.yaml`` (and ``schema.yaml``) into a validated :class:`Config`.

    Relative paths in the file are resolved against the project root, taken as the
    parent of the folder holding ``analysis.yaml``.
    """
    path = Path(path).resolve()
    root = path.parent.parent
    raw = yaml.safe_load(path.read_text())
    schema = load_schema(schema_path or path.parent / "schema.yaml")

    paths = _require(raw, "paths", "analysis.yaml")
    events = _require(raw, "events", "analysis.yaml")
    buckets = _require(raw, "buckets", "analysis.yaml")
    rates = _require(raw, "rates", "analysis.yaml")
    model = _require(raw, "model", "analysis.yaml")

    fico, ltv, inc = buckets["fico"], buckets["ltv"], buckets["incentive_bp"]
    _check_edges(fico["edges"], fico["labels"], "fico")
    _check_edges(ltv["edges"], ltv["labels"], "ltv")
    if inc["min"] >= inc["max"] or (inc["max"] - inc["min"]) % inc["step"]:
        raise ConfigError("incentive_bp: max must exceed min by a whole number of steps")

    frac = float(model["nonevent_sample_frac"])
    if not 0 < frac <= 1:
        raise ConfigError("model.nonevent_sample_frac must be in (0, 1]")
    train_end = str(model["train_end"])
    try:
        pd.Period(train_end, freq="M")
    except ValueError as exc:
        raise ConfigError(f"model.train_end must look like YYYY-MM, got {train_end}") from exc

    vintages = raw.get("vintages")
    return Config(
        root=root,
        raw_dir=root / paths["raw"],
        interim_dir=root / paths["interim"],
        processed_dir=root / paths["processed"],
        figures_dir=root / paths["figures"],
        tables_dir=root / paths["tables"],
        pmms_path=root / paths["pmms"],
        raw_files=tuple((f["orig_glob"], f["perf_template"]) for f in raw["raw_files"]),
        vintages=tuple(int(v) for v in vintages) if vintages else None,
        default_dq_months=int(events["default_dq_months"]),
        maturity_tolerance_months=int(events["maturity_tolerance_months"]),
        fico_edges=tuple(fico["edges"]),
        fico_labels=tuple(fico["labels"]),
        ltv_edges=tuple(ltv["edges"]),
        ltv_labels=tuple(ltv["labels"]),
        incentive_min_bp=int(inc["min"]),
        incentive_max_bp=int(inc["max"]),
        incentive_step_bp=int(inc["step"]),
        max_loan_age=int(buckets["max_loan_age"]),
        itm_threshold_bp=float(rates["itm_threshold_bp"]),
        incentive_lag_months=int(rates["incentive_lag_months"]),
        min_active_loans=int(_require(raw, "reporting", "analysis.yaml")["min_active_loans"]),
        train_end=train_end,
        nonevent_sample_frac=frac,
        random_seed=int(model["random_seed"]),
        incentive_clip_bp=tuple(model["incentive_clip_bp"]),
        age_cap=int(model["age_cap"]),
        model_formula=" ".join(str(model["formula"]).split()),
        schema=schema,
    )
