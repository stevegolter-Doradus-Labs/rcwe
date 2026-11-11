"""
Robust tabular telemetry ingestion.

`load_table` turns "whatever the user has" -- a CSV/TSV/JSON/JSONL file, a list of row
dicts, a dict of columns, a NumPy array or a pandas DataFrame -- into a clean
`TelemetryTable`: numeric sensor and control matrices, an optional time axis, optional
ground-truth labels, and a list of plain-English data-quality notes describing every
decision that was made on the way (columns ignored, gaps filled, rows re-ordered...).

Nothing here depends on pandas; DataFrames are supported by duck typing.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = ["TelemetryTable", "load_table", "DataError"]


class DataError(ValueError):
    """Raised when input data cannot be turned into a usable telemetry table."""


MISSING_TOKENS = {
    "", "na", "n/a", "nan", "null", "none", "nil", "-", "--", "?", "#n/a", "#na", "#value!",
    "#div/0!", "inf", "-inf", "+inf", "infinity", "-infinity", "missing", "bad", "error",
}
TIME_NAMES = (
    "timestamp", "time", "datetime", "date_time", "date", "ts", "t", "utc", "epoch",
    "timestamp_utc", "time_utc", "created_at", "recorded_at", "measured_at",
)
STEP_NAMES = ("timestamp_step", "step", "time_step", "timestep", "index", "idx", "sample", "tick", "row", "id", "#")
LABEL_NAMES = (
    "label", "labels", "anomaly", "is_anomaly", "anomalous", "fault", "is_fault", "attack",
    "is_attack", "regime_label", "regime", "ground_truth", "y_true", "outlier", "is_outlier",
)
NORMAL_LABEL_WORDS = {"", "0", "0.0", "normal", "ok", "false", "no", "nominal", "benign", "none", "n", "healthy"}
CONTROL_PREFIXES = ("act_", "u_", "ctrl_", "control_", "cmd_", "setpoint_", "sp_")
CONTROL_SUFFIXES = ("_setpoint", "_cmd", "_sp", "_command")

_DATETIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
    "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M", "%Y/%m/%d",
    "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M", "%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %I:%M %p", "%m/%d/%Y",
    "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y",
    "%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M", "%d.%m.%Y",
    "%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M", "%d-%b-%Y",
    "%Y%m%d%H%M%S", "%Y%m%dT%H%M%S", "%Y%m%d",
)


# --------------------------------------------------------------------------------------
# value parsing helpers
# --------------------------------------------------------------------------------------
def _to_float(value: Any) -> Optional[float]:
    """float for numbers, NaN for missing markers, None for genuinely non-numeric text."""
    if value is None:
        return math.nan
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else math.nan
    if isinstance(value, np.generic):
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return out if math.isfinite(out) else math.nan
    text = str(value).strip()
    if text.lower() in MISSING_TOKENS:
        return math.nan
    try:
        out = float(text)
        return out if math.isfinite(out) else math.nan
    except ValueError:
        pass
    # European decimal comma ("12,5") and thousands separators ("1,234.5" / "1.234,5").
    cleaned = text.replace(" ", "").replace(" ", "")
    if "," in cleaned:
        if "." in cleaned:
            if cleaned.rfind(",") > cleaned.rfind("."):
                cleaned = cleaned.replace(".", "").replace(",", ".")
            else:
                cleaned = cleaned.replace(",", "")
        elif cleaned.count(",") == 1:
            cleaned = cleaned.replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
        try:
            out = float(cleaned)
            return out if math.isfinite(out) else math.nan
        except ValueError:
            return None
    if cleaned.endswith("%"):
        try:
            return float(cleaned[:-1])
        except ValueError:
            return None
    return None


def _parse_datetime(text: str, fmt_hint: Optional[str] = None) -> Tuple[Optional[float], Optional[str]]:
    """Return (epoch_seconds, format_used). Naive datetimes are interpreted as UTC."""
    s = str(text).strip()
    if not s:
        return None, fmt_hint
    if fmt_hint and fmt_hint != "iso":
        try:
            return datetime.strptime(s, fmt_hint).replace(tzinfo=timezone.utc).timestamp(), fmt_hint
        except ValueError:
            pass
    iso = s[:-1] + "+00:00" if s.endswith(("Z", "z")) else s
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp(), "iso"
    except ValueError:
        pass
    for fmt in _DATETIME_FORMATS:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp(), fmt
        except ValueError:
            continue
    return None, fmt_hint


def _stamp_text(value: Any) -> str:
    """How a timestamp cell is shown: the user's own text, or compact ISO-8601 for datetime objects."""
    if hasattr(value, "isoformat"):
        text = str(value.isoformat())
        return text[:-6] + "Z" if text.endswith("+00:00") else text
    return str(value).strip()


def format_time(seconds: float) -> str:
    """Compact ISO-8601 UTC string for an epoch-seconds value."""
    dt = datetime.fromtimestamp(float(seconds), tz=timezone.utc)
    if dt.microsecond:
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_control_name(name: str) -> bool:
    low = name.lower()
    return low.startswith(CONTROL_PREFIXES) or low.endswith(CONTROL_SUFFIXES)


def _fill_gaps(column: np.ndarray) -> np.ndarray:
    """Linear interpolation across NaN runs; nearest valid value at the edges."""
    bad = np.isnan(column)
    if not bad.any():
        return column
    good = ~bad
    if not good.any():
        return np.zeros_like(column)
    idx = np.arange(column.shape[0])
    filled = column.copy()
    filled[bad] = np.interp(idx[bad], idx[good], column[good])
    return filled


# --------------------------------------------------------------------------------------
# raw readers: everything becomes an ordered {column_name: list_of_cells}
# --------------------------------------------------------------------------------------
def _dedupe(names: Sequence[Any]) -> List[str]:
    seen: Dict[str, int] = {}
    out: List[str] = []
    for i, raw in enumerate(names):
        name = str(raw).strip() if raw is not None else ""
        if name.startswith("﻿"):
            name = name.lstrip("﻿")
        if not name:
            name = f"column_{i}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        out.append(name)
    return out


def _rows_to_columns(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Any]]:
    names: List[str] = []
    for row in rows:
        if not isinstance(row, dict):
            raise DataError("Expected a list of row objects (one {column: value} mapping per time step).")
        for key in row:
            if key not in names:
                names.append(key)
    return {str(name): [row.get(name) for row in rows] for name in names}


def _read_delimited(text: str, notes: List[str]) -> Dict[str, List[Any]]:
    text = text.lstrip("﻿")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    # Skip leading comment lines (a common habit in logger exports).
    while lines and lines[0].lstrip().startswith("#") and "," not in lines[0] and ";" not in lines[0]:
        lines.pop(0)
    if len(lines) < 2:
        raise DataError("The file has no data rows (need a header line plus at least one row).")
    sample = "\n".join(lines[:25])
    delimiter = ","
    best = 0
    for cand in (",", ";", "\t", "|"):
        counts = [ln.count(cand) for ln in lines[:25]]
        if counts and min(counts) > 0 and min(counts) == max(counts) and counts[0] > best:
            delimiter, best = cand, counts[0]
    if best == 0:
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = ","
    reader = csv.reader(lines, delimiter=delimiter)
    header = _dedupe(next(reader))
    columns: Dict[str, List[Any]] = {name: [] for name in header}
    ragged = 0
    for cells in reader:
        if len(cells) != len(header):
            ragged += 1
        for i, name in enumerate(header):
            columns[name].append(cells[i] if i < len(cells) else None)
    if ragged:
        notes.append(f"{ragged} row(s) had a different number of fields than the header; missing cells were treated as gaps.")
    if delimiter != ",":
        label = {"\t": "tab", ";": "semicolon", "|": "pipe"}.get(delimiter, repr(delimiter))
        notes.append(f"Detected {label}-separated values.")
    return columns


def _read_json_payload(payload: Any) -> Dict[str, List[Any]]:
    if isinstance(payload, dict):
        for key in ("telemetry", "data", "rows", "records", "samples"):
            if isinstance(payload.get(key), list) and payload[key] and isinstance(payload[key][0], dict):
                return _rows_to_columns(payload[key])
        if isinstance(payload.get("columns"), list) and isinstance(payload.get("data"), list):
            names = _dedupe(payload["columns"])
            return {name: [row[i] if i < len(row) else None for row in payload["data"]] for i, name in enumerate(names)}
        if payload and all(isinstance(v, (list, tuple)) for v in payload.values()):
            return {str(k): list(v) for k, v in payload.items()}
        raise DataError(
            "Unrecognised JSON layout. Use a list of row objects, {\"telemetry\": [...]}, "
            "or an object mapping column names to lists."
        )
    if isinstance(payload, list):
        if not payload:
            raise DataError("The JSON file contains an empty list.")
        if isinstance(payload[0], dict):
            return _rows_to_columns(payload)
        if isinstance(payload[0], (list, tuple)):
            width = max(len(r) for r in payload)
            return {f"sensor_{i}": [r[i] if i < len(r) else None for r in payload] for i in range(width)}
    raise DataError("Unrecognised JSON layout: expected an object or a list of rows.")


def _read_path(path: Path, notes: List[str]) -> Dict[str, List[Any]]:
    if not path.exists():
        raise DataError(f"File not found: {path}")
    if path.is_dir():
        raise DataError(f"{path} is a directory; pass a data file (.csv, .tsv, .json, .jsonl).")
    suffixes = [s.lower() for s in path.suffixes]
    kind = suffixes[-1] if suffixes else ""
    compressed = kind == ".gz"
    if compressed:
        kind = suffixes[-2] if len(suffixes) > 1 else ".csv"

    if kind in (".xlsx", ".xls", ".parquet", ".feather"):
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:
            raise DataError(
                f"Reading {kind} files needs pandas. Install it with `pip install pandas openpyxl pyarrow`, "
                "or export the data to CSV."
            ) from exc
        try:
            frame = pd.read_excel(path) if kind in (".xlsx", ".xls") else (
                pd.read_parquet(path) if kind == ".parquet" else pd.read_feather(path)
            )
        except ImportError as exc:
            raise DataError(f"pandas could not read {path.name}: {exc}. Exporting to CSV always works.") from exc
        return _frame_to_columns(frame)

    opener = gzip.open if compressed else open
    try:
        with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:  # type: ignore[operator]
            text = handle.read()
    except UnicodeDecodeError:
        with opener(path, "rt", encoding="latin-1", newline="") as handle:  # type: ignore[operator]
            text = handle.read()
        notes.append("File is not UTF-8; decoded it as Latin-1.")
    if not text.strip():
        raise DataError(f"{path.name} is empty.")

    if kind in (".jsonl", ".ndjson"):
        rows = []
        for n, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise DataError(f"{path.name}, line {n}: invalid JSON ({exc.msg}).") from exc
        return _rows_to_columns(rows)
    if kind == ".json" or text.lstrip()[:1] in ("{", "["):
        try:
            return _read_json_payload(json.loads(text))
        except json.JSONDecodeError as exc:
            if kind == ".json":
                raise DataError(f"{path.name} is not valid JSON ({exc.msg} at line {exc.lineno}).") from exc
    return _read_delimited(text, notes)


def _frame_to_columns(frame: Any) -> Dict[str, List[Any]]:
    columns: Dict[str, List[Any]] = {}
    index = getattr(frame, "index", None)
    if index is not None and type(index).__name__ == "DatetimeIndex":
        columns[str(getattr(index, "name", None) or "timestamp")] = [_stamp_text(v) for v in index]
    names = _dedupe(list(frame.columns))
    for name, original in zip(names, list(frame.columns)):
        series = frame[original]
        values = series.tolist() if hasattr(series, "tolist") else list(series)
        columns[name] = [_stamp_text(v) if hasattr(v, "isoformat") else v for v in values]
    return columns


def _to_columns(source: Any, notes: List[str]) -> Tuple[Dict[str, List[Any]], str]:
    if isinstance(source, (str, Path)):
        text = str(source)
        if isinstance(source, str) and ("\n" in text or "\r" in text):
            stripped = text.lstrip()
            if stripped[:1] in ("{", "["):
                return _read_json_payload(json.loads(text)), "<text>"
            return _read_delimited(text, notes), "<text>"
        path = Path(text).expanduser()
        return _read_path(path, notes), str(path)
    if isinstance(source, io.IOBase) or hasattr(source, "read"):
        text = source.read()
        if isinstance(text, bytes):
            text = text.decode("utf-8-sig", errors="replace")
        if text.lstrip()[:1] in ("{", "["):
            return _read_json_payload(json.loads(text)), getattr(source, "name", "<stream>")
        return _read_delimited(text, notes), getattr(source, "name", "<stream>")
    if hasattr(source, "columns") and hasattr(source, "__getitem__") and not isinstance(source, dict):
        return _frame_to_columns(source), "<DataFrame>"
    if hasattr(source, "detach") and hasattr(source, "cpu"):  # torch.Tensor
        source = source.detach().cpu().numpy()
    if isinstance(source, np.ndarray):
        arr = np.asarray(source, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:
            raise DataError(f"Expected a 2-D array shaped [time_steps, sensors], got shape {arr.shape}.")
        return {f"sensor_{i}": arr[:, i].tolist() for i in range(arr.shape[1])}, "<array>"
    if isinstance(source, dict):
        if not source:
            raise DataError("The data mapping is empty.")
        if all(isinstance(v, (list, tuple, np.ndarray)) for v in source.values()):
            return {str(k): list(v) for k, v in source.items()}, "<dict>"
        return _read_json_payload(source), "<dict>"
    if isinstance(source, (list, tuple)):
        if not source:
            raise DataError("The data list is empty.")
        if isinstance(source[0], dict):
            return _rows_to_columns(list(source)), "<rows>"
        return _read_json_payload(list(source)), "<rows>"
    raise DataError(
        f"Unsupported data source of type {type(source).__name__}. Pass a file path, a list of row dicts, "
        "a dict of columns, a NumPy array or a pandas DataFrame."
    )


# --------------------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------------------
@dataclass
class TelemetryTable:
    """A cleaned, model-ready view of a telemetry log."""

    sensors: List[str]
    values: np.ndarray
    controls: List[str] = field(default_factory=list)
    control_values: Optional[np.ndarray] = None
    timestamps: Optional[List[str]] = None
    time_seconds: Optional[np.ndarray] = None
    time_column: Optional[str] = None
    labels: Optional[np.ndarray] = None
    label_column: Optional[str] = None
    missing: Optional[np.ndarray] = None
    notes: List[str] = field(default_factory=list)
    ignored_columns: List[str] = field(default_factory=list)
    source: str = "<memory>"

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float)
        if self.values.ndim != 2:
            raise DataError("values must be a 2-D array shaped [time_steps, sensors].")
        if self.values.shape[1] != len(self.sensors):
            raise DataError("values has a different number of columns than there are sensor names.")
        rows = self.values.shape[0]
        if self.control_values is None:
            self.control_values = np.zeros((rows, len(self.controls)), dtype=float)
        self.control_values = np.asarray(self.control_values, dtype=float).reshape(rows, len(self.controls))
        if self.missing is None:
            self.missing = np.zeros(self.values.shape, dtype=bool)

    # ---- shape & time helpers -----------------------------------------------------
    @property
    def n_rows(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_sensors(self) -> int:
        return len(self.sensors)

    def __len__(self) -> int:
        return self.n_rows

    @property
    def sample_interval(self) -> Optional[float]:
        """Median spacing between samples in seconds (None when there is no real time axis)."""
        if self.time_seconds is None or self.n_rows < 2:
            return None
        deltas = np.diff(self.time_seconds)
        deltas = deltas[deltas > 0]
        return float(np.median(deltas)) if deltas.size else None

    def time_label(self, index: int) -> str:
        if self.timestamps is not None and 0 <= index < len(self.timestamps):
            return str(self.timestamps[index])
        return f"step {index}"

    def column(self, name: str) -> np.ndarray:
        if name in self.sensors:
            return self.values[:, self.sensors.index(name)]
        if name in self.controls:
            return self.control_values[:, self.controls.index(name)]
        raise KeyError(name)

    def slice(self, start: Optional[int] = None, stop: Optional[int] = None) -> "TelemetryTable":
        """Row-range copy (same schema), e.g. `table.slice(0, 500)` for the first 500 steps."""
        sl = slice(start, stop)
        return TelemetryTable(
            sensors=list(self.sensors),
            values=self.values[sl].copy(),
            controls=list(self.controls),
            control_values=self.control_values[sl].copy(),
            timestamps=None if self.timestamps is None else list(self.timestamps[sl]),
            time_seconds=None if self.time_seconds is None else self.time_seconds[sl].copy(),
            time_column=self.time_column,
            labels=None if self.labels is None else self.labels[sl].copy(),
            label_column=self.label_column,
            missing=self.missing[sl].copy(),
            notes=list(self.notes),
            ignored_columns=list(self.ignored_columns),
            source=self.source,
        )

    def split(self, fraction: float = 0.5) -> Tuple["TelemetryTable", "TelemetryTable"]:
        """Chronological split: the first `fraction` of rows, and the rest."""
        if not 0.0 < fraction < 1.0:
            raise ValueError("fraction must be between 0 and 1.")
        cut = max(1, min(self.n_rows - 1, int(round(self.n_rows * fraction))))
        return self.slice(0, cut), self.slice(cut, None)

    # ---- reporting ----------------------------------------------------------------
    def describe(self) -> Dict[str, Any]:
        """Schema and data-quality summary (what `rcwe inspect` prints)."""
        stats = []
        for i, name in enumerate(self.sensors):
            col = self.values[:, i]
            stats.append({
                "name": name,
                "min": float(np.min(col)), "max": float(np.max(col)),
                "mean": float(np.mean(col)), "std": float(np.std(col)),
                "missing": int(self.missing[:, i].sum()),
            })
        interval = self.sample_interval
        return {
            "source": self.source,
            "rows": self.n_rows,
            "sensors": list(self.sensors),
            "controls": list(self.controls),
            "time_column": self.time_column,
            "start": self.time_label(0) if self.n_rows else None,
            "end": self.time_label(self.n_rows - 1) if self.n_rows else None,
            "sample_interval_seconds": interval,
            "label_column": self.label_column,
            "labeled_anomalous_rows": None if self.labels is None else int(self.labels.sum()),
            "missing_cells": int(self.missing.sum()),
            "ignored_columns": list(self.ignored_columns),
            "sensor_stats": stats,
            "notes": list(self.notes),
        }

    def to_rows(self) -> List[Dict[str, Any]]:
        rows = []
        for t in range(self.n_rows):
            row: Dict[str, Any] = {}
            if self.timestamps is not None:
                row[self.time_column or "timestamp"] = self.timestamps[t]
            for i, name in enumerate(self.sensors):
                row[name] = float(self.values[t, i])
            for i, name in enumerate(self.controls):
                row[name] = float(self.control_values[t, i])
            if self.labels is not None:
                row[self.label_column or "anomaly"] = int(self.labels[t])
            rows.append(row)
        return rows

    def to_csv(self, path: Union[str, Path]) -> str:
        rows = self.to_rows()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
        return str(out.resolve())


# --------------------------------------------------------------------------------------
# public loader
# --------------------------------------------------------------------------------------
def load_table(
    source: Any,
    *,
    sensors: Optional[Sequence[str]] = None,
    controls: Optional[Sequence[str]] = None,
    time_column: Optional[str] = None,
    label_column: Optional[str] = None,
    ignore: Optional[Iterable[str]] = None,
) -> TelemetryTable:
    """
    Load telemetry from a file or an in-memory object and infer its schema.

    Parameters
    ----------
    source : path | str | list[dict] | dict[str, list] | numpy.ndarray | pandas.DataFrame | TelemetryTable
        CSV/TSV (any common delimiter, optional .gz), JSON, JSON-lines, or in-memory data.
    sensors, controls : optional explicit column lists. By default every numeric column is a
        sensor, except columns that look like controls (`act_*`, `u_*`, `ctrl_*`, `cmd_*`,
        `setpoint_*`, `*_setpoint`, `*_cmd`), which become control inputs.
    time_column, label_column : optional explicit names; auto-detected otherwise.
    ignore : column names to leave out entirely.
    """
    if isinstance(source, TelemetryTable):
        return source

    notes: List[str] = []
    columns, origin = _to_columns(source, notes)
    if not columns:
        raise DataError("No columns found in the data.")
    lengths = {len(v) for v in columns.values()}
    if len(lengths) != 1:
        raise DataError("Columns have different lengths; every column needs one value per time step.")
    n_rows = lengths.pop()
    if n_rows < 2:
        raise DataError(f"Only {n_rows} row(s) of data found; at least 2 time steps are needed.")

    names = list(columns.keys())
    ignore_set = {str(c) for c in (ignore or [])}
    for label, wanted in (("sensors", sensors), ("controls", controls), ("time_column", [time_column] if time_column else None),
                          ("label_column", [label_column] if label_column else None)):
        for col in wanted or []:
            if col not in columns:
                close = [n for n in names if n.lower() == str(col).lower()]
                hint = f" Did you mean '{close[0]}'?" if close else f" Available columns: {', '.join(names[:12])}{'...' if len(names) > 12 else ''}"
                raise DataError(f"Column '{col}' (given in {label}) is not in the data.{hint}")

    # ---- classify every column as numeric or text ---------------------------------
    numeric: Dict[str, np.ndarray] = {}
    text_cols: List[str] = []
    for name in names:
        cells = columns[name]
        parsed = [_to_float(c) for c in cells]
        bad = sum(1 for p in parsed if p is None)
        present = sum(1 for p in parsed if p is not None and not math.isnan(p))
        if present > 0 and bad <= 0.1 * (present + bad):
            numeric[name] = np.array([math.nan if p is None else p for p in parsed], dtype=float)
            if bad:
                notes.append(f"'{name}': {bad} non-numeric cell(s) treated as missing.")
        else:
            text_cols.append(name)

    ignored: List[str] = []
    used: set = set(ignore_set)
    ignored.extend(sorted(ignore_set & set(names)))

    # ---- time axis ----------------------------------------------------------------
    time_name: Optional[str] = None
    timestamps: Optional[List[str]] = None
    time_seconds: Optional[np.ndarray] = None

    def _try_datetime(col: str) -> bool:
        nonlocal timestamps, time_seconds
        cells = columns[col]
        fmt: Optional[str] = None
        secs: List[float] = []
        failures = 0
        for c in cells:
            value, fmt = _parse_datetime("" if c is None else str(c), fmt)
            if value is None:
                failures += 1
                secs.append(math.nan)
            else:
                secs.append(value)
        if failures > 0.1 * len(cells):
            return False
        arr = _fill_gaps(np.array(secs, dtype=float))
        # Rows keep the user's own timestamp text; a row whose timestamp could not be read is
        # labelled with the time interpolated for it rather than left blank.
        timestamps = [format_time(filled) if math.isnan(parsed) else _stamp_text(c)
                      for c, parsed, filled in zip(cells, secs, arr)]
        time_seconds = arr
        if failures:
            notes.append(f"'{col}': {failures} unparseable timestamp(s) were interpolated.")
        return True

    def _try_numeric_time(col: str) -> bool:
        nonlocal timestamps, time_seconds
        arr = numeric[col]
        if np.isnan(arr).any():
            return False
        finite_min = float(arr.min())
        if finite_min > 1e11:      # epoch milliseconds
            time_seconds = arr / 1000.0
            timestamps = [format_time(v) for v in time_seconds]
        elif finite_min > 1e8:     # epoch seconds
            time_seconds = arr.copy()
            timestamps = [format_time(v) for v in time_seconds]
        else:                      # plain step counter
            time_seconds = None
            timestamps = [str(int(v)) if float(v).is_integer() else f"{v:g}" for v in arr]
        return True

    candidates: List[str] = []
    if time_column:
        candidates = [time_column]
    else:
        lowered = {n: n.lower().strip() for n in names if n not in used}
        candidates += [n for n, low in lowered.items() if low in TIME_NAMES]
        candidates += [n for n, low in lowered.items() if low in STEP_NAMES and n not in candidates]
        candidates += [n for n in text_cols if n not in candidates and n not in used
                       and any(tok in lowered.get(n, "") for tok in ("time", "date"))]
        candidates += [n for n in text_cols if n not in candidates and n not in used]
    for cand in candidates:
        ok = False
        if cand in text_cols:
            ok = _try_datetime(cand)
        elif cand in numeric:
            arr = numeric[cand]
            monotonic = not np.isnan(arr).any() and bool(np.all(np.diff(arr) >= 0))
            if time_column or monotonic:
                ok = _try_numeric_time(cand)
        if ok:
            time_name = cand
            used.add(cand)
            break
        if time_column:
            raise DataError(f"Could not read '{time_column}' as a time column (values are neither dates nor numbers).")

    # ---- labels -------------------------------------------------------------------
    labels: Optional[np.ndarray] = None
    label_name: Optional[str] = None
    label_candidates = [label_column] if label_column else [n for n in names if n.lower().strip() in LABEL_NAMES and n not in used]
    for cand in label_candidates:
        if cand in numeric:
            arr = numeric[cand]
            finite = arr[~np.isnan(arr)]
            if label_column or (finite.size and np.unique(finite).size <= 12 and np.all(finite == np.round(finite))):
                labels = (np.nan_to_num(arr, nan=0.0) != 0).astype(int)
                label_name = cand
        elif cand in text_cols:
            labels = np.array([0 if str(c or "").strip().lower() in NORMAL_LABEL_WORDS else 1 for c in columns[cand]], dtype=int)
            label_name = cand
        if label_name:
            used.add(label_name)
            break

    # ---- controls & sensors -------------------------------------------------------
    if controls is not None:
        control_names = [c for c in controls]
        for c in control_names:
            if c not in numeric:
                raise DataError(f"Control column '{c}' is not numeric.")
    else:
        control_names = [n for n in numeric if n not in used and _is_control_name(n)
                         and not (sensors is not None and n in sensors)]
    used.update(control_names)

    if sensors is not None:
        sensor_names = list(sensors)
        for s in sensor_names:
            if s not in numeric:
                raise DataError(f"Sensor column '{s}' is not numeric.")
            if s in control_names:
                raise DataError(f"Column '{s}' is listed as both a sensor and a control.")
    else:
        sensor_names = [n for n in numeric if n not in used]
        # Legacy RCWE logs carry explicit momentum columns p_i next to q_i: velocities are derived
        # from the sensors themselves, so the p_* columns would only duplicate information.
        q_like = {n for n in sensor_names if n.startswith("q_")}
        p_like = [n for n in sensor_names if n.startswith("p_") and ("q_" + n[2:]) in q_like]
        if p_like and len(p_like) == len(q_like):
            sensor_names = [n for n in sensor_names if n not in p_like]
            ignored.extend(p_like)
            notes.append(f"Ignored {len(p_like)} explicit momentum column(s) (p_*); velocities are derived from the sensors.")
        # A leftover unnamed counter column (0, 1, 2, ...) is an index, not a measurement.
        for n in list(sensor_names):
            low = n.lower().strip()
            if (low.startswith(("unnamed", "column_")) or low in STEP_NAMES) and n in numeric:
                arr = numeric[n]
                if not np.isnan(arr).any() and arr.size > 2 and np.allclose(np.diff(arr), 1.0):
                    sensor_names.remove(n)
                    ignored.append(n)
                    if timestamps is None:
                        timestamps = [str(int(v)) for v in arr]
                        time_name = n
    for n in text_cols:
        if n not in used and n != time_name and n != label_name:
            ignored.append(n)
    if not sensor_names:
        raise DataError(
            "No numeric sensor columns were found. Check the delimiter and header row, or name the "
            "columns explicitly with sensors=[...]."
        )

    # ---- order rows by time when the log is shuffled ---------------------------------
    order: Optional[np.ndarray] = None
    if time_seconds is not None and np.any(np.diff(time_seconds) < 0):
        order = np.argsort(time_seconds, kind="stable")
        notes.append("Rows were not in chronological order; they have been sorted by time.")
    if time_seconds is not None:
        ts = time_seconds if order is None else time_seconds[order]
        dup = int(np.sum(np.diff(ts) == 0))
        if dup:
            notes.append(f"{dup} duplicate timestamp(s) found (kept as separate samples).")
        deltas = np.diff(ts)
        deltas = deltas[deltas > 0]
        if deltas.size > 4:
            lo, hi = np.percentile(deltas, [5, 95])
            if hi > 1.5 * lo:
                notes.append("Sampling interval is irregular; the model treats rows as evenly spaced steps.")

    def _col(arr: np.ndarray) -> np.ndarray:
        return arr if order is None else arr[order]

    raw = np.stack([_col(numeric[n]) for n in sensor_names], axis=1)
    missing = np.isnan(raw)
    keep = ~missing.all(axis=0)
    if not keep.all():
        dropped = [n for n, k in zip(sensor_names, keep) if not k]
        notes.append(f"Dropped column(s) with no numeric values: {', '.join(dropped)}.")
        ignored.extend(dropped)
        sensor_names = [n for n, k in zip(sensor_names, keep) if k]
        raw, missing = raw[:, keep], missing[:, keep]
        if not sensor_names:
            raise DataError("Every sensor column is empty.")
    total_missing = int(missing.sum())
    if total_missing:
        worst = sorted(((int(missing[:, i].sum()), n) for i, n in enumerate(sensor_names)), reverse=True)[:3]
        detail = ", ".join(f"{n} ({k})" for k, n in worst if k)
        notes.append(f"Filled {total_missing} missing value(s) by interpolation: {detail}.")
    values = np.stack([_fill_gaps(raw[:, i]) for i in range(raw.shape[1])], axis=1)

    constant = [n for i, n in enumerate(sensor_names) if np.ptp(values[:, i]) == 0.0]
    if constant:
        notes.append(f"Constant column(s) (no variation): {', '.join(constant)}.")

    if control_names:
        ctrl_raw = np.stack([_col(numeric[n]) for n in control_names], axis=1)
        ctrl = np.stack([np.nan_to_num(_fill_gaps(ctrl_raw[:, i]), nan=0.0) for i in range(ctrl_raw.shape[1])], axis=1)
    else:
        ctrl = np.zeros((n_rows, 0), dtype=float)

    if ignored:
        shown = ", ".join(ignored[:8]) + ("..." if len(ignored) > 8 else "")
        notes.append(f"Ignored column(s): {shown}.")

    return TelemetryTable(
        sensors=sensor_names,
        values=values,
        controls=list(control_names),
        control_values=ctrl,
        timestamps=None if timestamps is None else (timestamps if order is None else [timestamps[i] for i in order]),
        time_seconds=None if time_seconds is None else _col(time_seconds),
        time_column=time_name,
        labels=None if labels is None else _col(labels),
        label_column=label_name,
        missing=missing,
        notes=notes,
        ignored_columns=ignored,
        source=origin,
    )
