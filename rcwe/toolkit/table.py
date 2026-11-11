"""
Robust tabular telemetry ingestion.

`load_table` turns "whatever the user has" -- a CSV/TSV/JSON/JSONL file, a list of row
dicts, a dict of columns, a NumPy array or a pandas DataFrame -- into a clean
`TelemetryTable`: numeric sensor and control matrices, an optional time axis, optional
ground-truth labels, and a list of plain-English data-quality notes describing every
decision that was made on the way (columns ignored, gaps filled, rows re-ordered...).

Decisions that cannot be made one cell at a time are made per column or per file: whether
"01/03/2026" is the first of March or the third of January, whether "1,002" is a thousand
and two or one point zero zero two, which character separates the fields.

Nothing here depends on pandas; DataFrames are supported by duck typing.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = ["TelemetryTable", "load_table", "DataError"]


class DataError(ValueError):
    """Raised when input data cannot be turned into a usable telemetry table."""


MISSING_TOKENS = {
    "", "na", "n/a", "nan", "null", "none", "nil", "-", "--", "?", "#n/a", "#na", "#value!",
    "#div/0!", "inf", "-inf", "+inf", "infinity", "-infinity", "missing", "bad", "error", "<na>", "nat",
}
TIME_NAMES = (
    "timestamp", "time", "datetime", "date_time", "date", "ts", "t", "utc", "epoch",
    "timestamp_utc", "time_utc", "created_at", "recorded_at", "measured_at",
)
# Words that mark a column as a clock when they appear inside a longer name ("timestamp_ms", "Time [s]").
TIME_TOKENS = {"timestamp", "time", "datetime", "date", "epoch", "unix", "utc"}
STEP_NAMES = ("timestamp_step", "step", "time_step", "timestep", "index", "idx", "sample", "tick", "row", "id", "#")
# Names that cannot be a measurement. ("t" may be a temperature and "id" a device number, so those are not here.)
UNMISTAKABLE_TIME_NAMES = {
    "timestamp", "time", "datetime", "date_time", "date", "epoch", "timestamp_utc", "time_utc", "created_at",
    "recorded_at", "measured_at", "timestamp_step", "step", "time_step", "timestep", "index", "idx", "tick", "row", "#",
}
LABEL_NAMES = (
    "label", "labels", "anomaly", "is_anomaly", "anomalous", "fault", "is_fault", "attack",
    "is_attack", "regime_label", "regime", "ground_truth", "y_true", "outlier", "is_outlier",
)
NORMAL_LABEL_WORDS = {
    "0", "0.0", "normal", "ok", "okay", "false", "no", "nominal", "benign", "none", "n", "healthy",
    "good", "pass", "passed", "clean", "inlier", "negative", "f",
}
UNLABELLED_WORDS = {"", "na", "n/a", "nan", "null", "nil", "-", "--", "?", "<na>", "nat", "#n/a", "unknown", "unlabeled", "unlabelled"}
CONTROL_PREFIXES = ("act_", "u_", "ctrl_", "control_", "cmd_", "setpoint_", "sp_")
CONTROL_SUFFIXES = ("_setpoint", "_cmd", "_sp", "_command")
DATE_COLUMN_NAMES = ("date", "datum", "day")
CLOCK_COLUMN_NAMES = ("time", "zeit", "uhrzeit", "clock", "hour")

_THOUSANDS_COMMA = re.compile(r"^[+-]?[1-9]\d{0,2}(,\d{3})+$")
_THOUSANDS_DOT = re.compile(r"^[+-]?[1-9]\d{0,2}(\.\d{3})+$")
_CLOCK = re.compile(r"^\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?(\s?[AaPp][Mm])?$")
_DELIMITERS = (",", ";", "\t", "|")
MAX_COLUMNS = 10000

# One entry per way of writing a calendar date. A column is read with exactly one of them,
# because "01/03/2026" cannot be told apart from its day/month twin one cell at a time.
_DATE_FAMILIES = (
    ("ymd-", "%Y-%m-%d"), ("ymd/", "%Y/%m/%d"), ("ymd.", "%Y.%m.%d"),
    ("mdy/", "%m/%d/%Y"), ("dmy/", "%d/%m/%Y"),
    ("mdy-", "%m-%d-%Y"), ("dmy-", "%d-%m-%Y"),
    ("dmy.", "%d.%m.%Y"),
    ("mdy/2", "%m/%d/%y"), ("dmy/2", "%d/%m/%y"), ("dmy.2", "%d.%m.%y"),
    ("dby-", "%d-%b-%Y"), ("dby ", "%d %b %Y"), ("bdy ", "%b %d %Y"), ("bdy,", "%b %d, %Y"),
    ("ymd", "%Y%m%d"),
)
_DAY_MONTH_TWINS = {"mdy/": "dmy/", "dmy/": "mdy/", "mdy-": "dmy-", "dmy-": "mdy-", "mdy/2": "dmy/2", "dmy/2": "mdy/2"}
_TIME_PARTS = ("", " %H:%M:%S", " %H:%M", " %H:%M:%S.%f", "T%H:%M:%S", "T%H:%M", "T%H:%M:%S.%f", " %I:%M:%S %p", " %I:%M %p")
_COMPACT_TIME_PARTS = ("%H%M%S", "T%H%M%S", "%H%M", "T%H%M")


# --------------------------------------------------------------------------------------
# numbers
# --------------------------------------------------------------------------------------
def _to_float(value: Any) -> Optional[float]:
    """
    float for numbers, NaN for missing markers, None for genuinely non-numeric text.

    This looks at one cell only, so a lone comma is read as a decimal mark; `_numeric_column`
    revisits cells such as "1,002" with the rest of their column in view.
    """
    if value is None:
        return math.nan
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, str):                       # before np.generic: numpy strings are both
        return _text_to_float(value)
    if isinstance(value, (int, float, np.generic)):
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return out if math.isfinite(out) else math.nan
    return _text_to_float(str(value))


def _text_to_float(raw: str) -> Optional[float]:
    text = raw.strip()
    if text.lower() in MISSING_TOKENS:
        return math.nan
    try:
        out = float(text)
        return out if math.isfinite(out) else math.nan
    except ValueError:
        pass
    cleaned = text.replace(" ", "").replace(" ", "").replace(" ", "")
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1]
    if "," in cleaned:
        if "." in cleaned:                           # both marks: whichever comes last is the decimal mark
            if cleaned.rfind(",") > cleaned.rfind("."):
                cleaned = cleaned.replace(".", "").replace(",", ".")
            else:
                cleaned = cleaned.replace(",", "")
        elif cleaned.count(",") == 1:
            cleaned = cleaned.replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif cleaned.count(".") > 1 and _THOUSANDS_DOT.match(cleaned):
        cleaned = cleaned.replace(".", "")           # 1.234.567
    try:
        out = float(cleaned)
    except ValueError:
        return None
    return out if math.isfinite(out) else math.nan


def _decimal_marks_in_use(columns: Dict[str, List[Any]]) -> Tuple[bool, bool]:
    """
    Does any cell of the file use a comma / a dot in a way that can only be a decimal mark?
    ("21,5" and "0,994" can; "1,002" cannot, it may be a thousand and two.)
    """
    comma = dot = False
    for cells in columns.values():
        for c in cells:
            if not isinstance(c, str) or (comma and dot):
                continue
            has_comma, has_dot = "," in c, "." in c
            if has_comma == has_dot:                 # neither, or both (then the cell settles itself)
                continue
            t = c.strip().rstrip("%").strip()
            if has_comma and not comma:
                if t.count(",") == 1 and not _THOUSANDS_COMMA.match(t) and _text_to_float(t) is not None:
                    comma = True
            elif has_dot and not dot:
                if t.count(".") == 1 and not _THOUSANDS_DOT.match(t):
                    try:
                        float(t)
                        dot = True
                    except ValueError:
                        pass
        if comma and dot:
            break
    return comma, dot


def _numeric_column(cells: Sequence[Any], comma_delimited: Optional[bool], decimal_comma: bool,
                    decimal_dot: bool) -> Tuple[List[Optional[float]], Optional[str]]:
    """
    Parse one column. Returns the cells as floats (NaN = missing, None = not a number) and a
    note whenever a mark such as the comma in "1,002" had to be read as a thousands separator
    or could not be told apart from one.
    """
    parsed = [_to_float(c) for c in cells]
    note: Optional[str] = None
    for sep, pattern, is_decimal_here, other_is_decimal in (
        (",", _THOUSANDS_COMMA, decimal_comma, decimal_dot),
        (".", _THOUSANDS_DOT, decimal_dot, decimal_comma),
    ):
        other = "." if sep == "," else ","
        candidates: List[int] = []
        for i, c in enumerate(cells):
            if isinstance(c, str) and sep in c and other not in c:
                t = c.strip().rstrip("%").strip()
                if len(t) - t.rfind(sep) == 4 and pattern.match(t):
                    candidates.append(i)
        if not candidates or is_decimal_here:
            continue                                 # the file uses this mark for decimals: nothing to revisit
        texts = {i: str(cells[i]).strip().rstrip("%").strip() for i in candidates}
        as_thousands = {i: float(texts[i].replace(sep, "")) for i in candidates}
        as_decimal = {i: (float(texts[i].replace(",", ".")) if texts[i].count(sep) == 1 else as_thousands[i]) for i in candidates}
        chosen: Optional[bool] = None
        doubtful = False
        if any(texts[i].count(sep) > 1 for i in candidates):
            chosen = True                            # "1,234,567" leaves no doubt
        else:
            picked = set(candidates)
            others = [abs(p) for i, p in enumerate(parsed) if i not in picked and p is not None and not math.isnan(p) and p != 0.0]
            if others:
                # Which reading makes the column hang together? 994, 998, "1,002" is 1002, not 1.002.
                spread_t = max(others + [abs(v) for v in as_thousands.values()]) / min(others + [abs(v) for v in as_thousands.values()])
                spread_d = max(others + [abs(v) for v in as_decimal.values()]) / min(others + [abs(v) for v in as_decimal.values()])
                if spread_t * 10.0 <= spread_d:
                    chosen = True
                elif spread_d * 10.0 <= spread_t:
                    chosen = False
            if chosen is None:
                if other_is_decimal:
                    chosen = True                    # the other mark is this file's decimal mark
                elif sep == ",":
                    chosen, doubtful = bool(comma_delimited), True      # quoted "1,002" in a comma-separated file
                else:
                    chosen = False                   # a dot is a decimal point unless shown otherwise
        example = candidates[0]
        if chosen:
            for i in candidates:
                parsed[i] = as_thousands[i]
            note = f"'{texts[example]}' was read as {as_thousands[example]:g} ('{sep}' taken as a thousands separator)"
        elif doubtful:
            note = (f"'{texts[example]}' could be {as_decimal[example]:g} or {as_thousands[example]:g}; "
                    f"it was read as {as_decimal[example]:g}")
    return parsed, note


# --------------------------------------------------------------------------------------
# dates and times
# --------------------------------------------------------------------------------------
def _iso_seconds(text: str) -> Optional[Tuple[float, Optional[float]]]:
    """(epoch seconds, UTC offset in seconds) for an ISO-8601 stamp; the offset is None for naive stamps."""
    s = text[:-1] + "+00:00" if text[-1:] in ("Z", "z") else text
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    try:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc).timestamp(), None
        return dt.timestamp(), dt.utcoffset().total_seconds()       # type: ignore[union-attr]
    except (OverflowError, OSError, ValueError):
        return None


class _DateFamily:
    """One way of writing dates, with any of the usual ways of writing the time of day after it."""

    __slots__ = ("key", "formats", "hint")

    def __init__(self, key: str, date_format: str):
        self.key = key
        parts = _TIME_PARTS + (_COMPACT_TIME_PARTS if key == "ymd" else ())
        self.formats = tuple(date_format + part for part in parts)
        self.hint: Optional[str] = None

    def parse(self, text: str) -> Optional[float]:
        if self.hint is not None:
            try:
                return datetime.strptime(text, self.hint).replace(tzinfo=timezone.utc).timestamp()
            except (ValueError, OverflowError, OSError):
                pass
        for fmt in self.formats:
            if fmt == self.hint:
                continue
            try:
                value = datetime.strptime(text, fmt).replace(tzinfo=timezone.utc).timestamp()
            except (ValueError, OverflowError, OSError):
                continue
            self.hint = fmt
            return value
        return None


def _parse_datetime(text: str, fmt_hint: Optional[str] = None) -> Tuple[Optional[float], Optional[str]]:
    """Epoch seconds for a single stamp (first reading that fits). Naive stamps are taken as UTC."""
    s = str(text).strip()
    if not s:
        return None, fmt_hint
    iso = _iso_seconds(s)
    if iso is not None:
        return iso[0], "iso"
    for key, date_format in _DATE_FAMILIES:
        value = _DateFamily(key, date_format).parse(s)
        if value is not None:
            return value, key
    return None, fmt_hint


def _parse_time_column(cells: Sequence[Any]) -> Optional[Dict[str, Any]]:
    """
    Read a column of date/time text with ONE format for the whole column.

    Every format that fits a sample of the cells is tried on all of them; the winner is the one
    that reads the most cells, then the one that leaves the rows in order, then the one with
    the most even spacing. That is what tells 11/03, 12/03, 13/03 (day first) from month first.
    Returns None when the column does not hold dates.
    """
    texts = ["" if c is None else str(c).strip() for c in cells]
    present = [t for t in texts if t and any(ch.isdigit() for ch in t)]
    if not present:
        return None
    sample = present[::max(1, len(present) // 48)][:64]
    allowed = len(sample) - max(1, math.ceil(0.9 * len(sample)))        # sample cells a format may fail on

    def fits(parse: Any) -> bool:
        misses = 0
        for t in sample:
            if parse(t) is None:
                misses += 1
                if misses > allowed:
                    return False
        return True

    families = {key: _DateFamily(key, date_format) for key, date_format in _DATE_FAMILIES}
    keys: List[str] = ["iso"] if fits(_iso_seconds) else []
    keys += [key for key, family in families.items() if fits(family.parse)]
    if not keys:
        return None

    def read_all(key: str) -> Tuple[np.ndarray, set]:
        secs = np.full(len(texts), np.nan)
        offsets: set = set()
        for i, t in enumerate(texts):
            if not t:
                continue
            if key == "iso":
                got = _iso_seconds(t)
                if got is not None:
                    secs[i] = got[0]
                    offsets.add(got[1])
            else:
                value = families[key].parse(t)
                if value is not None:
                    secs[i] = value
        return secs, offsets

    scored = []
    for order, key in enumerate(keys):
        secs, offsets = read_all(key)
        valid = secs[~np.isnan(secs)]
        steps = np.diff(valid) if valid.size > 1 else np.zeros(0)
        scored.append((int(np.isnan(secs).sum()), int((steps < 0).sum()), float(steps.max()) if steps.size else 0.0,
                       order, key, secs, offsets))
    scored.sort(key=lambda item: item[:4])
    failures, _, _, _, key, secs, offsets = scored[0]

    if failures:                                     # a few stragglers written another way
        for i, t in enumerate(texts):
            if t and math.isnan(secs[i]):
                got = _iso_seconds(t)
                if got is not None:
                    secs[i] = got[0]
                    offsets.add(got[1])
                    continue
                for other, family in families.items():
                    if other in (key, _DAY_MONTH_TWINS.get(key)):
                        continue
                    value = family.parse(t)
                    if value is not None:
                        secs[i] = value
                        break
        failures = int(np.isnan(secs).sum())
    if failures > 0.1 * len(texts):
        return None

    note = None
    twin = _DAY_MONTH_TWINS.get(key)
    if twin is not None:
        order_words = "day/month/year" if key.startswith("d") else "month/day/year"
        rival = next((item for item in scored if item[4] == twin), None)
        if rival is not None and rival[:3] == scored[0][:3]:
            note = f"dates read as {order_words}; they would read the other way round just as well, so use ISO dates (YYYY-MM-DD) to remove the doubt"
        else:
            note = f"dates read as {order_words}"
    aware = {o for o in offsets if o is not None}
    return {
        "seconds": secs, "failures": failures, "note": note,
        "naive": None in offsets or not offsets,
        "offset": next(iter(aware)) if len(aware) == 1 and None not in offsets else None,
    }


def _numeric_clock(arr: np.ndarray) -> Optional[np.ndarray]:
    """
    Epoch seconds for a numeric column that holds clock time: compact stamps such as
    20260301105800, or Unix time in seconds, milli-, micro- or nanoseconds. None otherwise.
    """
    finite = arr[~np.isnan(arr)]
    if finite.size == 0:
        return None
    low, high = float(finite.min()), float(finite.max())
    if low > 0 and np.all(finite == np.round(finite)):
        for digits, fmt in ((14, "%Y%m%d%H%M%S"), (12, "%Y%m%d%H%M"), (8, "%Y%m%d")):
            if low >= 10 ** (digits - 1) and high < 10 ** digits:
                secs = np.full(arr.shape, np.nan)
                try:
                    for i, v in enumerate(arr):
                        if not math.isnan(v):
                            dt = datetime.strptime(str(int(v)), fmt)
                            if not 1900 <= dt.year <= 2200:
                                raise ValueError
                            secs[i] = dt.replace(tzinfo=timezone.utc).timestamp()
                except (ValueError, OverflowError, OSError):
                    break
                return secs
    for bound, scale in ((1e17, 1e9), (1e14, 1e6), (1e11, 1e3), (1e8, 1.0)):
        if low > bound:
            return arr / scale if high / scale < 1e11 else None
    return None


def _clock_text(cell: Any) -> Optional[str]:
    """Time of day ("23:58", "23:58:00", or zero-padded "2358" / "235800") as HH:MM[:SS], else None."""
    if not isinstance(cell, str):
        return None
    t = cell.strip()
    if _CLOCK.match(t):
        return t
    if t.isdigit() and len(t) == 4:
        return f"{t[:2]}:{t[2:]}"
    if t.isdigit() and len(t) == 6:
        return f"{t[:2]}:{t[2:4]}:{t[4:]}"
    return None


def _stamp_text(value: Any) -> str:
    """How a timestamp cell is shown: the user's own text, or compact ISO-8601 for datetime objects."""
    if hasattr(value, "isoformat"):
        text = str(value.isoformat())
        return text[:-6] + "Z" if text.endswith("+00:00") else text
    return str(value).strip()


def format_time(seconds: float, offset: Optional[float] = None) -> str:
    """
    Compact ISO-8601 string for an epoch-seconds value: UTC ("...Z") by default, or local
    time with its offset ("...+02:00") when `offset` (seconds east of UTC) is given.
    """
    try:
        dt = datetime.fromtimestamp(float(seconds), tz=timezone.utc)
    except (OverflowError, OSError, ValueError) as exc:
        raise DataError(f"{seconds!r} is not a usable timestamp.") from exc
    suffix = "Z"
    if offset:
        dt = dt + timedelta(seconds=offset)
        sign = "+" if offset > 0 else "-"
        minutes = int(round(abs(offset) / 60.0))
        suffix = f"{sign}{minutes // 60:02d}:{minutes % 60:02d}"
    if dt.microsecond % 1000:
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%f") + suffix
    if dt.microsecond:
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + suffix
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + suffix


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
    """Usable, unique column names: blanks become column_<i>, repeats get _2, _3, ..."""
    cleaned = []
    for i, raw in enumerate(names):
        name = str(raw).strip().lstrip("\ufeff").strip() if raw is not None else ""
        cleaned.append(name or f"column_{i}")
    taken = set()
    given = set(cleaned)
    out: List[str] = []
    for name in cleaned:
        unique, n = name, 1
        while unique in taken or (n > 1 and unique in given):
            n += 1
            unique = f"{name}_{n}"
        taken.add(unique)
        out.append(unique)
    return out


def _rows_to_columns(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Any]]:
    names: Dict[Any, None] = {}                       # insertion-ordered, with constant-time lookup
    for row in rows:
        if not isinstance(row, dict):
            raise DataError("Expected a list of row objects (one {column: value} mapping per time step).")
        for key in row:
            if key not in names:
                names[key] = None
    return {str(name): [row.get(name) for row in rows] for name in names}


def _lines(text: str) -> List[str]:
    """Lines of a text file. Only CR and LF end a line: `str.splitlines` would also break on
    U+2028 and other separators that may legitimately sit inside a field."""
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def _width(cells: Sequence[str]) -> int:
    """Number of fields, not counting one empty field left by a trailing delimiter."""
    return len(cells) - (1 if cells and not cells[-1].strip() else 0)


def _looks_like_data(cells: Sequence[str]) -> bool:
    """True when every cell of a line is a number, a timestamp or empty: a data row, not a header."""
    seen = False
    for c in cells:
        t = c.strip()
        if not t:
            continue
        seen = True
        if _to_float(t) is None and _parse_datetime(t)[0] is None:
            return False
    return seen


def _sniff_delimiter(lines: Sequence[str]) -> Tuple[Optional[str], bool]:
    """
    Field separator of a delimited file, or None for a single column. The second value is
    True when the data rows carry one field more than the header (an unnamed leading index).

    A separator qualifies when the header and at least 80% of the first rows agree on the
    number of fields; among those, the one that yields the most fields wins. Decimal commas
    in a semicolon file therefore never make the comma look like the separator.
    """
    head = list(lines[:60])
    best: Optional[Tuple[int, float, str, bool]] = None
    fallback: Optional[Tuple[float, str]] = None
    for cand in _DELIMITERS:
        if not any(cand in ln for ln in head):
            continue
        rows = list(csv.reader(head, delimiter=cand))
        width = _width(rows[0])
        body = rows[1:] or rows
        same = sum(1 for r in body if _width(r) == width) / len(body)
        shifted = sum(1 for r in body if _width(r) == width + 1) / len(body)
        if fallback is None or same > fallback[0]:
            fallback = (same, cand)
        if width >= 2 and same >= 0.8:
            entry = (width, same, cand, False)
        elif width >= 2 and shifted >= 0.8:
            entry = (width, shifted, cand, True)
        else:
            continue
        if best is None or entry[:2] > best[:2]:
            best = entry
    if best is not None:
        return best[2], best[3]
    if not any(cand in head[0] for cand in _DELIMITERS):
        return None, False                           # one column; commas in the rows are decimal marks
    try:
        return csv.Sniffer().sniff("\n".join(head), delimiters="".join(_DELIMITERS)).delimiter, False
    except csv.Error:
        return (fallback[1] if fallback else ","), False


def _read_delimited(text: str, notes: List[str], hints: Dict[str, Any]) -> Dict[str, List[Any]]:
    text = text.lstrip("\ufeff")
    lines = [ln for ln in _lines(text) if ln.strip()]
    # Comment lines ("# exported by ...") are a common habit in logger exports, before and after the header.
    lead: List[str] = []
    while lines and lines[0].lstrip().startswith("#"):
        lead.append(lines.pop(0))
    kept = [ln for ln in lines if not ln.lstrip().startswith("#")]
    skipped = len(lead) + len(lines) - len(kept)
    lines = kept
    if not lines:
        raise DataError("The file has no data rows (need a header line plus at least one row).")

    delimiter, indexed = _sniff_delimiter(lines)
    hints["delimiter"] = delimiter
    rows = [[ln] for ln in lines] if delimiter is None else list(csv.reader(lines, delimiter=delimiter))

    header: Optional[List[str]] = None
    if _looks_like_data(rows[0]):
        # No header row. A commented header just above the data ("# time,temp,pressure") still counts.
        if lead:
            titles = lead[-1].lstrip().lstrip("#").strip()
            cand = [titles] if delimiter is None else next(csv.reader([titles], delimiter=delimiter))
            if _width(cand) == _width(rows[0]) and not _looks_like_data(cand):
                header, skipped = list(cand), skipped - 1
        if header is None:
            header = [""] * _width(rows[0])
            notes.append("The file has no header row; columns were named column_0, column_1, ...")
    else:
        header = rows.pop(0)
        if indexed:
            header = [""] + list(header)             # the rows' unnamed first field is an index
    if len(rows) < 1:
        raise DataError("The file has no data rows (need a header line plus at least one row).")
    if header and not header[-1].strip() and len(header) > 1 and all(len(r) < len(header) or not r[len(header) - 1].strip() for r in rows):
        header = header[:-1]                         # a trailing delimiter, not a column
    names = _dedupe(header)

    columns: Dict[str, List[Any]] = {name: [] for name in names}
    longer = shorter = 0
    for cells in rows:
        if len(cells) < len(names):
            shorter += 1
        elif any(c.strip() for c in cells[len(names):]):
            longer += 1
        for i, name in enumerate(names):
            columns[name].append(cells[i] if i < len(cells) else None)
    if shorter:
        notes.append(f"{shorter} row(s) had fewer fields than the header; the missing cells were treated as gaps.")
    if longer:
        notes.append(f"{longer} row(s) had more fields than the header; the extra cells were left out.")
    if skipped:
        notes.append(f"Skipped {skipped} comment line(s) starting with '#'.")
    if delimiter not in (",", None):
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


def _json_lines(text: str) -> Optional[List[Any]]:
    """Rows of a JSON-lines document (one object per line), or None when it is not one."""
    rows = []
    for line in _lines(text):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            return None
        if not isinstance(row, dict):
            return None
        rows.append(row)
    return rows or None


def _read_text(text: str, kind: str, name: str, notes: List[str], hints: Dict[str, Any]) -> Dict[str, List[Any]]:
    """Columns from the text of a file: JSON, JSON lines or delimited, going by content as well as by name."""
    if not text.strip():
        raise DataError(f"{name} is empty.")
    if kind in (".jsonl", ".ndjson"):
        rows = []
        for n, line in enumerate(_lines(text), 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise DataError(f"{name}, line {n}: invalid JSON ({exc.msg}).") from exc
        return _rows_to_columns(rows)
    if kind == ".json" or text.lstrip()[:1] in ("{", "["):
        try:
            return _read_json_payload(json.loads(text))
        except json.JSONDecodeError as exc:
            rows = _json_lines(text)                 # one object per line, whatever the file is called
            if rows is not None:
                return _rows_to_columns(rows)
            if kind == ".json" or text.lstrip()[:1] == "{":
                raise DataError(f"{name} is not valid JSON ({exc.msg} at line {exc.lineno}).") from exc
    return _read_delimited(text, notes, hints)


def _decode(data: bytes, notes: List[str]) -> str:
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16", errors="replace")
    head = data[:4000]
    if head.count(b"\x00") > len(head) // 4:         # UTF-16 without a byte-order mark
        return data.decode("utf-16-le" if head[1:2] == b"\x00" else "utf-16-be", errors="replace")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        notes.append("File is not UTF-8; decoded it as Latin-1.")
        return data.decode("latin-1")


def _read_path(path: Path, notes: List[str], hints: Dict[str, Any]) -> Dict[str, List[Any]]:
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

    try:
        data = path.read_bytes()
    except OSError as exc:
        raise DataError(f"Could not read {path}: {exc.strerror or exc}.") from exc
    if compressed:
        try:
            data = gzip.decompress(data)
        except (OSError, EOFError):
            notes.append(f"{path.name} is named .gz but is not compressed; read it as plain text.")
    return _read_text(_decode(data, notes), kind, path.name, notes, hints)


def _frame_to_columns(frame: Any) -> Dict[str, List[Any]]:
    index = getattr(frame, "index", None)
    if index is not None and type(index).__name__ != "RangeIndex" and hasattr(frame, "reset_index"):
        # A meaningful index (timestamps, a named counter) is a column like any other.
        named = getattr(index, "name", None) is not None or type(index).__name__ == "MultiIndex"
        frame = frame.reset_index()
        if not named:
            first = list(frame.columns)[0]
            frame = frame.rename(columns={first: "timestamp" if type(index).__name__ == "DatetimeIndex" else "index"})
    columns: Dict[str, List[Any]] = {}
    names = _dedupe(list(frame.columns))
    for name, original in zip(names, list(frame.columns)):
        series = frame[original]
        values = series.tolist() if hasattr(series, "tolist") else list(series)
        columns[name] = [_stamp_text(v) if hasattr(v, "isoformat") else v for v in values]
    return columns


def _to_columns(source: Any, notes: List[str], hints: Dict[str, Any]) -> Tuple[Dict[str, List[Any]], str]:
    if isinstance(source, (str, Path)):
        text = str(source)
        if isinstance(source, str) and ("\n" in text or "\r" in text):
            return _read_text(text, "", "The text", notes, hints), "<text>"
        path = Path(text).expanduser()
        return _read_path(path, notes, hints), str(path)
    if isinstance(source, io.IOBase) or hasattr(source, "read"):
        text = source.read()
        if isinstance(text, bytes):
            text = _decode(text, notes)
        label = str(getattr(source, "name", "<stream>"))
        return _read_text(text, "", label, notes, hints), label
    if hasattr(source, "columns") and hasattr(source, "__getitem__") and not isinstance(source, dict):
        return _frame_to_columns(source), "<DataFrame>"
    if hasattr(source, "detach") and hasattr(source, "cpu"):  # torch.Tensor
        source = source.detach().cpu().numpy()
    if isinstance(source, np.ndarray):
        try:
            arr = np.asarray(source, dtype=float)
        except (TypeError, ValueError) as exc:
            raise DataError("The array holds values that are not numbers.") from exc
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
    time_seconds: Optional[np.ndarray] = None       # epoch seconds, when the log has a real clock
    time_column: Optional[str] = None
    labels: Optional[np.ndarray] = None
    label_column: Optional[str] = None
    missing: Optional[np.ndarray] = None            # True where a sensor value was absent and has been filled in
    notes: List[str] = field(default_factory=list)
    ignored_columns: List[str] = field(default_factory=list)
    source: str = "<memory>"
    step_values: Optional[np.ndarray] = None        # numeric time axis that is not a clock (a counter, elapsed time)
    time_offset: Optional[float] = None             # seconds east of UTC, when every stamp carried the same offset

    def __post_init__(self) -> None:
        try:
            self.values = np.asarray(self.values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise DataError("values must be numeric.") from exc
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
        # A table is clean by construction: readings that are not finite numbers are gaps.
        bad = ~np.isfinite(self.values)
        if bad.any():
            self.missing = np.asarray(self.missing, dtype=bool) | bad
            holes = np.where(bad, np.nan, self.values)
            self.values = np.stack([_fill_gaps(holes[:, i]) for i in range(holes.shape[1])], axis=1)
        bad = ~np.isfinite(self.control_values)
        if bad.any():
            holes = np.where(bad, np.nan, self.control_values)
            self.control_values = np.stack([_fill_gaps(holes[:, i]) for i in range(holes.shape[1])], axis=1)

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

    def labels_after(self, steps: int, row_offset: int = 0) -> List[str]:
        """
        Time labels for the `steps` rows that would follow the last one, in the log's own style.
        `row_offset` is the number of rows that came before this table (when it is the tail of a
        longer log without a time axis, so that rows are numbered from the log's start).
        """
        if self.time_seconds is not None and self.sample_interval:
            last, interval = float(self.time_seconds[-1]), self.sample_interval
            return [format_time(last + interval * k, self.time_offset) for k in range(1, steps + 1)]
        if self.step_values is not None and self.n_rows:
            deltas = np.diff(self.step_values)
            deltas = deltas[deltas > 0]
            pace = float(np.median(deltas)) if deltas.size else 1.0
            last = float(self.step_values[-1])
            return [_number_label(last + pace * k) for k in range(1, steps + 1)]
        return [f"step {row_offset + self.n_rows + k}" for k in range(steps)]

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
            step_values=None if self.step_values is None else self.step_values[sl].copy(),
            time_offset=self.time_offset,
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
        """
        Write the table as CSV. A CSV file cannot say which column plays which part, so when
        it is read back the roles are worked out from the column names again; pass
        `sensors=`, `controls=`, `time_column=` and `label_column=` to `load_table` to keep
        choices that the names alone would not give.
        """
        rows = self.to_rows()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
        return str(out.resolve())


def _number_label(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _missing_column(col: Any, role: str, names: Sequence[str]) -> DataError:
    close = [n for n in names if n.lower() == str(col).lower()]
    hint = f" Did you mean '{close[0]}'?" if close else f" Available columns: {', '.join(names[:12])}{'...' if len(names) > 12 else ''}"
    return DataError(f"Column '{col}' (given in {role}) is not in the data.{hint}")


def _reselect(table: TelemetryTable, sensors: Optional[Sequence[str]], controls: Optional[Sequence[str]],
              time_column: Optional[str], label_column: Optional[str], ignore: Iterable[str]) -> TelemetryTable:
    """Apply column choices to a table that is already loaded."""
    if time_column not in (None, table.time_column) or label_column not in (None, table.label_column):
        raise DataError("This table already has its time axis and labels; load the original data again to choose other columns for them.")
    names = list(table.sensors) + list(table.controls)
    dropped = {str(c) for c in ignore}
    for role, wanted in (("sensors", sensors), ("controls", controls), ("ignore", sorted(dropped))):
        for col in wanted or []:
            if col not in names:
                raise _missing_column(col, role, names)
    new_controls = [c for c in (table.controls if controls is None else controls) if c not in dropped]
    if sensors is None:
        new_sensors = [s for s in names if s not in new_controls and s not in dropped and (controls is not None or s in table.sensors)]
    else:
        new_sensors = [s for s in sensors if s not in dropped]
        new_controls = [c for c in new_controls if c not in new_sensors] if controls is None else new_controls
    if set(new_sensors) & set(new_controls):
        raise DataError(f"Column '{sorted(set(new_sensors) & set(new_controls))[0]}' is listed as both a sensor and a control.")
    if not new_sensors:
        raise DataError("No sensor columns are left.")
    missing = np.stack([table.missing[:, table.sensors.index(s)] if s in table.sensors else np.zeros(table.n_rows, dtype=bool)
                        for s in new_sensors], axis=1)
    left_out = [n for n in names if n not in new_sensors and n not in new_controls]
    return TelemetryTable(
        sensors=new_sensors, values=np.stack([table.column(s) for s in new_sensors], axis=1),
        controls=new_controls,
        control_values=np.stack([table.column(c) for c in new_controls], axis=1) if new_controls else None,
        timestamps=None if table.timestamps is None else list(table.timestamps),
        time_seconds=None if table.time_seconds is None else table.time_seconds.copy(),
        time_column=table.time_column, labels=None if table.labels is None else table.labels.copy(),
        label_column=table.label_column, missing=missing, notes=list(table.notes),
        ignored_columns=list(table.ignored_columns) + left_out, source=table.source,
        step_values=None if table.step_values is None else table.step_values.copy(), time_offset=table.time_offset,
    )


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
    ignore = [str(c) for c in (ignore or [])]
    if isinstance(source, TelemetryTable):
        if sensors is None and controls is None and not ignore and time_column in (None, source.time_column) \
                and label_column in (None, source.label_column):
            return source
        return _reselect(source, sensors, controls, time_column, label_column, ignore)
    notes: List[str] = []
    hints: Dict[str, Any] = {}
    columns, origin = _to_columns(source, notes, hints)
    return _build(columns, origin, notes, hints, sensors=sensors, controls=controls, time_column=time_column,
                  label_column=label_column, ignore=ignore)


def load_like(source: Any, *, sensors: Sequence[str], controls: Sequence[str] = (), time_column: Optional[str] = None,
              label_column: Optional[str] = None, min_rows: int = 2) -> TelemetryTable:
    """
    Read `source` the way an earlier log was read: the named sensor, control, time and label
    columns are used wherever the data has them, so choices made for that log carry over to
    this one, and whatever is absent is inferred as usual.
    """
    if isinstance(source, TelemetryTable):
        return source
    notes: List[str] = []
    hints: Dict[str, Any] = {}
    columns, origin = _to_columns(source, notes, hints)
    have_sensors = bool(sensors) and all(s in columns for s in sensors)
    return _build(
        columns, origin, notes, hints,
        sensors=list(sensors) if have_sensors else None,
        controls=[c for c in controls if c in columns] if have_sensors else None,
        time_column=time_column if time_column in columns else None,
        label_column=label_column if label_column in columns else None,
        ignore=[], min_rows=min_rows,
    )


def _read_labels(name: str, numbers: Optional[np.ndarray], cells: Sequence[Any], explicit: bool,
                 notes: List[str]) -> Optional[np.ndarray]:
    """0/1 ground truth from a label column, or None when the column does not look like labels."""
    if numbers is not None:
        finite = numbers[~np.isnan(numbers)]
        if not explicit and not (finite.size and np.unique(finite).size <= 12 and np.all(finite == np.round(finite))):
            return None
        if finite.size and set(np.unique(finite).tolist()) == {-1.0, 1.0}:
            notes.append(f"Label column '{name}' uses -1/1; -1 was taken as anomalous.")
            return (np.nan_to_num(numbers, nan=1.0) == -1.0).astype(int)
        return (np.nan_to_num(numbers, nan=0.0) != 0).astype(int)
    words = [str(c if c is not None else "").strip().lower() for c in cells]
    labelled = [w for w in words if w not in UNLABELLED_WORDS]
    normal = set(NORMAL_LABEL_WORDS)
    if labelled and not any(w in normal for w in labelled):
        counts = Counter(labelled)
        if len(counts) > 1:
            usual = counts.most_common(1)[0][0]
            normal = {usual}
            notes.append(f"Label column '{name}': '{usual}', its most frequent value, was taken as normal.")
    return np.array([0 if (w in UNLABELLED_WORDS or w in normal) else 1 for w in words], dtype=int)


def _build(
    columns: Dict[str, List[Any]],
    origin: str,
    notes: List[str],
    hints: Dict[str, Any],
    *,
    sensors: Optional[Sequence[str]],
    controls: Optional[Sequence[str]],
    time_column: Optional[str],
    label_column: Optional[str],
    ignore: Sequence[str],
    min_rows: int = 2,
) -> TelemetryTable:
    if not columns:
        raise DataError("No columns found in the data.")
    if len(columns) > MAX_COLUMNS:
        raise DataError(f"The data has {len(columns):,} columns; RCWE reads up to {MAX_COLUMNS:,}. "
                        "Check the delimiter, or keep only the columns you need.")
    lengths = {len(v) for v in columns.values()}
    if len(lengths) != 1:
        raise DataError("Columns have different lengths; every column needs one value per time step.")
    n_rows = lengths.pop()
    if n_rows < min_rows:
        raise DataError(f"Only {n_rows} row(s) of data found; at least {min_rows} time steps are needed.")

    names = list(columns.keys())
    for role, wanted in (("sensors", sensors), ("controls", controls), ("time_column", [time_column] if time_column else None),
                         ("label_column", [label_column] if label_column else None), ("ignore", ignore)):
        for col in wanted or []:
            if col not in columns:
                raise _missing_column(col, role, names)

    # ---- classify every column as numeric or text ---------------------------------
    decimal_comma, decimal_dot = _decimal_marks_in_use(columns)
    delimiter = hints.get("delimiter", "unknown")
    comma_delimited = None if delimiter == "unknown" else delimiter == ","
    numeric: Dict[str, np.ndarray] = {}
    text_cols: List[str] = []
    for name in names:
        parsed, doubt = _numeric_column(columns[name], comma_delimited, decimal_comma, decimal_dot)
        bad = sum(1 for p in parsed if p is None)
        present = sum(1 for p in parsed if p is not None and not math.isnan(p))
        if present > 0 and bad <= 0.1 * (present + bad):
            numeric[name] = np.array([math.nan if p is None else p for p in parsed], dtype=float)
            if bad:
                notes.append(f"'{name}': {bad} non-numeric cell(s) treated as missing.")
            if doubt:
                notes.append(f"'{name}': {doubt}.")
        else:
            text_cols.append(name)

    ignored: List[str] = sorted(set(ignore))
    # Columns that are spoken for: left out on request, or given a part by name.
    used: set = set(ignore) | set(sensors or []) | set(controls or [])

    # ---- time axis ----------------------------------------------------------------
    time_name: Optional[str] = None
    timestamps: Optional[List[str]] = None
    time_seconds: Optional[np.ndarray] = None
    step_values: Optional[np.ndarray] = None
    time_offset: Optional[float] = None
    naive_clock = False
    sort_by_steps = False

    def clock_from_text(col: str, cells: Sequence[Any]) -> bool:
        nonlocal timestamps, time_seconds, time_offset, naive_clock
        read = _parse_time_column(cells)
        if read is None:
            return False
        secs, failures = read["seconds"], read["failures"]
        filled = _fill_gaps(secs)
        time_offset, naive_clock = read["offset"], bool(read["naive"])
        # Rows keep the user's own timestamp text; a row whose timestamp could not be read is
        # labelled with the time interpolated for it rather than left blank.
        timestamps = [format_time(f, time_offset) if math.isnan(s) else _stamp_text(c) for c, s, f in zip(cells, secs, filled)]
        time_seconds = filled
        if failures:
            notes.append(f"'{col}': {failures} unparseable timestamp(s) were interpolated.")
        if read["note"]:
            notes.append(f"'{col}': {read['note']}.")
        return True

    def clock_from_numbers(col: str) -> bool:
        nonlocal timestamps, time_seconds
        arr = numeric[col]
        secs = _numeric_clock(arr)
        gaps = int(np.isnan(arr).sum())
        if secs is None or gaps > 0.1 * arr.size:
            return False
        time_seconds = _fill_gaps(secs)
        timestamps = [format_time(v) for v in time_seconds]
        if gaps:
            notes.append(f"'{col}': {gaps} missing timestamp(s) were interpolated.")
        return True

    def counter_from_numbers(col: str, explicit: bool) -> bool:
        nonlocal timestamps, step_values, sort_by_steps
        arr = numeric[col]
        if np.isnan(arr).any():
            if not explicit or np.isnan(arr).sum() > 0.1 * arr.size:
                return False
            arr = _fill_gaps(arr)
        rising = bool(np.all(np.diff(arr) >= 0))
        if not explicit and (not rising or np.ptp(arr) == 0.0):
            return False
        step_values, sort_by_steps = arr, not rising
        timestamps = [_number_label(v) for v in arr]
        return True

    if time_column:
        if time_column in text_cols:
            done = clock_from_text(time_column, columns[time_column])
        else:
            done = clock_from_numbers(time_column) or counter_from_numbers(time_column, explicit=True)
        if not done:
            raise DataError(f"Could not read '{time_column}' as a time column (values are neither dates nor numbers).")
        time_name = time_column
        used.add(time_column)
    else:
        lowered = {n: n.lower().strip() for n in names if n not in used}
        # A date column next to a time-of-day column is one time axis written in two parts.
        date_col = next((n for n, low in lowered.items() if low in DATE_COLUMN_NAMES), None)
        hour_col = next((n for n, low in lowered.items() if low in CLOCK_COLUMN_NAMES), None)
        if date_col is not None and hour_col is not None:
            clocks = [_clock_text(c) for c in columns[hour_col]]
            if sum(1 for c in clocks if c is not None) >= 0.9 * n_rows:
                joined = [f"{str(d).strip()} {c}" if d is not None and c is not None else None
                          for d, c in zip(columns[date_col], clocks)]
                if clock_from_text(date_col, joined):
                    time_name = date_col
                    used.update((date_col, hour_col))
                    ignored.append(hour_col)
                    notes.append(f"Joined '{date_col}' and '{hour_col}' into one time axis.")
        if time_name is None:
            exact = [n for n, low in lowered.items() if low in TIME_NAMES]
            partial = [n for n, low in lowered.items() if n not in exact and low not in STEP_NAMES
                       and set(re.findall(r"[a-z]+", low)) & TIME_TOKENS]
            counters = [n for n, low in lowered.items() if low in STEP_NAMES and n not in exact]
            named = set(exact) | set(partial) | set(counters)
            textual = set(text_cols)
            other_text = [n for n in text_cols if n in lowered and n not in named]
            for cand in exact + partial + counters + other_text:
                done = False
                if cand in textual:
                    done = clock_from_text(cand, columns[cand])
                elif cand in numeric:
                    done = cand not in counters and clock_from_numbers(cand)
                    if not done and (cand in exact or cand in counters):
                        done = counter_from_numbers(cand, explicit=False)
                        if not done and lowered[cand] in UNMISTAKABLE_TIME_NAMES:
                            # Named like a clock or a counter, so never a measurement.
                            used.add(cand)
                            ignored.append(cand)
                            notes.append(f"'{cand}' is named like a time or step column but does not keep increasing; "
                                         "it was left out and the rows were kept in file order.")
                if done:
                    time_name = cand
                    used.add(cand)
                    break

    # ---- labels -------------------------------------------------------------------
    labels: Optional[np.ndarray] = None
    label_name: Optional[str] = None
    label_like = [n for n in names if n.lower().strip() in LABEL_NAMES and n not in used]
    for cand in ([label_column] if label_column else label_like):
        if cand in numeric or cand in text_cols:
            labels = _read_labels(cand, numeric.get(cand), columns[cand], bool(label_column), notes)
        if labels is not None:
            label_name = cand
            used.add(cand)
            break
    spare = [n for n in label_like if n not in used and n in numeric
             and _read_labels(n, numeric[n], columns[n], False, []) is not None]
    if spare:
        # Ground truth must never reach the model as a measurement.
        used.update(spare)
        ignored.extend(spare)
        notes.append(f"Label-like column(s) left out: {', '.join(spare)}"
                     + (f" ('{label_name}' is used as the labels)." if label_name else "."))

    # ---- controls & sensors -------------------------------------------------------
    if controls is not None:
        control_names = list(controls)
        for c in control_names:
            if c not in numeric:
                raise DataError(f"Control column '{c}' is not numeric.")
    else:
        control_names = [n for n in numeric if n not in used and _is_control_name(n)]
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
        # Logs written by earlier RCWE versions carry explicit momentum columns p_i next to q_i.
        # Velocities are derived from the sensors themselves, so those would only repeat them.
        # (Only for that layout: p_1/q_1 in someone else's log are active and reactive power.)
        if "timestamp_step" in names:
            q_like = {n for n in sensor_names if re.fullmatch(r"q_\d+", n)}
            p_like = [n for n in sensor_names if re.fullmatch(r"p_\d+", n) and ("q_" + n[2:]) in q_like]
            if p_like and len(p_like) == len(q_like):
                sensor_names = [n for n in sensor_names if n not in p_like]
                ignored.extend(p_like)
                notes.append(f"Ignored {len(p_like)} explicit momentum column(s) (p_*); velocities are derived from the sensors.")
        # A leftover unnamed counter column (0, 1, 2, ...) is an index, not a measurement.
        for n in list(sensor_names):
            low = n.lower().strip()
            if low.startswith(("unnamed", "column_")) or low in STEP_NAMES:
                arr = numeric[n]
                if not np.isnan(arr).any() and arr.size > 2 and arr[0] in (0.0, 1.0) and np.array_equal(np.diff(arr), np.ones(arr.size - 1)):
                    sensor_names.remove(n)
                    ignored.append(n)
                    if timestamps is None:
                        timestamps = [str(int(v)) for v in arr]
                        step_values = arr
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
    if time_seconds is not None:
        steps = np.diff(time_seconds)
        backward = np.flatnonzero(steps < 0)
        if backward.size == 1 and naive_clock and steps.size > 3:
            # Local clock time repeats one hour when daylight saving ends. Sorting would shuffle
            # that hour's rows together; moving the later rows on by an hour keeps them in order.
            pace = float(np.median(steps[steps > 0])) if np.any(steps > 0) else 0.0
            j = int(backward[0])
            if pace > 0 and abs(steps[j] + 3600.0 - pace) <= 0.02 * pace:
                time_seconds = time_seconds.copy()
                time_seconds[j + 1:] += 3600.0
                backward = np.zeros(0, dtype=int)
                notes.append("The clock steps back by an hour once (daylight saving ending?); rows were kept in file order.")
        if backward.size:
            order = np.argsort(time_seconds, kind="stable")
            notes.append("Rows were not in chronological order; they have been sorted by time.")
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
    elif step_values is not None and sort_by_steps:
        order = np.argsort(step_values, kind="stable")
        notes.append(f"Rows were not in the order of '{time_name}'; they have been sorted by it.")

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
        step_values=None if step_values is None else _col(step_values),
        time_offset=time_offset,
    )
