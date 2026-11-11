"""
Result objects returned by `TelemetryModel`: plain data plus convenience exporters.

Every result can describe itself in plain English (`summary()`), turn itself into
JSON-friendly dictionaries (`to_dict()`), and write CSV / JSON / HTML files.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

__all__ = ["AnomalyEvent", "DetectionResult", "Forecast", "FitSummary"]


def _round(value: Any, digits: int = 7) -> Any:
    """
    JSON-friendly copy of `value` with floats rounded to `digits` significant digits
    (not decimal places: readings of 3e-9 must not all turn into 0.0) and non-finite
    numbers replaced by None.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [_round(v, digits) for v in value]
    if isinstance(value, np.ndarray):
        arr = np.asarray(value, dtype=float)
        out = arr.copy()
        ok = np.isfinite(arr) & (arr != 0.0)
        if ok.any():
            factor = 10.0 ** (digits - 1 - np.floor(np.log10(np.abs(arr[ok]))))
            out[ok] = np.round(arr[ok] * factor) / factor
        cleaned = np.where(np.isfinite(out), out, np.nan).astype(object)
        cleaned[~np.isfinite(out)] = None
        return cleaned.tolist()
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(f"{float(value):.{digits}g}")
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _fmt(value: float) -> str:
    """Compact number formatting for prose: two decimals in the everyday range, whole digits always kept."""
    value = float(value)
    if not math.isfinite(value):
        return "n/a"
    magnitude = abs(value)
    if magnitude >= 1e12 or 0 < magnitude < 0.01:
        return f"{value:.3g}"
    if magnitude >= 1e6:
        return f"{value:.0f}"
    text = f"{value:.1f}" if magnitude >= 1000 else f"{value:.2f}"
    return text.rstrip("0").rstrip(".")


def _fmt_pair(first: float, second: float) -> List[str]:
    """Two numbers written so that they can be told apart ("1000012.3 vs 1000007.9", not "1e+06 vs 1e+06")."""
    a, b = _fmt(first), _fmt(second)
    if a == b and first != second and math.isfinite(first) and math.isfinite(second):
        decimals = min(12, max(0, int(math.ceil(-math.log10(abs(first - second)))) + 1))
        a, b = f"{first:.{decimals}f}", f"{second:.{decimals}f}"
    return [a, b]


def _plural(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


# --------------------------------------------------------------------------------------
@dataclass
class FitSummary:
    """What happened during `TelemetryModel.fit` -- an honest model card."""

    rows: int
    train_rows: int
    validation_rows: int
    sensors: List[str]
    controls: List[str]
    steps_run: int
    best_step: int
    seconds: float
    device: str
    parameters: int
    skill: float                       # 1 - MSE(model) / MSE(persistence), one step ahead, validation
    sensor_skill: Dict[str, float] = field(default_factory=dict)
    forecast_skill: Dict[str, float] = field(default_factory=dict)   # {"1": .., "12": .., ...} by horizon
    forecast_horizon: int = 0
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    source: str = ""
    commands: List[str] = field(default_factory=list)   # controls known ahead of the reading they act on

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rows": self.rows, "train_rows": self.train_rows, "validation_rows": self.validation_rows,
            "sensors": list(self.sensors), "controls": list(self.controls),
            "steps_run": self.steps_run, "best_step": self.best_step, "seconds": round(self.seconds, 2),
            "device": self.device, "parameters": self.parameters,
            "skill_vs_persistence": _round(self.skill, 4),
            "sensor_skill": {k: _round(v, 4) for k, v in self.sensor_skill.items()},
            "forecast_skill_by_horizon": {k: _round(v, 4) for k, v in self.forecast_skill.items()},
            "forecast_horizon": self.forecast_horizon,
            "warnings": list(self.warnings), "notes": list(self.notes), "source": self.source,
            "commands": list(self.commands),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FitSummary":
        return cls(
            rows=int(data["rows"]), train_rows=int(data["train_rows"]), validation_rows=int(data["validation_rows"]),
            sensors=list(data["sensors"]), controls=list(data["controls"]),
            steps_run=int(data["steps_run"]), best_step=int(data["best_step"]), seconds=float(data["seconds"]),
            device=str(data["device"]), parameters=int(data["parameters"]),
            skill=float(data["skill_vs_persistence"]) if data.get("skill_vs_persistence") is not None else float("nan"),
            sensor_skill={k: (float("nan") if v is None else float(v)) for k, v in data.get("sensor_skill", {}).items()},
            forecast_skill={k: (float("nan") if v is None else float(v)) for k, v in data.get("forecast_skill_by_horizon", {}).items()},
            forecast_horizon=int(data.get("forecast_horizon", 0)),
            warnings=list(data.get("warnings", [])), notes=list(data.get("notes", [])), source=str(data.get("source", "")),
            commands=list(data.get("commands", [])),
        )

    def summary(self) -> str:
        held_out = self.validation_rows > 0
        lines = [
            f"Trained on {self.train_rows:,} rows ({_plural(len(self.sensors), 'sensor')}, "
            f"{_plural(len(self.controls), 'control')}) in {self.seconds:.1f}s on {self.device}; "
            + (f"validated on the last {self.validation_rows:,} rows." if held_out
               else "too short to hold rows back, so the figures below are measured on the training rows themselves."),
        ]
        # Without held-out rows the comparison flatters the model; say where it was measured.
        where = "on held-out data" if held_out else "on its own training rows (optimistic)"
        if np.isfinite(self.skill):
            direction = "better" if self.skill >= 0 else "worse"
            lines.append(
                f"One-step-ahead accuracy: {abs(self.skill) * 100:.0f}% {direction} than the naive "
                f"'next value = current value' baseline {where}."
            )
        if self.forecast_skill:
            parts = [f"{h} step{'' if h == '1' else 's'}: {v * 100:+.0f}%" for h, v in self.forecast_skill.items() if np.isfinite(v)]
            if parts:
                lines.append("Forecast skill vs naive baseline -- " + ", ".join(parts) + ".")
        if self.commands:
            lines.append("Commands (the model is told their value for the reading it predicts): " + ", ".join(self.commands) + ".")
        for warning in self.warnings:
            lines.append(f"Warning: {warning}")
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
@dataclass
class AnomalyEvent:
    """One detected incident: when it happened, how severe it was, and which sensors drove it."""

    index: int
    start: int
    end: int
    start_time: str
    end_time: str
    peak: int
    peak_time: str
    peak_score: float
    severity: str                      # low | medium | high | critical
    kind: str                          # spike | sustained deviation | out of range | flatline
    sensors: List[Dict[str, Any]] = field(default_factory=list)
    description: str = ""              # full sentence, including when it happened
    what: str = ""                     # the same finding without the time span (for tables)
    control_changes: List[Dict[str, Any]] = field(default_factory=list)   # control inputs that changed as it began

    @property
    def duration(self) -> int:
        return self.end - self.start + 1

    @property
    def top_sensor(self) -> Optional[str]:
        return self.sensors[0]["name"] if self.sensors else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index, "start": self.start, "end": self.end, "duration_steps": self.duration,
            "start_time": self.start_time, "end_time": self.end_time,
            "peak": self.peak, "peak_time": self.peak_time, "peak_score": _round(self.peak_score, 3),
            "severity": self.severity, "kind": self.kind,
            "sensors": [{k: _round(v, 5) for k, v in s.items()} for s in self.sensors],
            "description": self.description, "what": self.what,
            "control_changes": [{k: _round(v, 6) for k, v in c.items()} for c in self.control_changes],
        }


# --------------------------------------------------------------------------------------
@dataclass
class Forecast:
    """Multi-step forecast in the sensors' original units, with empirical prediction intervals."""

    sensors: List[str]
    values: np.ndarray                 # [steps, n]
    lower: np.ndarray                  # [steps, n] -- 80% interval
    upper: np.ndarray
    lower95: np.ndarray                # [steps, n] -- 95% interval
    upper95: np.ndarray
    timestamps: List[str]
    time_seconds: Optional[np.ndarray] = None
    origin: str = ""                   # label of the last observed row
    history: Optional[np.ndarray] = None            # last observed rows, for context     [k, n]
    history_timestamps: Optional[List[str]] = None
    sensor_skill: Dict[str, float] = field(default_factory=dict)    # backtested skill at the final horizon
    model_weight: Dict[str, float] = field(default_factory=dict)    # 1 = pure model, 0 = falls back to "last value"
    notes: List[str] = field(default_factory=list)

    @property
    def steps(self) -> int:
        return int(self.values.shape[0])

    def to_rows(self) -> List[Dict[str, Any]]:
        rows = []
        for h in range(self.steps):
            row: Dict[str, Any] = {"step_ahead": h + 1, "time": self.timestamps[h]}
            for i, name in enumerate(self.sensors):
                row[name] = _round(self.values[h, i])
                row[f"{name}_lower80"] = _round(self.lower[h, i])
                row[f"{name}_upper80"] = _round(self.upper[h, i])
            rows.append(row)
        return rows

    def to_csv(self, path: Union[str, Path]) -> str:
        rows = self.to_rows()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return str(out.resolve())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sensors": list(self.sensors), "steps": self.steps, "origin": self.origin,
            "timestamps": list(self.timestamps),
            "values": _round(self.values), "lower80": _round(self.lower), "upper80": _round(self.upper),
            "lower95": _round(self.lower95), "upper95": _round(self.upper95),
            "history": None if self.history is None else _round(self.history),
            "history_timestamps": self.history_timestamps,
            "sensor_skill": {k: _round(v, 4) for k, v in self.sensor_skill.items()},
            "model_weight": {k: _round(v, 3) for k, v in self.model_weight.items()},
            "notes": list(self.notes),
        }

    def to_json(self, path: Union[str, Path]) -> str:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return str(out.resolve())

    def summary(self) -> str:
        lines = [f"Forecast: {_plural(self.steps, 'step')} ahead from {self.origin} (through {self.timestamps[-1]})."]
        for i, name in enumerate(self.sensors):
            skill = self.sensor_skill.get(name)
            trust = ""
            if skill is not None and np.isfinite(skill):
                trust = f"  [backtested skill {skill * 100:+.0f}% vs last-value]" if self.model_weight.get(name, 1.0) > 0.05 \
                    else "  [no skill over last-value: holding the last reading]"
            lines.append(
                f"  {name}: {_fmt(self.values[-1, i])} "
                f"(80% interval {_fmt(self.lower[-1, i])} to {_fmt(self.upper[-1, i])}) at the final step{trust}"
            )
        for note in self.notes:
            lines.append(f"Note: {note}")
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
@dataclass
class DetectionResult:
    """Anomaly-detection output for one telemetry log."""

    sensors: List[str]
    timestamps: List[str]
    values: np.ndarray                 # [T, n] actual readings
    expected: np.ndarray               # [T, n] what the model expected one step earlier
    band: np.ndarray                   # [n]    half-width of the "normal" envelope around `expected`
    residual_z: np.ndarray             # [T, n] signed surprise in sigma units
    sensor_scores: np.ndarray          # [T, n] per-sensor anomaly score
    scores: np.ndarray                 # [T]    overall anomaly score (worst sensor)
    threshold: float
    events: List[AnomalyEvent]
    mode: str = "reference"            # reference (frozen model) | adaptive (online plasticity)
    time_seconds: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    metrics: Optional[Dict[str, Any]] = None
    notes: List[str] = field(default_factory=list)
    source: str = ""
    model_info: Dict[str, Any] = field(default_factory=dict)
    forecast: Optional[Forecast] = None
    adaptation: Optional[Dict[str, Any]] = None

    # ---- convenience ---------------------------------------------------------------
    @property
    def n_rows(self) -> int:
        return int(self.scores.shape[0])

    @property
    def n_events(self) -> int:
        return len(self.events)

    @property
    def is_anomaly(self) -> np.ndarray:
        mask = np.zeros(self.n_rows, dtype=bool)
        for event in self.events:
            mask[event.start:event.end + 1] = True
        return mask

    @property
    def anomalous_fraction(self) -> float:
        return float(self.is_anomaly.mean()) if self.n_rows else 0.0

    def sensor_ranking(self) -> List[Dict[str, Any]]:
        """Sensors ordered by how much they contributed to the detected events."""
        mask = self.is_anomaly
        ranking = []
        for i, name in enumerate(self.sensors):
            excess = np.maximum(self.sensor_scores[:, i] - self.threshold, 0.0)
            ranking.append({
                "name": name,
                "peak_score": float(self.sensor_scores[:, i].max()) if self.n_rows else 0.0,
                "anomalous_steps": int(np.sum((self.sensor_scores[:, i] > self.threshold) & mask)),
                "involvement": float(excess[mask].sum()) if mask.any() else 0.0,
            })
        ranking.sort(key=lambda r: (r["involvement"], r["peak_score"]), reverse=True)
        return ranking

    def summary(self) -> str:
        span = f"{self.timestamps[0]} to {self.timestamps[-1]}" if self.timestamps else "the log"
        head = f"Scanned {self.n_rows:,} rows x {_plural(len(self.sensors), 'sensor')} ({span})."
        if not self.events:
            body = [head, f"No anomalies detected (peak score {self.scores.max():.1f} vs threshold {self.threshold:.1f})."]
        else:
            body = [
                head,
                f"{_plural(self.n_events, 'anomaly event')} detected, covering "
                f"{self.anomalous_fraction * 100:.1f}% of the log (threshold {self.threshold:.1f} sigma).",
            ]
            for event in self.events[:12]:
                body.append(f"  #{event.index} [{event.severity.upper()}] {event.description}")
            if self.n_events > 12:
                body.append(f"  ... and {self.n_events - 12} more.")
        if self.metrics:
            m = self.metrics
            if m.get("labeled_incidents"):
                delay = m.get("median_detection_delay_steps")
                body.append(
                    f"Against the labels in the file: {m['incidents_detected']} of {m['labeled_incidents']} incidents "
                    f"caught, {_plural(m['false_alarm_events'], 'false alarm')}"
                    + (f", median delay {delay:.0f} step(s)" if delay is not None else "")
                    + (f", AUROC {m['auroc']:.3f}" if m.get("auroc") is not None else "") + "."
                )
        for note in self.notes:
            body.append(f"Note: {note}")
        return "\n".join(body)

    # ---- exporters -----------------------------------------------------------------
    def to_dict(self, include_series: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "source": self.source,
            "mode": self.mode,
            "rows": self.n_rows,
            "sensors": list(self.sensors),
            "start": self.timestamps[0] if self.timestamps else None,
            "end": self.timestamps[-1] if self.timestamps else None,
            "threshold": _round(self.threshold, 3),
            "peak_score": _round(float(self.scores.max()) if self.n_rows else 0.0, 3),
            "event_count": self.n_events,
            "anomalous_fraction": _round(self.anomalous_fraction, 5),
            "events": [e.to_dict() for e in self.events],
            "sensor_ranking": [{k: _round(v, 4) for k, v in r.items()} for r in self.sensor_ranking()],
            "metrics": None if self.metrics is None else {k: _round(v, 4) for k, v in self.metrics.items()},
            "notes": list(self.notes),
            "model": dict(self.model_info),
        }
        if self.forecast is not None:
            out["forecast"] = self.forecast.to_dict()
        if include_series:
            out["series"] = {
                "timestamps": list(self.timestamps),
                "score": _round(self.scores, 4),
                "is_anomaly": [int(v) for v in self.is_anomaly],
                "values": {n: _round(self.values[:, i]) for i, n in enumerate(self.sensors)},
                "expected": {n: _round(self.expected[:, i]) for i, n in enumerate(self.sensors)},
                "sensor_score": {n: _round(self.sensor_scores[:, i], 4) for i, n in enumerate(self.sensors)},
            }
        return out

    def to_json(self, path: Union[str, Path], include_series: bool = True) -> str:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(include_series=include_series), indent=2) + "\n", encoding="utf-8")
        return str(out.resolve())

    def to_rows(self) -> List[Dict[str, Any]]:
        mask = self.is_anomaly
        event_of = np.zeros(self.n_rows, dtype=int)
        for event in self.events:
            event_of[event.start:event.end + 1] = event.index
        # Column names stay unique whatever the sensors are called (a sensor named "event", or
        # one named "flow_score" next to "flow").
        taken = set(self.sensors)
        general = {}
        for key in ("time", "anomaly_score", "is_anomaly", "event", "top_sensor"):
            general[key] = key if key not in taken else f"rcwe_{key}"
            taken.add(general[key])
        derived = []
        for name in self.sensors:
            pair = []
            for suffix in ("expected", "score"):
                column = f"{name}_{suffix}"
                while column in taken:
                    column += "_"
                taken.add(column)
                pair.append(column)
            derived.append(pair)
        values, expected, per_sensor = _round(self.values), _round(self.expected), _round(self.sensor_scores, 4)
        overall = _round(self.scores, 4)
        rows = []
        for t in range(self.n_rows):
            row: Dict[str, Any] = {
                general["time"]: self.timestamps[t],
                general["anomaly_score"]: overall[t],
                general["is_anomaly"]: int(mask[t]),
                general["event"]: int(event_of[t]) if event_of[t] else "",
                general["top_sensor"]: self.sensors[int(np.argmax(self.sensor_scores[t]))] if mask[t] else "",
            }
            for i, name in enumerate(self.sensors):
                row[name] = values[t][i]
                row[derived[i][0]] = expected[t][i]
                row[derived[i][1]] = per_sensor[t][i]
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

    def to_html(self, path: Union[str, Path], title: Optional[str] = None) -> str:
        """Write a self-contained interactive HTML report (no internet connection needed to view it)."""
        from rcwe.toolkit.report import write_report
        return write_report(self, path, title=title)
