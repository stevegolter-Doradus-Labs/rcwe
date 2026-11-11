"""
Built-in demo telemetry.

`make_pump_station` simulates a small water pumping station: six coupled sensors driven by
two operator controls, a daily demand cycle and measurement noise. The "incident" part of
the log contains five labelled faults of different kinds (drift, level shift, spike, stuck
sensor, correlation break), so the quickstart can show detections next to ground truth.

The data is synthetic and exists only so that new users have something realistic to try
before pointing RCWE at their own files.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from rcwe.toolkit.table import TelemetryTable

__all__ = ["DemoDataset", "make_pump_station", "write_demo_files", "PUMP_SENSORS", "PUMP_CONTROLS"]

PUMP_SENSORS = ["flow_m3h", "pressure_bar", "motor_power_kw", "bearing_temp_c", "vibration_mm_s", "tank_level_m"]
PUMP_CONTROLS = ["act_valve_open", "act_pump_speed"]
_NOISE = {"flow_m3h": 0.8, "pressure_bar": 0.03, "motor_power_kw": 0.35, "bearing_temp_c": 0.08,
          "vibration_mm_s": 0.03, "tank_level_m": 0.004}


@dataclass
class DemoDataset:
    """A demo table plus the ground-truth faults that were injected into it."""

    table: TelemetryTable
    events: List[Dict[str, Any]] = field(default_factory=list)


def make_pump_station(
    rows: int = 4200,
    faults: bool = True,
    fault_start: Optional[int] = None,
    seed: int = 7,
    start: str = "2026-03-02T00:00:00Z",
    interval_minutes: float = 5.0,
) -> DemoDataset:
    """
    Simulate `rows` samples of pump-station telemetry.

    With `faults=True`, five labelled faults are injected after row `fault_start`
    (default: 70% of the way through), leaving the first part clean for training.
    """
    if rows < 200:
        raise ValueError("rows must be at least 200 for the pump-station demo.")
    rng = np.random.default_rng(seed)
    burn_in = 400                       # simulated and discarded, so the log starts mid-operation
    wanted, rows = rows, rows + burn_in
    t = np.arange(rows)
    day = 2.0 * np.pi * t * interval_minutes / (24.0 * 60.0)

    valve = 0.55 + 0.15 * np.sin(day - 0.6) + 0.05 * np.sin(2.0 * np.pi * t / 37.0)
    speed = 0.70 + 0.10 * np.sin(day + 0.3) + 0.03 * np.cumsum(rng.standard_normal(rows)) / np.sqrt(t + 1.0)

    wander = np.zeros(rows)
    shocks = rng.standard_normal(rows) * 0.03
    for i in range(1, rows):
        wander[i] = 0.97 * wander[i - 1] + shocks[i]
    demand = 1.0 + 0.35 * np.sin(day - 1.2) + wander

    flow = 120.0 * speed * (0.4 + valve) * (0.6 + 0.4 * demand) / 1.3
    pressure = 6.0 + 4.0 * speed ** 2 - 0.012 * flow
    power = 15.0 + 0.11 * flow * pressure / 6.0 + 20.0 * speed ** 3
    temp = np.zeros(rows)
    temp[0] = 45.0
    for i in range(1, rows):
        temp[i] = temp[i - 1] + 0.05 * (30.0 + 0.35 * power[i] - temp[i - 1])
    vibration = 1.2 + 2.2 * speed ** 2 + 0.004 * (temp - 45.0)
    # Tank level: a leaky integrator of inflow minus demand-driven outflow (time constant ~4 h),
    # scaled to swing by roughly +/- 1 m around a 3 m operating point.
    net = flow - demand * (flow.mean() / demand.mean())
    raw = np.zeros(rows)
    for i in range(1, rows):
        raw[i] = 0.98 * raw[i - 1] + net[i]
    level = 3.0 + 0.45 * (raw - raw.mean()) / max(float(raw.std()), 1e-9)

    signal = {
        "flow_m3h": flow, "pressure_bar": pressure, "motor_power_kw": power,
        "bearing_temp_c": temp, "vibration_mm_s": vibration, "tank_level_m": level,
    }
    noise = {name: rng.standard_normal(rows) * _NOISE[name] for name in PUMP_SENSORS}
    for name in list(signal):
        signal[name] = signal[name][burn_in:].copy()
        noise[name] = noise[name][burn_in:]
    valve, speed, rows = valve[burn_in:], speed[burn_in:], wanted
    labels = np.zeros(rows, dtype=int)
    events: List[Dict[str, Any]] = []

    if faults:
        base = int(rows * 0.70) if fault_start is None else int(fault_start)
        span = rows - base
        if span < 400:
            raise ValueError("Not enough rows after fault_start to place the demo faults (need >= 400).")

        def at(fraction: float) -> int:
            return base + int(span * fraction)

        def record(a: int, b: int, name: str, sensors: List[str], description: str) -> None:
            labels[a:b] = 1
            events.append({"name": name, "start": a, "end": b - 1, "sensors": sensors, "description": description})

        a, ramp, cool = at(0.10), 60, 40
        signal["bearing_temp_c"][a:a + ramp] += np.linspace(0.0, 9.0, ramp)
        signal["bearing_temp_c"][a + ramp:a + ramp + cool] += np.linspace(9.0, 0.0, cool)
        signal["vibration_mm_s"][a:a + ramp + cool] += 0.35
        record(a, a + ramp + cool, "bearing overheating", ["bearing_temp_c", "vibration_mm_s"],
               "Bearing temperature ramps up 9 degC with elevated vibration, then recovers.")

        a, width = at(0.34), 45
        signal["pressure_bar"][a:a + width] -= 0.9
        record(a, a + width, "pressure drop", ["pressure_bar"], "Discharge pressure steps down by 0.9 bar (leak).")

        a = at(0.52)
        signal["flow_m3h"][a:a + 2] += 28.0
        record(a, a + 2, "flow spike", ["flow_m3h"], "Two-sample flow transient of +28 m3/h.")

        a, width = at(0.68), 50
        signal["vibration_mm_s"][a:a + width] = signal["vibration_mm_s"][a]
        noise["vibration_mm_s"][a:a + width] = noise["vibration_mm_s"][a]
        record(a, a + width, "stuck sensor", ["vibration_mm_s"], "Vibration sensor freezes at its last reading.")

        a, width = at(0.86), 40
        signal["motor_power_kw"][a:a + width] *= 1.18
        record(a, a + width, "power anomaly", ["motor_power_kw"], "Motor draws 18% more power than flow and pressure explain.")

    values = np.stack([signal[name] + noise[name] for name in PUMP_SENSORS], axis=1)
    decimals = {"flow_m3h": 2, "pressure_bar": 3, "motor_power_kw": 2, "bearing_temp_c": 2, "vibration_mm_s": 3, "tank_level_m": 4}
    for i, name in enumerate(PUMP_SENSORS):
        values[:, i] = np.round(values[:, i], decimals[name])
    control_values = np.round(np.stack([valve, speed], axis=1), 4)

    t0 = datetime.fromisoformat(start.replace("Z", "+00:00"))
    if t0.tzinfo is None:
        t0 = t0.replace(tzinfo=timezone.utc)
    step = timedelta(minutes=interval_minutes)
    stamps = [(t0 + i * step).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(rows)]
    seconds = np.array([(t0 + i * step).timestamp() for i in range(rows)], dtype=float)

    table = TelemetryTable(
        sensors=list(PUMP_SENSORS),
        values=values,
        controls=list(PUMP_CONTROLS),
        control_values=control_values,
        timestamps=stamps,
        time_seconds=seconds,
        time_column="timestamp",
        labels=labels if faults else None,
        label_column="anomaly" if faults else None,
        source="demo:pump_station",
    )
    return DemoDataset(table=table, events=events)


def write_demo_files(out_dir: Union[str, Path] = "rcwe_demo", seed: int = 7) -> Dict[str, str]:
    """
    Write the two-file demo used by `rcwe quickstart`:

    - `pump_station_normal.csv`    -- about ten days of healthy operation (train on this)
    - `pump_station_incident.csv`  -- the following four days, with five labelled faults
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    demo = make_pump_station(rows=4200, faults=True, fault_start=3000, seed=seed)
    normal, incident = demo.table.slice(0, 3000), demo.table.slice(3000, None)
    normal.labels, normal.label_column = None, None
    paths = {
        "normal": normal.to_csv(out / "pump_station_normal.csv"),
        "incident": incident.to_csv(out / "pump_station_incident.csv"),
    }
    return paths


def split_demo(seed: int = 7) -> Tuple[TelemetryTable, TelemetryTable, List[Dict[str, Any]]]:
    """In-memory version of `write_demo_files`: (normal_table, incident_table, incident_events)."""
    demo = make_pump_station(rows=4200, faults=True, fault_start=3000, seed=seed)
    normal, incident = demo.table.slice(0, 3000), demo.table.slice(3000, None)
    normal.labels, normal.label_column = None, None
    events = [dict(e, start=e["start"] - 3000, end=e["end"] - 3000) for e in demo.events]
    return normal, incident, events
