"""
`TelemetryModel` on data that is less tidy than the demo.

Each test pins a defect that an independent review reproduced: a single wild reading
blinding a sensor, a quiet stretch at the end of training flooding the live log with alarms,
gaps scored with knowledge of what followed them, constant sensors whose changes were missed.
"""
import json
import math

import numpy as np
import pytest

from rcwe.toolkit import DataError, TelemetryModel, TelemetryTable
from rcwe.toolkit.demo import make_pump_station
from rcwe.toolkit.results import _fmt, _fmt_pair
from rcwe.toolkit.scoring import detection_metrics, resolve_threshold


def _fit(table, **kwargs):
    return TelemetryModel(max_steps=80, seed=0, device="cpu").fit(table, **kwargs)


@pytest.fixture(scope="module")
def plant():
    """1,000 rows of healthy operation, the 500 rows after them with five faults, and the same 500 rows without."""
    demo = make_pump_station(rows=1500, faults=True, fault_start=1000, seed=3)
    normal, incident = demo.table.slice(0, 1000), demo.table.slice(1000, None)
    normal.labels, normal.label_column = None, None
    clean = make_pump_station(rows=1500, faults=False, seed=3).table.slice(1000, None)
    return {"normal": normal, "incident": incident, "clean": clean, "model": _fit(normal), "sensors": normal.sensors}


def _copy(table):
    return table.slice(0, None)


# ---- training data that is not clean -----------------------------------------------------
def test_one_wild_reading_in_the_training_log_does_not_blind_a_sensor(plant):
    flow = plant["sensors"].index("flow_m3h")
    spoiled = _copy(plant["normal"])
    spoiled.values[300, flow] = 1.0e9
    model = _fit(spoiled)
    result = model.detect(plant["incident"])
    assert result.metrics["incidents_detected"] == 5
    assert np.abs(result.residual_z[258:264, flow]).max() > 8                     # the +28 m3/h spike is as plain as ever
    notes = " ".join(model.fit_summary_.notes)
    assert "extreme reading" in notes and "Free-running" not in notes
    one_sigma = model._cal.sigma[flow] * model._vs[flow] * model._std[flow]
    reference = plant["model"]
    assert one_sigma < 2 * reference._cal.sigma[flow] * reference._vs[flow] * reference._std[flow]


def test_a_sensor_that_sat_still_at_the_end_of_training_does_not_flood_the_live_log(plant):
    vibration = plant["sensors"].index("vibration_mm_s")
    quiet = _copy(plant["normal"])
    quiet.values[800:, vibration] = quiet.values[800, vibration]                   # stuck for the whole held-out part
    assert _fit(quiet).detect(plant["clean"]).n_events == 0


def test_status_flags_may_switch_and_coarse_readings_may_step(plant):
    data = make_pump_station(rows=2600, faults=False, seed=5).table
    switching = ((np.arange(2600) // 150) % 2).astype(float)                       # a pump that is on, then off, ...
    values = np.column_stack([data.values, switching])

    def part(a, b, block):
        return TelemetryTable(sensors=list(block[0]), values=block[1][a:b], controls=list(data.controls),
                              control_values=data.control_values[a:b])

    flagged = (data.sensors + ["pump2_running"], values)
    model = _fit(part(0, 1700, flagged))
    assert "status or mode flags" in " ".join(model.fit_summary_.notes)
    assert not [e for e in model.detect(part(1700, 2600, flagged)).events if e.top_sensor == "pump2_running"]

    step = np.array([0.2, 2, 0.5, 0.5, 4, 4.0])                                    # e.g. pressure in half-bar steps
    coarse = (data.sensors, np.round(data.values * step) / step)
    assert _fit(part(705, 1700, coarse)).detect(part(1700, 2600, coarse)).n_events <= 1


def test_a_cycle_or_two_of_training_does_not_switch_checks_off(plant):
    model = _fit(plant["normal"].slice(280, None))                                 # 720 rows: two and a half daily cycles
    assert "Free-running" not in " ".join(model.fit_summary_.notes)
    drop = [e for e in model.detect(plant["incident"]).events if e.top_sensor == "pressure_bar"]
    assert drop and max(e.duration for e in drop) >= 40                            # the whole drop, not just its edges


# ---- gaps and broken readings ----------------------------------------------------------
def test_gaps_are_held_not_scored_and_never_filled_with_what_follows(plant):
    model, flow = plant["model"], plant["sensors"].index("flow_m3h")
    rows = plant["clean"].to_rows()

    gappy = [dict(row) for row in rows]
    gappy[199]["flow_m3h"] = None
    later = [dict(row) for row in gappy]
    later[200]["flow_m3h"] += 40.0
    first, second = model.detect(gappy), model.detect(later)
    assert first.expected[200, flow] == pytest.approx(second.expected[200, flow])   # row 200 is not known in advance
    assert first.sensor_scores[199, flow] == 0.0                                    # nothing to score in the gap
    assert any("not scored" in note for note in first.notes)

    for missing in (0, 12):
        stepped = [dict(row) for row in rows]
        for t in range(300, 500):
            stepped[t]["flow_m3h"] += 28.0
        for t in range(300 - missing, 300):
            stepped[t]["flow_m3h"] = None
        batch = model.detect(stepped).sensor_scores[:306, flow]
        monitor = model.stream()
        live = np.array([monitor.update(row).sensor_scores["flow_m3h"] for row in stepped[:306]])
        assert np.allclose(batch, live, atol=1e-3)                                  # one answer, replayed or live
        assert batch[300] > 4.5                                                     # and the step is still caught


def test_readings_that_are_not_numbers_do_not_disable_scoring(plant):
    model = plant["model"]
    rows = plant["clean"].to_rows()
    monitor = model.stream()
    steps = []
    for k, row in enumerate(rows[:120]):
        row = dict(row)
        if k in (40, 41):
            row["flow_m3h"] = float("inf")
        if k >= 80:
            row["pressure_bar"] += 5.0
        steps.append(monitor.update(row))
    assert all(math.isfinite(step.score) for step in steps)
    assert sum(step.is_anomaly for step in steps[80:]) >= 30
    json.dumps(steps[41].to_dict(), allow_nan=False)                                # still valid JSON

    edited = _copy(plant["incident"])
    edited.values[30, 0] = np.nan                                                   # written into a table by hand
    result = model.detect(edited)
    assert np.isfinite(result.scores).all() and result.metrics["incidents_detected"] == 5


def test_a_change_in_a_constant_sensor_is_flagged_in_any_units_and_in_both_modes():
    data = make_pump_station(rows=1000, faults=False, seed=5).table
    for before, after in ((3.0, 3.2), (3000.0, 3200.0), (0.003, 0.004)):
        mode = np.full(1000, before)
        mode[600:] = after
        table = TelemetryTable(sensors=data.sensors + ["mode"], values=np.column_stack([data.values, mode]),
                               controls=list(data.controls), control_values=data.control_values)
        model = _fit(table.slice(0, 240))                                           # too short for the usual range check
        for adapt in (False, True):
            events = [e for e in model.detect(table.slice(240, None), adapt=adapt).events if e.top_sensor == "mode"]
            assert events and events[0].start == 360, (before, after, adapt)


# ---- explanations ----------------------------------------------------------------------
def test_event_descriptions_give_the_extreme_reading_and_the_right_side(plant):
    model, incident = plant["model"], plant["incident"]
    result = model.detect(incident)
    overheating = [e for e in result.events if e.top_sensor == "bearing_temp_c"][0]
    temp = plant["sensors"].index("bearing_temp_c")
    assert overheating.sensors[0]["actual"] == pytest.approx(incident.values[overheating.start:overheating.end + 1, temp].max())

    midway = model.detect(incident.slice(190, None))                                # the log begins inside the pressure drop
    first = midway.events[0]
    assert first.start <= 1 and first.top_sensor == "pressure_bar"
    assert "went below the range seen in training" in first.description


def test_events_point_at_a_control_that_changed_as_they_began(plant):
    stepped = make_pump_station(rows=1500, faults=False, seed=3, operator_steps=3)
    log = stepped.table.slice(1000, None)
    moves = [m for m in stepped.operator_steps if 1000 < m["start"] < 1450]
    assert moves                                                                    # the generator did place steps here
    result = plant["model"].detect(log)                                             # trained without ever seeing one
    assert result.n_events and all(e.control_changes for e in result.events)
    assert "coincides with a change in act_" in result.events[0].description
    monitor = plant["model"].stream()
    alerts = [step for step in (monitor.update(row) for row in log.to_rows()) if step.alert and step.controls_changed]
    assert alerts and alerts[0].controls_changed[0].startswith("act_")


# ---- commands --------------------------------------------------------------------------
def test_declaring_commands_lets_a_forecast_respond_to_a_plan():
    data = make_pump_station(rows=3000, faults=False, seed=7, operator_steps=14).table
    history, last = data.slice(2800, None), data.to_rows()[-1]
    hold = np.tile([last["act_valve_open"], last["act_pump_speed"]], (24, 1))
    opened = hold.copy()
    opened[:, 0] += 0.10
    plant_response = last["flow_m3h"] * 0.10 / (0.4 + last["act_valve_open"])      # what the simulated station does

    def response(model):
        flow = model.sensors.index("flow_m3h")
        a, b = model.forecast(24, history, future_controls=hold), model.forecast(24, history, future_controls=opened)
        return float(b.values[-1, flow] - a.values[-1, flow]), b

    as_feedback = TelemetryModel(max_steps=250, seed=7, device="cpu").fit(data)
    as_commands = TelemetryModel(max_steps=250, seed=7, device="cpu").fit(data, commands=True)
    assert as_feedback.commands == [] and as_commands.commands == ["act_valve_open", "act_pump_speed"]
    blind, forecast = response(as_feedback)
    seen, _ = response(as_commands)
    assert abs(blind) < 0.2 * plant_response                       # told nothing about the next row, it cannot react
    assert 0.3 * plant_response < seen < 1.5 * plant_response       # told the plan, it moves the right way, by a fair amount
    assert any("commands" in note for note in forecast.notes)       # and the blind model says why it did not
    assert as_commands.fit_summary_.skill > as_feedback.fit_summary_.skill
    assert "Commands" in as_commands.summary()


def test_commands_are_chosen_by_name_or_named_explicitly():
    rows = [{"temp": 20 + math.sin(i / 9), "setpoint_temp": 20.0 + (i // 40) % 2, "act_heater": 0.5 + 0.1 * math.cos(i / 9)}
            for i in range(160)]
    assert TelemetryModel(max_steps=20, device="cpu").fit(rows).commands == ["setpoint_temp"]
    assert TelemetryModel(max_steps=20, device="cpu").fit(rows, commands="none").commands == []
    assert TelemetryModel(max_steps=20, device="cpu").fit(rows, commands=["act_heater"]).commands == ["act_heater"]
    with pytest.raises(DataError, match="commands must be control columns"):
        TelemetryModel(max_steps=20, device="cpu").fit(rows, commands=["temp"])


# ---- reading new data the way the training data was read -------------------------------------
def test_detect_and_forecast_use_the_columns_chosen_for_fit():
    n = 400
    data = {
        "clock": [f"2026-05-28T20:{(i // 60) % 60:02d}:{i % 60:02d}Z" for i in range(n)],
        "temp": [20 + math.sin(i / 15) for i in range(n)], "press": [5 + math.cos(i / 11) for i in range(n)],
        "valve": [0.5 + 0.1 * math.sin(i / 20) for i in range(n)], "bad": [1 if 300 <= i < 320 else 0 for i in range(n)],
        "spare": [float(i % 7) for i in range(n)],
    }
    model = TelemetryModel(max_steps=40, device="cpu").fit(data, sensors=["temp", "press"], controls=["valve"],
                                                           time_column="clock", label_column="bad")
    again = model.detect(data)                                         # no column options repeated here
    assert again.sensors == ["temp", "press"] and again.metrics is not None
    assert again.timestamps[0] == "2026-05-28T20:00:00Z"
    assert model.forecast(3, data).timestamps[0] == "2026-05-28T20:06:40Z"

    feeders = {"p_a": [1 + math.sin(i / 9) for i in range(200)], "q_a": [2 + math.cos(i / 7) for i in range(200)],
               "regime": [3 + math.sin(i / 5) for i in range(200)]}
    kept = TelemetryModel(max_steps=20, device="cpu").fit(feeders, sensors=["p_a", "q_a", "regime"])
    assert kept.detect(feeders).sensors == ["p_a", "q_a", "regime"]


def test_forecast_labels_continue_the_logs_own_time_axis():
    wave = [math.sin(i / 9) for i in range(400)]
    for axis, expected in (
        ([5000 + 10 * i for i in range(400)], ["9000", "9010"]),
        ([0.5 * i for i in range(400)], ["200", "200.5"]),
    ):
        model = TelemetryModel(max_steps=20, device="cpu").fit({"t": axis, "v": wave}, time_column="t")
        assert model.forecast(2).timestamps == expected
    local = [f"2026-06-01T06:{i // 60:02d}:{i % 60:02d}+02:00" for i in range(400)]
    model = TelemetryModel(max_steps=20, device="cpu").fit({"when": local, "v": wave})
    assert model.forecast(2).timestamps == ["2026-06-01T06:06:40+02:00", "2026-06-01T06:06:41+02:00"]
    assert TelemetryModel(max_steps=20, device="cpu").fit({"v": wave}).forecast(2).timestamps == ["step 400", "step 401"]


# ---- numbers and files -------------------------------------------------------------------
def test_tiny_and_huge_values_survive_exports_and_descriptions(plant, tmp_path):
    scaled = _copy(plant["normal"])
    scaled.values *= 1e-9
    live = _copy(plant["incident"])
    live.values *= 1e-9
    result = _fit(scaled).detect(live)
    assert np.allclose(result.scores, plant["model"].detect(plant["incident"]).scores, rtol=1e-6, atol=1e-6)   # units do not matter
    row = result.to_rows()[260]
    assert 0 < row["flow_m3h"] < 1e-6 and row["flow_m3h_expected"] != 0.0          # not rounded away to 0.0
    event = result.events[0].to_dict()["sensors"][0]
    assert event["actual"] != 0.0 and event["expected"] != 0.0

    assert [_fmt(v) for v in (1013.25, 1009.8, 45.56, 6.5, 0.00123)] == ["1013.2", "1009.8", "45.56", "6.5", "0.00123"]
    assert _fmt_pair(1e9 + 12.3, 1e9 + 7.9) == ["1000000012", "1000000008"]         # not "1e+09 vs 1e+09"
    assert _fmt_pair(3.0, 3.0000002) == ["3.00000000", "3.00000020"]


def test_export_columns_stay_distinct_whatever_the_sensors_are_called():
    n = 300
    data = {"event": [math.sin(i / 9) for i in range(n)], "flow": [math.cos(i / 7) for i in range(n)],
            "flow_score": [math.sin(i / 5) for i in range(n)]}
    model = TelemetryModel(max_steps=20, device="cpu").fit(data, sensors=["event", "flow", "flow_score"])
    columns = list(model.detect(data).to_rows()[0])
    assert len(columns) == len(set(columns)) and {"event", "flow", "flow_score", "rcwe_event"} <= set(columns)


def test_damaged_model_files_are_refused(plant, tmp_path):
    path = tmp_path / "plant.rcwe"
    plant["model"].save(path)
    assert TelemetryModel.load(path).sensors == plant["sensors"]

    (tmp_path / "empty.rcwe").write_bytes(b"")
    with pytest.raises(DataError, match="not a readable RCWE model file"):
        TelemetryModel.load(tmp_path / "empty.rcwe")

    damaged = bytearray(path.read_bytes())
    middle = len(damaged) // 2
    for k in range(middle, middle + 64):
        damaged[k] ^= 0xFF
    (tmp_path / "flipped.rcwe").write_bytes(bytes(damaged))
    with pytest.raises(DataError, match="damaged|not a readable"):
        TelemetryModel.load(tmp_path / "flipped.rcwe")

    named_by_numpy = TelemetryModel(max_steps=20, device="cpu").fit(
        {"a": [math.sin(i / 9) for i in range(120)], "b": [math.cos(i / 7) for i in range(120)]},
        sensors=list(np.array(["a", "b"])))
    named_by_numpy.save(tmp_path / "np.rcwe")
    assert TelemetryModel.load(tmp_path / "np.rcwe").sensors == ["a", "b"]


def test_small_input_mistakes_are_refused_clearly(plant):
    model = plant["model"]
    values = plant["clean"].values[0]
    for controls in ([0.55], [0.5, 0.7, 0.1]):
        with pytest.raises(DataError, match="Expected 2 control value"):
            model.stream().update(values, controls=controls)
    assert resolve_threshold("5") == 5.0
    for bad in (None, float("inf"), "extreme", 0):
        with pytest.raises(ValueError, match="sensitivity must be"):
            resolve_threshold(bad)


def test_detection_metrics_handle_ties_and_empty_cases():
    assert detection_metrics(np.array([1, 1, 0, 0]), np.ones(4), [])["auroc"] == 0.5
    assert detection_metrics(np.array([0, 0, 1, 1]), np.ones(4), [])["auroc"] == 0.5        # row order does not matter
    scores = np.zeros(600)
    scores[300:500] = 13.5                                                                 # a saturated stretch
    early, late = np.zeros(600, dtype=int), np.zeros(600, dtype=int)
    early[300:340], late[460:500] = 1, 1
    assert detection_metrics(early, scores, [])["auroc"] == pytest.approx(detection_metrics(late, scores, [])["auroc"])
    quiet = detection_metrics(np.zeros(50, dtype=int), np.zeros(50), [])
    assert quiet["point_precision"] is None and quiet["point_recall"] is None and quiet["point_f1"] is None


def test_a_counter_that_ticks_evenly_is_flagged_when_it_skips():
    rng = np.random.default_rng(0)
    rows = [{"sequence": 1000 + i + (7 if i >= 300 else 0), "temp": 20 + math.sin(i / 9) + rng.normal(0, 0.05)} for i in range(400)]
    model = TelemetryModel(max_steps=60, device="cpu").fit(rows[:240])
    assert "exactly the same amount" in " ".join(model.fit_summary_.notes)
    skipped = [e for e in model.detect(rows).events if e.top_sensor == "sequence"]
    assert len(skipped) == 1 and skipped[0].start == 300 and skipped[0].severity == "critical"
