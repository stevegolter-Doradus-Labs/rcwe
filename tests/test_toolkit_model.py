"""
End-user API: TelemetryModel fit / detect / forecast / save / load / stream.

One model is trained once per module on the built-in pump-station demo (a clean period)
and then asked to find five labelled faults in the period that follows.
"""
import math

import numpy as np
import pytest
import torch

from rcwe import TelemetryModel
from rcwe.toolkit import DataError, NotFittedError, load_table
from rcwe.toolkit.demo import make_pump_station
from rcwe.toolkit.scoring import detection_metrics, find_events, resolve_threshold


@pytest.fixture(scope="module")
def demo():
    data = make_pump_station(rows=2600, faults=True, fault_start=1700, seed=7)
    normal, incident = data.table.slice(0, 1700), data.table.slice(1700, None)
    normal.labels, normal.label_column = None, None
    truth = [dict(e, start=e["start"] - 1700, end=e["end"] - 1700) for e in data.events]
    return normal, incident, truth


@pytest.fixture(scope="module")
def model(demo):
    normal, _, _ = demo
    return TelemetryModel(max_steps=200, seed=0, device="cpu").fit(normal)


def _matching(result, fault):
    return [e for e in result.events if not (e.end < fault["start"] - 3 or e.start > fault["end"] + 3)]


# --------------------------------------------------------------------------------------
def test_fit_reports_honest_skill_against_the_naive_baseline(model):
    card = model.fit_summary_
    assert card.train_rows + card.validation_rows == card.rows == 1700
    assert card.validation_rows > 0                               # calibration uses held-out rows
    assert 0.05 < card.skill < 0.9                                # better than "next = current", not magic
    assert set(card.sensor_skill) == set(model.sensors)
    assert card.forecast_skill and all(v > 0 for v in card.forecast_skill.values())
    assert "naive" in model.summary()
    assert model.engine.training is False
    assert float(model.engine.metabolic_synapse.fast_weight.abs().sum()) == 0.0   # no runtime adaptation left behind


def test_detect_finds_every_labelled_fault_and_blames_the_right_sensor(model, demo):
    _, incident, truth = demo
    result = model.detect(incident)

    assert result.metrics["labeled_incidents"] == 5
    assert result.metrics["incidents_detected"] == 5
    assert result.metrics["false_alarm_events"] <= 1
    assert result.metrics["auroc"] > 0.95
    for fault in truth:
        events = _matching(result, fault)
        assert events, f"missed: {fault['name']}"
        assert events[0].top_sensor in fault["sensors"], f"{fault['name']}: blamed {events[0].top_sensor}"

    kinds = {fault["name"]: _matching(result, fault)[0].kind for fault in truth}
    assert kinds["flow spike"] == "spike"
    assert kinds["stuck sensor"] == "flatline"
    assert kinds["pressure drop"] == "sustained deviation"
    drop = _matching(result, next(f for f in truth if f["name"] == "pressure drop"))[0]
    assert drop.sensors[0]["direction"] == "below" and drop.sensors[0]["actual"] < drop.sensors[0]["expected"]
    assert "pressure_bar ran below expectations" in drop.description
    assert "5 of 5 incidents caught" in result.summary()


def test_clean_data_raises_no_alarms(model):
    clean = make_pump_station(rows=2600, faults=False, seed=7).table.slice(1700, None)
    result = model.detect(clean)
    assert result.n_events == 0
    assert "No anomalies detected" in result.summary()
    assert result.metrics is None and result.anomalous_fraction == 0.0


def test_result_exports_are_consistent(model, demo, tmp_path):
    _, incident, _ = demo
    result = model.detect(incident)
    payload = result.to_dict()
    assert payload["event_count"] == result.n_events and len(payload["series"]["score"]) == incident.n_rows
    assert payload["sensor_ranking"][0]["involvement"] >= payload["sensor_ranking"][-1]["involvement"]

    csv_table = load_table(result.to_csv(tmp_path / "scores.csv"), sensors=["anomaly_score", "is_anomaly"])
    assert csv_table.n_rows == incident.n_rows
    assert int(csv_table.column("is_anomaly").sum()) == int(result.is_anomaly.sum())
    assert (tmp_path / "scores.csv").read_text().splitlines()[0].startswith("time,anomaly_score,is_anomaly,event,top_sensor")
    assert '"events"' in open(result.to_json(tmp_path / "result.json")).read()

    # one-step expectations are in the sensors' own units and track the readings
    relative = np.abs(result.expected - result.values) / (np.abs(result.values) + 1e-9)
    assert float(np.median(relative)) < 0.05


def test_sensitivity_levels_are_ordered(model, demo):
    _, incident, _ = demo
    low, high = model.detect(incident, sensitivity="low"), model.detect(incident, sensitivity="high")
    assert low.threshold > model.detect(incident).threshold > high.threshold
    assert high.is_anomaly.sum() >= low.is_anomaly.sum()
    assert model.detect(incident, sensitivity=5.0).threshold == 5.0
    with pytest.raises(ValueError):
        resolve_threshold("extreme")


def test_adaptive_mode_flags_onsets_then_accepts_the_new_normal(model, demo):
    _, incident, truth = demo
    reference = model.detect(incident)
    before = {k: v.clone() for k, v in model.engine.state_dict().items()}
    adaptive = model.detect(incident, adapt=True)

    assert adaptive.mode == "adaptive" and reference.mode == "reference"
    drop = next(f for f in truth if f["name"] == "pressure drop")
    span = slice(drop["start"] + 12, drop["end"] - 4)                 # well inside the 45-step fault
    assert reference.is_anomaly[span].mean() > 0.9                    # frozen model: flagged throughout
    assert adaptive.is_anomaly[span].mean() < 0.5                     # plastic model: re-learned the level
    assert adaptive.is_anomaly[drop["start"]:drop["start"] + 3].any() # ...but it did flag the onset
    after = model.engine.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)      # the fitted model itself is untouched


def test_forecast_has_calibrated_intervals_and_real_timestamps(model, demo):
    normal, _, _ = demo
    forecast = model.forecast(steps=36)
    assert forecast.values.shape == (36, 6) and forecast.steps == 36
    assert forecast.origin == normal.timestamps[-1]
    assert forecast.timestamps[0] == "2026-03-07T21:40:00Z"           # five minutes after the last training row
    assert np.all(forecast.lower95 <= forecast.lower) and np.all(forecast.lower <= forecast.values)
    assert np.all(forecast.values <= forecast.upper) and np.all(forecast.upper <= forecast.upper95)
    widths = forecast.upper - forecast.lower
    assert np.all(widths[-1] >= widths[0] - 1e-9)                     # uncertainty grows with the horizon
    assert set(forecast.model_weight) == set(model.sensors)
    assert forecast.to_rows()[0]["step_ahead"] == 1 and "flow_m3h_lower80" in forecast.to_rows()[0]

    # backtest on unseen data: most of the truth should fall inside the 95% interval
    future = make_pump_station(rows=2600, faults=False, seed=7).table.slice(1700, 1736)
    inside = (future.values >= forecast.lower95) & (future.values <= forecast.upper95)
    assert inside.mean() > 0.8

    longer = model.forecast(steps=150, future_controls=np.tile([[0.6, 0.7]], (150, 1)))
    assert longer.values.shape == (150, 6)
    assert any("not backtested" in note for note in longer.notes)
    with pytest.raises(ValueError):
        model.forecast(steps=0)


def test_save_and_load_reproduce_results_exactly(model, demo, tmp_path):
    _, incident, _ = demo
    path = model.save(tmp_path / "pump.rcwe")
    clone = TelemetryModel.load(path, device="cpu")

    assert clone.sensors == model.sensors and clone.controls == model.controls
    assert clone.fit_summary_.skill == pytest.approx(model.fit_summary_.skill, abs=1e-4)
    assert np.array_equal(clone.detect(incident).scores, model.detect(incident).scores)
    assert np.allclose(clone.forecast(24).values, model.forecast(24).values)
    with pytest.raises(DataError, match="training log"):
        clone.detect()                                              # a loaded model has no training log to rescan

    engine_ckpt = tmp_path / "engine.pt"
    model.engine.save_pretrained(engine_ckpt)
    with pytest.raises(DataError, match="load_pretrained"):
        TelemetryModel.load(engine_ckpt)
    with pytest.raises(FileNotFoundError):
        TelemetryModel.load(tmp_path / "missing.rcwe")


def test_streaming_matches_batch_scoring(model, demo):
    _, incident, _ = demo
    batch = model.detect(incident)
    monitor = model.stream()
    steps = [monitor.update(row) for row in incident.to_rows()[:260]]

    assert np.allclose([s.score for s in steps], batch.scores[:260], atol=1e-3)
    first_alarm = next(s for s in steps if s.is_anomaly)
    assert first_alarm.index == batch.events[0].start
    assert first_alarm.sensors[0]["name"] == model.sensors[int(np.argmax(batch.sensor_scores[first_alarm.index]))]
    assert first_alarm.time == incident.timestamps[first_alarm.index]
    assert set(first_alarm.to_dict()) >= {"score", "is_anomaly", "sensors", "expected", "values"}

    positional = model.stream()
    assert positional.update(incident.values[0], controls=incident.control_values[0]).score == 0.0
    with pytest.raises(DataError, match="Expected 6 sensor values"):
        positional.update([1.0, 2.0])


def test_unfitted_and_mismatched_usage_fail_clearly(model):
    with pytest.raises(NotFittedError):
        TelemetryModel().detect({"a": [1, 2, 3]})
    with pytest.raises(DataError, match="at least 12 time steps"):
        TelemetryModel().fit({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(DataError, match="Sensor 'flow_m3h'"):
        model.detect({"something_else": [1.0, 2.0, 3.0, 4.0]})


# --------------------------------------------------------------------------------------
def test_awkward_columns_are_handled_not_fatal():
    """Constants, running totals, gaps and a wild glitch should each be dealt with on their own terms."""
    data = make_pump_station(rows=2600, faults=False, seed=3).table
    train_rows, live_rows = data.slice(0, 1700).to_rows(), data.slice(1700, None).to_rows()
    for i, row in enumerate(train_rows):
        row["firmware_rev"] = 3.0
        row["energy_total_kwh"] = 1000.0 + 0.7 * i + 0.01 * math.sin(i)
        if i % 97 == 0:
            row["pressure_bar"] = None
    for i, row in enumerate(live_rows):
        row["firmware_rev"] = 3.0 if i < 600 else 4.0
        row["energy_total_kwh"] = 1000.0 + 0.7 * (1700 + i) + 0.01 * math.sin(1700 + i)
    live_rows[300]["flow_m3h"] = 1.0e9

    model = TelemetryModel(max_steps=150, seed=0, device="cpu").fit(train_rows)
    notes = " ".join(model.fit_summary_.notes)
    assert "missing value" in notes and "firmware_rev" in notes and "energy_total_kwh" in notes
    assert "Free-running" in notes                                   # the running total is modelled by its rate only

    result = model.detect(live_rows)
    by_sensor = {e.top_sensor: e for e in result.events}
    assert set(by_sensor) == {"flow_m3h", "firmware_rev"}            # the ever-growing counter is NOT an anomaly
    assert by_sensor["flow_m3h"].start == 300 and by_sensor["flow_m3h"].duration <= 8   # the glitch does not linger
    assert by_sensor["firmware_rev"].start == 600
    assert np.isfinite(result.scores).all() and result.scores.max() <= 1.0e4


def test_very_short_logs_work_and_say_so():
    rows = []
    for t in range(45):
        fault = t >= 22
        rows.append({
            "timestamp": f"2026-11-10T06:00:{t:02d}Z",
            "voltage_kV": 138.0 + 1.2 * math.sin(0.2 * t) - (4.8 if fault else 0.0),
            "current_A": 420.0 + 8.0 * math.sin(0.3 * t),
            "anomaly": int(fault),
        })
    model = TelemetryModel(seed=0, device="cpu").fit(rows[:22])
    assert any("Very short log" in w for w in model.fit_summary_.warnings)
    result = model.detect(rows)
    assert result.metrics["incidents_detected"] == 1 and result.metrics["false_alarm_events"] == 0
    assert result.events[0].top_sensor == "voltage_kV" and result.events[0].start == 22
    assert model.forecast(5).values.shape == (5, 2)


def test_event_grouping_and_metrics_helpers():
    score = np.array([0, 5, 0, 0, 5.2, 0, 0, 0, 0, 0, 0, 4.9, 0, 0, 0, 0, 0, 9, 3.5, 3.4, 8, 0], dtype=float)
    # 1 & 4 merge (gap 2 <= 3); the lone 4.9 blip is dropped; 17 & 20 merge through the elevated 3.5/3.4
    assert find_events(score, threshold=4.5, merge_gap=3) == [(1, 4), (17, 20)]
    assert find_events(score, threshold=4.5, merge_gap=0, exit_factor=1.0) == [(17, 17), (20, 20)]
    assert find_events(np.zeros(5), 4.5) == []

    labels = np.zeros(22, dtype=int)
    labels[16:21] = 1
    metrics = detection_metrics(labels, score, [(1, 4), (17, 20)])
    assert metrics["labeled_incidents"] == 1 and metrics["incidents_detected"] == 1
    assert metrics["false_alarm_events"] == 1 and metrics["median_detection_delay_steps"] == 1.0
    assert 0.0 < metrics["point_precision"] < 1.0 and metrics["point_recall"] == pytest.approx(0.8)
    assert 0.5 < metrics["auroc"] <= 1.0
