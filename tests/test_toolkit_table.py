"""Data ingestion: `load_table` must cope with whatever shape real telemetry arrives in."""
import json

import numpy as np
import pytest

from rcwe.toolkit import DataError, TelemetryTable, load_table
from rcwe.toolkit.demo import make_pump_station, write_demo_files


def test_demo_files_round_trip_with_schema_inference(tmp_path):
    paths = write_demo_files(tmp_path / "demo")
    normal, incident = load_table(paths["normal"]), load_table(paths["incident"])

    assert normal.n_rows == 3000 and incident.n_rows == 1200
    assert normal.sensors == ["flow_m3h", "pressure_bar", "motor_power_kw", "bearing_temp_c", "vibration_mm_s", "tank_level_m"]
    assert normal.controls == ["act_valve_open", "act_pump_speed"]
    assert normal.time_column == "timestamp" and normal.sample_interval == 300.0
    assert normal.labels is None                        # the training file carries no labels
    assert incident.label_column == "anomaly" and 0 < incident.labels.sum() < incident.n_rows
    assert incident.time_seconds[0] - normal.time_seconds[-1] == 300.0   # the incident log continues the normal one


def test_legacy_rcwe_logs_ignore_explicit_momentum_columns():
    table = load_table("sample_grid_telemetry.csv")
    assert table.n_sensors == 16 and all(name.startswith("q_") for name in table.sensors)
    assert len(table.controls) == 8
    assert table.time_column == "timestamp_step" and table.label_column == "regime_label"
    assert any("momentum" in note for note in table.notes)
    as_json = load_table("sample_grid_telemetry.json")
    assert np.allclose(table.values, as_json.values)


def test_messy_european_csv_is_cleaned(tmp_path):
    path = tmp_path / "eu.csv"
    path.write_text(
        "Zeit;Temp C;Druck;Status;act_ventil\n"
        "01.03.2026 10:00;21,5;1.013,2;OK;0,5\n"
        "01.03.2026 10:05;21,7;;OK;0,5\n"
        "01.03.2026 10:10;n/a;1.013,9;FAULT;0,6\n"
        "01.03.2026 10:15;22,1;1.014,4;OK;0,6\n",
        encoding="utf-8",
    )
    table = load_table(path)
    assert table.sensors == ["Temp C", "Druck"] and table.controls == ["act_ventil"]
    assert table.time_column == "Zeit" and table.sample_interval == 300.0
    assert table.values[:, 0].tolist() == pytest.approx([21.5, 21.7, 21.9, 22.1])      # gap interpolated
    assert table.values[:, 1].tolist() == pytest.approx([1013.2, 1013.55, 1013.9, 1014.4])
    assert int(table.missing.sum()) == 2
    assert "Status" in table.ignored_columns
    assert any("semicolon" in note for note in table.notes)


def test_in_memory_sources_and_epoch_timestamps(tmp_path):
    lines = [json.dumps({"ts": 1767225600 + 60 * i, "a": 0.5 * i, "b": None if i == 2 else float(i)}) for i in range(6)]
    path = tmp_path / "stream.jsonl"
    path.write_text("\n".join(lines), encoding="utf-8")
    table = load_table(path)
    assert table.sensors == ["a", "b"] and table.timestamps[0] == "2026-01-01T00:00:00Z"
    assert table.sample_interval == 60.0 and table.values[2, 1] == pytest.approx(2.0)

    assert load_table(np.random.rand(10, 3)).sensors == ["sensor_0", "sensor_1", "sensor_2"]
    from_dict = load_table({"x": [1, 2, 3], "y": [3, 2, 1], "anomaly": [0, 0, 1]})
    assert from_dict.sensors == ["x", "y"] and from_dict.labels.tolist() == [0, 0, 1]
    rows = [{"t": "2026-01-01 00:00:02", "v": 3}, {"t": "2026-01-01 00:00:00", "v": 1}, {"t": "2026-01-01 00:00:01", "v": 2}]
    shuffled = load_table(rows)
    assert shuffled.values.ravel().tolist() == [1.0, 2.0, 3.0]
    assert any("sorted" in note for note in shuffled.notes)
    assert load_table(shuffled) is shuffled             # tables pass straight through


def test_explicit_schema_overrides_and_table_helpers():
    demo = make_pump_station(rows=600, faults=False).table
    picked = load_table(demo.to_rows(), sensors=["flow_m3h", "pressure_bar"], controls=["act_pump_speed"])
    assert picked.sensors == ["flow_m3h", "pressure_bar"] and picked.controls == ["act_pump_speed"]

    head, rest = demo.split(0.25)
    assert head.n_rows == 150 and rest.n_rows == 450 and rest.timestamps[0] == demo.timestamps[150]
    summary = demo.describe()
    assert summary["rows"] == 600 and summary["sample_interval_seconds"] == 300.0
    assert {s["name"] for s in summary["sensor_stats"]} == set(demo.sensors)
    assert isinstance(demo.slice(10, 20), TelemetryTable) and demo.slice(10, 20).n_rows == 10


@pytest.mark.parametrize(
    "bad, fragment",
    [
        ("does_not_exist.csv", "File not found"),
        ("a,b\n1,2\n", "at least 2 time steps"),
        ({"x": ["a", "b", "c"]}, "No numeric sensor columns"),
        ([], "empty"),
    ],
)
def test_unusable_input_gives_actionable_errors(bad, fragment):
    with pytest.raises(DataError, match=fragment):
        load_table(bad)


def test_misspelled_column_suggests_the_right_one():
    with pytest.raises(DataError, match="Did you mean 'x'"):
        load_table({"x": [1, 2, 3]}, sensors=["X"])
