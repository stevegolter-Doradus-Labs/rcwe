"""The `rcwe` command: every subcommand an end user touches, driven through `main(argv)`."""
import json
import os
from pathlib import Path

import pytest

from rcwe.main import EXIT_ANOMALY, EXIT_ERROR, EXIT_OK, build_parser, main
from rcwe.toolkit import TelemetryModel, load_table
from rcwe.toolkit.demo import make_pump_station


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """Demo CSVs plus a small trained model, shared by the tests in this file."""
    root = tmp_path_factory.mktemp("cli")
    data = make_pump_station(rows=2400, faults=True, fault_start=1500, seed=7).table
    normal, incident = data.slice(0, 1500), data.slice(1500, None)
    normal.labels, normal.label_column = None, None
    files = {"root": root, "normal": normal.to_csv(root / "normal.csv"), "incident": incident.to_csv(root / "incident.csv"),
             "model": str(root / "plant.rcwe")}
    assert main(["fit", files["normal"], "-o", files["model"], "--steps", "150", "--quiet", "--device", "cpu"]) == EXIT_OK
    return files


def test_help_lists_every_command(capsys):
    assert main([]) == EXIT_OK
    out = capsys.readouterr().out
    for command in ("quickstart", "analyze", "fit", "detect", "forecast", "watch", "inspect", "serve", "ui", "bench", "doctor"):
        assert command in out
    with pytest.raises(SystemExit) as stop:
        build_parser().parse_args(["detect", "--help"])
    assert stop.value.code == 0
    assert "--fail-on-anomaly" in capsys.readouterr().out


def test_version_and_bench_list(capsys):
    assert main(["version"]) == EXIT_OK
    assert capsys.readouterr().out.startswith("rcwe 1.")
    assert main(["bench", "--list"]) == EXIT_OK
    listing = capsys.readouterr().out
    assert "realworld" in listing and "v09" in listing
    assert main(["bench", "nope"]) == EXIT_ERROR


def test_fit_saves_a_loadable_model_and_reports_skill(workspace, capsys):
    capsys.readouterr()
    model = TelemetryModel.load(workspace["model"])
    assert model.sensors[0] == "flow_m3h" and model.time_column == "timestamp"
    assert main(["fit", workspace["normal"], "-o", str(workspace["root"] / "again.rcwe"), "--steps", "50", "--quiet",
                 "--sensors", "flow_m3h,pressure_bar"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "naive" in out and "Model saved to" in out
    assert TelemetryModel.load(workspace["root"] / "again.rcwe").sensors == ["flow_m3h", "pressure_bar"]


def test_inspect_describes_the_schema(workspace, capsys):
    assert main(["inspect", workspace["incident"]]) == EXIT_OK
    text = capsys.readouterr().out
    assert "rows        900" in text and "every 300s" in text and "act_valve_open, act_pump_speed" in text
    assert "anomaly (" in text
    assert main(["inspect", workspace["incident"], "--json"]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["rows"] == 900


def test_detect_writes_report_json_and_csv_and_signals_anomalies(workspace, capsys):
    out = workspace["root"] / "detect"
    argv = ["detect", workspace["incident"], "-m", workspace["model"], "--quiet", "--forecast", "12",
            "--report", str(out / "report.html"), "--json", str(out / "result.json"), "--csv", str(out / "scores.csv")]
    assert main(argv) == EXIT_OK
    text = capsys.readouterr().out
    assert "anomaly event" in text and "incidents caught" in text and "Forecast: 12 steps ahead" in text
    assert (out / "report.html").read_text(encoding="utf-8").startswith("<!DOCTYPE html>")
    result = json.loads((out / "result.json").read_text())
    assert result["event_count"] >= 4 and "series" not in result and result["forecast"]["steps"] == 12
    assert load_table(out / "scores.csv", sensors=["anomaly_score"]).n_rows == 900

    assert main(argv + ["--fail-on-anomaly"]) == EXIT_ANOMALY            # scripts can branch on the exit status
    capsys.readouterr()
    quiet_log = workspace["root"] / "quiet.csv"
    load_table(workspace["normal"]).slice(1000, 1400).to_csv(quiet_log)
    assert main(["detect", str(quiet_log), "-m", workspace["model"], "--quiet", "--fail-on-anomaly"]) == EXIT_OK
    assert "No anomalies detected" in capsys.readouterr().out


def test_forecast_command_writes_csv(workspace, capsys):
    target = workspace["root"] / "forecast.csv"
    assert main(["forecast", "-m", workspace["model"], "--steps", "6", "--csv", str(target)]) == EXIT_OK
    assert "Forecast: 6 steps ahead" in capsys.readouterr().out
    lines = target.read_text().splitlines()
    assert len(lines) == 7 and lines[0].startswith("step_ahead,time,flow_m3h,flow_m3h_lower80,flow_m3h_upper80")


def test_analyze_single_file_produces_a_results_folder(workspace, capsys, monkeypatch):
    monkeypatch.chdir(workspace["root"])
    full = workspace["root"] / "full_log.csv"
    make_pump_station(rows=2400, faults=True, fault_start=1500, seed=7).table.to_csv(full)
    assert main(["analyze", "full_log.csv", "--steps", "150", "--quiet", "--out", "results", "--train-rows", "1500"]) == EXIT_OK
    text = capsys.readouterr().out
    assert "the first 1,500 of 2,400 rows" in text and "incidents caught" in text
    produced = {p.name for p in (workspace["root"] / "results").iterdir()}
    assert produced == {"report.html", "result.json", "scores.csv", "model.rcwe"}
    assert "report   results/report.html" in text                      # paths are shown the way the user would type them


def test_watch_once_raises_debounced_alerts(workspace, capsys):
    alerts = workspace["root"] / "alerts.jsonl"
    code = main(["watch", workspace["incident"], "-m", workspace["model"], "--once", "--jsonl", str(alerts), "--fail-on-anomaly"])
    assert code == EXIT_ANOMALY
    text = capsys.readouterr().out
    assert "ANOMALY" in text and "cleared" in text and "Processed 900 row(s)" in text
    events = [json.loads(line) for line in alerts.read_text().splitlines()]
    assert events[0]["type"] == "anomaly" and events[0]["sensors"][0]["detail"]
    assert sum(e["type"] == "anomaly" for e in events) == sum(e["type"] == "cleared" for e in events)
    stuck = [e for e in events if e["type"] == "anomaly" and "stuck" in e["sensors"][0]["detail"]]
    assert stuck and stuck[0]["sensors"][0]["name"] == "vibration_mm_s"


def test_demo_data_and_friendly_errors(tmp_path, capsys):
    assert main(["demo-data", "--out", str(tmp_path / "demo")]) == EXIT_OK
    assert {p.name for p in (tmp_path / "demo").iterdir()} == {"pump_station_normal.csv", "pump_station_incident.csv"}
    capsys.readouterr()

    assert main(["inspect", str(tmp_path / "missing.csv")]) == EXIT_ERROR
    assert "error: File not found" in capsys.readouterr().err
    assert main(["detect", str(tmp_path / "demo" / "pump_station_incident.csv"), "-m", str(tmp_path / "nope.rcwe")]) == EXIT_ERROR
    assert "Model file not found" in capsys.readouterr().err
    (tmp_path / "words.csv").write_text("name,city\nann,oslo\nbob,rome\n")
    assert main(["fit", str(tmp_path / "words.csv")]) == EXIT_ERROR
    assert "No numeric sensor columns" in capsys.readouterr().err
    with pytest.raises(Exception):
        main(["--debug", "inspect", str(tmp_path / "missing.csv")])       # --debug re-raises for a traceback


def test_doctor_reports_a_healthy_installation(capsys):
    assert main(["doctor"]) == EXIT_OK
    text = capsys.readouterr().out
    assert "[ ok ] PyTorch" in text and "[ ok ] End-to-end test" in text and "[FAIL]" not in text
