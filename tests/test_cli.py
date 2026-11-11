"""The `rcwe` command: every subcommand an end user touches, driven through `main(argv)`."""
import json
from pathlib import Path

import pytest

from rcwe.main import EXIT_ANOMALY, EXIT_ERROR, EXIT_OK, build_parser, main
from rcwe.toolkit import DataError, TelemetryModel, load_table
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


def test_watch_reads_semicolon_files_and_an_unterminated_last_row(workspace, capsys):
    rows = Path(workspace["incident"]).read_text().splitlines()[:201]          # header + 200 rows
    european = workspace["root"] / "european.csv"
    european.write_text("\n".join(line.replace(",", ";") for line in rows))    # no newline after the last row
    assert main(["watch", str(european), "-m", workspace["model"], "--once"]) == EXIT_OK
    assert "Processed 200 row(s)" in capsys.readouterr().out


def test_watch_explains_a_file_that_lacks_the_models_sensors(workspace, capsys):
    other = workspace["root"] / "other_plant.csv"
    other.write_text("timestamp,flow_m3h,speed\n2026-01-01T00:00:00Z,1.0,2.0\n2026-01-01T00:05:00Z,1.1,2.1\n")
    assert main(["watch", str(other), "-m", workspace["model"], "--once"]) == EXIT_ERROR
    err = capsys.readouterr().err
    assert "no column named 'pressure_bar'" in err and "Columns found: timestamp, flow_m3h, speed" in err


def test_watch_max_rows_also_counts_the_rows_used_to_warm_up(workspace, capsys):
    import threading

    codes = []
    # Without --once the command follows the file forever, so only --max-rows can end it.
    worker = threading.Thread(
        target=lambda: codes.append(main(["watch", workspace["incident"], "-m", workspace["model"], "--max-rows", "40"])),
        daemon=True,
    )
    worker.start()
    worker.join(timeout=60)
    assert codes == [EXIT_OK]
    assert "Processed 40 row(s); 0 alert(s)." in capsys.readouterr().out


def test_follow_csv_waits_for_complete_rows_and_survives_rotation(tmp_path):
    from rcwe.main import _follow_csv

    log = tmp_path / "live.csv"
    log.write_text("a,b\n1,2\n")
    rows = _follow_csv(log, interval=0.01, once=False)
    assert next(rows) == {"a": "1", "b": "2"}
    assert next(rows) is None                                  # caught up with what was already there
    with log.open("a") as handle:
        handle.write("3,4")                                    # a row that is still being written ...
        handle.flush()
        handle.write("\n5,6\n")                                # ... and is then completed
    assert next(rows) == {"a": "3", "b": "4"}
    assert next(rows) == {"a": "5", "b": "6"}
    log.write_text("a;b\n7;8\n")                               # rotated: shorter file, different delimiter
    assert next(rows) == {"a": "7", "b": "8"}
    rows.close()


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
    with pytest.raises(DataError):
        main(["--debug", "inspect", str(tmp_path / "missing.csv")])       # --debug re-raises for a traceback


def test_doctor_reports_a_healthy_installation(capsys):
    assert main(["doctor"]) == EXIT_OK
    text = capsys.readouterr().out
    assert "[ ok ] PyTorch" in text and "[ ok ] End-to-end test" in text and "[FAIL]" not in text


def test_mistakes_on_the_command_line_never_look_like_anomalies(workspace, capsys):
    """Status 2 means "anomalies found", so a typo or a missing option must not exit with it."""
    assert main(["detect", workspace["incident"], "-m", workspace["model"], "--fail-on-anomoly"]) == EXIT_ERROR
    assert main(["detect", workspace["incident"]]) == EXIT_ERROR
    assert main(["detect", "--help"]) == EXIT_OK
    capsys.readouterr()
    for option, value, message in [
        ("--train-fraction", "0", "between 0 and 1"), ("--train-fraction", "60", "between 0 and 1"),
        ("--train-rows", "-5", "at least 12"), ("--steps", "0", "at least 1"), ("--steps", "2.5", "not a whole number"),
        ("--forecast", "-3", "at least 0"),
    ]:
        assert main(["analyze", workspace["incident"], option, value]) == EXIT_ERROR
        assert message in capsys.readouterr().err, (option, value)


def test_output_paths_of_the_wrong_kind_are_refused_before_any_work(workspace, capsys):
    root = workspace["root"]
    (root / "a_file.txt").write_text("x")
    (root / "a_folder").mkdir()
    cases = [
        (["analyze", workspace["incident"], "--out", str(root / "a_file.txt")], "is a file; give a folder"),
        (["fit", workspace["normal"], "-o", str(root / "a_folder")], "is a folder; give a file name"),
        (["detect", workspace["incident"], "-m", workspace["model"], "--report", str(root / "a_folder")], "is a folder; give a file name"),
        (["demo-data", "--out", str(root / "a_file.txt")], "is a file; give a folder"),
    ]
    for argv, message in cases:
        assert main(argv) == EXIT_ERROR
        captured = capsys.readouterr()
        assert message in captured.err and "Training" not in captured.err + captured.out, argv


def test_analyze_scores_against_labels_even_when_the_training_log_has_none(workspace, capsys):
    # `--label-column` names a column of the log being scanned; a known-good training log does not have it.
    code = main(["analyze", workspace["incident"], "--train", workspace["normal"], "--label-column", "anomaly",
                 "--steps", "120", "--quiet", "--device", "cpu", "--out", str(workspace["root"] / "with_labels")])
    assert code == EXIT_OK
    assert "incidents caught" in capsys.readouterr().out


def test_suggested_commands_can_be_pasted_when_paths_have_spaces(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["demo-data", "--out", "my demo data"]) == EXIT_OK
    hint = [line for line in capsys.readouterr().out.splitlines() if line.startswith("Try:")][0]
    assert "my demo data/pump_station_incident.csv" in hint and ("'" in hint or '"' in hint)


def test_watch_skips_comments_and_rows_caught_mid_write(workspace, capsys):
    lines = Path(workspace["normal"]).read_text().splitlines()
    log = workspace["root"] / "half_written.csv"
    log.write_text("# pump station export\n" + "\n".join(lines[:151]) + "\n" + lines[151][:40])   # last row cut off
    assert main(["watch", str(log), "-m", workspace["model"], "--once", "--fail-on-anomaly"]) == EXIT_OK
    assert "Processed 150 row(s); 0 alert(s)." in capsys.readouterr().out

    headerless = workspace["root"] / "headerless.csv"
    headerless.write_text("\n".join(lines[1:40]) + "\n")
    assert main(["watch", str(headerless), "-m", workspace["model"], "--once"]) == EXIT_ERROR
    assert "no header row" in capsys.readouterr().err


def test_follow_csv_keeps_its_columns_when_a_log_is_truncated_in_place(tmp_path):
    from rcwe.main import _follow_csv

    headers = []
    log = tmp_path / "live.csv"
    log.write_text("a,b\n1,2\n3,4\n")
    rows = _follow_csv(log, interval=0.01, once=False, on_header=headers.append)
    assert [next(rows), next(rows), next(rows)] == [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}, None]
    log.write_text("5,6\n")                         # logrotate copytruncate: the writer just carries on with data
    assert next(rows) == {"a": "5", "b": "6"}       # not mistaken for a header, and not lost
    with log.open("a") as handle:
        handle.write("7,8\n")
    assert next(rows) == {"a": "7", "b": "8"}
    assert headers == [["a", "b"]]
    rows.close()


def test_a_device_that_is_not_there_is_explained(capsys, monkeypatch):
    monkeypatch.setenv("RCWE_DEVICE", "gpu0")
    assert main(["doctor"]) == EXIT_ERROR
    assert "[FAIL] Accelerator: RCWE_DEVICE asks for 'gpu0'" in capsys.readouterr().out
    monkeypatch.delenv("RCWE_DEVICE")
    with pytest.raises(ValueError, match="not a device PyTorch knows"):
        TelemetryModel(device="warp-drive").fit(make_pump_station(rows=300, faults=False).table)


def test_demo_generator_limits_are_stated_up_front():
    with pytest.raises(ValueError, match="at least 600 rows"):
        make_pump_station(rows=500)
    assert make_pump_station(rows=700).table.labels.sum() > 0
    assert make_pump_station(rows=300, faults=False).table.n_rows == 300
    shifted = make_pump_station(rows=300, faults=False, start="2026-03-02T00:00:00+02:00").table
    assert shifted.timestamps[0] == "2026-03-01T22:00:00Z"            # labelled in UTC, matching time_seconds
