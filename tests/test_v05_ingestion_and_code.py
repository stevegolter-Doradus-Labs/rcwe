from pathlib import Path
from rcwe.ingestion.telemetry_loader import TelemetryIngestionLoader
from rcwe.symbolic.code_reasoning import run_code_reasoning_benchmark
from rcwe.diagnostics.dashboard_v05 import run_v05_pipeline_and_build_dashboard


def test_csv_and_json_telemetry_ingestion_parity(tmp_path: Path):
    loader = TelemetryIngestionLoader(num_nodes=16, d_action=8, dt=0.05)
    csv_p, json_p = loader.generate_sample_grid_telemetry(
        csv_path=str(tmp_path / "grid.csv"),
        json_path=str(tmp_path / "grid.json"),
        num_steps=16,
    )
    d_csv = loader.load_csv(csv_p)
    d_json = loader.load_json(json_p)

    assert d_csv["x_seq"].shape == (1, 15, 32)
    assert d_json["x_seq"].shape == (1, 15, 32)
    assert (d_csv["x_seq"] - d_json["x_seq"]).abs().max().item() < 1e-6


def test_tokenized_code_reasoning_and_dashboard_generation(tmp_path: Path):
    dash_file = tmp_path / "dash.html"
    json_file = tmp_path / "report.json"
    summary = run_v05_pipeline_and_build_dashboard(
        dashboard_path=str(dash_file),
        json_report_path=str(json_file),
    )
    assert dash_file.exists()
    assert json_file.exists()
    assert summary["post_anomaly_rcwe_mse"] < summary["post_anomaly_tf_mse"]
    assert summary["code_reasoning"]["rcwe_code_trace_mse"] < summary["code_reasoning"]["tf_code_trace_mse"]
    assert set(summary["code_reasoning"]["flagged_bug_registers"]) == {5, 6}
