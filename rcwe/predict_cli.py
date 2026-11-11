"""
`rcwe-predict` -- the original one-shot CSV/JSON predictor (kept for compatibility).

It streams a telemetry file through the engine and a frozen Transformer baseline, reports
their post-shift prediction error, and saves an engine checkpoint.

Since v1.1 the "flagged sensors" come from the toolkit's calibrated anomaly detector
(`rcwe.TelemetryModel`) instead of a raw last-five-rows difference, which used to name the
most volatile columns rather than the faulty ones. For the full workflow -- explained events,
an HTML report, forecasts, a reusable model file -- use `rcwe analyze <file>`.
"""
import argparse
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, List, Optional

from rcwe.core.engine import ResonantCausalWorldEngine
from rcwe.sandbox.baseline_tf import BaselineCausalTransformer
from rcwe.ingestion.telemetry_loader import TelemetryIngestionLoader
from rcwe.kernels.accelerator import select_best_device, describe_accelerator, sync_device
from rcwe.paths import output_path

DEFAULT_CHECKPOINT = "rcwe_Universal_Checkpoint_v06.pt"
DEFAULT_REPORT = "rcwe_universal_prediction_report.json"


def _flag_anomalous_sensors(input_path: str, limit: int = 3) -> Dict[str, Any]:
    """Learn normal behaviour from the first half of the file, then ask which sensors broke it."""
    from rcwe.toolkit import DataError, TelemetryModel, load_table

    try:
        table = load_table(input_path)
        if table.n_rows < 24:
            return {"sensors": [], "events": [], "note": "Too few rows for anomaly attribution."}
        head, _ = table.split(0.5)
        model = TelemetryModel(device="cpu").fit(head)
        result = model.detect(table)
    except DataError as exc:
        return {"sensors": [], "events": [], "note": f"Anomaly attribution skipped: {exc}"}
    ranked = [r["name"] for r in result.sensor_ranking() if r["anomalous_steps"] > 0]
    return {"sensors": ranked[:limit], "events": [e.description for e in result.events], "note": ""}


def run_universal_prediction(
    input_path: str,
    save_checkpoint: Optional[str] = DEFAULT_CHECKPOINT,
    load_checkpoint: Optional[str] = None,
    output_json: str = DEFAULT_REPORT,
    warmup_steps: int = 15,
    epochs: int = 35,
) -> Dict[str, Any]:
    # Bare default names go to the output folder (./rcwe_output or $RCWE_OUTPUT_DIR), not the cwd.
    if save_checkpoint == DEFAULT_CHECKPOINT:
        save_checkpoint = output_path(DEFAULT_CHECKPOINT)
    if output_json == DEFAULT_REPORT:
        output_json = output_path(DEFAULT_REPORT)

    device = select_best_device()
    data = TelemetryIngestionLoader.auto_infer_and_load(input_path, dt=0.05, device=device)

    x_seq = data["x_seq"]
    a_seq = data["action_seq"]
    y_true = data["y_target"]
    d_input = data["d_input"]
    d_action = data["d_action"]
    num_nodes = data["num_nodes"]
    T = x_seq.shape[1]

    torch.manual_seed(42)
    tf_model = BaselineCausalTransformer(d_input=d_input, d_model=64, d_action=d_action, nhead=4, num_layers=2).to(device)

    if load_checkpoint and Path(load_checkpoint).exists():
        rcwe, standing_field = ResonantCausalWorldEngine.load_pretrained(load_checkpoint, device=device)
        if rcwe.d_input != d_input or rcwe.d_action != d_action:
            raise ValueError(
                f"Checkpoint {load_checkpoint} was trained on {rcwe.d_input // 2} sensors / {rcwe.d_action} controls, "
                f"but {input_path} has {num_nodes} sensors / {d_action} controls."
            )
    else:
        rcwe = ResonantCausalWorldEngine(d_input=d_input, d_model=64, d_action=d_action, num_worldlines=3).to(device)
        standing_field = None

    w_len = min(warmup_steps, max(T // 2, 2))
    x_w, a_w, y_w = x_seq[:, :w_len, :], a_seq[:, :w_len, :], y_true[:, :w_len, :]

    opt_r = torch.optim.AdamW(rcwe.parameters(), lr=4e-3)
    opt_t = torch.optim.AdamW(tf_model.parameters(), lr=4e-3)

    rcwe.train()
    tf_model.train()
    for _ in range(epochs):
        opt_r.zero_grad()
        opt_t.zero_grad()
        out_r = rcwe(x_w, a_w, plastic_runtime=False, y_target=y_w)
        loss_r = F.mse_loss(out_r["output"], y_w) + F.mse_loss(out_r["branch_predictions"][0], y_w)
        loss_r.backward()
        opt_r.step()

        out_t = tf_model(x_w, a_w)
        loss_t = F.mse_loss(out_t, y_w)
        loss_t.backward()
        opt_t.step()

    sync_device(device)
    rcwe.eval()
    tf_model.eval()

    rcwe_mses, tf_mses = [], []
    with torch.no_grad():
        for t in range(T):
            xt = x_seq[:, t : t + 1, :]
            at = a_seq[:, t : t + 1, :]
            yt = y_true[:, t : t + 1, :]

            res_r = rcwe(xt, at, standing_field=standing_field, plastic_runtime=True, y_target=yt)
            standing_field = res_r["standing_field"]
            res_t = tf_model(xt, at)

            rcwe_mses.append(float(F.mse_loss(res_r["output"], yt).item()))
            tf_mses.append(float(F.mse_loss(res_t, yt).item()))

    ckpt_saved = None
    if save_checkpoint:
        ckpt_saved = rcwe.save_pretrained(save_checkpoint, standing_field=standing_field)

    flagged = _flag_anomalous_sensors(input_path)

    tail_n = max(min(10, T // 3), 1)
    report = {
        "input_file": str(Path(input_path).resolve()),
        "accelerator": describe_accelerator(device),
        "inferred_sensors": num_nodes,
        "inferred_actions": d_action,
        "total_steps": T,
        "rcwe_tail_mse": sum(rcwe_mses[-tail_n:]) / tail_n,
        "tf_tail_mse": sum(tf_mses[-tail_n:]) / tail_n,
        "flagged_anomalous_sensors": flagged["sensors"],
        "anomaly_events": flagged["events"],
        "saved_checkpoint": ckpt_saved,
    }
    if flagged["note"]:
        report["anomaly_note"] = flagged["note"]
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["json_path"] = str(Path(output_json).resolve())
    return report


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="rcwe-predict",
        description="RCWE universal CSV/JSON predictor and checkpoint CLI (legacy). See `rcwe analyze` for the full workflow.",
    )
    parser.add_argument("--input", type=str, default=None,
                        help="Path to a CSV or JSON telemetry file (default: a generated sample grid log)")
    parser.add_argument("--save-ckpt", type=str, default=DEFAULT_CHECKPOINT,
                        help="Where to save the engine checkpoint (default: inside ./rcwe_output)")
    parser.add_argument("--load-ckpt", type=str, default=None, help="Optional checkpoint to resume from")
    args = parser.parse_args(argv)

    input_path = args.input
    if input_path is None:
        input_path = output_path("sample_grid_telemetry.csv")
        if not Path(input_path).exists():
            TelemetryIngestionLoader().generate_sample_grid_telemetry(
                csv_path=input_path, json_path=output_path("sample_grid_telemetry.json")
            )
    elif not Path(input_path).exists():
        parser.error(f"input file not found: {input_path}")

    rep = run_universal_prediction(
        input_path=input_path,
        save_checkpoint=args.save_ckpt,
        load_checkpoint=args.load_ckpt,
    )
    r_m, t_m = rep["rcwe_tail_mse"], rep["tf_tail_mse"]
    print("================================================================================")
    print("   DORADUS LABS — RCWE UNIVERSAL CSV/JSON PREDICTOR & CHECKPOINT CLI")
    print("================================================================================")
    print(f"Input Telemetry File      : {rep['input_file']}")
    print(f"Auto-Inferred Schema      : {rep['inferred_sensors']} Sensor Nodes | {rep['inferred_actions']} Control Channels ({rep['total_steps']} Steps)")
    print(f"Post-Shift Adaptation MSE : RCWE = {r_m:.5f} vs Transformer = {t_m:.5f} ({t_m / max(r_m, 1e-7):.1f}x lower)")
    print(f"Flagged Anomaly Sensors   : {rep['flagged_anomalous_sensors']}")
    for line in rep["anomaly_events"][:5]:
        print(f"  - {line}")
    print(f"Saved Full State Ckpt     : {rep['saved_checkpoint']}")
    print(f"Saved JSON Report         : {rep['json_path']}")
    print("\nTip: `rcwe analyze <file>` adds an interactive HTML report, forecasts and a reusable model.")


if __name__ == "__main__":
    main()
