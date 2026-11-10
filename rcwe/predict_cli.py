import argparse
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, Optional

from rcwe.core.engine import ResonantCausalWorldEngine
from rcwe.sandbox.baseline_tf import BaselineCausalTransformer
from rcwe.ingestion.telemetry_loader import TelemetryIngestionLoader
from rcwe.kernels.accelerator import select_best_device, describe_accelerator, sync_device


def run_universal_prediction(
    input_path: str,
    save_checkpoint: Optional[str] = "rcwe_Universal_Checkpoint_v06.pt",
    load_checkpoint: Optional[str] = None,
    output_json: str = "rcwe_universal_prediction_report.json",
    warmup_steps: int = 15,
    epochs: int = 35,
) -> Dict[str, Any]:
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

        # Identify top anomalous sensor columns in the tail window
        tail_err = (y_true[:, -5:, :num_nodes] - x_seq[:, -5:, :num_nodes]).abs().mean(dim=(0, 1))
        top_k = min(3, num_nodes)
        top_indices = torch.topk(tail_err, k=top_k).indices.tolist()
        flagged_sensors = [data["sensor_columns"][i] for i in top_indices]

    ckpt_saved = None
    if save_checkpoint:
        ckpt_saved = rcwe.save_pretrained(save_checkpoint, standing_field=standing_field)

    tail_n = max(min(10, T // 3), 1)
    report = {
        "input_file": str(Path(input_path).resolve()),
        "accelerator": describe_accelerator(device),
        "inferred_sensors": num_nodes,
        "inferred_actions": d_action,
        "total_steps": T,
        "rcwe_tail_mse": sum(rcwe_mses[-tail_n:]) / tail_n,
        "tf_tail_mse": sum(tf_mses[-tail_n:]) / tail_n,
        "flagged_anomalous_sensors": flagged_sensors,
        "saved_checkpoint": ckpt_saved,
    }
    Path(output_json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="RCWE v0.6.0 Universal CSV/JSON Predictor & Checkpoint CLI")
    parser.add_argument("--input", type=str, default="sample_grid_telemetry.csv", help="Path to CSV or JSON telemetry file")
    parser.add_argument("--save-ckpt", type=str, default="rcwe_Universal_Checkpoint_v06.pt", help="Path to save pretrained RCWE checkpoint")
    parser.add_argument("--load-ckpt", type=str, default=None, help="Optional checkpoint to resume from")
    args = parser.parse_args()

    if not Path(args.input).exists():
        TelemetryIngestionLoader().generate_sample_grid_telemetry(csv_path=args.input)

    rep = run_universal_prediction(
        input_path=args.input,
        save_checkpoint=args.save_ckpt,
        load_checkpoint=args.load_ckpt,
    )
    r_m, t_m = rep["rcwe_tail_mse"], rep["tf_tail_mse"]
    print("================================================================================")
    print("   DORADUS LABS — RCWE v0.6.0 UNIVERSAL CSV/JSON PREDICTOR & CHECKPOINT CLI")
    print("================================================================================")
    print(f"Input Telemetry File      : {rep['input_file']}")
    print(f"Auto-Inferred Schema      : {rep['inferred_sensors']} Sensor Nodes | {rep['inferred_actions']} Control Channels ({rep['total_steps']} Steps)")
    print(f"Post-Shift Adaptation MSE : RCWE = {r_m:.5f} vs Transformer = {t_m:.5f} ({t_m / max(r_m, 1e-7):.1f}x lower)")
    print(f"Flagged Anomaly Sensors   : {rep['flagged_anomalous_sensors']}")
    print(f"Saved Full State Ckpt     : {rep['saved_checkpoint']}")


if __name__ == "__main__":
    main()
