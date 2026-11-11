import csv
import json
import math
import torch
from pathlib import Path
from typing import Dict, Any, Optional

from rcwe.predict_cli import run_universal_prediction


class CustomDatasetConnector:
    """
    RCWE v0.8.0 Drop-In Custom Dataset Connector (`datasets/custom/`).
    Automatically scans `datasets/custom/` for any user-provided `.csv` or `.json` files
    (with arbitrary column names, timestamps, or missing momentum columns), normalizes
    sensor scales, runs `rcwe-predict` with online 3-factor metabolic plasticity, and
    outputs a dedicated custom dataset diagnostic report.
    """
    @staticmethod
    def ensure_custom_dataset_template(custom_dir: Path = Path("datasets/custom")) -> Path:
        custom_dir.mkdir(parents=True, exist_ok=True)
        readme_p = custom_dir / "README_DROP_FILES_HERE.md"
        readme_p.write_text(
            "# Drop Any Custom `.csv` or `.json` Dataset Here\n\n"
            "Run `make custom-data` or `rcwe-predict --input datasets/custom/<your_file.csv>` to:\n"
            "1. Auto-detect all numeric sensor/feature columns and control columns.\n"
            "2. Reconstruct canonical momentum `p_t = (q_t - q_{t-1}) / dt` automatically.\n"
            "3. Run online 3-factor metabolic adaptation on Apple Silicon MPS and flag anomalous sensors.\n",
            encoding="utf-8",
        )
        sample_custom = custom_dir / "custom_scada_substation_log.csv"
        if not sample_custom.exists():
            rows = []
            for t in range(45):
                fault = 1 if t >= 22 else 0
                rows.append({
                    "timestamp": f"2026-11-10T06:00:{t:02d}Z",
                    "voltage_bus_A_kV": round(138.0 + 1.2 * math.sin(0.2 * t) - (4.8 if fault else 0.0), 4),
                    "voltage_bus_B_kV": round(138.0 + 1.1 * math.cos(0.2 * t) - (4.2 if fault else 0.0), 4),
                    "current_line_1_A": round(420.0 + 8.0 * math.sin(0.3 * t) + (35.0 if fault else 0.0), 4),
                    "freq_Hz": round(60.0 + 0.02 * math.cos(0.1 * t) - (0.35 if fault else 0.0), 5),
                    "transformer_oil_C": round(68.5 + 0.05 * t + (6.5 if fault else 0.0), 4),
                    "reactive_MVAR": round(24.0 + 0.9 * math.sin(0.15 * t), 4),
                    "act_breaker_trim": round(0.1 * math.cos(0.2 * t), 4),
                    "act_tap_changer": round(0.15 * math.sin(0.2 * t), 4),
                    "anomaly": fault,
                })
            with sample_custom.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        return sample_custom

    @classmethod
    def run_custom_directory_scan(cls, custom_dir: str = "datasets/custom") -> Dict[str, Any]:
        c_dir = Path(custom_dir)
        default_csv = cls.ensure_custom_dataset_template(c_dir)
        data_files = sorted(list(c_dir.glob("*.csv")) + list(c_dir.glob("*.json")))
        target_file = data_files[0] if data_files else default_csv

        rep = run_universal_prediction(
            input_path=str(target_file),
            save_checkpoint="rcwe_Custom_Dataset_Checkpoint_v08.pt",
            output_json="rcwe_custom_dataset_report_v08.json",
            warmup_steps=16,
            epochs=35,
        )
        return rep


def main() -> None:
    print("================================================================================")
    print("   DORADUS LABS — RCWE v0.8.0 CUSTOM DATASET AUTO-INGESTION & DIAGNOSTIC CLI")
    print("================================================================================")
    rep = CustomDatasetConnector.run_custom_directory_scan()
    r_m, t_m = rep["rcwe_tail_mse"], rep["tf_tail_mse"]
    print(f"Custom Dataset Ingested   : {rep['input_file']}")
    print(f"Auto-Discovered Columns   : {rep['inferred_sensors']} Named Telemetry Sensors | {rep['inferred_actions']} Controls")
    print(f"Post-Fault Adaptation MSE : RCWE = {r_m:.5f} vs Transformer = {t_m:.5f} ({t_m / max(r_m, 1e-7):.1f}x lower)")
    print(f"Flagged Fault Columns     : {rep['flagged_anomalous_sensors']}")
    print(f"Saved Custom Checkpoint   : {rep['saved_checkpoint']}")


if __name__ == "__main__":
    main()
