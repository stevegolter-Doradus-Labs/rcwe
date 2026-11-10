import csv
import json
import math
import torch
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


class TelemetryIngestionLoader:
    """
    RCWE v0.6.0 Universal Physical Telemetry CSV & JSON Ingestion Pipeline.
    Supports both fixed schema (`q_0..q_N`, `p_0..p_N`, `act_0..act_K`) and automatic
    column schema inference (`auto_infer_and_load`) for arbitrary real-world CSV/JSON files.
    """
    def __init__(self, num_nodes: int = 16, d_action: int = 8, dt: float = 0.05):
        self.num_nodes = num_nodes
        self.d_action = d_action
        self.dt = dt

    def generate_sample_grid_telemetry(
        self,
        csv_path: str = "sample_grid_telemetry.csv",
        json_path: str = "sample_grid_telemetry.json",
        num_steps: int = 40,
    ) -> Tuple[str, str]:
        torch.manual_seed(42)
        rows: List[Dict[str, Any]] = []
        q = [math.sin(0.3 * i) * 0.25 for i in range(self.num_nodes)]
        p = [0.0 for _ in range(self.num_nodes)]

        for t in range(num_steps):
            regime = 0 if t < (num_steps // 2) else 1
            actions = [math.cos(0.2 * t + k) * 0.15 for k in range(self.d_action)]

            next_q = [0.0] * self.num_nodes
            next_p = [0.0] * self.num_nodes
            for i in range(self.num_nodes):
                left = q[(i - 1) % self.num_nodes]
                right = q[(i + 1) % self.num_nodes]
                force = 1.4 * (left + right - 2.0 * q[i]) - 0.15 * q[i]
                if i < self.d_action:
                    force += actions[i] * 0.25
                if regime == 1 and i in (5, 6):
                    force -= 1.45 + 0.40 * q[i]

                next_p[i] = p[i] + self.dt * force
                next_q[i] = q[i] + self.dt * next_p[i]
                if regime == 1 and i < 8:
                    next_q[i] += 0.015

            row: Dict[str, Any] = {"timestamp_step": t, "regime_label": regime}
            for i in range(self.num_nodes):
                row[f"q_{i}"] = round(next_q[i], 6)
            for i in range(self.num_nodes):
                row[f"p_{i}"] = round(next_p[i], 6)
            for k in range(self.d_action):
                row[f"act_{k}"] = round(actions[k], 6)

            rows.append(row)
            q, p = next_q, next_p

        csv_file = Path(csv_path)
        fieldnames = list(rows[0].keys())
        with csv_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        json_file = Path(json_path)
        json_file.write_text(
            json.dumps({"metadata": {"num_nodes": self.num_nodes, "dt": self.dt}, "telemetry": rows}, indent=2) + "\n",
            encoding="utf-8",
        )
        return str(csv_file.resolve()), str(json_file.resolve())

    def load_csv(
        self, csv_path: str, device: torch.device = torch.device("cpu")
    ) -> Dict[str, torch.Tensor]:
        path = Path(csv_path)
        with path.open("r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        return self._rows_to_tensors(reader, device=device)

    def load_json(
        self, json_path: str, device: torch.device = torch.device("cpu")
    ) -> Dict[str, torch.Tensor]:
        path = Path(json_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload["telemetry"] if isinstance(payload, dict) and "telemetry" in payload else payload
        return self._rows_to_tensors(rows, device=device)

    @classmethod
    def auto_infer_and_load(
        cls, file_path: str, dt: float = 0.05, device: torch.device = torch.device("cpu")
    ) -> Dict[str, Any]:
        """
        Universal schema inference for any custom CSV or JSON file.
        Automatically discovers q_* / sensor columns, p_* columns (or computes finite-difference
        velocity p_t = (q_t - q_{t-1})/dt), and act_* / control columns.
        """
        path = Path(file_path)
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload["telemetry"] if isinstance(payload, dict) and "telemetry" in payload else payload
        else:
            with path.open("r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        if len(rows) < 2:
            raise ValueError("Input file must contain at least 2 time steps.")

        keys = list(rows[0].keys())
        q_cols = sorted([k for k in keys if k.startswith("q_")], key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else x)
        p_cols = sorted([k for k in keys if k.startswith("p_")], key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else x)
        act_cols = sorted([k for k in keys if k.startswith("act_") or k.startswith("u_")], key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else x)

        # If custom CSV has arbitrary numeric columns (like ETT/SWaT/Turbine sensors), auto-assign them as q_cols
        if not q_cols:
            ignore = {"timestamp", "timestamp_step", "date", "time", "regime", "regime_label", "label", "anomaly"}
            numeric_cols = []
            for k in keys:
                if k.lower() not in ignore:
                    try:
                        float(rows[0][k])
                        numeric_cols.append(k)
                    except (ValueError, TypeError):
                        pass
            q_cols = numeric_cols

        num_nodes = len(q_cols)
        d_action = max(len(act_cols), 4)

        q_list, p_list, act_list, regimes = [], [], [], []
        has_p = len(p_cols) == num_nodes and num_nodes > 0

        for idx, r in enumerate(rows):
            q_row = [float(r[c]) for c in q_cols]
            q_list.append(q_row)
            if has_p:
                p_row = [float(r[c]) for c in p_cols]
            elif idx == 0:
                p_row = [0.0] * num_nodes
            else:
                prev_q = q_list[idx - 1]
                p_row = [(q_row[i] - prev_q[i]) / dt for i in range(num_nodes)]
            p_list.append(p_row)

            if act_cols:
                a_row = [float(r[c]) for c in act_cols]
            else:
                a_row = [0.0] * d_action
            act_list.append(a_row)
            regimes.append(int(float(r.get("regime_label", r.get("anomaly", 0)))))

        q_t = torch.tensor(q_list, dtype=torch.float32, device=device)
        p_t = torch.tensor(p_list, dtype=torch.float32, device=device)
        state_all = torch.cat([q_t, p_t], dim=-1).unsqueeze(0)
        act_all = torch.tensor(act_list, dtype=torch.float32, device=device).unsqueeze(0)

        return {
            "x_seq": state_all[:, :-1, :],
            "action_seq": act_all[:, :-1, :],
            "y_target": state_all[:, 1:, :],
            "regimes": torch.tensor(regimes[1:], dtype=torch.long, device=device),
            "num_nodes": num_nodes,
            "d_input": num_nodes * 2,
            "d_action": d_action,
            "sensor_columns": q_cols,
        }

    def _rows_to_tensors(
        self, rows: List[Dict[str, Any]], device: torch.device
    ) -> Dict[str, torch.Tensor]:
        if len(rows) < 2:
            raise ValueError("Telemetry log must contain at least 2 time steps.")

        q_list, p_list, act_list, regimes = [], [], [], []
        has_explicit_p = "p_0" in rows[0]

        for idx, r in enumerate(rows):
            q_row = [float(r.get(f"q_{i}", 0.0)) for i in range(self.num_nodes)]
            q_list.append(q_row)

            if has_explicit_p:
                p_row = [float(r.get(f"p_{i}", 0.0)) for i in range(self.num_nodes)]
            elif idx == 0:
                p_row = [0.0] * self.num_nodes
            else:
                prev_q = q_list[idx - 1]
                p_row = [(q_row[i] - prev_q[i]) / self.dt for i in range(self.num_nodes)]
            p_list.append(p_row)

            act_row = [float(r.get(f"act_{k}", 0.0)) for k in range(self.d_action)]
            act_list.append(act_row)
            regimes.append(int(r.get("regime_label", 0)))

        q_tensor = torch.tensor(q_list, dtype=torch.float32, device=device)
        p_tensor = torch.tensor(p_list, dtype=torch.float32, device=device)
        state_all = torch.cat([q_tensor, p_tensor], dim=-1).unsqueeze(0)
        act_all = torch.tensor(act_list, dtype=torch.float32, device=device).unsqueeze(0)

        return {
            "x_seq": state_all[:, :-1, :],
            "action_seq": act_all[:, :-1, :],
            "y_target": state_all[:, 1:, :],
            "regimes": torch.tensor(regimes[1:], dtype=torch.long, device=device),
        }
