import json
import math
import time
import platform
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, List, Optional

from rcwe.core.engine import ResonantCausalWorldEngine
from rcwe.sandbox.cpfn_env import CyberPhysicalFlowSandbox
from rcwe.sandbox.baseline_tf import BaselineCausalTransformer
from rcwe.diagnostics.nl_codec import CausalGraphNLCodec
from rcwe.diagnostics.profiler import HardwareProfilingHarness
from rcwe.kernels.accelerator import select_best_device, describe_accelerator, sync_device
from rcwe.paths import resolve_output


class ScientificValidationBenchmark:
    def __init__(self, seed: int = 42, prefer_device: Optional[str] = None):
        self.seed = seed
        self.device = select_best_device(prefer_device)
        self._init_models(seed)

    def _init_models(self, seed: int) -> None:
        torch.manual_seed(seed)
        self.sandbox = CyberPhysicalFlowSandbox(num_nodes=16, dt=0.05, device=self.device)
        self.rcwe = ResonantCausalWorldEngine(d_input=32, d_model=64, d_action=8, num_worldlines=3).to(self.device)
        self.baseline_tf = BaselineCausalTransformer(d_input=32, d_model=64, d_action=8, nhead=4, num_layers=2).to(self.device)
        self.codec = CausalGraphNLCodec(d_model=64, num_nodes=16)

    @staticmethod
    def count_parameters(model: torch.nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def compute_ci95(values: List[float]) -> Dict[str, float]:
        n = max(len(values), 1)
        mean_val = sum(values) / n
        if n < 2:
            return {"mean": mean_val, "std": 0.0, "ci95_halfwidth": 0.0, "ci95_low": mean_val, "ci95_high": mean_val}
        var_val = sum((v - mean_val) ** 2 for v in values) / (n - 1)
        std_val = math.sqrt(var_val)
        hw = 1.96 * (std_val / math.sqrt(n))
        return {
            "mean": mean_val,
            "std": std_val,
            "ci95_halfwidth": hw,
            "ci95_low": mean_val - hw,
            "ci95_high": mean_val + hw,
        }

    def _collect_regime_rollout(self, batch_size: int, seq_len: int, regime: int):
        obs = self.sandbox.reset(batch_size=batch_size, worldline_regime=regime)
        xs, acts, ys, energies = [], [], [], []
        for _ in range(seq_len):
            act = torch.randn(batch_size, 8, device=self.device) * 0.3
            next_obs, info = self.sandbox.step(act)
            xs.append(obs)
            acts.append(act)
            ys.append(next_obs)
            energies.append(info["true_energy"])
            obs = next_obs
        return (
            torch.stack(xs, dim=1),
            torch.stack(acts, dim=1),
            torch.stack(ys, dim=1),
            torch.stack(energies, dim=1),
        )

    def train_and_calibrate(self, epochs: int = 55) -> Dict[str, float]:
        self.rcwe.train()
        self.baseline_tf.train()
        opt_rcwe = torch.optim.AdamW(self.rcwe.parameters(), lr=4e-3)
        opt_tf = torch.optim.AdamW(self.baseline_tf.parameters(), lr=4e-3)

        rcwe_final_loss = 0.0
        tf_final_loss = 0.0

        for ep in range(epochs):
            opt_rcwe.zero_grad()
            opt_tf.zero_grad()

            total_rcwe_loss = torch.tensor(0.0, device=self.device)
            total_tf_loss = torch.tensor(0.0, device=self.device)

            for regime_id in range(3):
                x_seq, a_seq, y_seq, _ = self._collect_regime_rollout(batch_size=6, seq_len=10, regime=regime_id)
                out = self.rcwe(x_seq, a_seq, plastic_runtime=False, y_target=y_seq)

                true_vel = (y_seq - x_seq) / 0.05
                pred_vel_consensus = (out["output"] - x_seq) / 0.05
                consensus_loss = F.mse_loss(pred_vel_consensus, true_vel)

                branch_k_vel = (out["branch_predictions"][regime_id] - x_seq) / 0.05
                spec_loss = F.mse_loss(branch_k_vel, true_vel)

                other_losses = []
                for other_k in range(3):
                    if other_k != regime_id:
                        other_vel = (out["branch_predictions"][other_k] - x_seq) / 0.05
                        other_mse = F.mse_loss(other_vel, true_vel)
                        other_losses.append(F.relu(0.45 - (other_mse - spec_loss)))
                margin_loss = torch.stack(other_losses).mean()

                total_rcwe_loss = total_rcwe_loss + consensus_loss + spec_loss + 0.8 * margin_loss

                tf_pred = self.baseline_tf(x_seq, a_seq)
                total_tf_loss = total_tf_loss + F.mse_loss((tf_pred - x_seq) / 0.05, true_vel)

            total_rcwe_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.rcwe.parameters(), 2.0)
            opt_rcwe.step()

            total_tf_loss.backward()
            opt_tf.step()

            rcwe_final_loss = float(total_rcwe_loss.item() / 3.0)
            tf_final_loss = float(total_tf_loss.item() / 3.0)

        sync_device(self.device)
        return {"rcwe_train_loss": rcwe_final_loss, "tf_train_loss": tf_final_loss}

    @torch.no_grad()
    def evaluate_topology_mutation_and_conservation(self) -> Dict[str, float]:
        self.rcwe.eval()
        self.baseline_tf.eval()
        self.rcwe.reset_runtime_adaptation()
        obs = self.sandbox.reset(batch_size=4, worldline_regime=1)
        standing_field = None
        rcwe_mse_list, tf_mse_list = [], []
        rcwe_env_drift, tf_env_drift = [], []

        disturbance = torch.zeros(4, 1, 32, device=self.device)
        disturbance[:, :, :16] = 0.40

        for step_i in range(20):
            act = torch.randn(4, 1, 8, device=self.device) * 0.3
            x_curr = obs.unsqueeze(1)
            next_obs, _ = self.sandbox.step(act.squeeze(1))
            y_true = next_obs.unsqueeze(1) + disturbance

            rcwe_out = self.rcwe(x_curr, act, standing_field=standing_field, plastic_runtime=True, y_target=y_true)
            standing_field = rcwe_out["standing_field"]
            y_rcwe = rcwe_out["output"]
            y_tf = self.baseline_tf(x_curr, act)

            rcwe_mse_list.append(float(F.mse_loss(y_rcwe, y_true).item()))
            tf_mse_list.append(float(F.mse_loss(y_tf, y_true).item()))

            N = self.sandbox.N
            H_true = self.sandbox.compute_true_hamiltonian(y_true[:, 0, :N], y_true[:, 0, N:])
            H_rcwe = self.sandbox.compute_true_hamiltonian(y_rcwe[:, 0, :N], y_rcwe[:, 0, N:])
            H_tf = self.sandbox.compute_true_hamiltonian(y_tf[:, 0, :N], y_tf[:, 0, N:])

            rcwe_env_drift.append(float(torch.abs(H_rcwe - H_true).mean().item()))
            tf_env_drift.append(float(torch.abs(H_tf - H_true).mean().item()))
            obs = next_obs

        self.rcwe.metabolic_synapse.consolidate_sleep_phase()
        self.rcwe.reset_runtime_adaptation()

        return {
            "rcwe_post_mutation_mse": sum(rcwe_mse_list[-5:]) / 5.0,
            "tf_post_mutation_mse": sum(tf_mse_list[-5:]) / 5.0,
            "rcwe_hamiltonian_drift": sum(rcwe_env_drift) / len(rcwe_env_drift),
            "tf_hamiltonian_drift": sum(tf_env_drift) / len(tf_env_drift),
        }

    @torch.no_grad()
    def evaluate_causal_regime_identification(self, episodes_per_regime: int = 5) -> Dict[str, Any]:
        self.rcwe.eval()
        self.baseline_tf.eval()
        correct = 0
        total = 0

        for true_regime in range(3):
            for _ in range(episodes_per_regime):
                x_seq, a_seq, y_seq, _ = self._collect_regime_rollout(batch_size=2, seq_len=8, regime=true_regime)
                out = self.rcwe(x_seq, a_seq, plastic_runtime=False, y_target=y_seq)
                wl_viol = out["worldline_violations"].mean(dim=(1, 2))
                pred_regime = int(torch.argmin(wl_viol).item())
                if pred_regime == true_regime:
                    correct += 1
                total += 1

        x_s, a_s, y_s, _ = self._collect_regime_rollout(batch_size=2, seq_len=8, regime=1)
        rcwe_out = self.rcwe(x_s, a_s, plastic_runtime=True, y_target=y_s)

        candidates = torch.randn(6, 8, device=self.device) * 0.05
        candidates[4, 5:7] = 2.5
        z_enc = self.rcwe.codec.encode(x_s)
        mdo_idx, mdo_scores = self.rcwe.forage_minimum_distinguishing_observation(z_enc, candidates)

        pkt = self.codec.extract_diagnostic_packet(
            rcwe_output=rcwe_out,
            y_true=y_s,
            candidate_actions=candidates,
            mdo_idx=mdo_idx,
            mdo_scores=mdo_scores,
            atp_vector=self.rcwe.metabolic_synapse.atp_budget,
        )

        return {
            "regime_classification_accuracy": correct / max(total, 1),
            "diagnostic_packet": pkt,
            "advisory_text": self.codec.render_engineering_advisory(pkt),
        }

    def run_multiseed_validation(
        self,
        seeds: Optional[List[int]] = None,
        epochs: int = 55,
        output_json_path: Optional[str] = None,
        seq_lengths: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        output_json_path = resolve_output(output_json_path, "rcwe_validation_report_v03.json")
        if seeds is None:
            seeds = [42, 101, 202]
        t0 = time.perf_counter()
        scaling_rows = HardwareProfilingHarness.run_scaling_sweep(
            d_model=64, batch_size=2, seq_lengths=seq_lengths, prefer_device=str(self.device)
        )

        rcwe_mses, tf_mses = [], []
        rcwe_drifts, tf_drifts = [], []
        accuracies, w1_posteriors = [], []
        last_causal_eval = None
        last_train_metrics = None

        for s in seeds:
            self._init_models(s)
            last_train_metrics = self.train_and_calibrate(epochs=epochs)
            mut = self.evaluate_topology_mutation_and_conservation()
            ceval = self.evaluate_causal_regime_identification(episodes_per_regime=5)

            rcwe_mses.append(mut["rcwe_post_mutation_mse"])
            tf_mses.append(mut["tf_post_mutation_mse"])
            rcwe_drifts.append(mut["rcwe_hamiltonian_drift"])
            tf_drifts.append(mut["tf_hamiltonian_drift"])
            accuracies.append(ceval["regime_classification_accuracy"])
            w1_posteriors.append(ceval["diagnostic_packet"].worldline_posteriors[1])
            last_causal_eval = ceval

        total_time_s = time.perf_counter() - t0
        pkt = last_causal_eval["diagnostic_packet"]

        multiseed_stats = {
            "seeds": seeds,
            "rcwe_post_mutation_mse": self.compute_ci95(rcwe_mses),
            "tf_post_mutation_mse": self.compute_ci95(tf_mses),
            "rcwe_hamiltonian_drift": self.compute_ci95(rcwe_drifts),
            "tf_hamiltonian_drift": self.compute_ci95(tf_drifts),
            "regime_id_accuracy": self.compute_ci95(accuracies),
            "w1_posterior_confidence": self.compute_ci95(w1_posteriors),
        }

        report = {
            "version": "0.3.0",
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "accelerator": describe_accelerator(self.device),
                "device_type": self.device.type,
                "rcwe_params": self.count_parameters(self.rcwe),
                "baseline_tf_params": self.count_parameters(self.baseline_tf),
                "wall_time_sec": round(total_time_s, 3),
            },
            "hardware_scaling": scaling_rows,
            "training_calibration": last_train_metrics,
            "head_to_head_mutation_benchmark": {
                "rcwe_post_mutation_mse": multiseed_stats["rcwe_post_mutation_mse"]["mean"],
                "tf_post_mutation_mse": multiseed_stats["tf_post_mutation_mse"]["mean"],
                "rcwe_hamiltonian_drift": multiseed_stats["rcwe_hamiltonian_drift"]["mean"],
                "tf_hamiltonian_drift": multiseed_stats["tf_hamiltonian_drift"]["mean"],
            },
            "multiseed_statistics": multiseed_stats,
            "causal_identification": {
                "accuracy": multiseed_stats["regime_id_accuracy"]["mean"],
                "regime_1_posteriors": pkt.worldline_posteriors,
                "regime_1_residuals": pkt.worldline_violations,
                "dominant_regime_id": pkt.dominant_regime_id,
                "mdo_action_idx": pkt.mdo_action_idx,
                "mdo_target_nodes": pkt.mdo_target_nodes,
                "mdo_divergence_score": pkt.mdo_divergence_score,
            },
        }

        Path(output_json_path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        report["advisory_text"] = last_causal_eval["advisory_text"]
        report["json_path"] = str(Path(output_json_path).resolve())
        return report

    def run_full_validation(
        self,
        output_json_path: Optional[str] = None,
        seq_lengths: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        return self.run_multiseed_validation(
            seeds=[self.seed],
            epochs=55,
            output_json_path=output_json_path,
            seq_lengths=seq_lengths,
        )
