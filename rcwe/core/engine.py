import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Tuple, Optional, Union

from rcwe.core.wave_codec import WavepacketCodec
from rcwe.core.resonance import HolographicResonanceLayer
from rcwe.core.symplectic import SymplecticCausalBlock
from rcwe.core.metabolic import MetabolicPlasticSynapse


class ResonantCausalWorldEngine(nn.Module):
    def __init__(
        self,
        d_input: int = 32,
        d_model: int = 64,
        d_action: int = 8,
        num_worldlines: int = 3,
        num_freq_bands: int = 16,
    ):
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.d_action = d_action
        self.num_worldlines = num_worldlines
        self.num_freq_bands = num_freq_bands

        self.codec = WavepacketCodec(d_input, d_model)
        self.resonance = HolographicResonanceLayer(d_model, num_freq_bands=num_freq_bands)
        self.worldlines = nn.ModuleList(
            [SymplecticCausalBlock(d_model, d_action) for _ in range(num_worldlines)]
        )
        self.branch_decoders = nn.ModuleList(
            [nn.Linear(d_model * 2, d_input) for _ in range(num_worldlines)]
        )
        self.metabolic_synapse = MetabolicPlasticSynapse(d_model, lr_local=0.015)
        self.register_buffer("fast_sensory_bias", torch.zeros(d_input))

    def reset_runtime_adaptation(self) -> None:
        self.fast_sensory_bias.zero_()
        self.metabolic_synapse.fast_weight.zero_()
        self.metabolic_synapse.atp_budget.fill_(1.0)

    def save_pretrained(
        self,
        checkpoint_path: Union[str, Path],
        standing_field: Optional[torch.Tensor] = None,
    ) -> str:
        """
        v0.6.0 Full State Serialization.
        Persists slow weights, fast synaptic plastic weights, ATP metabolic budget,
        fast sensory bias, and the live O(1) complex standing wavefield |Psi|.
        """
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "d_input": self.d_input,
                "d_model": self.d_model,
                "d_action": self.d_action,
                "num_worldlines": self.num_worldlines,
                "num_freq_bands": self.num_freq_bands,
            },
            "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()},
            "standing_field_real": standing_field.real.detach().cpu() if standing_field is not None else None,
            "standing_field_imag": standing_field.imag.detach().cpu() if standing_field is not None else None,
        }
        torch.save(payload, path)
        return str(path.resolve())

    @classmethod
    def load_pretrained(
        cls,
        checkpoint_path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
    ) -> Tuple["ResonantCausalWorldEngine", Optional[torch.Tensor]]:
        """
        v0.6.0 Full State Deserialization.
        Restores model architecture, slow/fast weights, ATP budget, and O(1) standing wavefield.
        """
        dev = torch.device(device)
        payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
        cfg = payload["config"]
        model = cls(
            d_input=cfg["d_input"],
            d_model=cfg["d_model"],
            d_action=cfg["d_action"],
            num_worldlines=cfg["num_worldlines"],
            num_freq_bands=cfg["num_freq_bands"],
        )
        model.load_state_dict(payload["state_dict"])
        model.to(dev)

        sf = None
        if payload.get("standing_field_real") is not None and payload.get("standing_field_imag") is not None:
            sf_r = payload["standing_field_real"].to(dev)
            sf_i = payload["standing_field_imag"].to(dev)
            sf = torch.complex(sf_r, sf_i)
        return model, sf

    def forward(
        self,
        x_seq: torch.Tensor,
        action_seq: torch.Tensor,
        standing_field: Optional[torch.Tensor] = None,
        plastic_runtime: bool = True,
        y_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        z = self.codec.encode(x_seq)
        z_res, new_standing_field = self.resonance(z, standing_field)

        worldline_spl_r, worldline_spl_i = [], []
        branch_predictions = []
        energy_violations = []

        for k, branch in enumerate(self.worldlines):
            z_branch, e_viol = branch(z_res, action_seq)
            worldline_spl_r.append(z_branch.real)
            worldline_spl_i.append(z_branch.imag)
            delta_pred = self.branch_decoders[k](torch.cat([z_branch.real, z_branch.imag], dim=-1))
            branch_pred = x_seq + delta_pred
            branch_predictions.append(branch_pred)

            if y_target is not None:
                velocity_err = (((branch_pred - y_target) / 0.05) ** 2).mean(dim=-1)
                total_anomaly = velocity_err + 0.05 * e_viol
            else:
                total_anomaly = e_viol
            energy_violations.append(total_anomaly)

        stacked_r = torch.stack(worldline_spl_r, dim=0)
        stacked_i = torch.stack(worldline_spl_i, dim=0)
        stacked_preds = torch.stack(branch_predictions, dim=0)
        stacked_viol = torch.stack(energy_violations, dim=0)

        branch_weights = F.softmax(-stacked_viol.mean(dim=-1) * 4.0, dim=0)
        consensus_r = torch.einsum("kb,kbld->bld", branch_weights, stacked_r)
        consensus_i = torch.einsum("kb,kbld->bld", branch_weights, stacked_i)
        z_consensus = torch.complex(consensus_r, consensus_i)

        z_plastic = self.metabolic_synapse(z_consensus, plastic_runtime=(plastic_runtime and y_target is None))
        y_pred = x_seq + self.codec.decode(z_plastic) + self.fast_sensory_bias

        if plastic_runtime and y_target is not None:
            with torch.no_grad():
                sensory_err_seq = y_target - y_pred
                err_encoded = self.codec.encode(sensory_err_seq).real
                self.metabolic_synapse.local_metabolic_update(
                    pre_q=z_consensus.real, post_q=z_plastic.real, sensory_error_proj=err_encoded * 0.1
                )
                sensory_err = sensory_err_seq.mean(dim=(0, 1))
                gate = self.metabolic_synapse.atp_budget.mean().clamp(0.2, 1.5)
                self.fast_sensory_bias.add_(0.35 * gate * sensory_err)

        return {
            "output": y_pred,
            "branch_predictions": stacked_preds,
            "branch_weights": branch_weights,
            "standing_field": new_standing_field,
            "worldline_states": torch.complex(stacked_r, stacked_i),
            "worldline_violations": stacked_viol,
            "energy_violation": stacked_viol.mean(),
            "mean_atp": self.metabolic_synapse.atp_budget.mean(),
        }

    def forage_minimum_distinguishing_observation(
        self, z_current: torch.Tensor, candidate_actions: torch.Tensor
    ) -> Tuple[int, torch.Tensor]:
        num_candidates = candidate_actions.shape[0]
        zero_act = torch.zeros(
            z_current.shape[0], z_current.shape[1], candidate_actions.shape[-1], device=z_current.device
        )
        base_outputs = []
        for k, branch in enumerate(self.worldlines):
            z0, _ = branch(z_current, zero_act)
            base_outputs.append(self.branch_decoders[k](torch.cat([z0.real, z0.imag], dim=-1)))
        base_stack = torch.stack(base_outputs, dim=0)

        divergence_scores = []
        for c in range(num_candidates):
            act = candidate_actions[c].expand(z_current.shape[0], z_current.shape[1], -1)
            branch_outputs = []
            for k, branch in enumerate(self.worldlines):
                z_next, _ = branch(z_current, act)
                pred_k = self.branch_decoders[k](torch.cat([z_next.real, z_next.imag], dim=-1))
                branch_outputs.append(pred_k)
            delta_out = (torch.stack(branch_outputs, dim=0) - base_stack) / 0.005
            divergence_scores.append(delta_out.var(dim=0).mean())

        scores_tensor = torch.stack(divergence_scores)
        best_idx = int(torch.argmax(scores_tensor).item())
        return best_idx, scores_tensor
