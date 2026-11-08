import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MetabolicPlasticSynapse(nn.Module):
    def __init__(self, d_model: int, lr_local: float = 0.01):
        super().__init__()
        self.d_model = d_model
        self.lr_local = lr_local

        self.slow_weight = nn.Linear(d_model, d_model, bias=False)
        self.predictor = nn.Linear(d_model, d_model, bias=False)

        self.register_buffer("fast_weight", torch.zeros(d_model, d_model))
        self.register_buffer("atp_budget", torch.ones(d_model))
        self.register_buffer("prev_error_norm", torch.tensor(1.0))

    @torch.no_grad()
    def local_metabolic_update(
        self,
        pre_q: torch.Tensor,
        post_q: torch.Tensor,
        sensory_error_proj: Optional[torch.Tensor] = None,
    ) -> None:
        pred_q = self.predictor(post_q)
        local_error = (post_q - pred_q) if sensory_error_proj is None else sensory_error_proj
        current_err_norm = local_error.norm(dim=-1).mean()

        firing_cost = 0.02 * (post_q ** 2).mean(dim=(0, 1))
        compression_reward = 0.1 * F.relu(self.prev_error_norm - current_err_norm)
        self.atp_budget.add_(0.01 + compression_reward - firing_cost).clamp_(0.0, 2.0)
        self.prev_error_norm.copy_(current_err_norm)

        metabolic_gate = torch.sigmoid(10.0 * (self.atp_budget - 0.25))
        surprise_modulator = torch.tanh(current_err_norm)

        norm_factor = max(pre_q.shape[0] * pre_q.shape[1], 1)
        hebbian_outer = torch.einsum("bli,blj->ij", local_error, pre_q) / norm_factor
        delta_w = self.lr_local * surprise_modulator * (metabolic_gate.unsqueeze(1) * hebbian_outer)
        self.fast_weight.mul_(0.995).add_(delta_w)

    @torch.no_grad()
    def consolidate_sleep_phase(self, consolidation_rate: float = 0.05) -> None:
        active_mask = (self.atp_budget > 0.5).float().unsqueeze(1)
        self.slow_weight.weight.add_(consolidation_rate * (self.fast_weight * active_mask))
        self.fast_weight.zero_()

    def forward(
        self,
        z: torch.Tensor,
        plastic_runtime: bool = True,
        sensory_error_proj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = z.real
        gate = torch.sigmoid(10.0 * (self.atp_budget - 0.25))
        effective_weight = self.slow_weight.weight + self.fast_weight
        q_out = F.linear(q, effective_weight) * gate

        if plastic_runtime:
            self.local_metabolic_update(pre_q=q, post_q=q_out, sensory_error_proj=sensory_error_proj)

        return torch.complex(q_out, z.imag)
