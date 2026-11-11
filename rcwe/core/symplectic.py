import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SymplecticCausalBlock(nn.Module):
    def __init__(self, d_model: int, d_action: int, step_size: float = 0.05):
        super().__init__()
        self.d_model = d_model
        self.d_action = d_action
        self.dt = step_size

        self.q_coupling = nn.Linear(d_model, d_model, bias=False)
        self.a_coupling = nn.Linear(d_action, d_model, bias=False) if d_action > 0 else nn.Linear(8, d_model, bias=False)
        self.potential_net = nn.Sequential(
            nn.Linear(d_model + max(d_action, 8), d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, 1),
        )
        self.log_inv_mass = nn.Parameter(torch.zeros(d_model))

    def _scalar_potential(self, q_intervened: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        in_f_action = action.shape[-1]
        in_f_q = q_intervened.shape[-1]

        if in_f_q != self.q_coupling.in_features:
            device = action.device
            new_qc = nn.Linear(in_f_q, self.q_coupling.out_features, bias=False).to(device)
            nn.init.zeros_(new_qc.weight)
            new_qc.weight.data[:, :min(in_f_q, self.q_coupling.in_features)] = self.q_coupling.weight.data[:, :min(in_f_q, self.q_coupling.in_features)]
            self.q_coupling = new_qc

        if in_f_action != self.a_coupling.in_features:
            device = action.device
            new_ac = nn.Linear(in_f_action, self.a_coupling.out_features, bias=False).to(device)
            nn.init.zeros_(new_ac.weight)
            new_ac.weight.data[:, :min(in_f_action, self.a_coupling.in_features)] = self.a_coupling.weight.data[:, :min(in_f_action, self.a_coupling.in_features)]
            self.a_coupling = new_ac
            
        old_linear = self.potential_net[0]
        if old_linear.in_features != in_f_q + in_f_action:
            device = action.device
            new_linear = nn.Linear(in_f_q + in_f_action, old_linear.out_features).to(device)
            nn.init.zeros_(new_linear.weight)
            new_linear.weight.data[:, :min(in_f_q + in_f_action, old_linear.in_features)] = old_linear.weight.data[:, :min(in_f_q + in_f_action, old_linear.in_features)]
            new_linear.bias.data = old_linear.bias.data
            self.potential_net[0] = new_linear
            
        coupled = self.q_coupling(q_intervened) + self.a_coupling(action)
        quad_pot = 0.5 * (coupled ** 2).mean(dim=-1)
        nl_pot = self.potential_net(torch.cat([q_intervened, action], dim=-1)).squeeze(-1)
        return quad_pot + nl_pot

    def compute_potential_and_force(
        self, q: torch.Tensor, action: torch.Tensor, causal_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.enable_grad():
            q_req = q.detach().requires_grad_(True)
            q_intervened = q_req * causal_mask
            V = self._scalar_potential(q_intervened, action).sum()
            grad_q = torch.autograd.grad(V, q_req, create_graph=self.training)[0]
        return V, grad_q

    def hamiltonian_energy(
        self, q: torch.Tensor, p: torch.Tensor, action: torch.Tensor, causal_mask: torch.Tensor
    ) -> torch.Tensor:
        inv_mass = F.softplus(self.log_inv_mass)
        kinetic = 0.5 * torch.sum((p ** 2) * inv_mass, dim=-1)
        q_intervened = q * causal_mask
        potential = self._scalar_potential(q_intervened, action)
        return kinetic + potential

    def leapfrog_step(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        action: torch.Tensor,
        causal_mask: torch.Tensor,
        dt: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        step_dt = self.dt if dt is None else dt
        inv_mass = F.softplus(self.log_inv_mass)

        _, grad_q0 = self.compute_potential_and_force(q, action, causal_mask)
        p_half = p - 0.5 * step_dt * grad_q0
        q_next = q + step_dt * (p_half * inv_mass)
        _, grad_q1 = self.compute_potential_and_force(q_next, action, causal_mask)
        p_next = p_half - 0.5 * step_dt * grad_q1
        return q_next, p_next

    def forward(
        self, z: torch.Tensor, action: torch.Tensor, causal_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q, p = z.real, z.imag
        if causal_mask is None:
            causal_mask = torch.ones_like(q)

        H_init = self.hamiltonian_energy(q, p, action, causal_mask)
        q_next, p_next = self.leapfrog_step(q, p, action, causal_mask)
        H_final = self.hamiltonian_energy(q_next, p_next, action, causal_mask)

        energy_violation = torch.abs(H_final - H_init)
        return torch.complex(q_next, p_next), energy_violation
