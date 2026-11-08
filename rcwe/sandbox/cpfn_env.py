import math
import torch
from typing import Dict, Tuple, Optional, Union


class CyberPhysicalFlowSandbox:
    def __init__(self, num_nodes: int = 16, dt: float = 0.05, device: Union[str, torch.device] = "cpu"):
        self.N = num_nodes
        self.dt = dt
        self.device = torch.device(device)

        B = torch.zeros(num_nodes, num_nodes, device=self.device)
        for i in range(num_nodes):
            B[i, (i + 1) % num_nodes] = 1.5
            B[(i + 1) % num_nodes, i] = 1.5
            B[i, (i + 4) % num_nodes] = 0.6
            B[(i + 4) % num_nodes, i] = 0.6
        self.base_B = B
        self.reset(worldline_regime=0)

    def reset(self, batch_size: int = 4, worldline_regime: int = 0) -> torch.Tensor:
        self.regime = worldline_regime
        self.t_step = 0
        self.B = self.base_B.clone()
        self.sink_mask = torch.zeros(self.N, device=self.device)

        if worldline_regime == 1:
            self.sink_mask[5:7] = 0.45
        elif worldline_regime == 2:
            self.B[4, 8] *= 0.35
            self.B[8, 4] *= 0.35

        self.q = torch.randn(batch_size, self.N, device=self.device) * 0.25
        self.p = torch.randn(batch_size, self.N, device=self.device) * 0.25
        return self._pack_obs()

    def _pack_obs(self) -> torch.Tensor:
        return torch.cat([self.q, self.p], dim=-1)

    def compute_true_hamiltonian(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        kinetic = 0.5 * (p ** 2).sum(dim=-1)
        diff = q.unsqueeze(2) - q.unsqueeze(1)
        coupling_pot = 0.25 * (self.B.unsqueeze(0) * (diff ** 2)).sum(dim=(1, 2))
        self_pot = 0.1 * (q ** 2).sum(dim=-1)
        return kinetic + coupling_pot + self_pot

    def step(
        self, action: torch.Tensor, do_clamp_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B_size = self.q.shape[0]
        self.t_step += 1

        u_inject = torch.zeros(B_size, self.N, device=self.device)
        u_inject[:, :action.shape[-1]] = action * 0.25

        if self.regime == 0:
            u_inject[:, 0:3] += 1.40 + 0.35 * math.sin(0.5 * self.t_step)
        elif self.regime == 1:
            u_inject[:, 5:7] -= 1.60
        elif self.regime == 2:
            u_inject[:, 4] += 1.35
            u_inject[:, 8] -= 1.35

        laplacian_force = lambda q_s: torch.matmul(q_s, self.B) - q_s * self.B.sum(dim=-1) - 0.2 * q_s

        f0 = laplacian_force(self.q) + u_inject - self.q * self.sink_mask
        p_half = self.p + 0.5 * self.dt * f0

        q_next = self.q + self.dt * p_half
        if do_clamp_mask is not None:
            q_next = q_next * do_clamp_mask + (1.0 - do_clamp_mask) * u_inject

        f1 = laplacian_force(q_next) + u_inject - q_next * self.sink_mask
        p_next = p_half + 0.5 * self.dt * f1

        self.q, self.p = q_next, p_next
        true_energy = self.compute_true_hamiltonian(self.q, self.p)
        return self._pack_obs(), {"true_energy": true_energy, "regime": torch.tensor(self.regime, device=self.device)}
