import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List


class DistributedWavefieldSwarm(nn.Module):
    """
    RCWE v0.9.0 Track 3: Distributed Multi-Agent Wavefield Swarm Consensus.
    Connects N_agents regional RCWE world models over a sparse graph topology.
    Agents exchange ONLY their O(1) standing wavefields Psi_i (8.0 KB per agent)
    via phase-coherent Laplacian gossip—achieving preemptive cascading-fault immunity
    without streaming raw high-frequency telemetry.
    """
    def __init__(self, num_agents: int = 4, d_node: int = 8, d_model: int = 64, num_bands: int = 16):
        super().__init__()
        self.num_agents = num_agents
        self.d_node = d_node
        self.d_model = d_model
        self.num_bands = num_bands

        self.enc_proj = nn.Linear(d_node, d_model)
        self.dec_proj = nn.Linear(d_model * 2, d_node)
        nn.init.zeros_(self.dec_proj.weight)
        nn.init.zeros_(self.dec_proj.bias)

        self.register_buffer("fast_swarm_weights", torch.zeros(num_agents, d_node, d_model * 2))
        self.register_buffer("psi_swarm", torch.zeros(num_agents, num_bands, d_model, dtype=torch.cfloat))

        adj = torch.zeros(num_agents, num_agents)
        for i in range(num_agents):
            adj[i, (i - 1) % num_agents] = 0.5
            adj[i, (i + 1) % num_agents] = 0.5
        self.register_buffer("adjacency", adj)

    @torch.no_grad()
    def gossip_wavefields(self, coupling_alpha: float = 0.35) -> float:
        neighbor_psi = torch.einsum("ij,jbd->ibd", self.adjacency.to(torch.cfloat), self.psi_swarm)
        self.psi_swarm.copy_((1.0 - coupling_alpha) * self.psi_swarm + coupling_alpha * neighbor_psi)
        per_agent_kb = (self.psi_swarm[0].numel() * self.psi_swarm[0].element_size()) / 1024.0
        return float(per_agent_kb)

    def step_swarm(
        self,
        x_t: torch.Tensor,
        y_target: torch.Tensor,
        active_fault_mask: torch.Tensor,
        enable_swarm_gossip: bool = True,
    ) -> torch.Tensor:
        h = self.enc_proj(x_t)
        z_t = torch.polar(F.softplus(h), torch.tanh(h) * math.pi)

        with torch.no_grad():
            self.psi_swarm.copy_(0.85 * self.psi_swarm + 0.15 * z_t.unsqueeze(1))
            if enable_swarm_gossip:
                self.gossip_wavefields(coupling_alpha=0.45)
                # Consensus diffusion of fast plastic weights from fault-experienced agents to neighbors
                neighbor_fast = torch.einsum("ij,jdf->idf", self.adjacency, self.fast_swarm_weights)
                self.fast_swarm_weights.copy_(0.50 * self.fast_swarm_weights + 0.50 * neighbor_fast)

        psi_mean = self.psi_swarm.mean(dim=1)
        feat = torch.cat([(z_t + psi_mean).real, (z_t + psi_mean).imag], dim=-1)
        fast_out = torch.einsum("if,idf->id", feat, self.fast_swarm_weights)
        y_hat = x_t + self.dec_proj(feat) + fast_out

        with torch.no_grad():
            err = (y_target - y_hat) * active_fault_mask.unsqueeze(-1)
            feat_norm_sq = (feat ** 2).sum(dim=-1, keepdim=True).clamp(min=1e-4)
            norm_feat = feat / feat_norm_sq
            hebb = torch.einsum("id,if->idf", err, norm_feat)
            self.fast_swarm_weights.add_(0.80 * hebb)

        return y_hat


def run_distributed_swarm_benchmark(steps: int = 30) -> Dict[str, Any]:
    torch.manual_seed(42)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    N_agents, D_n = 4, 8

    swarm_coupled = DistributedWavefieldSwarm(num_agents=N_agents, d_node=D_n, d_model=64, num_bands=16).to(device)
    swarm_isolated = DistributedWavefieldSwarm(num_agents=N_agents, d_node=D_n, d_model=64, num_bands=16).to(device)
    swarm_isolated.load_state_dict(swarm_coupled.state_dict())

    coupled_errors, isolated_errors = [], []

    for t in range(steps):
        base_sig = 0.3 * torch.sin(torch.linspace(0, 3.14, D_n, device=device) + 0.15 * t).unsqueeze(0).expand(N_agents, D_n)
        fault_offset = torch.zeros(N_agents, D_n, device=device)
        active_mask = torch.zeros(N_agents, device=device)

        # Cascading substation fault hits Agent 0 at t=5, then cascades to Agents 1, 2, 3 at t=14
        if t >= 5:
            fault_offset[0] = 1.45
            active_mask[0] = 1.0
        if t >= 14:
            fault_offset[1:] = 1.45
            active_mask[1:] = 1.0

        x_t = base_sig
        y_true = base_sig + fault_offset

        y_c = swarm_coupled.step_swarm(x_t, y_true, active_mask, enable_swarm_gossip=True)
        y_i = swarm_isolated.step_swarm(x_t, y_true, active_mask, enable_swarm_gossip=False)

        # Evaluate first-impact cascading fault error on downstream substations (Agents 1..3) at t=14, 15
        if t in (14, 15):
            coupled_errors.append(float(F.mse_loss(y_c[1:], y_true[1:]).item()))
            isolated_errors.append(float(F.mse_loss(y_i[1:], y_true[1:]).item()))

    c_mse = sum(coupled_errors) / max(len(coupled_errors), 1)
    i_mse = sum(isolated_errors) / max(len(isolated_errors), 1)
    per_agent_kb = swarm_coupled.gossip_wavefields(0.0)

    return {
        "num_agents": N_agents,
        "per_agent_wavefield_kb": per_agent_kb,
        "total_swarm_wavefield_kb": per_agent_kb * N_agents,
        "swarm_coupled_cascade_mse": round(c_mse, 6),
        "isolated_agents_cascade_mse": round(i_mse, 6),
        "swarm_consensus_gain": round(i_mse / max(c_mse, 1e-7), 2),
    }
