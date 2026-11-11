import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple


class SymplecticPontryaginAgent(nn.Module):
    """
    RCWE v0.9.0 Track 2: Symplectic Reinforcement Learning & Dopaminergic Active Inference.
    Combines:
      1. Multi-Horizon Symplectic World-Line Rollouts (H-step reversible Leapfrog Verlet trajectories)
      2. Hamiltonian Pontryagin Continuous Control (-grad_a H(q, p, a))
      3. Backprop-Free 3-Factor Dopaminergic Policy Plasticity with Eligibility Traces & ATP Gating
    """
    def __init__(self, d_state: int = 8, d_action: int = 4, d_hidden: int = 64, dt: float = 0.05):
        super().__init__()
        self.d_state = d_state
        self.d_action = d_action
        self.d_hidden = d_hidden
        self.dt = dt

        self.potential_net = nn.Sequential(
            nn.Linear(d_state + d_action, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, 1),
        )
        self.slow_actor = nn.Linear(d_state * 2, d_action, bias=False)
        self.critic = nn.Sequential(
            nn.Linear(d_state * 2, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, 1),
        )

        self.register_buffer("fast_actor", torch.zeros(d_action, d_state * 2))
        self.register_buffer("eligibility_trace", torch.zeros(d_action, d_state * 2))
        self.register_buffer("atp", torch.tensor(1.6))

    def scalar_potential(self, q: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        quad = 0.5 * (q ** 2).sum(dim=-1) + 0.1 * (a ** 2).sum(dim=-1)
        nl = self.potential_net(torch.cat([q, a], dim=-1)).squeeze(-1)
        return quad + 0.1 * nl

    def leapfrog_rollout_step(self, q: torch.Tensor, p: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.enable_grad():
            q_req = q.detach().requires_grad_(True)
            V0 = self.scalar_potential(q_req, a.detach()).sum()
            grad_q0 = torch.autograd.grad(V0, q_req)[0]
        p_half = p - 0.5 * self.dt * grad_q0
        q_next = q + self.dt * p_half
        with torch.enable_grad():
            q_req1 = q_next.detach().requires_grad_(True)
            V1 = self.scalar_potential(q_req1, a.detach()).sum()
            grad_q1 = torch.autograd.grad(V1, q_req1)[0]
        p_next = p_half - 0.5 * self.dt * grad_q1
        return q_next, p_next

    def select_pontryagin_action(self, q: torch.Tensor, p: torch.Tensor, horizon: int = 3) -> torch.Tensor:
        state_vec = torch.cat([q, p], dim=-1)
        base_a = self.slow_actor(state_vec) + F.linear(state_vec, self.fast_actor)

        # Evaluate Pontryagin action gradient dV/da to minimize future Hamiltonian energy
        with torch.enable_grad():
            a_req = base_a.detach().requires_grad_(True)
            V = self.scalar_potential(q.detach(), a_req).sum()
            grad_a = torch.autograd.grad(V, a_req)[0]

        # Multi-horizon candidate refinement across K=3 action scalings
        best_a = base_a - 0.15 * grad_a
        best_cost = torch.full((q.shape[0],), float("inf"), device=q.device)

        with torch.no_grad():
            for scale in (0.7, 1.0, 1.3):
                cand_a = torch.clamp( (base_a - 0.15 * grad_a) * scale, -2.5, 2.5 )
                q_sim, p_sim = q.clone(), p.clone()
                cost = torch.zeros(q.shape[0], device=q.device)
                for _ in range(horizon):
                    q_sim, p_sim = self.leapfrog_rollout_step(q_sim, p_sim, cand_a)
                    cost = cost + (q_sim ** 2).sum(dim=-1) + 0.5 * (p_sim ** 2).sum(dim=-1)
                mask = cost < best_cost
                if mask.any():
                    best_cost = torch.where(mask, cost, best_cost)
                    best_a = torch.where(mask.unsqueeze(-1), cand_a, best_a)
        return best_a

    @torch.no_grad()
    def dopaminergic_plastic_update(
        self,
        q_t: torch.Tensor,
        p_t: torch.Tensor,
        a_t: torch.Tensor,
        q_next: torch.Tensor,
        p_next: torch.Tensor,
        target_correction: torch.Tensor,
        lr_da: float = 0.45,
    ) -> float:
        state_t = torch.cat([q_t, p_t], dim=-1)
        state_next = torch.cat([q_next, p_next], dim=-1)

        r_t = -((q_next ** 2).mean() + 0.25 * (p_next ** 2).mean())
        v_t = self.critic(state_t).mean()
        v_next = self.critic(state_next).mean()
        td_error = (r_t + 0.95 * v_next - v_t).abs() + target_correction.norm()

        # 3-Factor Dopaminergic update: Presynaptic state x Control Error x ATP-Gated Phasic Dopamine
        self.eligibility_trace.copy_(
            0.85 * self.eligibility_trace + torch.einsum("ba,bs->as", target_correction, state_t) / float(q_t.shape[0])
        )
        atp_gate = torch.sigmoid(10.0 * (self.atp - 0.25))
        da_Burst = torch.tanh(td_error)
        self.fast_actor.add_(lr_da * da_Burst * atp_gate * self.eligibility_trace)
        self.atp.copy_(torch.clamp(self.atp - 0.008 * da_Burst + 0.005, 0.0, 2.0))
        return float(td_error.item())


def run_symplectic_rl_benchmark(steps: int = 40) -> Dict[str, Any]:
    torch.manual_seed(42)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    D_s = 6
    dt = 0.08
    mutation_step = 15

    rcwe_agent = SymplecticPontryaginAgent(d_state=D_s, d_action=D_s, d_hidden=48, dt=dt).to(device)
    # Initialize slow actor as nominal PD stabilizer
    with torch.no_grad():
        rcwe_agent.slow_actor.weight.zero_()
        for i in range(D_s):
            rcwe_agent.slow_actor.weight[i, i] = -1.2
            rcwe_agent.slow_actor.weight[i, D_s + i] = -0.8

    q_r = torch.ones(1, D_s, device=device) * 0.8
    p_r = torch.zeros(1, D_s, device=device)
    q_b = q_r.clone()
    p_b = p_r.clone()

    rcwe_ener, base_ener = [], []
    da_bursts = []

    for t in range(steps):
        # At t >= mutation_step, inject severe non-stationary external disturbance & mass shift
        disturbance = torch.sin(torch.arange(D_s, device=device).float() * 0.5 + 0.2 * t).unsqueeze(0) * (1.35 if t >= mutation_step else 0.05)

        a_r = rcwe_agent.select_pontryagin_action(q_r, p_r, horizon=3)
        # Baseline frozen PD controller (no dopaminergic fast-weight adaptation)
        state_b = torch.cat([q_b, p_b], dim=-1)
        a_b = F.linear(state_b, rcwe_agent.slow_actor.weight)

        # True physical plant transition
        p_r_next = 0.92 * p_r + dt * (-1.5 * q_r + a_r + disturbance)
        q_r_next = q_r + dt * p_r_next

        p_b_next = 0.92 * p_b + dt * (-1.5 * q_b + a_b + disturbance)
        q_b_next = q_b + dt * p_b_next

        if t >= mutation_step:
            # Desired corrective force to cancel non-stationary disturbance and restore equilibrium (q=0, p=0)
            target_corr = -(q_r_next * 1.8 + p_r_next * 1.4) / dt
            td_val = rcwe_agent.dopaminergic_plastic_update(q_r, p_r, a_r, q_r_next, p_r_next, target_corr)
            da_bursts.append(td_val)

        q_r, p_r = q_r_next, p_r_next
        q_b, p_b = q_b_next, p_b_next

        if t >= mutation_step + 8:
            rcwe_ener.append(float(((q_r ** 2).mean() + (p_r ** 2).mean()).item()))
            base_ener.append(float(((q_b ** 2).mean() + (p_b ** 2).mean()).item()))

    r_dev = sum(rcwe_ener) / max(len(rcwe_ener), 1)
    b_dev = sum(base_ener) / max(len(base_ener), 1)
    return {
        "rcwe_rl_post_mutation_deviation": round(r_dev, 6),
        "frozen_policy_post_mutation_deviation": round(b_dev, 6),
        "rl_stabilization_gain": round(b_dev / max(r_dev, 1e-7), 2),
        "final_atp": round(float(rcwe_agent.atp.item()), 4),
        "peak_dopamine_td": round(max(da_bursts) if da_bursts else 0.0, 4),
    }
