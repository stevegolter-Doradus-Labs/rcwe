import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class CausalDiagnosticPacket:
    worldline_posteriors: List[float]
    worldline_violations: List[float]
    dominant_regime_id: int
    top_surprise_nodes: List[Tuple[int, float]]
    depleted_atp_nodes: List[Tuple[int, float]]
    mdo_action_idx: int
    mdo_target_nodes: List[int]
    mdo_divergence_score: float
    standing_field_coherence: float


class ConstrainedNeuralVerbalizer(nn.Module):
    def __init__(self, d_model: int = 64, vocab_size: int = 64):
        super().__init__()
        self.state_kv_proj = nn.Linear(d_model * 2, d_model)
        self.action_q_proj = nn.Linear(8, d_model)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        self.concept_head = nn.Linear(d_model, vocab_size)

    def forward(self, z_consensus: torch.Tensor, mdo_action: torch.Tensor) -> torch.Tensor:
        kv_feat = self.state_kv_proj(torch.cat([z_consensus.real, z_consensus.imag], dim=-1))
        q_feat = self.action_q_proj(mdo_action).unsqueeze(1)
        attn_out, _ = self.cross_attn(query=q_feat, key=kv_feat, value=kv_feat)
        return self.concept_head(attn_out.squeeze(1))


class CausalGraphNLCodec:
    REGIME_ONTOLOGY = {
        0: (
            "W0: Harmonic Upstream Micro-Burst",
            "Conservative oscillatory ingress injection detected; total Kirchhoff flux remains invariant.",
        ),
        1: (
            "W1: Covert Byzantine Sinkhole / Exfiltration Tap",
            "Non-conservative symplectic sink detected on interior relay nodes; flux vanishes across cut-sets.",
        ),
        2: (
            "W2: Symplectic Trunk Susceptance Degradation",
            "Conservative impedance shift along primary chord links; phase velocity is throttled without packet loss.",
        ),
    }

    def __init__(self, d_model: int = 64, num_nodes: int = 16):
        self.num_nodes = num_nodes
        self.neural_verbalizer = ConstrainedNeuralVerbalizer(d_model=d_model)

    @torch.no_grad()
    def extract_diagnostic_packet(
        self,
        rcwe_output: Dict[str, torch.Tensor],
        y_true: torch.Tensor,
        candidate_actions: torch.Tensor,
        mdo_idx: int,
        mdo_scores: torch.Tensor,
        atp_vector: torch.Tensor,
    ) -> CausalDiagnosticPacket:
        wl_viol = rcwe_output["worldline_violations"].mean(dim=(1, 2))
        posteriors = F.softmax(-wl_viol * 8.0, dim=0)
        dom_id = int(torch.argmax(posteriors).item())

        pred_q = rcwe_output["output"][:, -1, :self.num_nodes]
        true_q = y_true[:, -1, :self.num_nodes]
        node_err = (pred_q - true_q).abs().mean(dim=0)
        top_err_vals, top_err_idx = torch.topk(node_err, k=3)
        surprise_nodes = [(int(i.item()), float(v.item())) for i, v in zip(top_err_idx, top_err_vals)]

        node_atp = atp_vector[:self.num_nodes]
        low_atp_vals, low_atp_idx = torch.topk(-node_atp, k=3)
        depleted_nodes = [(int(i.item()), float(-v.item())) for i, v in zip(low_atp_idx, low_atp_vals)]

        winning_action = candidate_actions[mdo_idx]
        active_ctrl_nodes = torch.where(winning_action.abs() > 0.5)[0].tolist()
        if not active_ctrl_nodes:
            active_ctrl_nodes = [int(torch.argmax(winning_action.abs()).item())]

        sf_coherence = float(rcwe_output["standing_field"].abs().mean().item())

        return CausalDiagnosticPacket(
            worldline_posteriors=[round(float(p), 4) for p in posteriors],
            worldline_violations=[round(float(v), 5) for v in wl_viol],
            dominant_regime_id=dom_id,
            top_surprise_nodes=surprise_nodes,
            depleted_atp_nodes=depleted_nodes,
            mdo_action_idx=mdo_idx,
            mdo_target_nodes=active_ctrl_nodes,
            mdo_divergence_score=float(mdo_scores[mdo_idx].item()),
            standing_field_coherence=sf_coherence,
        )

    def render_engineering_advisory(self, pkt: CausalDiagnosticPacket) -> str:
        regime_name, regime_desc = self.REGIME_ONTOLOGY.get(
            pkt.dominant_regime_id, ("Unknown Regime", "Unclassified manifold trajectory.")
        )
        probs = [max(p, 1e-9) for p in pkt.worldline_posteriors]
        entropy = -sum(p * math.log2(p) for p in probs)
        ambiguity_ratio = (entropy / math.log2(len(probs))) * 100.0

        surprise_str = ", ".join([f"Node v_{idx} (|e|={val:.3f})" for idx, val in pkt.top_surprise_nodes])
        atp_str = ", ".join([f"Channel c_{idx} (ATP={val:.2f}J)" for idx, val in pkt.depleted_atp_nodes])
        target_str = ", ".join([f"v_{idx}" for idx in pkt.mdo_target_nodes])

        status_flag = (
            "AMBIGUOUS LOOK-ALIKE SUPERPOSITION (ACTIVE MDO INTERVENTION REQUIRED)"
            if ambiguity_ratio > 80.0
            else "CONVERGED CAUSAL IDENTIFICATION"
        )

        lines = [
            "================================================================================",
            f"[RCWE CAUSAL DIAGNOSTIC ADVISORY] | STATUS: {status_flag}",
            "================================================================================",
            f"1. SYMPLECTIC WORLD-LINE SUPERPOSITION (Epistemic Ambiguity: {ambiguity_ratio:.1f}%)",
            f"   - Posterior Weights (pi_k) : W0={pkt.worldline_posteriors[0]:.3f} | W1={pkt.worldline_posteriors[1]:.3f} | W2={pkt.worldline_posteriors[2]:.3f}",
            f"   - Residual Anomaly (E_k)   : W0={pkt.worldline_violations[0]:.4f} | W1={pkt.worldline_violations[1]:.4f} | W2={pkt.worldline_violations[2]:.4f}",
            f"   - Leading Hypothesis       : {regime_name}",
            f"   - Physical Interpretation  : {regime_desc}",
            "",
            "2. SPATIAL TOPOLOGY & METABOLIC PLASTICITY TELEMETRY",
            f"   - Standing Wavefield |Psi| : {pkt.standing_field_coherence:.4f} (O(1) Phase-Locked Buffer)",
            f"   - Peak Predictive Surprise : {surprise_str}",
            f"   - Fast Synaptic Rewiring   : {atp_str} (Active 3-Factor Adaptation)",
            "",
            "3. ACTIVE EPISTEMIC FORAGING PRESCRIPTION (MINIMUM DISTINGUISHING OBSERVATION)",
            f"   - Recommended Intervention : Execute do(Action #{pkt.mdo_action_idx}) on Control Nodes [{target_str}]",
            f"   - Counterfactual Leverage  : Expected World-Line Variance Split = {pkt.mdo_divergence_score:.5f}",
            "================================================================================",
        ]
        return "\n".join(lines)
