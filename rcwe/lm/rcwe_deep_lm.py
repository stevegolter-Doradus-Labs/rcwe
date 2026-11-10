import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple

from rcwe.core.resonance import HolographicResonanceLayer
from rcwe.core.symplectic import SymplecticCausalBlock
from rcwe.core.metabolic import MetabolicPlasticSynapse
from rcwe.kernels.accelerator import select_best_device, sync_device


class DeepRCWEBlock(nn.Module):
    """Single hierarchical layer of RCWE-Deep with residual complex phase-space skip connections."""
    def __init__(self, d_model: int = 64, num_bands: int = 16):
        super().__init__()
        self.resonance = HolographicResonanceLayer(d_model=d_model, num_freq_bands=num_bands)
        self.symplectic = SymplecticCausalBlock(d_model=d_model, d_action=d_model, step_size=0.04)
        self.metabolic = MetabolicPlasticSynapse(d_model=d_model, lr_local=0.015)
        self.norm_r = nn.LayerNorm(d_model)
        self.norm_i = nn.LayerNorm(d_model)

    def forward(
        self,
        z: torch.Tensor,
        standing_field: Optional[torch.Tensor] = None,
        plastic_runtime: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_res, new_sf = self.resonance(z, standing_field)
        # Self-directed phase-space action from real component
        act_self = torch.tanh(z_res.real)
        z_sym, e_viol = self.symplectic(z_res, act_self)
        z_pl = self.metabolic(z_sym, plastic_runtime=plastic_runtime)

        out_r = self.norm_r(z.real + z_pl.real)
        out_i = self.norm_i(z.imag + z_pl.imag)
        return torch.complex(out_r, out_i), new_sf, e_viol


class RCWEDeepLanguageWorldModel(nn.Module):
    """
    Track C (v0.7.0): Hierarchical Multi-Layer Non-Autoregressive Language & Code World Model.
    Stacks N_layers of DeepRCWEBlock over byte/character token streams, maintaining a strictly
    constant O(1) standing wavefield state (N_layers * 16.0 KB) regardless of sequence length.
    """
    def __init__(self, vocab_size: int = 128, d_model: int = 64, num_layers: int = 4, num_bands: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_bands = num_bands

        self.amp_emb = nn.Embedding(vocab_size, d_model)
        self.phase_emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([DeepRCWEBlock(d_model=d_model, num_bands=num_bands) for _ in range(num_layers)])
        self.lm_head = nn.Linear(d_model * 2, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        standing_fields: Optional[List[torch.Tensor]] = None,
        plastic_runtime: bool = False,
    ) -> Dict[str, Any]:
        amp = F.softplus(self.amp_emb(input_ids))
        pha = torch.tanh(self.phase_emb(input_ids)) * 3.14159265
        z = torch.complex(amp * torch.cos(pha), amp * torch.sin(pha))

        new_fields: List[torch.Tensor] = []
        total_viol = []

        for idx, layer in enumerate(self.layers):
            sf_in = standing_fields[idx] if (standing_fields is not None and idx < len(standing_fields)) else None
            z, sf_out, ev = layer(z, standing_field=sf_in, plastic_runtime=plastic_runtime)
            new_fields.append(sf_out)
            total_viol.append(ev.mean())

        logits = self.lm_head(torch.cat([z.real, z.imag], dim=-1))
        total_psi_kb = sum((f.numel() * f.element_size()) / 1024.0 for f in new_fields)

        return {
            "logits": logits,
            "standing_fields": new_fields,
            "total_psi_kb": total_psi_kb,
            "mean_energy_violation": torch.stack(total_viol).mean(),
        }


class BaselineTransformerLM(nn.Module):
    def __init__(self, vocab_size: int = 128, d_model: int = 64, num_layers: int = 4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model * 2, dropout=0.0, batch_first=True
        )
        self.tf = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        L = input_ids.shape[1]
        h = self.emb(input_ids)
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=input_ids.device)
        return self.head(self.tf(h, mask=mask))


def run_deep_lm_benchmark(seed: int = 42, epochs: int = 45) -> Dict[str, Any]:
    torch.manual_seed(seed)
    device = select_best_device()

    # Structured algorithmic code stream corpus
    code_corpus = (
        "def conserve_flux(q, p, dt): "
        "p_half = p - 0.5 * dt * grad_V(q); "
        "q_next = q + dt * p_half; "
        "assert abs(sum(q_next) - sum(q)) < 1e-6; "
        "return q_next, p_half || "
    ) * 4

    raw_bytes = [min(ord(c), 127) for c in code_corpus]
    seq_tensor = torch.tensor(raw_bytes, dtype=torch.long, device=device).unsqueeze(0).expand(2, -1)
    x_ids = seq_tensor[:, :-1]
    y_ids = seq_tensor[:, 1:]

    rcwe_lm = RCWEDeepLanguageWorldModel(vocab_size=128, d_model=64, num_layers=4, num_bands=16).to(device)
    tf_lm = BaselineTransformerLM(vocab_size=128, d_model=64, num_layers=4).to(device)

    opt_r = torch.optim.AdamW(rcwe_lm.parameters(), lr=5e-3)
    opt_t = torch.optim.AdamW(tf_lm.parameters(), lr=5e-3)

    rcwe_lm.train()
    tf_lm.train()
    for _ in range(epochs):
        opt_r.zero_grad()
        opt_t.zero_grad()

        out_r = rcwe_lm(x_ids, plastic_runtime=False)
        loss_r = F.cross_entropy(out_r["logits"].reshape(-1, 128), y_ids.reshape(-1))
        loss_r.backward()
        torch.nn.utils.clip_grad_norm_(rcwe_lm.parameters(), 2.0)
        opt_r.step()

        logits_t = tf_lm(x_ids)
        loss_t = F.cross_entropy(logits_t.reshape(-1, 128), y_ids.reshape(-1))
        loss_t.backward()
        opt_t.step()

    sync_device(device)
    rcwe_lm.eval()
    tf_lm.eval()

    with torch.no_grad():
        res_r = rcwe_lm(x_ids, plastic_runtime=True)
        res_t = tf_lm(x_ids)
        acc_r = float((res_r["logits"].argmax(dim=-1) == y_ids).float().mean().item())
        acc_t = float((res_t.argmax(dim=-1) == y_ids).float().mean().item())
        psi_kb = float(res_r["total_psi_kb"])

    return {
        "layers": 4,
        "sequence_tokens": int(x_ids.shape[1]),
        "rcwe_deep_token_accuracy": round(acc_r, 4),
        "tf_deep_token_accuracy": round(acc_t, 4),
        "rcwe_deep_total_psi_kb": psi_kb,
    }
