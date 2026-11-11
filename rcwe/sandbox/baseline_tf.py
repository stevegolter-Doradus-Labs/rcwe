import torch
import torch.nn as nn


class BaselineCausalTransformer(nn.Module):
    """Parameter-matched causal Transformer baseline (residual next-state predictor)."""

    def __init__(self, d_input: int = 32, d_model: int = 64, d_action: int = 8, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.d_input = d_input
        self.d_action = d_action
        self.d_model = d_model
        self.in_proj = nn.Linear(d_input + d_action, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=0.0, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, d_input)

    def forward(self, x_seq: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        if x_seq.shape[-1] != self.d_input or action_seq.shape[-1] != self.d_action:
            raise ValueError(
                f"expected x_seq[..., {self.d_input}] and action_seq[..., {self.d_action}], "
                f"got x_seq[..., {x_seq.shape[-1]}] and action_seq[..., {action_seq.shape[-1]}]."
            )
        h = self.in_proj(torch.cat([x_seq, action_seq], dim=-1))
        mask = nn.Transformer.generate_square_subsequent_mask(h.shape[1], device=h.device)
        return x_seq + self.out_proj(self.encoder(h, mask=mask))
