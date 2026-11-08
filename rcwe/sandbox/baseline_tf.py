import torch
import torch.nn as nn


class BaselineCausalTransformer(nn.Module):
    def __init__(
        self, d_input: int = 32, d_model: int = 64, d_action: int = 8, nhead: int = 4, num_layers: int = 2
    ):
        super().__init__()
        self.in_proj = nn.Linear(d_input + d_action, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, dropout=0.0, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, d_input)

    def forward(self, x_seq: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        L = x_seq.shape[1]
        h = self.in_proj(torch.cat([x_seq, action_seq], dim=-1))
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L, device=x_seq.device)
        return self.out_proj(self.transformer(h, mask=causal_mask))
