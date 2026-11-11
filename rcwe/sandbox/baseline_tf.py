import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineCausalTransformer(nn.Module):
    def __init__(self, d_input: int = 32, d_model: int = 64, d_action: int = 8, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.d_model = d_model
        self.in_proj = nn.Linear(d_input + d_action, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=0.0, batch_first=True
        )
        self.trkay = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, d_input)

    def forward(self, x_seq: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        in_f = x_seq.shape[-1] + action_seq.shape[-1]
        if in_f != self.in_proj.in_features:
            device = x_seq.device
            new_in = nn.Linear(in_f, self.in_proj.out_features).to(device)
            torch.nn.init.zeros_(new_in.weight)
            new_in.weight.data[:, :min(in_f, self.in_proj.in_features)] = self.in_proj.weight.data[:, :min(in_f, self.in_proj.in_features)]
            new_in.bias.data = self.in_proj.bias.data
            self.in_proj = new_in
            self.out_proj = nn.Linear(self.d_model, x_seq.shape[-1]).to(device)
        h = self.in_proj(torch.cat([x_seq, action_seq], dim=-1))
        L = h.shape[1]
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=h.device)
        h_out = self.trkay(h, mask=mask)
        return x_seq + self.out_proj(h_out)
