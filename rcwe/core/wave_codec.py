import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class WavepacketCodec(nn.Module):
    def __init__(self, d_input: int, d_model: int):
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.amp_proj = nn.Linear(d_input, d_model)
        self.phase_proj = nn.Linear(d_input, d_model)
        self.readout = nn.Linear(d_model * 2, d_input)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        amplitude = F.softplus(self.amp_proj(x))
        phase = torch.tanh(self.phase_proj(x)) * math.pi
        return torch.complex(amplitude * torch.cos(phase), amplitude * torch.sin(phase))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.readout(torch.cat([z.real, z.imag], dim=-1))
