import math
import torch
import torch.nn as nn
from typing import Tuple, Optional
from rcwe.kernels.triton_holographic import fused_holographic_scan


class HolographicResonanceLayer(nn.Module):
    def __init__(self, d_model: int, num_freq_bands: int = 16):
        super().__init__()
        self.d_model = d_model
        self.num_bands = num_freq_bands

        scale = 1.0 / math.sqrt(d_model)
        self.spectral_weight_real = nn.Parameter(torch.randn(d_model, d_model) * scale)
        self.spectral_weight_imag = nn.Parameter(torch.randn(d_model, d_model) * scale)

        init_decay = torch.linspace(1.5, 6.0, num_freq_bands).unsqueeze(-1)
        self.logit_decay = nn.Parameter(init_decay)
        self.omega = nn.Parameter(torch.linspace(0.05, 2.0 * math.pi, num_freq_bands).unsqueeze(-1))
        self.band_drive = nn.Parameter(torch.randn(num_freq_bands, d_model) * 0.1)

    def _spectral_mix(self, z_seq: torch.Tensor) -> torch.Tensor:
        orig_device = z_seq.device
        try:
            z_freq = torch.fft.fft(z_seq, dim=1, norm="ortho")
            zf_r, zf_i = z_freq.real, z_freq.imag
            wr, wi = self.spectral_weight_real, self.spectral_weight_imag
            mix_r = torch.matmul(zf_r, wr) - torch.matmul(zf_i, wi)
            mix_i = torch.matmul(zf_r, wi) + torch.matmul(zf_i, wr)
            return torch.fft.ifft(torch.complex(mix_r, mix_i), dim=1, norm="ortho")
        except RuntimeError:
            z_cpu = z_seq.cpu()
            z_freq = torch.fft.fft(z_cpu, dim=1, norm="ortho")
            wr, wi = self.spectral_weight_real.cpu(), self.spectral_weight_imag.cpu()
            mix_r = torch.matmul(z_freq.real, wr) - torch.matmul(z_freq.imag, wi)
            mix_i = torch.matmul(z_freq.real, wi) + torch.matmul(z_freq.imag, wr)
            z_mixed_cpu = torch.fft.ifft(torch.complex(mix_r, mix_i), dim=1, norm="ortho")
            return torch.complex(z_mixed_cpu.real.to(orig_device), z_mixed_cpu.imag.to(orig_device))

    def forward(
        self, z_seq: torch.Tensor, standing_field: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = z_seq.shape
        has_init = standing_field is not None
        if standing_field is None:
            zeros = torch.zeros(B, self.num_bands, D, dtype=torch.float32, device=z_seq.device)
            standing_field = torch.complex(zeros, zeros.clone())

        z_mixed = self._spectral_mix(z_seq)
        gamma = torch.sigmoid(self.logit_decay)
        return fused_holographic_scan(
            z_mixed, standing_field, gamma, self.omega, self.band_drive, has_init_field=has_init
        )
