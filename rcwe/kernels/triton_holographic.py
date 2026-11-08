import torch
from typing import Tuple

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def fused_holographic_scan(
    z_mixed: torch.Tensor,
    standing_field: torch.Tensor,
    gamma: torch.Tensor,
    omega: torch.Tensor,
    band_drive: torch.Tensor,
    has_init_field: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, L, D = z_mixed.shape
    M = standing_field.shape[1]
    device = z_mixed.device
    z_r, z_i = z_mixed.real, z_mixed.imag
    psi0_r, psi0_i = standing_field.real, standing_field.imag

    if L == 1:
        rot_r = (gamma * torch.cos(omega)).squeeze(-1).view(1, M, 1)
        rot_i = (gamma * torch.sin(omega)).squeeze(-1).view(1, M, 1)
        drv = band_drive.unsqueeze(0)
        zrt, zit = z_r[:, 0, :], z_i[:, 0, :]
        new_psi_r = (psi0_r * rot_r - psi0_i * rot_i) + zrt.unsqueeze(1) * drv
        new_psi_i = (psi0_r * rot_i + psi0_i * rot_r) + zit.unsqueeze(1) * drv
        coherence = (zrt.unsqueeze(1) * new_psi_r + zit.unsqueeze(1) * new_psi_i).mean(dim=1)
        gain = (1.0 + torch.tanh(coherence)).unsqueeze(1)
        return torch.complex(z_r * gain, z_i * gain), torch.complex(new_psi_r, new_psi_i)

    steps = torch.arange(L, device=device, dtype=torch.float32)
    log_gamma = torch.log(gamma.clamp(min=1e-7))
    r_mag = torch.exp(log_gamma * steps.unsqueeze(0))
    r_ang = omega * steps.unsqueeze(0)
    r_pow_real = r_mag * torch.cos(r_ang)
    r_pow_imag = r_mag * torch.sin(r_ang)

    zr_t = z_r.transpose(1, 2).contiguous()
    zi_t = z_i.transpose(1, 2).contiguous()

    v_real_t = r_pow_real.flip(dims=[1]).t().contiguous()
    v_imag_t = r_pow_imag.flip(dims=[1]).t().contiguous()
    drv = band_drive.unsqueeze(0)

    final_psi_r = (torch.matmul(zr_t, v_real_t) - torch.matmul(zi_t, v_imag_t)).transpose(1, 2) * drv
    final_psi_i = (torch.matmul(zr_t, v_imag_t) + torch.matmul(zi_t, v_real_t)).transpose(1, 2) * drv

    if has_init_field:
        rL_mag = torch.exp(log_gamma.squeeze(-1) * float(L))
        rL_ang = omega.squeeze(-1) * float(L)
        rL_r = (rL_mag * torch.cos(rL_ang)).view(1, M, 1)
        rL_i = (rL_mag * torch.sin(rL_ang)).view(1, M, 1)
        final_psi_r = final_psi_r + (psi0_r * rL_r - psi0_i * rL_i)
        final_psi_i = final_psi_i + (psi0_r * rL_i + psi0_i * rL_r)

    inv_M = 1.0 / float(M)
    n_fft = 2 * L
    try:
        rp_rf = torch.fft.rfft(r_pow_real, n=n_fft, dim=-1)
        rp_if = torch.fft.rfft(r_pow_imag, n=n_fft, dim=-1)
        drv_t = band_drive.t() * inv_M
        kr_f = torch.complex(torch.matmul(drv_t, rp_rf.real), torch.matmul(drv_t, rp_rf.imag)).unsqueeze(0)
        ki_f = torch.complex(torch.matmul(drv_t, rp_if.real), torch.matmul(drv_t, rp_if.imag)).unsqueeze(0)

        zr_f = torch.fft.rfft(zr_t, n=n_fft, dim=-1)
        zi_f = torch.fft.rfft(zi_t, n=n_fft, dim=-1)

        pm_r_t = torch.fft.irfft(zr_f * kr_f - zi_f * ki_f, n=n_fft, dim=-1)[..., :L]
        pm_i_t = torch.fft.irfft(zr_f * ki_f + zi_f * kr_f, n=n_fft, dim=-1)[..., :L]
    except RuntimeError:
        rp_rf = torch.fft.rfft(r_pow_real.cpu(), n=n_fft, dim=-1)
        rp_if = torch.fft.rfft(r_pow_imag.cpu(), n=n_fft, dim=-1)
        drv_t = band_drive.detach().cpu().t() * inv_M
        kr_f = torch.complex(torch.matmul(drv_t, rp_rf.real), torch.matmul(drv_t, rp_rf.imag)).unsqueeze(0)
        ki_f = torch.complex(torch.matmul(drv_t, rp_if.real), torch.matmul(drv_t, rp_if.imag)).unsqueeze(0)
        zr_f = torch.fft.rfft(zr_t.cpu(), n=n_fft, dim=-1)
        zi_f = torch.fft.rfft(zi_t.cpu(), n=n_fft, dim=-1)
        pm_r_t = torch.fft.irfft(zr_f * kr_f - zi_f * ki_f, n=n_fft, dim=-1)[..., :L].to(device)
        pm_i_t = torch.fft.irfft(zr_f * ki_f + zi_f * kr_f, n=n_fft, dim=-1)[..., :L].to(device)

    if has_init_field:
        rot1_r = (gamma * torch.cos(omega)).view(1, M, 1)
        rot1_i = (gamma * torch.sin(omega)).view(1, M, 1)
        p1_r = ((psi0_r * rot1_r - psi0_i * rot1_i) * inv_M).transpose(1, 2)
        p1_i = ((psi0_r * rot1_i + psi0_i * rot1_r) * inv_M).transpose(1, 2)
        pm_r_t = pm_r_t + (torch.matmul(p1_r, r_pow_real) - torch.matmul(p1_i, r_pow_imag))
        pm_i_t = pm_i_t + (torch.matmul(p1_r, r_pow_imag) + torch.matmul(p1_i, r_pow_real))

    coherence = (zr_t * pm_r_t + zi_t * pm_i_t).transpose(1, 2)
    gain = 1.0 + torch.tanh(coherence)
    return torch.complex(z_r * gain, z_i * gain), torch.complex(final_psi_r, final_psi_i)


def sequential_holographic_scan(
    z_mixed: torch.Tensor,
    standing_field: torch.Tensor,
    gamma: torch.Tensor,
    omega: torch.Tensor,
    band_drive: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, L, D = z_mixed.shape
    rot_r = (gamma * torch.cos(omega)).squeeze(-1).view(1, -1, 1)
    rot_i = (gamma * torch.sin(omega)).squeeze(-1).view(1, -1, 1)
    drv = band_drive.unsqueeze(0)
    z_r, z_i = z_mixed.real, z_mixed.imag
    psi_r, psi_i = standing_field.real.clone(), standing_field.imag.clone()

    out_r_steps, out_i_steps = [], []
    for t in range(L):
        zrt = z_r[:, t, :]
        zit = z_i[:, t, :]
        new_psi_r = (psi_r * rot_r - psi_i * rot_i) + zrt.unsqueeze(1) * drv
        new_psi_i = (psi_r * rot_i + psi_i * rot_r) + zit.unsqueeze(1) * drv
        psi_r, psi_i = new_psi_r, new_psi_i

        coherence = (zrt.unsqueeze(1) * psi_r + zit.unsqueeze(1) * psi_i).mean(dim=1)
        gain = 1.0 + torch.tanh(coherence)
        out_r_steps.append(zrt * gain)
        out_i_steps.append(zit * gain)

    return torch.complex(torch.stack(out_r_steps, dim=1), torch.stack(out_i_steps, dim=1)), torch.complex(psi_r, psi_i)
