import time
import torch
from typing import Tuple, Dict, Any
from rcwe.kernels.triton_holographic import fused_holographic_scan
from rcwe.kernels.accelerator import select_best_device, sync_device, describe_accelerator

# Inline Apple Metal Shading Language (MSL) Kernel for Fused Phase-Coherent Wavefield Interference
MSL_HOLOGRAPHIC_SHADER = """
#include <metal_stdlib>
using namespace metal;

kernel void fused_holographic_coherence_gain(
    device const float* zr [[buffer(0)]],
    device const float* zi [[buffer(1)]],
    device const float* pm_r [[buffer(2)]],
    device const float* pm_i [[buffer(3)]],
    device float* out_r [[buffer(4)]],
    device float* out_i [[buffer(5)]],
    uint index [[thread_position_in_grid]])
{
    float r = zr[index];
    float i = zi[index];
    float coh = r * pm_r[index] + i * pm_i[index];
    float x = clamp(coh, -3.0f, 3.0f);
    float x2 = x * x;
    float gain = 1.0f + x * (27.0f + x2) / (27.0f + 9.0f * x2);
    out_r[index] = r * gain;
    out_i[index] = i * gain;
}
"""

_COMPILED_MSL_LIB = None
_MSL_Attempted = False


def get_compiled_metal_shader():
    global _COMPILED_MSL_LIB, _MSL_Attempted
    if _MSL_Attempted:
        return _COMPILED_MSL_LIB
    _MSL_Attempted = True
    if hasattr(torch, "mps") and hasattr(torch.mps, "compile_shader") and torch.backends.mps.is_available():
        try:
            _COMPILED_MSL_LIB = torch.mps.compile_shader(MSL_HOLOGRAPHIC_SHADER)
        except Exception:
            _COMPILED_MSL_LIB = None
    return _COMPILED_MSL_LIB


def msl_holographic_scan(
    z_mixed: torch.Tensor,
    standing_field: torch.Tensor,
    gamma: torch.Tensor,
    omega: torch.Tensor,
    band_drive: torch.Tensor,
    has_init_field: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Track A (v0.7.0): Apple Silicon Metal Shading Language (MSL) Accelerated Scan.
    Uses vectorized Vandermonde / Rank-M rfft projection and dispatches the custom
    MSL threadgroup kernel (`fused_holographic_coherence_gain`) when running on MPS.
    Capable of single-pass execution at L = 131,072+ with a constant 16.0 KB |Psi| state.
    """
    B, L, D = z_mixed.shape
    M = standing_field.shape[1]
    device = z_mixed.device

    if L == 1 or device.type != "mps":
        return fused_holographic_scan(
            z_mixed, standing_field, gamma, omega, band_drive, has_init_field=has_init_field
        )

    z_r, z_i = z_mixed.real, z_mixed.imag
    psi0_r, psi0_i = standing_field.real, standing_field.imag

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
        pm_r = torch.fft.irfft(zr_f * kr_f - zi_f * ki_f, n=n_fft, dim=-1)[..., :L].transpose(1, 2).contiguous()
        pm_i = torch.fft.irfft(zr_f * ki_f + zi_f * kr_f, n=n_fft, dim=-1)[..., :L].transpose(1, 2).contiguous()
    except RuntimeError:
        rp_rf = torch.fft.rfft(r_pow_real.cpu(), n=n_fft, dim=-1)
        rp_if = torch.fft.rfft(r_pow_imag.cpu(), n=n_fft, dim=-1)
        drv_t = band_drive.detach().cpu().t() * inv_M
        kr_f = torch.complex(torch.matmul(drv_t, rp_rf.real), torch.matmul(drv_t, rp_rf.imag)).unsqueeze(0)
        ki_f = torch.complex(torch.matmul(drv_t, rp_if.real), torch.matmul(drv_t, rp_if.imag)).unsqueeze(0)
        zr_f = torch.fft.rfft(zr_t.cpu(), n=n_fft, dim=-1)
        zi_f = torch.fft.rfft(zi_t.cpu(), n=n_fft, dim=-1)
        pm_r = torch.fft.irfft(zr_f * kr_f - zi_f * ki_f, n=n_fft, dim=-1)[..., :L].transpose(1, 2).contiguous().to(device)
        pm_i = torch.fft.irfft(zr_f * ki_f + zi_f * kr_f, n=n_fft, dim=-1)[..., :L].transpose(1, 2).contiguous().to(device)

    msl_lib = get_compiled_metal_shader()
    if msl_lib is not None and not z_mixed.requires_grad:
        try:
            zr_c = z_r.contiguous()
            zi_c = z_i.contiguous()
            out_r = torch.empty_like(zr_c)
            out_i = torch.empty_like(zi_c)
            msl_lib.fused_holographic_coherence_gain(zr_c, zi_c, pm_r, pm_i, out_r, out_i)
            return torch.complex(out_r, out_i), torch.complex(final_psi_r, final_psi_i)
        except Exception:
            pass

    coherence = z_r * pm_r + z_i * pm_i
    gain = 1.0 + torch.tanh(coherence)
    return torch.complex(z_r * gain, z_i * gain), torch.complex(final_psi_r, final_psi_i)


def benchmark_msl_131k_scaling(d_model: int = 64, batch_size: int = 1) -> Dict[str, Any]:
    device = select_best_device()
    msl_lib = get_compiled_metal_shader()
    results = []
    for L in [4096, 32768, 65536, 131072]:
        z = torch.complex(
            torch.randn(batch_size, L, d_model, device=device),
            torch.randn(batch_size, L, d_model, device=device),
        )
        psi0 = torch.complex(
            torch.zeros(batch_size, 16, d_model, device=device),
            torch.zeros(batch_size, 16, d_model, device=device),
        )
        gamma = torch.sigmoid(torch.linspace(1.5, 6.0, 16, device=device)).unsqueeze(-1)
        omega = torch.linspace(0.05, 6.28, 16, device=device).unsqueeze(-1)
        drv = torch.randn(16, d_model, device=device) * 0.1

        with torch.no_grad():
            _ = msl_holographic_scan(z, psi0, gamma, omega, drv)
        sync_device(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            _, psi_out = msl_holographic_scan(z, psi0, gamma, omega, drv)
        sync_device(device)
        ms = (time.perf_counter() - t0) * 1000.0

        psi_kb = (psi_out.numel() * psi_out.element_size()) / 1024.0
        tf_attn_gb = (batch_size * 4 * (L ** 2) * 4) / (1024.0 ** 3)
        results.append({
            "seq_len": L,
            "rcwe_msl_latency_ms": round(ms, 2),
            "measured_psi_kb": psi_kb,
            "tf_equivalent_attn_gb": round(tf_attn_gb, 2),
        })

    return {
        "accelerator": describe_accelerator(device),
        "msl_shader_compiled": msl_lib is not None,
        "sweeps": results,
    }
