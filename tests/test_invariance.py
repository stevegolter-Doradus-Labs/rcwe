import torch
from rcwe.core.symplectic import SymplecticCausalBlock
from rcwe.core.resonance import HolographicResonanceLayer


def test_symplectic_leapfrog_time_reversibility():
    torch.manual_seed(101)
    block = SymplecticCausalBlock(d_model=32, d_action=8, step_size=0.05)
    block.eval()

    q0 = torch.randn(4, 16, 32)
    p0 = torch.randn(4, 16, 32)
    action = torch.randn(4, 16, 8)
    mask = torch.ones_like(q0)

    q1, p1 = block.leapfrog_step(q0, p0, action, mask, dt=0.05)
    q_rev, p_rev = block.leapfrog_step(q1, p1, action, mask, dt=-0.05)

    assert torch.allclose(q0, q_rev, atol=1e-5), "Symplectic position failed time-reversibility!"
    assert torch.allclose(p0, p_rev, atol=1e-5), "Symplectic momentum failed time-reversibility!"


def test_hamiltonian_conservation_boundedness():
    torch.manual_seed(102)
    block = SymplecticCausalBlock(d_model=32, d_action=8, step_size=0.02)
    z0 = torch.complex(torch.randn(2, 8, 32) * 0.5, torch.randn(2, 8, 32) * 0.5)
    action = torch.zeros(2, 8, 8)

    z_next, energy_violation = block(z0, action)
    assert z_next.shape == z0.shape
    assert float(energy_violation.mean().item()) < 0.05, "Hamiltonian drift exceeded conservation tolerance!"


def test_holographic_o1_memory_invariance():
    torch.manual_seed(103)
    layer = HolographicResonanceLayer(d_model=64, num_freq_bands=16)

    z_short = torch.complex(torch.randn(2, 16, 64), torch.randn(2, 16, 64))
    z_long = torch.complex(torch.randn(2, 512, 64), torch.randn(2, 512, 64))

    _, field_short = layer(z_short)
    _, field_long = layer(z_long)

    assert field_short.shape == (2, 16, 64)
    assert field_long.shape == (2, 16, 64)
    assert field_short.numel() == field_long.numel(), "Standing field violated O(1) memory invariance!"


def test_vectorized_scan_matches_sequential_reference():
    from rcwe.kernels.triton_holographic import fused_holographic_scan, sequential_holographic_scan
    torch.manual_seed(104)
    B, L, M, D = 2, 128, 16, 32
    z_in = torch.complex(torch.randn(B, L, D), torch.randn(B, L, D))
    psi_0 = torch.complex(torch.randn(B, M, D) * 0.1, torch.randn(B, M, D) * 0.1)
    gamma = torch.sigmoid(torch.linspace(1.5, 6.0, M)).unsqueeze(-1)
    omega = torch.linspace(0.05, 6.28, M).unsqueeze(-1)
    drv = torch.randn(M, D) * 0.1

    out_seq, psi_seq = sequential_holographic_scan(z_in, psi_0, gamma, omega, drv)
    out_vec, psi_vec = fused_holographic_scan(z_in, psi_0, gamma, omega, drv, has_init_field=True)

    assert torch.allclose(out_seq, out_vec, atol=1e-3), "Vectorized scan output diverged from sequential loop!"
    assert torch.allclose(psi_seq, psi_vec, atol=1e-3), "Vectorized final Psi diverged from sequential loop!"
