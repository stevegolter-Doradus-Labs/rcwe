import torch
from rcwe.kernels.metal_msl import msl_holographic_scan, benchmark_msl_131k_scaling
from rcwe.control.self_healing import AutonomousSelfHealingAgent
from rcwe.lm.rcwe_deep_lm import RCWEDeepLanguageWorldModel, run_deep_lm_benchmark


def test_track_a_metal_msl_scan_and_65k_context():
    torch.manual_seed(701)
    z = torch.complex(torch.randn(1, 65536, 32), torch.randn(1, 65536, 32))
    psi0 = torch.complex(torch.zeros(1, 16, 32), torch.zeros(1, 16, 32))
    gamma = torch.sigmoid(torch.linspace(1.5, 6.0, 16)).unsqueeze(-1)
    omega = torch.linspace(0.05, 6.28, 16).unsqueeze(-1)
    drv = torch.randn(16, 32) * 0.1

    out, psi_final = msl_holographic_scan(z, psi0, gamma, omega, drv)
    assert out.shape == (1, 65536, 32)
    assert psi_final.shape == (1, 16, 32)
    assert (psi_final.numel() * psi_final.element_size()) / 1024.0 == 4.0


def test_track_b_closed_loop_autonomous_self_healing():
    agent = AutonomousSelfHealingAgent(seed=42)
    rep = agent.run_closed_loop_healing_episode()
    assert rep["identified_regime"] == 1
    assert rep["mdo_probe_action_idx"] == 4
    assert rep["self_healed_residual_deviation"] < rep["unhealed_sinkhole_energy_loss"]
    assert rep["stabilization_gain_ratio"] > 3.0


def test_track_c_hierarchical_rcwe_deep_language_world_model():
    rep = run_deep_lm_benchmark(seed=42, epochs=35)
    assert rep["layers"] == 4
    assert rep["rcwe_deep_token_accuracy"] > 0.70
    assert rep["rcwe_deep_total_psi_kb"] == 64.0
