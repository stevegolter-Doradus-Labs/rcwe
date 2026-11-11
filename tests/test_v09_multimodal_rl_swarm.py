from pathlib import Path
from rcwe.multimodal.spectral_fusion import run_multimodal_substation_benchmark
from rcwe.rl.symplectic_rl import run_symplectic_rl_benchmark
from rcwe.swarm.wavefield_swarm import run_distributed_swarm_benchmark
from rcwe.v09_runner import run_all_v09_tracks


def test_track1_multimodal_2d_spectral_fusion_and_nl_grounding():
    res = run_multimodal_substation_benchmark(epochs=25)
    assert res["psi_kb"] == 8.0
    assert res["rcwe_multimodal_mse"] < res["tf_multimodal_mse"]
    assert res["improvement_ratio"] > 10.0
    assert "CRITICAL MULTIMODAL RESONANCE" in res["diagnostic_nl"]


def test_track2_symplectic_pontryagin_rl_and_dopaminergic_plasticity():
    res = run_symplectic_rl_benchmark(steps=40)
    assert res["rcwe_rl_post_mutation_deviation"] < res["frozen_policy_post_mutation_deviation"]
    assert res["rl_stabilization_gain"] > 2.5
    assert 0.0 < res["final_atp"] <= 2.0


def test_track3_distributed_wavefield_swarm_cascading_fault_immunity(tmp_path: Path):
    res = run_distributed_swarm_benchmark(steps=30)
    assert res["per_agent_wavefield_kb"] == 8.0
    assert res["total_swarm_wavefield_kb"] == 32.0
    assert res["swarm_coupled_cascade_mse"] < res["isolated_agents_cascade_mse"]
    assert res["swarm_consensus_gain"] > 5.0

    rep_file = tmp_path / "v09_report.json"
    full_rep = run_all_v09_tracks(output_json=str(rep_file))
    assert rep_file.exists()
    assert full_rep["version"] == "0.9.0"
