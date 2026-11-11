import json
from pathlib import Path
from typing import Dict, Any

from rcwe.multimodal.spectral_fusion import run_multimodal_substation_benchmark
from rcwe.rl.symplectic_rl import run_symplectic_rl_benchmark
from rcwe.swarm.wavefield_swarm import run_distributed_swarm_benchmark


def run_all_v09_tracks(output_json: str = "rcwe_v09_multimodal_rl_swarm_report.json") -> Dict[str, Any]:
    print("================================================================================")
    print("   DORADUS LABS — RCWE v0.9.0 MULTIMODAL, SYMPLECTIC RL & SWARM SUITE")
    print("================================================================================")

    mm = run_multimodal_substation_benchmark(epochs=30)
    print(
        f"[Track 1: 2D Thermal + SCADA Multimodal Fusion] "
        f"RCWE MSE = {mm['rcwe_multimodal_mse']:.5f} vs TF = {mm['tf_multimodal_mse']:.5f} "
        f"({mm['improvement_ratio']}x gain | |Psi| = {mm['psi_kb']:.1f} KB)"
    )
    print(f"  -> Grounded Diagnostic: {mm['diagnostic_nl']}")

    rl = run_symplectic_rl_benchmark(steps=40)
    print(
        f"[Track 2: Symplectic Pontryagin RL & Dopamine ] "
        f"RCWE Dev = {rl['rcwe_rl_post_mutation_deviation']:.5f} vs Frozen = {rl['frozen_policy_post_mutation_deviation']:.5f} "
        f"({rl['rl_stabilization_gain']}x stabilization | ATP = {rl['final_atp']:.3f})"
    )

    sw = run_distributed_swarm_benchmark(steps=30)
    print(
        f"[Track 3: Distributed Wavefield Swarm Gossip  ] "
        f"Swarm Cascade MSE = {sw['swarm_coupled_cascade_mse']:.5f} vs Isolated = {sw['isolated_agents_cascade_mse']:.5f} "
        f"({sw['swarm_consensus_gain']}x immunity | {sw['per_agent_wavefield_kb']:.1f} KB/agent)"
    )

    report = {
        "version": "0.9.0",
        "track1_multimodal": mm,
        "track2_symplectic_rl": rl,
        "track3_wavefield_swarm": sw,
    }
    Path(output_json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[Saved RCWE v0.9.0 Unified Report -> {Path(output_json).resolve()}]")
    return report


if __name__ == "__main__":
    run_all_v09_tracks()
