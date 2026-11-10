import json
from pathlib import Path
from typing import Dict, Any

from rcwe.kernels.metal_msl import benchmark_msl_131k_scaling
from rcwe.control.self_healing import AutonomousSelfHealingAgent
from rcwe.lm.rcwe_deep_lm import run_deep_lm_benchmark


def run_all_phase2_tracks(output_json: str = "rcwe_phase2_v07_report.json") -> Dict[str, Any]:
    print("================================================================================")
    print("   DORADUS LABS — RCWE v0.7.0 PHASE 2 SUITE (METAL MSL, SELF-HEALING & DEEP-LM)")
    print("================================================================================")

    # Track A: Apple Metal MSL Shader & L=131,072 Extreme Context Scaling
    print("\n=== TRACK A: APPLE METAL SHADING LANGUAGE (MSL) & L=131,072 SCALING ===")
    msl_rep = benchmark_msl_131k_scaling(d_model=64, batch_size=1)
    print(f"Accelerator         : {msl_rep['accelerator']}")
    print(f"Custom MSL Compiled : {msl_rep['msl_shader_compiled']}")
    header = f"{'L':<8} | {'RCWE MSL Latency':<18} | {'Measured |Psi|':<15} | {'TF Attention Equiv.'}"
    print(header)
    print("-" * len(header))
    for r in msl_rep["sweeps"]:
        print(
            f"{r['seq_len']:<8} | {r['rcwe_msl_latency_ms']:>12.2f} ms   | "
            f"{r['measured_psi_kb']:>10.1f} KB  | {r['tf_equivalent_attn_gb']:>12.2f} GB (OOM)"
        )

    # Track B: Closed-Loop Autonomous Self-Healing Controller
    print("\n=== TRACK B: CLOSED-LOOP AUTONOMOUS SELF-HEALING AGENT (DO-CALCULUS CONTROL) ===")
    agent = AutonomousSelfHealingAgent(seed=42)
    heal_rep = agent.run_closed_loop_healing_episode()
    u_loss = heal_rep["unhealed_sinkhole_energy_loss"]
    h_loss = heal_rep["self_healed_residual_deviation"]
    gain = heal_rep["stabilization_gain_ratio"]
    print(f"Autonomous Regime ID & Probe : Identified W{heal_rep['identified_regime']} via MDO Probe Action #{heal_rep['mdo_probe_action_idx']} on [v_5, v_6]")
    print(f"Unhealed Sinkhole Deviation  : {u_loss:.5f}")
    print(f"Self-Healed Node Deviation   : {h_loss:.5f} ({gain:.1f}x stabilization / fault neutralization)")

    # Track C: Hierarchical 4-Layer RCWE-Deep Language & Code World Model
    print("\n=== TRACK C: HIERARCHICAL 4-LAYER RCWE-DEEP LANGUAGE & CODE WORLD MODEL ===")
    lm_rep = run_deep_lm_benchmark(seed=42, epochs=45)
    print(f"Architecture Depth & Context : {lm_rep['layers']} Stacked DeepRCWEBlocks ({lm_rep['sequence_tokens']} Code Tokens)")
    print(f"Code Next-Token Accuracy     : RCWE-Deep = {lm_rep['rcwe_deep_token_accuracy']*100:.1f}% vs Transformer-LM = {lm_rep['tf_deep_token_accuracy']*100:.1f}%")
    print(f"Total 4-Layer O(1) |Psi|     : {lm_rep['rcwe_deep_total_psi_kb']:.1f} KB (Fixed 8.0 KB per layer, B=1)")

    full_report = {
        "version": "0.7.0",
        "repository": "https://github.com/stevegolter-Doradus-Labs/Doradus-RCWE",
        "track_a_metal_msl_131k": msl_rep,
        "track_b_self_healing_control": heal_rep,
        "track_c_rcwe_deep_lm": lm_rep,
    }
    out_p = Path(output_json)
    out_p.write_text(json.dumps(full_report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[Saved Phase 2 v0.7.0 Telemetry Report -> {out_p.resolve()}]")
    return full_report


if __name__ == "__main__":
    run_all_phase2_tracks()
