from rcwe.diagnostics.validation import ScientificValidationBenchmark


def main() -> None:
    print("================================================================================")
    print("   DORADUS LABS — RCWE v0.3.0 MULTI-SEED & MPS ACCELERATOR BENCHMARK SUITE")
    print("================================================================================")
    bench = ScientificValidationBenchmark(seed=42)
    report = bench.run_multiseed_validation(seeds=[42, 101, 202], epochs=55)

    env = report["environment"]
    print(f"Environment : Python {env['python']} | PyTorch {env['torch']} | {env['platform']}")
    print(f"Accelerator : {env['accelerator']}")
    print(f"Model Parity: RCWE Params = {env['rcwe_params']:,} | Baseline Transformer Params = {env['baseline_tf_params']:,}\n")

    print("=== 1. INSTRUMENTED HARDWARE & MEMORY SCALING (SYNCHRONIZED ACCELERATOR TIMERS) ===")
    header = (
        f"{'L':<5} | {'TF Latency':<11} | {'RCWE Latency':<12} | "
        f"{'TF Mask+KV KB':<14} | {'Measured |Psi|':<14} | {'State Ratio':<11}"
    )
    print(header)
    print("-" * len(header))
    for row in report["hardware_scaling"]:
        print(
            f"{int(row['seq_len']):<5} | {row['tf_latency_ms']:>8.2f} ms | {row['rcwe_latency_ms']:>9.2f} ms | "
            f"{row['tf_peak_alloc_kb']:>10.1f} KB | {row['measured_psi_kb']:>11.1f} KB | {row['compression_ratio']:>9.1f}x"
        )

    ms = report["multiseed_statistics"]
    r_mse = ms["rcwe_post_mutation_mse"]
    t_mse = ms["tf_post_mutation_mse"]
    r_drf = ms["rcwe_hamiltonian_drift"]
    t_drf = ms["tf_hamiltonian_drift"]
    acc = ms["regime_id_accuracy"]

    print(f"\n=== 2. MULTI-SEED STATISTICAL VALIDATION (Seeds={ms['seeds']}, Mean ± SD [95% CI]) ===")
    print(
        f"Post-Mutation MSE      : RCWE = {r_mse['mean']:.5f} ± {r_mse['std']:.5f} "
        f"vs TF = {t_mse['mean']:.5f} ± {t_mse['std']:.5f}"
    )
    print(
        f"Hamiltonian Drift (dH) : RCWE = {r_drf['mean']:.5f} ± {r_drf['std']:.5f} "
        f"vs TF = {t_drf['mean']:.5f} ± {t_drf['std']:.5f}"
    )
    print(
        f"Causal Regime ID Acc   : {acc['mean']*100.0:.1f}% ± {acc['std']*100.0:.1f}% "
        f"(95% CI: [{acc['ci95_low']*100.0:.1f}%, {min(acc['ci95_high']*100.0, 100.0):.1f}%])"
    )

    print("\n" + report["advisory_text"])
    print(f"\n[Saved Multi-Seed JSON Telemetry Report -> {report['json_path']}]")


if __name__ == "__main__":
    main()
