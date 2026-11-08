from pathlib import Path
import torch
from rcwe.kernels.accelerator import select_best_device, sync_device, get_device_memory_kb, describe_accelerator
from rcwe.diagnostics.profiler import HardwareProfilingHarness
from rcwe.diagnostics.validation import ScientificValidationBenchmark


def test_accelerator_device_and_sync():
    dev = select_best_device()
    assert isinstance(dev, torch.device)
    sync_device(dev)
    mem_kb = get_device_memory_kb(dev)
    assert mem_kb >= 0.0
    desc = describe_accelerator(dev)
    assert len(desc) > 0


def test_instrumented_hardware_profiler_measurements():
    rows = HardwareProfilingHarness.run_scaling_sweep(d_model=32, batch_size=2, seq_lengths=[128, 256])
    assert len(rows) == 2
    for r in rows:
        assert r["tf_latency_ms"] > 0.0
        assert r["rcwe_latency_ms"] > 0.0
        assert r["measured_psi_kb"] == 8.0


def test_worldline_specialization_breaks_posterior_tie(tmp_path: Path):
    bench = ScientificValidationBenchmark(seed=42)
    report = bench.run_full_validation(
        output_json_path=str(tmp_path / "val_report.json"),
        seq_lengths=[128, 256],
    )

    c_id = report["causal_identification"]
    posteriors = c_id["regime_1_posteriors"]
    h2h = report["head_to_head_mutation_benchmark"]

    assert c_id["dominant_regime_id"] == 1, f"Expected Regime 1 (Sinkhole) to dominate, got {c_id['dominant_regime_id']}"
    assert posteriors[1] > 0.50, f"Expected confident W1 posterior > 0.50, got {posteriors}"
    assert c_id["accuracy"] >= 0.75, f"Expected >=75% regime ID accuracy, got {c_id['accuracy']}"
    assert h2h["rcwe_post_mutation_mse"] < h2h["tf_post_mutation_mse"], "Expected plastic RCWE to beat frozen TF post-mutation!"
    assert c_id["mdo_divergence_score"] > 1e-4, "Expected non-zero counterfactual variance split!"
    assert (tmp_path / "val_report.json").exists()


def test_multiseed_statistical_confidence_intervals(tmp_path: Path):
    bench = ScientificValidationBenchmark(seed=42)
    report = bench.run_multiseed_validation(
        seeds=[42, 101],
        epochs=35,
        output_json_path=str(tmp_path / "multiseed_report.json"),
        seq_lengths=[128],
    )
    ms = report["multiseed_statistics"]
    assert ms["rcwe_post_mutation_mse"]["mean"] < ms["tf_post_mutation_mse"]["mean"]
    assert ms["rcwe_hamiltonian_drift"]["mean"] < ms["tf_hamiltonian_drift"]["mean"]
    assert ms["regime_id_accuracy"]["mean"] >= 0.75
    assert "ci95_low" in ms["rcwe_post_mutation_mse"]
