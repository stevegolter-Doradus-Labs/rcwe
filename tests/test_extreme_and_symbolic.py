from rcwe.diagnostics.extreme_bench import run_extreme_context_sweep
from rcwe.symbolic.algebra_task import run_symbolic_reasoning_benchmark


def test_extreme_context_o1_survival_past_transformer_wall():
    rows = run_extreme_context_sweep(
        seq_lengths=[1024, 16384],
        d_model=64,
        batch_size=2,
        tf_memory_wall_kb=1048576.0,
    )
    assert len(rows) == 2
    assert rows[0]["tf_latency_ms"] is not None
    assert rows[0]["measured_psi_kb"] == 16.0
    assert rows[1]["tf_latency_ms"] is None
    assert "OOM" in rows[1]["tf_status"]
    assert rows[1]["measured_psi_kb"] == 16.0
    assert rows[1]["rcwe_latency_ms"] > 0.0


def test_symbolic_conservation_and_rule_mutation():
    metrics = run_symbolic_reasoning_benchmark(seed=42, epochs=35)
    assert metrics["rcwe_rule_mutation_mse"] < metrics["tf_rule_mutation_mse"]
    assert metrics["rcwe_symbolic_conservation_error"] < metrics["tf_symbolic_conservation_error"]
