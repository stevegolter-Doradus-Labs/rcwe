import time
import json
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional

from rcwe.core.engine import ResonantCausalWorldEngine
from rcwe.sandbox.baseline_tf import BaselineCausalTransformer
from rcwe.diagnostics.profiler import HardwareProfilingHarness
from rcwe.kernels.accelerator import select_best_device, sync_device, describe_accelerator
from rcwe.symbolic.algebra_task import run_symbolic_reasoning_benchmark


def run_extreme_context_sweep(
    seq_lengths: Optional[List[int]] = None,
    d_model: int = 64,
    batch_size: int = 2,
    tf_memory_wall_kb: float = 1048576.0,
) -> List[Dict[str, Any]]:
    if seq_lengths is None:
        seq_lengths = [256, 1024, 4096, 8192, 16384, 32768]

    device = select_best_device()
    rcwe = ResonantCausalWorldEngine(d_input=32, d_model=d_model, d_action=8, num_worldlines=3).to(device)
    tf_model = BaselineCausalTransformer(d_input=32, d_model=d_model, d_action=8, nhead=4, num_layers=2).to(device)
    rcwe.eval()
    tf_model.eval()

    rows = []
    for L in seq_lengths:
        tf_state_kb, rcwe_state_kb = HardwareProfilingHarness.state_memory_kb(L, batch_size, d_model)
        x = torch.randn(batch_size, L, 32, device=device)
        a = torch.randn(batch_size, L, 8, device=device)

        with torch.no_grad():
            _ = rcwe(x, a, plastic_runtime=False)
        sync_device(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            out_rcwe = rcwe(x, a, plastic_runtime=False)
        sync_device(device)
        rcwe_ms = (time.perf_counter() - t0) * 1000.0

        psi_tensor = out_rcwe["standing_field"]
        measured_psi_kb = (psi_tensor.numel() * psi_tensor.element_size()) / 1024.0

        tf_ms: Optional[float] = None
        tf_status = "OK"
        tf_mask_kv_kb = (L * L * 4 + 2 * 2 * batch_size * L * d_model * 4) / 1024.0

        if tf_state_kb > tf_memory_wall_kb:
            tf_status = f"OOM / MEMORY WALL ({tf_state_kb / 1024.0:.0f} MB > {tf_memory_wall_kb / 1024.0:.0f} MB Limit)"
        else:
            try:
                with torch.no_grad():
                    _ = tf_model(x, a)
                sync_device(device)
                t1 = time.perf_counter()
                with torch.no_grad():
                    _ = tf_model(x, a)
                sync_device(device)
                tf_ms = (time.perf_counter() - t1) * 1000.0
            except RuntimeError as e:
                tf_status = f"OOM CRASH ({str(e)[:32]}...)"

        rows.append({
            "seq_len": L,
            "tf_status": tf_status,
            "tf_latency_ms": tf_ms,
            "rcwe_latency_ms": rcwe_ms,
            "tf_workspace_kb": tf_state_kb,
            "tf_mask_kv_kb": tf_mask_kv_kb,
            "measured_psi_kb": measured_psi_kb,
            "compression_ratio": tf_state_kb / rcwe_state_kb,
        })

    return rows


def main_extreme_and_symbolic() -> None:
    device = select_best_device()
    print("================================================================================")
    print("   DORADUS LABS — RCWE v0.4.0 EXTREME CONTEXT (L=32,768) & SYMBOLIC BENCHMARK")
    print("================================================================================")
    print(f"Accelerator : {describe_accelerator(device)}\n")

    print("=== 1. EXTREME CONTEXT SCALING SWEEP (L = 256 -> 32,768) ===")
    header = (
        f"{'L':<7} | {'TF Status / Latency':<26} | {'RCWE Latency':<13} | "
        f"{'TF Workspace':<15} | {'RCWE |Psi|':<11} | {'State Ratio':<12}"
    )
    print(header)
    print("-" * len(header))

    sweep_rows = run_extreme_context_sweep()
    for r in sweep_rows:
        if r["tf_latency_ms"] is not None:
            tf_col = f"{r['tf_latency_ms']:>8.2f} ms (OK)"
            speedup_str = f"({r['tf_latency_ms'] / r['rcwe_latency_ms']:.1f}x)"
        else:
            tf_col = "OOM / QUADRATIC WALL"
            speedup_str = "(INF)"
        tf_mem_mb = r["tf_workspace_kb"] / 1024.0
        print(
            f"{r['seq_len']:<7} | {tf_col:<26} | {r['rcwe_latency_ms']:>6.2f} ms {speedup_str:<5}| "
            f"{tf_mem_mb:>11.1f} MB | {r['measured_psi_kb']:>7.1f} KB | {r['compression_ratio']:>10.0f}x"
        )

    print("\n=== 2. SYMBOLIC ALGEBRAIC CONSERVATION & RULE-MUTATION BENCHMARK ===")
    sym_metrics = run_symbolic_reasoning_benchmark(seed=42, epochs=45)
    c_r = sym_metrics["rcwe_symbolic_conservation_error"]
    c_t = sym_metrics["tf_symbolic_conservation_error"]
    m_r = sym_metrics["rcwe_rule_mutation_mse"]
    m_t = sym_metrics["tf_rule_mutation_mse"]

    print(f"Symbolic Sum Conservation Error : RCWE = {c_r:.5f} vs Transformer = {c_t:.5f} ({c_t / max(c_r, 1e-7):.2f}x lower)")
    print(f"Symbolic Rule-Mutation MSE      : RCWE = {m_r:.5f} vs Transformer = {m_t:.5f} ({m_t / max(m_r, 1e-7):.1f}x lower)")

    out_path = Path("rcwe_extreme_and_symbolic_v04.json")
    out_path.write_text(
        json.dumps({"extreme_context_sweep": sweep_rows, "symbolic_benchmark": sym_metrics}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\n[Saved Extreme Scaling & Symbolic Report -> {out_path.resolve()}]")


if __name__ == "__main__":
    main_extreme_and_symbolic()
