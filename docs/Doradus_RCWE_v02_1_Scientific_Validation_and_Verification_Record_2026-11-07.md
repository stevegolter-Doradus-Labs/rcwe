# Doradus Labs — Resonant Causal World Engine (RCWE)
## v0.2.1 Scientific Validation, Calibration, and Hardware Benchmark Record

**Date:** November 7, 2026  
**Project:** Resonant Causal World Engine (RCWE) — Version 0.2.1  
**Environment:** MacBook Pro, macOS-27.2-arm64-arm-64bit-Mach-O (Apple Silicon), zsh, Homebrew Python 3.14.5, PyTorch 2.15.0, NumPy 2.5.4, pytest 9.2.0  
**Purpose:** Document the architectural progression from v0.1 to v0.2.1, preserve the mathematical repairs that resolved world-line posterior ties and runtime adaptation bottlenecks, and archive the verified empirical telemetry from `rcwe_validation_report_v02.json`.

> **Final Outcome:** RCWE v0.2.1 installed cleanly with zero build or runtime warnings, passed **7 of 7** automated unit and scientific validation tests in `3.20s`, and completed the calibrated head-to-head benchmark against a parameter-matched `BaselineCausalTransformer` (`80,611` vs `71,648` parameters). Across held-out test rollouts, RCWE v0.2.1 achieved **100.0% causal regime identification accuracy** (`W1=1.000` posterior confidence, `0.0%` epistemic ambiguity), **27.1x lower post-mutation MSE** (`0.00393` vs `0.10634`) without backpropagation, **3.62x lower physical Hamiltonian conservation drift** (`0.18184` vs `0.65792`), and **1.45x faster wall-clock execution** at sequence length `L=4096` (`103.25 ms` vs `149.20 ms`) while maintaining a constant `16.0 KB` standing wavefield buffer.

---

## 1. Executive Summary: Progression from v0.1 to v0.2.1

In the initial v0.1 setup record, the package installation and five basic operator unit tests succeeded, but `rcwe-bench` ran on untrained random weights and relied strictly on analytical memory formulas. As a result, v0.1 displayed tied world-line posteriors (`W0=0.333 | W1=0.333 | W2=0.333`), `100.0%` epistemic ambiguity, rounded `0.00000` counterfactual variance, and no empirical comparison against a Transformer baseline.

Work across v0.2.0 and v0.2.1 systematically resolved every packaging, syntax, and mathematical calibration limitation:

| Validation Dimension | v0.1 (Untrained Demo) | v0.2.0 (Raw State Decoder) | v0.2.1 (Velocity-Space + Sensory Plasticity) |
|---|---|---|---|
| **Build & Packaging Warnings** | Missing `README.md` + missing `numpy` | Resolved (`README.md` + `numpy-2.5.4`) | **Zero warnings (`rcwe-0.2.1` editable build)** |
| **Automated Pytest Suite** | 5 / 5 passed (`1.10s`) | 6 / 7 passed (`6.83s`) | **7 / 7 passed (`3.20s`)** |
| **Model Parameter Parity** | Unmatched | 80,611 (RCWE) vs 162,112 (TF) | **80,611 (RCWE) vs 71,648 (TF)** |
| **Regime 1 Posterior Weights** | `[0.333, 0.333, 0.333]` (Tied) | `[0.312, 0.346, 0.343]` (Weak split) | **`[0.000, 1.000, 0.000]` (Converged `W1`)** |
| **Epistemic Ambiguity** | `100.0%` | `99.9%` | **`0.0%` (`CONVERGED CAUSAL IDENTIFICATION`)** |
| **Held-Out Regime ID Accuracy** | Not evaluated | `53.3%` | **`100.0%` across held-out `(W0, W1, W2)` rollouts** |
| **Post-Mutation Adaptation MSE** | Not evaluated | `0.10465` (RCWE) vs `0.10356` (TF) | **`0.00393` (RCWE) vs `0.10634` (TF) (27.1x lower)** |
| **Physical Hamiltonian Drift** | Not evaluated | `0.58222` (RCWE) vs `0.70652` (TF) | **`0.18184` (RCWE) vs `0.65792` (TF) (3.62x lower)** |
| **Wall-Clock Latency at L=4096** | Not measured | `173.33 ms` (RCWE) vs `175.77 ms` (TF) | **`103.25 ms` (RCWE) vs `149.20 ms` (TF) (1.45x faster)** |
| **Measured State Tensor at L=4096** | Formula only (`32.0 KB`) | `16.0 KB` `|Psi|` | **`16.0 KB` `|Psi|` vs `73,728.0 KB` TF Mask+KV** |

---

## 2. Chronological Engineering & Mathematical Repairs (v0.2.0 -> v0.2.1)

### 2.1. Packaging & Multi-Line String Sanitization
1. **Added `README.md` and `numpy>=1.26.0`:** Included `README.md` in the root generator dictionary and added `numpy` to `pyproject.toml`, eliminating setuptools `BuildBackendWarning` and PyTorch's `functional_tensor` warning.
2. **Eliminated Multi-Line f-String Line-Break Corruption (`rcwe/diagnostics/nl_codec.py`):** An initial v0.2.0 draft embedded literal newlines inside f-string delimiters when serialized into `upgrade_rcwe_v02.py`, raising `SyntaxError: unterminated f-string literal` at line 119. Refactoring `render_engineering_advisory()` to assemble a list of single-line strings joined via `"\n".join(lines)` and adding an automated `python -m compileall -q rcwe tests` gate prior to `pip install` eliminated syntax corruption permanently.

### 2.2. Coupled Symplectic Action-State Potential (`rcwe/core/symplectic.py`)
In v0.1, `SymplecticCausalBlock` relied solely on a small randomly initialized MLP for its scalar potential $V(q, a)$, producing near-zero initial forces ($\sim 10^{-4}$) and causing counterfactual variance across candidate interventions to round to `0.00000`.

In v0.2+, the scalar potential was upgraded to a coupled quadratic-plus-nonlinear potential:
$$V_k(q, do(a)) = rac{1}{2d} \left\| W_q^{(k)}(M_a \odot q) + W_a^{(k)} a ight\|_2^2 + \Phi_{	ext{MLP}}^{(k)}(M_a \odot q, a)$$
This preserved exact Leapfrog Verlet time-reversibility (`atol=1e-5`) and bounded unforced energy drift (`< 0.05`) while increasing counterfactual MDO variance split to `0.04740`–`0.06624`.

### 2.3. Velocity-Space Transition Decoding (`rcwe/core/engine.py` & `rcwe/diagnostics/validation.py`)
In the first v0.2.0 run, `test_worldline_specialization_breaks_posterior_tie` failed because the posterior for `W1` reached only `0.3457` (`[0.3118, 0.3457, 0.3425]`), below the `> 0.50` threshold.

**Root Cause:** In discrete symplectic integration with time step $\Delta t = 0.05$, the next state $y_{t+1}$ is dominated by the current state $x_t$ ($y_{t+1} pprox x_t + \Delta t \, \dot{y}_t$). When branch decoders predicted raw $y_{t+1}$, 95% of the variance was shared position state across all three regimes, masking the regime-specific force anomaly.

**v0.2.1 Fix:** Each world-line decoder now predicts the state transition residual $\Delta y^{(k)} = \hat{y}_{t+1}^{(k)} - x_t$, and anomaly residuals are scored in normalized transition-velocity space:
$$\mathcal{E}_k = \left\| rac{(\hat{y}_{t+1}^{(k)} - y_{t+1}^*)}{\Delta t} ight\|_2^2 + 0.05 \, \Delta \mathcal{H}_k$$
With velocity-space calibration, the residual anomaly on Regime 1 separated sharply (`W0=2.7369 | W1=1.2408 | W2=2.6782`), driving the `W1` posterior to **`1.000`** (`[0.000, 1.000, 0.000]`) and achieving **100.0% classification accuracy** across held-out rollouts.

### 2.4. Sensory Predictive Coding for Runtime Topology Mutations (`rcwe/core/engine.py`)
In v0.2.0, post-mutation MSE between RCWE (`0.10465`) and the frozen Transformer (`0.10356`) was nearly tied because `MetabolicPlasticSynapse` was only updating on internal hidden-state self-prediction rather than the observable sensory surprise residual $e_{	ext{sensory}} = y_{	ext{true}} - \hat{y}$.

In v0.2.1, `ResonantCausalWorldEngine` routes the live sensory prediction error into an ATP-gated fast sensory residual buffer (`fast_sensory_bias`) under `torch.no_grad()`. When an unmodeled topology mutation and DC disturbance occur at inference time, RCWE adapts within 5–10 steps without backpropagation, reducing post-mutation MSE to **`0.00393`** versus **`0.10634`** for the frozen Transformer (**27.1x improvement**).

---

## 3. Verified Unit & Scientific Validation Test Suite (7 / 7 Passed)

Executed via `python -m pytest -v` on macOS Apple Silicon (Python 3.14.5, PyTorch 2.15.0):

| Test File | Test Function | Verified Property | Status |
|---|---|---|---|
| `tests/test_invariance.py` | `test_symplectic_leapfrog_time_reversibility` | Stepping $+dt$ then $-dt$ recovers $(q_0, p_0)$ within `1e-5` | **PASSED** |
| `tests/test_invariance.py` | `test_hamiltonian_conservation_boundedness` | Unforced symplectic step keeps $|\Delta \mathcal{H}| < 0.05` | **PASSED** |
| `tests/test_invariance.py` | `test_holographic_o1_memory_invariance` | Standing wavefield $\Psi$ is `(2, 16, 64)` at both $L=16$ and $L=512$ | **PASSED** |
| `tests/test_plasticity_and_mdo.py` | `test_backprop_free_metabolic_plasticity_and_sleep` | Fast weights adapt under `torch.no_grad()` and consolidate in sleep | **PASSED** |
| `tests/test_plasticity_and_mdo.py` | `test_mdo_epistemic_foraging_and_nl_codec` | Selects high-leverage probe and grounds target nodes in text | **PASSED** |
| `tests/test_scientific_validation.py` | `test_instrumented_hardware_profiler_measurements` | Verifies positive wall-clock timings and constant `|Psi|` byte size | **PASSED** |
| `tests/test_scientific_validation.py` | `test_worldline_specialization_breaks_posterior_tie` | Verifies `W1 > 0.50`, accuracy $\ge 75\%$, and RCWE MSE $<$ TF MSE | **PASSED** |

---

## 4. Empirical Hardware & Memory Scaling Results

From `rcwe-bench` on MacBook Pro (`d_model=64, batch_size=2`, parameter parity: `80,611` RCWE vs `71,648` Transformer):

| Sequence Length (L) | TF MFLOPs | RCWE MFLOPs | TF Latency (ms) | RCWE Latency (ms) | TF Mask+KV Tensor (KB) | Measured RCWE `|Psi|` (KB) | Analytical State Ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **256** | 25.17 | 51.12 | 1.92 ms | 9.87 ms | 768.0 KB | 16.0 KB | 80.0x |
| **1,024** | 301.99 | 205.78 | 13.20 ms | 30.28 ms | 6,144.0 KB | 16.0 KB | 1,088.0x |
| **2,048** | 1,140.85 | 412.88 | 44.29 ms | 53.59 ms | 20,480.0 KB | 16.0 KB | 4,224.0x |
| **4,096** | 4,429.19 | 828.38 | **149.20 ms** | **103.25 ms** | **73,728.0 KB** | **16.0 KB** | **16,640.0x** |

### Key Hardware Findings
1. **Empirical Latency Crossover Between $L=2048$ and $L=4096$:** Because standard causal self-attention scales quadratically as $\mathcal{O}(L^2)$, a $16	imes$ increase in sequence length ($256 	o 4096$) caused Transformer wall-clock latency to increase **77.8x** (`1.92 ms` $	o$ `149.20 ms`). Over the same $16	imes$ span, RCWE's $\mathcal{O}(L \log L)$ spectral mixing and $\mathcal{O}(1)$ oscillator recurrence increased only **10.5x** (`9.87 ms` $	o$ `103.25 ms`), making RCWE **1.45x faster** at $L=4096$.
2. **Constant Measured Standing Wavefield Memory:** The materialized PyTorch causal attention mask and KV activation tensors in the Transformer grew from `768.0 KB` to `73,728.0 KB` (`72.0 MB`), whereas the measured `complex64` standing wavefield tensor `|Psi|` remained invariant at exactly `16.0 KB` (`32.0 KB` combined with the fast synaptic weight matrix).

---

## 5. Archival JSON Telemetry (`rcwe_validation_report_v02.json`)

```json
{
  "version": "0.2.1",
  "environment": {
    "python": "3.14.5",
    "platform": "macOS-27.2-arm64-arm-64bit-Mach-O",
    "torch": "2.15.0",
    "rcwe_params": 80611,
    "baseline_tf_params": 71648,
    "wall_time_sec": 3.827
  },
  "hardware_scaling": [
    {
      "seq_len": 256.0,
      "tf_mflops": 25.165824,
      "rcwe_mflops": 51.11808,
      "tf_kb": 2560.0,
      "rcwe_kb": 32.0,
      "measured_psi_kb": 16.0,
      "tf_latency_ms": 1.9171943228381376,
      "rcwe_latency_ms": 9.866624992961684,
      "tf_peak_alloc_kb": 768.0,
      "rcwe_peak_alloc_kb": 16.0,
      "compression_ratio": 80.0
    },
    {
      "seq_len": 1024.0,
      "tf_mflops": 301.989888,
      "rcwe_mflops": 205.78304,
      "tf_kb": 34816.0,
      "rcwe_kb": 32.0,
      "measured_psi_kb": 16.0,
      "tf_latency_ms": 13.195541667907188,
      "rcwe_latency_ms": 30.278583348263055,
      "tf_peak_alloc_kb": 6144.0,
      "rcwe_peak_alloc_kb": 16.0,
      "compression_ratio": 1088.0
    },
    {
      "seq_len": 2048.0,
      "tf_mflops": 1140.850688,
      "rcwe_mflops": 412.8768,
      "tf_kb": 135168.0,
      "rcwe_kb": 32.0,
      "measured_psi_kb": 16.0,
      "tf_latency_ms": 44.29204166323567,
      "rcwe_latency_ms": 53.58500000632679,
      "tf_peak_alloc_kb": 20480.0,
      "rcwe_peak_alloc_kb": 16.0,
      "compression_ratio": 4224.0
    },
    {
      "seq_len": 4096.0,
      "tf_mflops": 4429.185024,
      "rcwe_mflops": 828.37504,
      "tf_kb": 532480.0,
      "rcwe_kb": 32.0,
      "measured_psi_kb": 16.0,
      "tf_latency_ms": 149.20390267313147,
      "rcwe_latency_ms": 103.25027798535302,
      "tf_peak_alloc_kb": 73728.0,
      "rcwe_peak_alloc_kb": 16.0,
      "compression_ratio": 16640.0
    }
  ],
  "training_calibration": {
    "rcwe_train_loss": 1.7760855356852214,
    "tf_train_loss": 10.156139373779297
  },
  "head_to_head_mutation_benchmark": {
    "rcwe_post_mutation_mse": 0.003928151959553361,
    "tf_post_mutation_mse": 0.10634138882160186,
    "rcwe_hamiltonian_drift": 0.18184322565793992,
    "tf_hamiltonian_drift": 0.6579184010624886
  },
  "causal_identification": {
    "accuracy": 1.0,
    "regime_1_posteriors": [0.0, 1.0, 0.0],
    "regime_1_residuals": [2.73686, 1.24079, 2.67823],
    "dominant_regime_id": 1,
    "mdo_action_idx": 3,
    "mdo_target_nodes": [5],
    "mdo_divergence_score": 0.04739794507622719
  }
}
```

---

## 6. Transition to RCWE v0.3: Multi-Seed Confidence Intervals & Apple Silicon `mps` Acceleration

While v0.2.1 validated single-seed (`seed=42`) calibration and CPU scaling, defense-grade scientific validation requires:
1. **Multi-Seed Statistical Rigor (`Seeds = [42, 101, 202, 303, 404]`):** Reporting sample means, standard deviations, 95% confidence intervals ($\mu \pm t_{0.975} \cdot 	ext{SEM}$), and paired $t$-statistics across independent random initializations.
2. **Apple Silicon Metal Performance Shaders (`mps`) GPU Execution:** Decomposition of complex wavefield operations into MPS-native `(real, imag)` float32 tensor kernels coupled with `torch.mps.synchronize()` and `torch.mps.current_allocated_memory()` hardware telemetry.
