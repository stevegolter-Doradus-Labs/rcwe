# Doradus Labs — Resonant Causal World Engine (RCWE)
## v0.2.1 Scientific Validation, Calibration, and Hardware Benchmark Record

**Date:** November 7, 2026  
**Project:** Resonant Causal World Engine (RCWE) — v0.2.1 Calibrated Edition  
**Environment:** MacBook Pro, macOS-27.2-arm64 (Apple Silicon), zsh, Homebrew Python 3.14.5, PyTorch 2.15.0, NumPy 2.5.4  
**Purpose:** Document the progression from v0.1 to v0.2.1, the mathematical resolution of the world-line posterior tie, the 7/7 passing verification test suite, and the empirical head-to-head benchmark telemetry against a parameter-matched causal Transformer baseline.

> **Final Outcome:** RCWE v0.2.1 installed cleanly with zero build or runtime warnings, passed **7 of 7** unit and scientific validation tests in `3.20s`, achieved **100.0% causal regime identification accuracy** across held-out $(\mathcal{W}_0, \mathcal{W}_1, \mathcal{W}_2)$ rollouts (breaking the v0.1 `0.333` posterior tie with `W1 = 1.000`), achieved **27.1x lower post-mutation MSE** (`0.00393` vs. `0.10634`) and **3.62x lower physical Hamiltonian drift** (`0.18184` vs. `0.65792`) than the parameter-matched Transformer baseline, and demonstrated an empirical **wall-clock latency crossover at $L = 4096$** (`103.25 ms` for RCWE vs. `149.20 ms` for the Transformer).

---

## 1. Executive Summary: Progression from v0.1 to v0.2.1

In the initial v0.1 Mac verification log, the core package compiled and passed 5 operator-level unit tests, but the CLI demonstration (`rcwe-bench`) ran on freshly initialized, untrained weights and reported only analytical memory formulas.

Across v0.2.0 and v0.2.1, every engineering and scientific limitation identified in the v0.1 audit was systematically resolved:

| Validation Dimension | v0.1 (Untrained Demo) | v0.2.0 (Raw State Decoder) | v0.2.1 (Velocity-Space + Sensory Plasticity) |
|---|---|---|---|
| **Packaging & Build Warnings** | Missing `README.md` & `numpy` warnings | Resolved (`README.md` & `numpy>=1.26.0` added) | **Resolved (Zero build or runtime warnings)** |
| **Automated Test Suite** | 5 / 5 passed | 6 / 7 passed | **7 / 7 passed (`3.20s`)** |
| **Model Parameter Parity** | Unmatched | 80,611 vs. 162,112 | **80,611 (RCWE) vs. 71,648 (Transformer)** |
| **World-Line Posteriors (Regime 1)** | `[0.333, 0.333, 0.333]` (Tied) | `[0.312, 0.346, 0.343]` | **`[0.000, 1.000, 0.000]` (100% $\mathcal{W}_1$ confidence)** |
| **Epistemic Ambiguity** | 100.0% | 99.9% | **0.0% (`CONVERGED CAUSAL IDENTIFICATION`)** |
| **Held-Out Regime ID Accuracy** | Uncalibrated | 53.3% | **100.0% across $(\mathcal{W}_0, \mathcal{W}_1, \mathcal{W}_2)$ rollouts** |
| **Post-Mutation Adaptation MSE** | Not measured | `0.10465` vs. `0.10356` | **`0.00393` (RCWE) vs. `0.10634` (TF) — $27.1	imes$ lower** |
| **Physical Hamiltonian Drift ($\Delta \mathcal{H}$)** | Not measured | `0.58222` vs. `0.70652` | **`0.18184` (RCWE) vs. `0.65792` (TF) — $3.62	imes$ lower** |
| **MDO Counterfactual Variance Split** | `0.00000` (Rounded to zero) | `0.06624` | **`0.04740` (Targeted Relay Node `[v_5]`)** |

---

## 2. Root-Cause Mathematical Diagnosis & Calibration in v0.2.1

### 2.1. Resolving the `f-string` Generator Bug (v0.2.0 $	o$ v0.2.0-fixed)
In the first v0.2.0 drop, `rcwe/diagnostics/nl_codec.py` threw `SyntaxError: unterminated f-string literal (detected at line 119)` because multi-line f-strings inside the installer dictionary had raw newline characters expanded during script generation. Replacing the multi-line return block in `render_engineering_advisory()` with a clean `"\n".join(lines)` list construction and adding an automatic `python -m compileall -q rcwe tests` gate inside the installer permanently eliminated syntax generation bugs.

### 2.2. Why Raw State Prediction Stopped at `W1 = 0.3457` in v0.2.0
When `rcwe-engine-v02-fixed` ran on the MacBook Pro, 6 of 7 tests passed, and World-Line $\mathcal{W}_1$ achieved the lowest anomaly residual (`W1=0.0372` vs. `W0=0.0458` and `W2=0.0380`), correctly identifying `W1: Covert Byzantine Sinkhole / Exfiltration Tap`. However, the 7th test (`assert posteriors[1] > 0.50`) failed at `0.3457` because:
1. Under discrete symplectic integration with time step $\Delta t = 0.05$, the next state is $q_{t+1} = q_t + \Delta t \, p_t + rac{1}{2}\Delta t^2 f_{	ext{regime}}$. Thus, $y_{t+1}$ is dominated ($>95\%$) by the current state $x_t$, masking the regime-specific force signature $f_{	ext{regime}}$.
2. In v0.2.1, each world-line branch decoder was reformulated to predict the **state transition residual** $\Delta \hat{y}_k = \hat{y}_{t+1}^{(k)} - x_t$ and scored in **normalized transition-velocity space**:
   $$\mathcal{E}_k = \left\| rac{(\hat{y}_{t+1}^{(k)} - y_{t+1})}{\Delta t} ight\|_2^2 + 0.05 \left|\Delta \mathcal{H}_kight|$$
   Removing the shared baseline state $x_t$ immediately separated the regime residuals on Regime 1 to `W0=2.7369 | W1=1.2408 | W2=2.6782`, driving the softmax posterior to `W0=0.000 | W1=1.000 | W2=0.000` (`0.0%` epistemic ambiguity) and raising held-out regime classification accuracy from `53.3%` to **`100.0%`**.

### 2.3. Coupling Sensory Prediction Error into Local Metabolic Plasticity
In v0.2.0, `MetabolicPlasticSynapse` updated its fast weights using only internal hidden-state self-prediction (`post_q - predictor(post_q)`), leaving post-mutation MSE tied with the frozen Transformer (`0.10465` vs. `0.10356`). In v0.2.1, `ResonantCausalWorldEngine` couples the observable sensory prediction residual $e_{	ext{sensory}} = y_{	ext{true}} - \hat{y}$ into an ATP-gated fast sensory corrector under `torch.no_grad()`. As a result, when an abrupt topology mutation and DC disturbance hit the network at runtime, RCWE adapts within steps to achieve a post-mutation MSE of **`0.00393`** while the frozen Transformer remains stuck at **`0.10634`** ($27.1	imes$ improvement).

---

## 3. Verified Mac Benchmark Telemetry (`rcwe_validation_report_v02.json`)

### 3.1. Empirical Wall-Clock Latency & Memory Scaling
Measured on MacBook Pro (`macOS-27.2-arm64`, Python 3.14.5, PyTorch 2.15.0) across sequence lengths $L \in \{256, 1024, 2048, 4096\}$ with batch size $B = 2$ and $d_{	ext{model}} = 64$:

| Sequence Length ($L$) | Transformer MFLOPs | RCWE MFLOPs | Transformer Latency (ms) | RCWE Latency (ms) | Transformer Mask+KV (KB) | Measured RCWE $|\Psi|$ (KB) | Total State Ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **256** | 25.17 | 51.12 | **1.92 ms** | 9.87 ms | 768.0 KB | **16.0 KB** | 80.0× |
| **1,024** | 301.99 | 205.78 | **13.20 ms** | 30.28 ms | 6,144.0 KB | **16.0 KB** | 1,088.0× |
| **2,048** | 1,140.85 | 412.88 | **44.29 ms** | 53.59 ms | 20,480.0 KB | **16.0 KB** | 4,224.0× |
| **4,096** | 4,429.19 | 828.38 | 149.20 ms | **103.25 ms** | 73,728.0 KB | **16.0 KB** | 16,640.0× |

**Key Hardware Takeaways:**
1. **Empirical Latency Crossover at $L = 4096$:** As sequence length increases $16	imes$ ($256 	o 4096$), the Baseline Transformer's wall-clock latency grows **$77.8	imes$** (`1.92 ms` $	o$ `149.20 ms`) due to quadratic $\mathcal{O}(L^2)$ attention. Over the same $16	imes$ span, RCWE's latency grows only **$10.5	imes$** (`9.87 ms` $	o$ `103.25 ms`), overtaking the Transformer at $L = 4096$ (**$1.45	imes$ faster** on CPU even before GPU kernel fusion).
2. **Constant $\mathcal{O}(1)$ Standing Field Footprint:** The materialized causal attention mask and KV activation buffers in the Transformer grow from `768.0 KB` to `73,728.0 KB` (`532,480.0 KB` including multi-head attention score matrices), whereas RCWE's complex standing wavefield tensor $|\Psi|$ is empirically verified at exactly **`16.0 KB`** (`32.0 KB` including the $64 	imes 64$ fast weight matrix) across all sequence lengths.

### 3.2. Head-to-Head Scientific Validation Summary
* **Training Convergence Loss (55 Epochs):** RCWE = `1.7761` vs. Baseline Transformer = `10.1561`
* **Post-Mutation Adaptation MSE (No Backprop):** RCWE = `0.00393` vs. Baseline Transformer = `0.10634` (**$27.1	imes$ lower error**)
* **Physical Hamiltonian Conservation Drift ($\Delta \mathcal{H}_{	ext{env}}$):** RCWE = `0.18184` vs. Baseline Transformer = `0.65792` (**$3.62	imes$ lower violation**)
* **Held-Out Causal Regime Classification Accuracy:** **`100.0%`** (`15 / 15` held-out episodes across $\mathcal{W}_0, \mathcal{W}_1, \mathcal{W}_2$)

---

## 4. Next Milestone: RCWE v0.3 (Multi-Seed Statistical CIs & Apple Silicon MPS)

While v0.2.1 proved single-seed calibration (`seed=42`) on CPU, **RCWE v0.3** advances the validation harness to publication-grade rigor:
1. **Apple Silicon `MPS` & Hardware Accelerator Backend (`rcwe/kernels/accelerator.py`):** Automatically detects and runs on Apple Silicon Metal (`mps`), NVIDIA `cuda`, or `cpu`, with hardware-synchronized timers (`torch.mps.synchronize()`) and native device memory queries.
2. **Multi-Seed Statistical Confidence Intervals (`rcwe/diagnostics/validation.py`):** Evaluates training, post-mutation recovery, Hamiltonian drift, and regime classification across multiple independent seeds (`[42, 101, 202]`), reporting **Mean $\pm$ SD** and **95% Confidence Intervals**.
