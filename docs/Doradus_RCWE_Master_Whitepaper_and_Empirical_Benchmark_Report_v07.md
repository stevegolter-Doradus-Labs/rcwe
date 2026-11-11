# Doradus Labs — Master Whitepaper & Empirical Benchmark Record (v0.7.0)
**Title:** Resonant Causal World Engine (RCWE): A Non-Autoregressive Symplectic, Holographic, and Metabolically Plastic World Model
**Authors:** Steve Golter & Doradus Labs
**Date:** November 9, 2026
**Version:** 0.7.0

---

## Abstract
Autoregressive Causal Transformers suffer from strict structural limits in real-time industrial telemetry, symbolic reasoning, and edge deployment: an $\mathcal{O}(L^2)$ quadratic memory wall, unconstrained physical and algebraic conservation drift, static inference weights that cannot adapt post-deployment without catastrophic forgetting or backpropagation, and passive correlation instead of interventional causality. This whitepaper formalizes the **Resonant Causal World Engine (RCWE v0.7.0)**, a non-autoregressive world model combining Apple Silicon Metal Shading Language (`MSL`) hardware acceleration, complex phase-space wavepackets ($\Psi$), spectral $O(L \log L)$ mixing with strict $O(1)$ standing wavefields, reversible Symplectic Leapfrog Verlet integration, and backprop-free 3-factor metabolic plasticity. Across extensive benchmarks on macOS Apple Silicon (`mps:0`), RCWE v0.7.0 achieves single-pass context scaling up to **$L = 131,072$ tokens in 32.22 ms** (a **33,554,432x state compression ratio** over equivalent Transformer attention workspaces), **160.1x lower post-mutation prediction error**, **8.94x lower physical Hamiltonian conservation drift**, **10.8x closed-loop self-healing fault neutralization**, and **100.0% code next-token accuracy** across a 4-layer hierarchical LM backbone.

---

## 1. Architectural Core & Mathematical Foundations

### 1.1 Phase-Space Wavepacket Codec (`rcwe/core/wave_codec.py`)
Inputs $x_t \in \mathbb{R}^{d_{\text{input}}}$ are mapped into complex phase space via softplus amplitudes and tanh-bounded phase angles:
$$z_t = A(x_t) \exp(i \theta(x_t)) \in \mathbb{C}^{d_{\text{model}}}$$

### 1.2 Spectral Mixing & $\mathcal{O}(1)$ Holographic Standing Wavefield (`rcwe/kernels/metal_msl.py`)
Rather than maintaining an $\mathcal{O}(L^2)$ attention matrix or an $\mathcal{O}(L)$ KV cache, RCWE evolves a fixed-dimension complex standing wavefield tensor $\Psi \in \mathbb{C}^{M \times d_{\text{model}}}$ ($M = 16$ frequency bands, occupying exactly **$8.0 \text{ KB}$ per layer**):
$$\Psi^{(t)} = \Gamma \odot \left( \Psi^{(t-1)} \exp(i \Omega) \right) + z_t \otimes \text{BandDrive}$$
Using contiguous $\text{rfft}$ causal convolution and BLAS Vandermonde state contraction, the recurrence is solved in zero loops and dispatched directly onto Apple Silicon SRAM via an inline **Apple Metal Shading Language (`MSL`) kernel** (`fused_holographic_coherence_gain`).

### 1.3 Symplectic Causal Manifolds & Hamiltonian Dynamics (`rcwe/core/symplectic.py`)
State transitions evolve on a learned Hamiltonian manifold governed by potential $V(q, a)$ and kinetic energy $T(p) = \frac{1}{2} p^2 m^{-1}$ using a reversible Symplectic Leapfrog Verlet integrator:
$$p_{t+\frac{1}{2}} = p_t - \frac{\Delta t}{2} \nabla_q V(q_t, a_t), \quad q_{t+1} = q_t + \Delta t \, p_{t+\frac{1}{2}} m^{-1}, \quad p_{t+1} = p_{t+\frac{1}{2}} - \frac{\Delta t}{2} \nabla_q V(q_{t+1}, a_{t+1})$$

### 1.4 Backprop-Free 3-Factor Metabolic Plasticity (`rcwe/core/metabolic.py`)
To prevent catastrophic forgetting and eliminate gradient backpropagation during deployment, RCWE maintains a slow backbone weight $W_{\text{slow}}$ and an online fast plastic weight $W_{\text{fast}}$ modulated by ATP homeostatic energy budgets ($ATP \in [0.0, 2.0]$) and local predictive surprise:
$$\Delta W_{\text{fast}} = \eta \cdot \tanh(\|e_t\|) \cdot \sigma(10(ATP - 0.25)) \cdot \left( e_t \otimes q_t^{\text{pre}} \right)$$

---

## 2. Empirical Benchmark Results (v0.1 — v0.7.0)

### 2.1 Extreme Context Scaling & Metal `MSL` Acceleration ($L = 256 \to 131,072$)

| Sequence Length ($L$) | Baseline Transformer (`mps`) | RCWE v0.7.0 Metal MSL (`mps`) | Transformer Attention Workspace | Measured RCWE $\|\Psi\|$ | State Compression Ratio |
|---:|---:|---:|---:|---:|---:|
| **256** | 1.45 ms (OK) | 4.12 ms | 2.5 MB | **8.0 KB** | 320x |
| **1,024** | 5.57 ms (OK) | 5.21 ms | 34.0 MB | **8.0 KB** | 4,352x |
| **4,096** | 60.13 ms (OK) | **3.39 ms (17.4x faster)** | 520.0 MB | **8.0 KB** | 66,560x |
| **32,768** | **OOM (Quadratic Wall)** | **8.07 ms** | 16.00 GB | **8.0 KB** | 2,097,152x |
| **65,536** | **OOM (Quadratic Wall)** | **15.29 ms** | 64.00 GB | **8.0 KB** | 8,388,608x |
| **131,072** | **OOM (Quadratic Wall)** | **32.22 ms** | 256.00 GB | **8.0 KB** | **33,554,432x** |

### 2.2 Multi-Seed Scientific, Self-Healing, and Real-World Domain Validation

| Benchmark Suite / Domain | Task / Schema | RCWE v0.7.0 Metric | Baseline Transformer | RCWE Advantage / Gain |
|---|---|---:|---:|---:|
| **Multi-Seed Runtime Topology Mutation** | `Seeds=[42, 101, 202]` | **`0.00071 ± 0.00012` (MSE)** | `0.11367` | **160.1x lower prediction error** |
| **Multi-Seed Hamiltonian Conservation** | `Seeds=[42, 101, 202]` | **`0.11865` ($\Delta \mathcal{H}$)** | `1.06028` | **8.94x lower physical drift** |
| **Closed-Loop Autonomous Self-Healing** | $\mathcal{W}_1$ Byzantine Sinkhole | **`0.03692` (Residual Dev)** | `0.39753` | **10.8x fault stabilization gain** |
| **Hierarchical 4-Layer `RCWE-Deep` LM** | 623-Token Code Stream | **`100.0%` Token Accuracy** | `57.8%` | **+42.2% accuracy (`64 KB` state)** |
| **Universal Grid Telemetry CLI** | `16q / 8u` CSV/JSON | **`0.00021` (MSE)** | `0.10139` | **472.4x lower MSE** |
| **ETT Electricity Transformer Thermal** | `8q / 4u` Overload Shift | **`0.00989` (MSE)** | `1.34143` | **135.6x lower MSE** |
| **SWaT Cyber-Physical Water Treatment** | `12q / 6u` Valve Attack | **`0.04841` (MSE)** | `3.55342` | **73.4x lower MSE** |
| **NASA Turbofan Bearing Degradation** | `10q / 4u` Vibration Drift | **`0.01309` (MSE)** | `1.22663` | **93.7x lower MSE** |

---

## 3. Conclusion & Publication Readiness
RCWE v0.7.0 transitions physics-informed world models to production-grade software. By combining native Apple Metal Shading Language (`MSL`) kernels, closed-loop MDO self-healing controllers, and hierarchical multi-layer backbones, RCWE achieves 131K+ context execution with constant $\mathcal{O}(1)$ memory footprints and zero backpropagation lag.

*Repository:* https://github.com/stevegolter-Doradus-Labs/Doradus-RCWE
