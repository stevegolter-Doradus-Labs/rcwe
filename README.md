# Resonant Causal World Engine (RCWE) — v0.3.0 Multi-Seed & MPS GPU Edition

RCWE v0.3.0 combines:
1. **Apple Silicon MPS / CUDA / CPU Hardware Accelerator (`rcwe/kernels/accelerator.py`)**
2. **Complex Phase-Space Wavepackets (`rcwe/core/wave_codec.py`)**
3. **O(N log N) Spectral Mixing & O(1) Standing Wavefields (`rcwe/core/resonance.py`)**
4. **Symplectic Causal Manifolds with Leapfrog Verlet Integration (`rcwe/core/symplectic.py`)**
5. **Backprop-Free 3-Factor Metabolic Plasticity & Sleep Consolidation (`rcwe/core/metabolic.py`)**
6. **Multi-Seed Statistical Validation & 95% Confidence Intervals (`rcwe/diagnostics/validation.py`)**

## Quickstart
```bash
python -m pip install -e '.[dev]'
python -m pytest -v
rcwe-bench
```
