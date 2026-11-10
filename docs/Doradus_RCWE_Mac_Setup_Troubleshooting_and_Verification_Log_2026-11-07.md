# Doradus Labs — Resonant Causal World Engine (RCWE)
## Complete Mac Setup, Troubleshooting, Repair, and Verification Record

**Date:** November 7, 2026  
**Project:** Resonant Causal World Engine (RCWE)  
**Environment:** MacBook Pro, macOS (Darwin), zsh, Homebrew Python 3.14.5  
**Purpose:** Preserve the steps, commands, code changes, observed results, and remaining work from the RCWE installation and first successful demonstration.

> **Final outcome:** RCWE was successfully installed in a Python virtual environment on the Mac, all five supplied tests passed, and `rcwe-bench` ran to completion. One nonfatal warning remained because NumPy had not been installed. The numerical memory figures printed by `rcwe-bench` are **theoretical estimates**, not measured proof of performance or architectural superiority.

---

## 1. Executive Summary

The goal was to run a Python bootstrap program that generates a prototype AI system called **Resonant Causal World Engine (RCWE)**. The bootstrap program writes out the project modules, installs the generated package in editable mode with development dependencies, and runs its tests.

The work proceeded through several separate problems:

1. **`python` was not found in zsh.** On this Mac, Python was installed as `python3` via Homebrew.
2. **Homebrew blocked direct package installation.** `pip install` raised `externally-managed-environment` under PEP 668. A Python virtual environment fixed this.
3. **The generated `pyproject.toml` was malformed.** The bootstrap script appended a literal backslash-n (`\n`) rather than a real newline to every generated file.
4. **The same file-writing bug caused syntax errors across the source code and tests.** A compiler check exposed the widespread pattern.
5. **A PyTorch complex/real dtype mismatch was discovered in the engine.** The branch weights needed casting to the complex dtype of the worldline state tensor before an `einsum` operation.
6. **Download links for prepared ZIP/installer files were not usable in the chat interface.** A Terminal-only script patched the user's existing bootstrap file instead.
7. **A freshly generated project installed and passed its tests.** Both the bootstrap-driven tests and a second manual test run reported **5 passed**.
8. **The demonstration CLI executed successfully.** It printed analytical memory estimates and a causal diagnostic advisory.

### Verified final state

| Item | Result | Evidence from the Mac run |
|---|---|---|
| Python command and virtual environment | Working | Python 3.14.5 at `~/Downloads/.venv/bin/python` |
| Editable project installation | Successful | `Successfully installed rcwe-0.1.0` |
| Python test collection | Successful | `collected 5 items` |
| RCWE tests | **5 / 5 passed** | Test suite completed twice |
| `rcwe-bench` CLI | Successful | Hardware summary and diagnostic advisory printed |
| NumPy | Not installed at time of run | Nonfatal PyTorch warning |
| Measured transformer-vs-RCWE superiority | **Not established** | Benchmark output is a formula-based estimate |
| Mac GPU/Triton accelerated kernel | **Not verified** | Mac run used non-CUDA path |

---

## 2. Source Files and Project Layout

### Input bootstrap script

The source provided for review was:

```text
~/Downloads/rcwe-code-1794019784929.py
```

The program is a repository generator. It defines a `FILES` dictionary containing source files and tests, then writes them to a `rcwe-engine` directory. After creating the files, it performs:

```python
subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", ".[dev]"])
subprocess.check_call(["pytest", "-v"])
```

### Original project files generated

The original bootstrap script defines **19 generated files**:

```text
rcwe-engine/
├── pyproject.toml
├── rcwe/
│   ├── __init__.py
│   ├── cli.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   ├── metabolic.py
│   │   ├── resonance.py
│   │   ├── symplectic.py
│   │   └── wave_codec.py
│   ├── diagnostics/
│   │   ├── __init__.py
│   │   ├── nl_codec.py
│   │   └── profiler.py
│   ├── kernels/
│   │   ├── __init__.py
│   │   └── triton_holographic.py
│   └── sandbox/
│       ├── __init__.py
│       ├── baseline_tf.py
│       └── cpfn_env.py
└── tests/
    ├── test_invariance.py
    └── test_plasticity_and_mdo.py
```

**Packaging discrepancy identified:** `pyproject.toml` says `readme = "README.md"`, but the original bootstrap's `FILES` dictionary does not create that README. As a result, the successful Mac installation still produced a **nonfatal build warning** about `README.md` being missing. A separately prepared corrected distribution included a README.

---

## 3. Chronological Troubleshooting and Repairs

### 3.1. Initial issue: `python` not found

**Observed error:**

```text
zsh: command not found: python
stevegolter@Mac Downloads %
```

**Diagnosis:** Current macOS/Homebrew installations commonly expose Python as `python3`, not necessarily `python` outside a virtual environment.

**Initial checks and commands recommended:**

```bash
python3 --version
python3 your_script.py
```

For the actual bootstrap script:

```bash
cd ~/Downloads
python3 rcwe-code-1794019784929.py
```

An optional shell alias was discussed for cases where `python` needs to resolve to `python3`:

```bash
echo 'alias python=python3' >> ~/.zshrc
source ~/.zshrc
```

The alias was **optional**, not required for the final solution. Activating a virtual environment provided a working `python` command.

### 3.2. Second issue: `externally-managed-environment`

**Observed error:**

```text
error: externally-managed-environment
× This environment is externally managed
```

The source was invoking `pip install -e .[dev]` through the Homebrew interpreter. The traceback showed Homebrew Python 3.14.5 and an error from the script's `main()` at the package-installation step.

**Diagnosis:** Homebrew protects its Python installation from ordinary system-wide pip changes (PEP 668).

**Resolution:** Create and activate a virtual environment in Downloads.

```bash
cd ~/Downloads
python3 -m venv .venv
source .venv/bin/activate
python --version
which python
python rcwe-code-1794019784929.py
```

Successful environment indicator:

```text
(.venv) stevegolter@Mac Downloads %
```

Python path observed during later testing:

```text
/Users/stevegolter/Downloads/.venv/bin/python
```

**Safety decision:** We did **not** use `--break-system-packages` or make changes to Homebrew's global Python installation.

### 3.3. Third issue: malformed TOML metadata

With the virtual environment active, pip progressed farther but failed while parsing the generated package configuration.

**Key error:**

```text
tomllib.TOMLDecodeError:
Expected newline or end of document after a statement
(at line 20, column 29)
```

**Inspection command:**

```bash
cd ~/Downloads/rcwe-engine
nl -ba pyproject.toml | sed -n '1,35p'
```

The beginning of the file showed valid metadata and dependencies:

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "rcwe"
version = "0.1.0"
description = "Resonant Causal World Engine: Symplectic, Phase-Coherent, and Metabolically Plastic AI Architecture"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.1.0",
]

[project.optional-dependencies]
gpu = ["triton>=2.1.0"]
dev = ["pytest>=7.4.0"]

[project.scripts]
rcwe-bench = "rcwe.cli:main"
```

The displayed line 20 in the corrupted output included additional literal trailing characters where the line should have ended.

**Temporary one-file correction proposed:**

```bash
sed -i '' '20c\
rcwe-bench = "rcwe.cli:main"
' pyproject.toml
```

**Next installation commands:**

```bash
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest -v
```

The quoting around `'.[dev]'` is appropriate in zsh. The pip upgrade was optional; the pip update notification was not the cause of the error.

### 3.4. Fourth issue: Python source and test syntax errors

After the TOML problem was bypassed, pytest failed to collect tests:

```text
collected 0 items / 2 errors
```

The two failing files were:

```text
tests/test_invariance.py
tests/test_plasticity_and_mdo.py
```

Both ended a Python statement with the literal characters `\n`:

```python
assert field_short.numel() == field_long.numel(), "Standing field violated O(1) memory invariance!"\n
```

and:

```python
assert f"Action #{best_idx}" in advisory_text\n
```

These are syntactically invalid when `\n` is outside a quoted string.

**Diagnostic commands requested:**

```bash
grep -RFn --exclude-dir='.git' --exclude-dir='__pycache__' '\n' .
grep -RFn --exclude-dir='.git' --exclude-dir='__pycache__' '\t' .
python -m compileall -q rcwe tests
```

The pasted Terminal result did not show matches from the two grep commands, but `compileall` identified many source files ending with the same invalid literal suffix. It is the compile results, not the empty grep output, that establish the widespread corruption.

**Affected generated Python files shown in the compiler output:**

```text
rcwe/__init__.py
rcwe/cli.py
rcwe/core/__init__.py
rcwe/core/engine.py
rcwe/core/metabolic.py
rcwe/core/resonance.py
rcwe/core/symplectic.py
rcwe/core/wave_codec.py
rcwe/diagnostics/__init__.py
rcwe/diagnostics/nl_codec.py
rcwe/diagnostics/profiler.py
rcwe/kernels/__init__.py
rcwe/kernels/triton_holographic.py
rcwe/sandbox/__init__.py
rcwe/sandbox/baseline_tf.py
rcwe/sandbox/cpfn_env.py
tests/test_invariance.py
tests/test_plasticity_and_mdo.py
```

This established that editing the two test files alone was not sufficient.

**Intermediate suggestion:** A backup-first, end-of-line-only repair was proposed for the existing files, avoiding a global replacement that might damage legitimate string escape sequences. The provided log does **not** establish that this particular intermediate repair was run; the ultimately verified resolution was to fix the generator itself and regenerate a clean copy.

### 3.5. Root-cause discovery: error in bootstrap file-writing logic

Review of the complete uploaded bootstrap script identified the root cause at approximately **line 918**.

**Original, incorrect code:**

```python
file_path.write_text(content.strip() + "\\n", encoding="utf-8")
```

The string `"\\n"` evaluates to a literal backslash followed by `n`, not a line break.

**Corrected code:**

```python
file_path.write_text(content.strip() + "\n", encoding="utf-8")
```

The string `"\n"` evaluates to a real newline, correctly terminating generated source and configuration files.

**Why this matters:** Because all generated files passed through the same `write_text()` call, one bootstrap defect corrupted the TOML file and every Python file. Fixing the generator prevents the problem from reappearing each time the repository is regenerated.

### 3.6. Additional code bug: real/complex tensor mismatch

During verification of the regenerated code, a separate numerical type mismatch was identified in `rcwe/core/engine.py` at approximately **line 427 of the original bootstrap source**.

**Original code:**

```python
z_consensus = torch.einsum("kb,kbld->bld", branch_weights, stacked_z)
```

`branch_weights` is real-valued, while `stacked_z` contains complex-valued worldline states. The relevant PyTorch operation expects compatible dtypes.

**Corrected code:**

```python
z_consensus = torch.einsum(
    "kb,kbld->bld",
    branch_weights.to(stacked_z.dtype),
    stacked_z,
)
```

A one-line equivalent was used in the patch:

```python
z_consensus = torch.einsum("kb,kbld->bld", branch_weights.to(stacked_z.dtype), stacked_z)
```

After these two fixes, the supplied five tests passed in the Mac environment.

### 3.7. Downloadable packages were prepared, but links failed

Corrected script/ZIP artifacts were prepared during the conversation, including names such as:

```text
rcwe-code-1794019784929-fixed.py
rcwe-engine-fixed.zip
Doradus_RCWE_Complete_Fixed.zip
Doradus_RCWE_Installer_Fixed.py
```

However, the chat download links did not work for the user. We switched to a **Terminal-only repair** using the original script already present on the Mac. This removed any dependence on downloading corrected files.

The separately prepared distribution ZIP contained the complete fixed project and installer. Its project had **20 generated files** (the 19 original files plus `README.md`); the ZIP contained **21 entries** including the installer. The Terminal-only regeneration used the patched **original** script, so its run still showed the missing-README warning.

### 3.8. Final successful solution: patch and regenerate in a new directory

The following Python script was run within the active virtual environment. It changed the two known source defects and wrote an independent, patched bootstrap script without overwriting the original:

```bash
cd ~/Downloads
source .venv/bin/activate

python - <<'PY'
from pathlib import Path

src = Path.home() / "Downloads" / "rcwe-code-1794019784929.py"
data = src.read_text(encoding="utf-8")

# Fix corrupted newline generation.
old = 'content.strip() + "\\\\n"'
new = 'content.strip() + "\\n"'
assert data.count(old) == 1, "Newline fix target not found"
data = data.replace(old, new)

# Fix PyTorch complex tensor operation.
old = 'torch.einsum("kb,kbld->bld", branch_weights, stacked_z)'
new = 'torch.einsum("kb,kbld->bld", branch_weights.to(stacked_z.dtype), stacked_z)'
assert data.count(old) == 1, "Tensor fix target not found"
data = data.replace(old, new)

dest = src.with_name("rcwe-code-fixed-local.py")
dest.write_text(data, encoding="utf-8")

print("SUCCESS: Created corrected RCWE installer")
print(dest)
PY
```

**Observed Mac result:**

```text
SUCCESS: Created corrected RCWE installer
/Users/stevegolter/Downloads/rcwe-code-fixed-local.py
```

Next, the patched bootstrap was executed from a new, uniquely named workspace, preventing the old corrupted project from being overwritten:

```bash
workdir=$(mktemp -d "$HOME/Downloads/rcwe-clean-XXXXXX")
cd "$workdir"
python ~/Downloads/rcwe-code-fixed-local.py
```

The successful workspace shown in the pasted Mac log was:

```text
/Users/stevegolter/Downloads/rcwe-clean-hJHYBV/
```

The generator wrote its output beneath:

```text
/Users/stevegolter/Downloads/rcwe-clean-hJHYBV/rcwe-engine/
```

The console printed the created files, installed the package in editable mode, then ran pytest.

---

## 4. Final Mac Installation Results

### 4.1. Installed versions observed

| Component | Mac result |
|---|---|
| Operating system | macOS / Darwin |
| Shell | zsh |
| Python | 3.14.5 |
| Python interpreter | `/Users/stevegolter/Downloads/.venv/bin/python` |
| PyTorch | 2.15.0 |
| pytest | 9.2.0 |
| pluggy | 1.7.0 |
| RCWE package | 0.1.0, installed editable |
| NumPy | Missing at the time of the captured output |

### 4.2. Installation output

The installer built an editable wheel and reported:

```text
Successfully built rcwe
Successfully installed rcwe-0.1.0
```

It also reported a nonfatal warning similar to:

```text
BuildBackendWarning: File '.../rcwe-engine/README.md' cannot be found
```

**Reason:** The original generator declared a README in package metadata without creating the file. This is a packaging-documentation omission, not a runtime failure.

### 4.3. The five tests — all passed

The following test cases ran successfully:

| Test | What it checks | Status |
|---|---|---|
| `test_symplectic_leapfrog_time_reversibility` | Forward/reverse step numerical agreement | PASS |
| `test_hamiltonian_conservation_boundedness` | Bounded energy drift in a test step | PASS |
| `test_holographic_o1_memory_invariance` | Fixed size of the returned standing-field tensor across two sequence lengths | PASS |
| `test_backprop_free_metabolic_plasticity_and_sleep` | Fast local weight adaptation and consolidation | PASS |
| `test_mdo_epistemic_foraging_and_nl_codec` | Action selection and advisory text generation | PASS |

Observed bootstrap test summary:

```text
collected 5 items
...
5 passed, 1 warning in 0.69s
```

The user then independently reran the test suite and the CLI:

```bash
cd rcwe-engine
python -m pytest -v
rcwe-bench
```

Second observed test summary:

```text
5 passed, 1 warning in 0.42s
```

Both runs are directly supported by the pasted Mac Terminal output.

### 4.4. Remaining warning: NumPy not installed

Both test and CLI output included a warning from PyTorch:

```text
UserWarning: Failed to initialize NumPy: No module named 'numpy'
```

The warning did **not** prevent the tests or demonstration from running.

**Recommended cleanup command (not yet confirmed executed):**

```bash
source ~/Downloads/.venv/bin/activate
python -m pip install numpy
python -m pytest -v
rcwe-bench
```

Do not treat this as an established fix until the command is run and the warning disappears.

---

## 5. `rcwe-bench` Output and Interpretation

### 5.1. Theoretical hardware scaling results

The CLI printed:

```text
=== RCWE HARDWARE SCALING SUMMARY ===
L=256   | TF VRAM:    2560.0 KB | RCWE O(1) VRAM: 32.0 KB | Compression:    80.0x
L=1024  | TF VRAM:   34816.0 KB | RCWE O(1) VRAM: 32.0 KB | Compression:  1088.0x
L=4096  | TF VRAM:  532480.0 KB | RCWE O(1) VRAM: 32.0 KB | Compression: 16640.0x
L=8192  | TF VRAM: 2113536.0 KB | RCWE O(1) VRAM: 32.0 KB | Compression: 66048.0x
```

| Sequence length | Formula-derived Transformer memory | Formula-derived RCWE state memory | Reported ratio |
|---:|---:|---:|---:|
| 256 | 2,560 KB | 32 KB | 80× |
| 1,024 | 34,816 KB | 32 KB | 1,088× |
| 4,096 | 532,480 KB | 32 KB | 16,640× |
| 8,192 | 2,113,536 KB | 32 KB | 66,048× |

**Critical interpretation:** These are estimates calculated by functions inside `rcwe/diagnostics/profiler.py`, **not measurements of actual GPU VRAM, CPU resident memory, training memory, inference memory, or end-to-end latency**. On the Mac, the labels still read `VRAM`, but this is not evidence of GPU usage. The RCWE standing-field state is fixed-size under the specific comparison, while other operations (for example FFT processing and storing output sequences) can still depend on input sequence length. The large ratios cannot be presented as experimentally validated efficiency gains.

### 5.2. Simulated diagnostic advisory

The run also printed:

```text
[RCWE CAUSAL DIAGNOSTIC ADVISORY]
STATUS: AMBIGUOUS LOOK-ALIKE SUPERPOSITION (ACTIVE MDO INTERVENTION REQUIRED)
```

Representative results:

```text
Epistemic Ambiguity: 100.0%
Posterior Weights: W0=0.333 | W1=0.333 | W2=0.333
Hamiltonian Drift: W0=0.0000 | W1=0.0000 | W2=0.0000
Leading Hypothesis: W2: Symplectic Trunk Susceptance Degradation
Standing Wavefield |Psi|: 0.0733
Peak Predictive Surprise: Node v_1 (|e|=0.756), Node v_3 (|e|=0.608), Node v_9 (|e|=0.416)
Fast Synaptic Rewiring: Channel c_6, Channel c_8, Channel c_13
Recommended Intervention: Execute do(Action #4) on Control Nodes [v_5, v_6]
Counterfactual Leverage: Expected World-Line Variance Split = 0.00000
```

**Interpretation and limitations:**

- All three displayed posterior weights were equal to three decimal places, so the system did **not** make a convincing distinction among competing worldline explanations.
- The reported `W2` leading hypothesis should not be treated as a reliable diagnosis when the displayed confidence is essentially tied and the ambiguity is 100%.
- Printed Hamiltonian drift `0.0000` is a rounded display value, not proof of exact energy conservation.
- Printed divergence `0.00000` is a rounded display value, not evidence that the chosen action has useful discriminatory power.
- The experiment uses simulated inputs and a sandbox. No production data, generalization validation, or comparative learning benchmark was shown in this session.

---

## 6. Clean Restart and Daily-Use Commands

### 6.1. Return to the verified Mac project

Use the exact directory seen in the successful Terminal log:

```bash
cd ~/Downloads/rcwe-clean-hJHYBV/rcwe-engine
source ~/Downloads/.venv/bin/activate
python -m pytest -v
rcwe-bench
```

If the workspace was moved or removed, locate it with:

```bash
find ~/Downloads -maxdepth 3 -type d -name rcwe-engine -print
```

### 6.2. Install NumPy and rerun (recommended)

```bash
cd ~/Downloads/rcwe-clean-hJHYBV/rcwe-engine
source ~/Downloads/.venv/bin/activate
python -m pip install numpy
python -m pytest -v
rcwe-bench
```

### 6.3. Recreate a clean project from the locally patched generator

This creates a **new** uniquely named folder each time and does not require downloading a ZIP:

```bash
cd ~/Downloads
source .venv/bin/activate
workdir=$(mktemp -d "$HOME/Downloads/rcwe-clean-XXXXXX")
cd "$workdir"
python ~/Downloads/rcwe-code-fixed-local.py
cd rcwe-engine
python -m pytest -v
rcwe-bench
```

**Caution:** The original bootstrap always writes a folder called `rcwe-engine` in its current working directory, so run it inside a new clean workspace instead of an important existing directory.

### 6.4. Useful diagnostics

```bash
# Check interpreter and package state.
python --version
which python
python -m pip show torch pytest rcwe

# Validate all project Python files without executing tests.
python -m compileall -q rcwe tests

# Run tests with full test names.
python -m pytest -v

# Run the built-in demonstration.
rcwe-bench
```

---

## 7. RCWE Prototype Components Seen in the Uploaded Source

The original bootstrap implemented a small experimental PyTorch architecture containing:

| Component | File(s) | Purpose in this prototype |
|---|---|---|
| Wavepacket codec | `rcwe/core/wave_codec.py` | Converts real-valued input features to complex-valued amplitude/phase representations and decodes them. |
| Holographic resonance layer | `rcwe/core/resonance.py` | Applies FFT-based mixing and maintains a frequency-band standing field. |
| Resonance kernel and fallback | `rcwe/kernels/triton_holographic.py` | Provides a Triton/CUDA-oriented path where supported and a PyTorch fallback. |
| Symplectic causal block | `rcwe/core/symplectic.py` | Applies leapfrog-style position/momentum updates and tracks an energy-drift measure. |
| Metabolic plastic synapse | `rcwe/core/metabolic.py` | Maintains fast weights and an ATP-like budget and supports consolidation. |
| World engine and MDO search | `rcwe/core/engine.py` | Combines multiple worldline branches and scores candidate actions by branch divergence. |
| Simulated environment | `rcwe/sandbox/cpfn_env.py` | Produces synthetic cyber-physical flow states and regime variants. |
| Baseline transformer | `rcwe/sandbox/baseline_tf.py` | Defines a reference transformer architecture; this session did not demonstrate a measured head-to-head training comparison. |
| Natural-language diagnostics | `rcwe/diagnostics/nl_codec.py` | Converts model signals into structured diagnostic packets and advisory text. |
| Profiling formulas | `rcwe/diagnostics/profiler.py` | Computes estimated FLOPs and state-memory comparisons. |
| Command-line demo | `rcwe/cli.py` | Runs the scaling summary and one simulated diagnostic demonstration. |
| Test suite | `tests/*.py` | Covers five basic behaviors. |

The project is an **experimental proof of concept**, not a trained or externally validated general AI model.

---

## 8. Remaining Work and Recommended Next Phase

The Mac installation problem is resolved. The next task is scientific validation of the architecture, not more setup work.

### Immediate cleanup

1. Install NumPy, rerun tests, and verify the warning disappears.
2. Add the missing `README.md` to the Mac-generated project and document how to install and run it.
3. Keep the corrected bootstrap under version control. The original source is known to regenerate corrupted files and should not be used unchanged.
4. Preserve test outputs, environment/package versions, and a copy of the successful project.

### Scientific and engineering validation

1. **Measure actual memory usage** — CPU RSS and peak allocated memory, plus MPS/CUDA memory where applicable, rather than only analytical state estimates.
2. **Measure wall-clock time and throughput** — warm-up, repeated trials, sequence-length scaling, data-transfer overhead, and inference versus training memory.
3. **Make comparisons fair** — equal or comparable parameter counts, task difficulty, training budget, batch sizes, and accuracy targets against transformer and other baselines.
4. **Validate learning** — predictive accuracy, held-out generalization, training stability, and sensitivity to random seeds.
5. **Validate causal inference** — known-ground-truth interventions, regime identification metrics, false positives, calibration, and whether recommended actions actually separate hypotheses.
6. **Profile sequence-dependent allocations** — FFT, intermediate activations, Python lists, and outputs need explicit accounting even if the standing-field tensor is constant-size.
7. **Separate CPU/Mac results from CUDA claims** — Triton/CUDA code paths were not exercised during this Mac run.
8. **Inspect numerical robustness** — gradient flow, real/complex dtype consistency, energy drift under longer rollouts, and reproducibility.
9. **Document model limitations** — avoid presenting simulated diagnostics as production evidence or theoretical memory estimates as benchmark measurements.

### Proposed next milestone

**RCWE Scientific Validation Benchmark, v0.2**: a reproducible evaluation harness that saves raw metrics, environment information, baseline comparisons, summary plots, and a defensible report on what RCWE does or does not improve.

---

## 9. Resolved vs. Unresolved Issues

| Issue | Status | Notes |
|---|---|---|
| `zsh: command not found: python` | **Resolved** | Use `python3` or activate `.venv`. |
| Homebrew `externally-managed-environment` | **Resolved** | Install within `~/Downloads/.venv`, not globally. |
| TOML parse failure at line 20 | **Resolved** | Fix bootstrap file-writing newline, regenerate. |
| Widespread literal `\n` syntax errors | **Resolved** | Root cause repaired in locally patched script. |
| Real/complex branch weighting operation | **Resolved for supplied tests** | Cast weights to worldline-state dtype. |
| Failed chat download links | **Worked around** | Patched source directly on Mac. |
| Package installation | **Resolved** | `rcwe-0.1.0` installed editable. |
| Test suite | **Passed** | 5/5 tests, twice on the Mac. |
| Built-in CLI demonstration | **Passed** | Printed memory summary and advisory. |
| Missing `README.md` warning | **Outstanding on Mac-generated copy** | Nonfatal; alternate corrected distribution includes README. |
| Missing NumPy warning | **Outstanding in captured log** | `python -m pip install numpy` recommended but not yet confirmed. |
| Actual speed or memory advantage | **Unverified** | Requires instrumented, accuracy-matched benchmarks. |
| Model's ability to identify causes | **Unverified** | Displayed posterior weights were effectively tied. |

---

## 10. Handoff Notes for a New AI Agent or Developer

**Current working state:** A clean RCWE project exists at `~/Downloads/rcwe-clean-hJHYBV/rcwe-engine` in the Mac session shown. The active virtual environment is `~/Downloads/.venv`. The patched generator created during the troubleshooting is `~/Downloads/rcwe-code-fixed-local.py`. Five tests and the command-line demonstration ran successfully.

**Known critical patches:**

```diff
- file_path.write_text(content.strip() + "\\n", encoding="utf-8")
+ file_path.write_text(content.strip() + "\n", encoding="utf-8")
```

```diff
- z_consensus = torch.einsum("kb,kbld->bld", branch_weights, stacked_z)
+ z_consensus = torch.einsum("kb,kbld->bld", branch_weights.to(stacked_z.dtype), stacked_z)
```

**Important constraints for follow-on work:**

- Do not replace the virtual environment with `--break-system-packages` installations.
- Do not re-run the **unpatched** generator expecting it to produce valid project files.
- Avoid overwriting the working project during experimentation; use a new directory or version control.
- Treat `rcwe-bench` memory and FLOPs figures as **modeling estimates** until replaced by instrumented measurements.
- Treat printed causal hypotheses as prototype outputs, not verified causal determinations.
- Preserve and expand the five passing tests before major refactoring.
- Establish a reproducible benchmark protocol and genuine baseline comparisons before making technical, commercial, or patent-related performance claims.

### Source material used for this record

- Original uploaded bootstrap script: `rcwe-code-1794019784929.py` (932 lines).
- User-provided command output containing the first source compilation failures.
- User-provided command output containing the successful clean bootstrap, two complete test runs, and `rcwe-bench` result.
- Code review of the original generator and the corrected distribution prepared during this chat.

**Record boundary:** This document covers the work and results visible in the conversation through the successful November 7, 2026 Mac demonstration and the subsequent request to document it. It does not assert that recommended future validation or warning cleanup steps have already been performed.
