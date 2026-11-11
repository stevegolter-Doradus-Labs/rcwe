# Changelog — Resonant Causal World Engine (RCWE)

## [1.1.0] — 2026-11-11

RCWE becomes usable by people who have telemetry rather than time to read the source: one command
or five lines of Python take a log file to explained anomalies, forecasts and a report.

### Added
- **Toolkit (`rcwe.toolkit`, re-exported from `rcwe`).** `TelemetryModel` with `fit`, `detect`,
  `forecast`, `stream`, `save` and `load`; `load_table` for CSV, TSV, JSON, JSON-lines and `.gz`
  files (Excel and Parquet with the `data` extra), lists of dicts, dicts of columns, NumPy arrays
  and pandas DataFrames.
  - Sensor, control, time and label columns are detected and reported, and every decision is
    listed. The loader works out, per column and from the data, whether dates are day-first or
    month-first, whether a comma is a decimal mark or a thousands separator, and whether a numeric
    clock is in seconds, milliseconds or a compact `YYYYMMDDhhmmss` stamp. Comma, semicolon, tab
    and pipe delimiters, header-less files, comment lines, separate date and time columns, UTF-16
    exports, shuffled rows and repeated timestamps are handled.
  - Anomaly scores in calibrated sigmas from four signals (sudden jump, sustained deviation, out of
    range, flatline), grouped into events with severity, kind, the sensors responsible, and actual
    versus expected values.
  - Calibration that a dirty log cannot bend: scales and ranges come from statistics that a few
    wild readings do not move, so one glitch in the training rows neither hides later faults nor
    floods the scan with alarms. Status flags, running totals, steadily ticking counters and
    constant columns each get the treatment they need.
  - Missing readings are held at the last known value and never scored; the reading after a gap is
    judged against how far a sensor can move in that many steps. Scanning a file and feeding the
    same rows one at a time give the same answer.
  - **Controls as commands.** By default a control is only used up to the row being predicted
    from, because it might be a controller reacting to the sensors. Controls declared with
    `fit(..., commands=...)` or `--commands` (and columns named like `setpoint_*` or `*_sp`) are
    also given for the row being predicted, so the model learns how the sensors respond: forecasts
    follow a planned change (`forecast(..., future_controls=...)`, `--planned-controls`), and
    one-step skill on the demo plant rises from about 25% to about 33%.
  - An event that begins as a control moves says so ("coincides with a change in act_pump_speed
    (0.6 to 0.65)"; `event.control_changes`, `step.controls_changed` when streaming), which tells
    an operator's step from a fault at a glance.
  - Reference mode and adaptive mode. Adaptive mode uses the engine's online plasticity, applied
    only after each prediction has been scored.
  - Forecasts with 80% and 95% intervals taken from backtests on held-out rows, and a per-sensor
    weight that falls back to "holds its latest value" where the model shows no skill.
  - A model card: skill against the naive "next value = current value" baseline on held-out rows,
    overall, per sensor and by forecast horizon.
- **The `rcwe` command** (also `python -m rcwe`): `quickstart`, `analyze`, `fit`, `detect`,
  `forecast`, `watch`, `inspect`, `serve`, `ui`, `bench`, `demo-data`, `doctor`, `version`.
  `--fail-on-anomaly` gives exit status 2 for scripts and CI; every error, including a mistyped
  option, gives 1, so a typo can never be read as "anomalies found". Options are checked before
  any work starts, and an output path that cannot be written is reported up front. `watch` follows
  a growing file like `tail -F`, including across log rotation and truncation.
- **Interactive HTML report** (`result.to_html()` or `--report`): one self-contained file that
  needs no network access, with light and dark themes, hover read-outs, zoom-to-event, a forecast
  panel and a per-sensor trust table.
- **RCWE Studio** (`rcwe serve`): drop a log in the browser, read the report, download the outputs.
  It listens on `127.0.0.1`, refuses requests that do not come from its own pages, cannot be
  framed by another site, keeps the 25 most recent analyses and deletes each upload once it has
  been read.
- Model files (`.rcwe`) carry a checksum and are refused if they are damaged or were written by an
  incompatible version, instead of loading into a model that quietly misbehaves.
- Exports are safe to open: CSV cells that a spreadsheet would run as a formula are neutralised,
  and readings keep twelve significant digits.
- **Built-in demo dataset**: a pump station with six sensors, two controls and five labelled
  faults, used by `quickstart`, the examples and the tests.
- **`rcwe doctor`**: checks Python, PyTorch, NumPy, the accelerator and the output folder, runs an
  end-to-end test, and says how to fix what it finds.
- `docs/USER_GUIDE.md`; `docs/BENCHMARKS.md`, generated from actual runs by
  `scripts/build_benchmarks_doc.py`; three runnable scripts in `examples/`.
- Engine: `ResonantCausalWorldEngine.sleep_consolidate()` and the `mean_atp` property.
- Continuous integration on Linux and macOS (Python 3.10 and 3.12).

### Changed
- **Mission Control (`rcwe ui`) shows what the engine computes.** In 1.0.0 the page animated
  scripted numbers in the browser and displayed fixed benchmark figures as if they were live. Every
  figure is now measured on each tick by `LiveMissionEngine`, which runs RCWE next to a frozen
  Transformer on the sandbox; the buttons act on that simulation; dropping a telemetry file runs a
  real analysis.
- **`rcwe-predict` and `rcwe bench realworld`** take their flagged sensors from the toolkit's
  attribution. The previous rule (largest change over the last five rows) named none of the
  injected sensors on the three regime-shift logs.
- **Nothing is written into the working directory any more.** Generated files go to
  `./rcwe_output/` (or `$RCWE_OUTPUT_DIR`). Reports that were tracked in the repository root moved
  to `docs/reference_results/`, sample inputs to `examples/data/`.
- README rewritten around the quickstart. Its headline figures now come from `docs/BENCHMARKS.md`
  and are measured against the residual Transformer baseline introduced in 0.9.0.
- `Makefile` no longer requires a `python3.12` binary; `make help` lists the targets.
- `pyproject.toml`: version 1.1.0 and the optional extras `data`, `gpu` and `dev`.
- The `RCWE_DEVICE` environment variable overrides accelerator selection.
- Checkpoints and model files are read with `torch.load(weights_only=True)`.
- Compiled files (`__pycache__`, `*.egg-info`) are no longer tracked.

### Fixed
- `SymplecticCausalBlock` rebuilt its layers at run time when the action width did not match its
  configuration. That silently removed six parameters from the optimizer and made
  `load_pretrained` fail for any `d_action` other than 8. Shapes are now checked, and the telemetry
  loader pads control rows to the configured width.
- The 1.0.0 Mission Control backend called `sleep_consolidate` and read `atp_energy`, neither of
  which existed, so its own tests could not pass.
- `BaselineCausalTransformer` had the same run-time resizing; it now checks its input widths.
- The code-trace test required RCWE to beat the Transformer by a margin that only held on one
  platform; it now asserts parity, which is what the benchmark shows.

### How this release was checked
- About 140 tests, run on Python 3.13 with PyTorch 2.15 and on Python 3.10 with PyTorch 2.1 and
  NumPy 1.26; the wheel was installed and exercised in a clean environment.
- Three independent reviews of the new code (ingestion and CLI, the modelling core, the web layer)
  reproduced 60 defects before release. All were fixed and pinned with regression tests, except
  the two listed below.
- Known gaps: an analysis running in Studio cannot be cancelled, and `TelemetryTable.to_csv` does
  not record which columns were chosen as controls or labels (pass them again when reloading).

### A note on earlier figures
The numbers quoted in the entries below are the ones reported when each version was released.
Figures measured with the current code are in `docs/BENCHMARKS.md`, together with how to read
them.

## [1.0.0], [0.9.0], [0.8.0] — recovered 2026-11-11

These versions existed only in the project's Drive folder and had no changelog entries. Their
sources were imported unmodified (commit `9d2a85b`) before the 1.1.0 work started.

- **1.0.0**: Mission Control UI (`rcwe/ui/mission_control.py`).
- **0.9.0**: 2D multimodal spectral fusion (`rcwe/multimodal/`), symplectic Pontryagin RL
  (`rcwe/rl/`), distributed wavefield swarm (`rcwe/swarm/`), `rcwe/v09_runner.py`, and a stronger
  Transformer baseline (residual output, wider feed-forward layer).
- **0.8.0** (preprint dated 2026-11-10): RCWE-Deep corpus trainer (`rcwe/lm/corpus_trainer.py`),
  drop-in custom dataset connector (`rcwe/ingestion/custom_connector.py`), arXiv preprint source
  (`paper/`), master whitepaper (`docs/`).

## [0.7.0] — 2026-11-09
### Added
- **Track A — Apple Metal Shading Language (MSL) Custom Kernel & $L=131,072$ Scaling (`rcwe/kernels/metal_msl.py`)**: Inline MSL shader (`torch.mps.compile_shader`) and single-pass execution up to $L=131,072$ (`256 GB` equivalent Transformer attention workspace compressed into a `8.0 KB` / `16.0 KB` standing wavefield).
- **Track B — Closed-Loop Autonomous Self-Healing Controller (`rcwe/control/self_healing.py`)**: Executes active MDO probing (`do(a_probe)`) followed by synthesized PD + feedforward counter-flux injection (`do(a_heal)`) to neutralize covert Byzantine sinkholes on relay nodes `[v_5, v_6]`.
- **Track C — Hierarchical 4-Layer `RCWE-Deep` Language & Code World Model (`rcwe/lm/rcwe_deep_lm.py`)**: Stacks 4 `DeepRCWEBlock` layers with complex phase-space skip connections and byte-level tokenization.

## [0.6.0] — 2026-11-09
### Added
- **Full-State Checkpoint Serialization (`save_pretrained` / `load_pretrained`)**: Saves and restores slow weights, 3-factor fast synaptic weights (`fast_weight`), ATP metabolic budgets (`atp_budget`), sensory residual bias (`fast_sensory_bias`), and the live $\mathcal{O}(1)$ complex standing wavefield tensor ($|\Psi|$).
- **Universal CSV/JSON CLI (`rcwe-predict`)**: Automatic column schema inference (`auto_infer_and_load`) for arbitrary external time-series files with finite-difference momentum reconstruction.
- **Multi-Domain Real-World Benchmark Suite (`rcwe/benchmarks/real_world_suite.py`)**: Added ETT Electricity Transformer Thermal (`135.6x` MSE gain), SWaT Cyber-Physical Water Treatment (`73.4x` MSE gain), and NASA Turbofan Bearing Degradation (`93.7x` MSE gain) benchmarks.
- **Live Interactive HTTP Control Server (`rcwe/server/live_server.py`)**: Real-time browser streaming server (`make serve`) with interactive anomaly injection, Sleep Consolidation, and live checkpoint saving.

## [0.5.0] — 2026-11-08
### Added
- **Real-World CSV & JSON Telemetry Ingestion Pipeline (`rcwe/ingestion/telemetry_loader.py`)**: Achieves `292.0x` lower post-anomaly stream MSE than frozen Transformers.
- **Tokenized Code & Invariant Reasoning Head (`rcwe/symbolic/code_reasoning.py`)**: Audits program execution traces (`14.5x` lower MSE) and isolates conservation-violating bug registers (`['r5', 'r6']`).
- **Interactive HTML/SVG Diagnostic Dashboard (`rcwe/diagnostics/dashboard_v05.py`)**.

## [0.4.0] — 2026-11-08
### Added
- **Extreme Context Scaling Sweep ($L = 256 \to 32,768$)**: Demonstrated survival past the Transformer 32.8 GB quadratic OOM wall at `96.98 ms` with a constant `16.0 KB` standing wavefield (`1,050,624x` state compression).
- **Symbolic Algebraic Conservation Benchmark (`rcwe/symbolic/algebra_task.py`)**: Achieved `6.43x` lower conservation sum error and `7.6x` lower post-rule-mutation MSE.

## [0.3.2] — 2026-11-07
### Added
- **Apple Silicon `MPS` / CUDA Accelerator Backend (`rcwe/kernels/accelerator.py`)**.
- **Zero-Loop Vectorized Spectral-Vandermonde Scan (`rcwe/kernels/triton_holographic.py`)**: Replaced sequential Python loop with contiguous `rfft` causal convolution and BLAS Vandermonde state contraction (`28.0x` speedup on Metal GPU; `5.02x` faster than Transformer at $L=4,096$).
- **Multi-Seed Statistical Validation (`Seeds=[42, 101, 202]`)**: Verified `160.1x` lower post-mutation MSE, `8.94x` lower Hamiltonian drift, and `97.8% ± 3.8%` causal regime ID accuracy.

## [0.2.1] — 2026-11-07
### Added
- **Velocity-Space World-Line Specialization & Sensory Residual Plasticity**: Broke the v0.1 posterior tie (`W1 = 1.000`, `0.0%` epistemic ambiguity).

## [0.1.0] — 2026-11-07
### Added
- Initial bootstrap release of `WavepacketCodec`, `HolographicResonanceLayer`, `SymplecticCausalBlock`, `MetabolicPlasticSynapse`, and `CyberPhysicalFlowSandbox`.
