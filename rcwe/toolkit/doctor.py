"""
`rcwe doctor` -- check that this machine can run RCWE, and say how to fix what cannot.

Each check returns a status (ok / warn / fail), a one-line finding and, when something is
wrong, the command or setting that fixes it.
"""
from __future__ import annotations

import importlib
import os
import platform
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple

__all__ = ["Check", "run_checks", "format_checks"]


@dataclass
class Check:
    name: str
    status: str          # ok | warn | fail
    detail: str
    fix: str = ""


def _python() -> Check:
    version = platform.python_version()
    if sys.version_info < (3, 10):
        return Check("Python", "fail", f"{version} is too old", "Install Python 3.10 or newer (https://www.python.org/downloads/).")
    return Check("Python", "ok", f"{version} ({platform.system()} {platform.machine()})")


def _torch() -> Check:
    try:
        import torch
    except ImportError:
        return Check("PyTorch", "fail", "not installed", "pip install torch   (see https://pytorch.org/get-started/locally/)")
    parts = torch.__version__.split("+")[0].split(".")
    try:
        too_old = (int(parts[0]), int(parts[1])) < (2, 1)
    except (ValueError, IndexError):
        too_old = False
    if too_old:
        return Check("PyTorch", "fail", f"{torch.__version__} is older than 2.1", "pip install --upgrade torch")
    return Check("PyTorch", "ok", f"{torch.__version__}, {torch.get_num_threads()} CPU threads")


def _numpy() -> Check:
    try:
        import numpy
    except ImportError:
        return Check("NumPy", "fail", "not installed", "pip install numpy")
    return Check("NumPy", "ok", numpy.__version__)


def _accelerator() -> Check:
    try:
        import torch
        from rcwe.core.engine import ResonantCausalWorldEngine
        from rcwe.kernels.accelerator import describe_accelerator, select_best_device
    except ImportError as exc:
        return Check("Accelerator", "fail", f"cannot import the engine ({exc})", "pip install -e .   from the repository root")
    try:
        device = select_best_device()
    except ValueError as exc:
        return Check("Accelerator", "fail", str(exc))
    label = describe_accelerator(device)
    try:
        engine = ResonantCausalWorldEngine(d_input=8, d_model=16, d_action=2, num_worldlines=2, num_freq_bands=4).to(device)
        x = torch.randn(2, 6, 8, device=device)
        a = torch.randn(2, 6, 2, device=device)
        out = engine(x, a, plastic_runtime=False)
        out["output"].pow(2).mean().backward()
    except Exception as exc:  # noqa: BLE001 - any backend failure is reported, not raised
        if device.type == "cpu":
            return Check("Accelerator", "fail", f"the engine failed on the CPU: {str(exc).splitlines()[0][:100]}",
                         "Reinstall PyTorch: pip install --force-reinstall torch")
        return Check("Accelerator", "warn", f"{label} is present but the engine failed on it: {str(exc).splitlines()[0][:90]}",
                     "RCWE will fall back to the CPU. To skip the attempt, pass --device cpu (or device='cpu').")
    if device.type == "cpu":
        return Check("Accelerator", "ok", f"{label}. No GPU found; the CPU is fine for logs up to millions of rows")
    return Check("Accelerator", "ok", f"{label} (engine forward and backward pass verified)")


def _package() -> Check:
    try:
        import rcwe
    except ImportError as exc:
        return Check("RCWE package", "fail", f"import failed ({exc})", "pip install -e .   from the repository root")
    return Check("RCWE package", "ok", f"version {rcwe.__version__} at {Path(rcwe.__file__).resolve().parent}")


def _optional() -> Check:
    found, missing = [], []
    for module, purpose in (("pandas", "Excel/Parquet files and DataFrames"), ("openpyxl", ".xlsx files"), ("pyarrow", ".parquet files")):
        try:
            importlib.import_module(module)
            found.append(module)
        except ImportError:
            missing.append(f"{module} ({purpose})")
    if missing:
        return Check("Optional extras", "ok", "CSV/TSV/JSON work out of the box. Not installed: " + "; ".join(missing),
                     "Only if you need them: pip install pandas openpyxl pyarrow")
    return Check("Optional extras", "ok", "installed: " + ", ".join(found))


def _output_dir() -> Check:
    from rcwe.paths import output_dir
    try:
        folder = output_dir()
        with tempfile.NamedTemporaryFile(dir=folder, prefix=".rcwe_write_test_", delete=True):
            pass
    except OSError as exc:
        return Check("Output folder", "fail", f"cannot write to {Path(os.environ.get('RCWE_OUTPUT_DIR') or 'rcwe_output').resolve()} ({exc.strerror})",
                     "Run from a folder you can write to, or set RCWE_OUTPUT_DIR to one.")
    return Check("Output folder", "ok", f"{folder.resolve()} is writable")


def _smoke() -> Check:
    try:
        from rcwe.toolkit import TelemetryModel
        from rcwe.toolkit.demo import make_pump_station
        started = time.perf_counter()
        table = make_pump_station(rows=1000, faults=True, fault_start=500, seed=1).table
        model = TelemetryModel(max_steps=60, seed=0, device="cpu").fit(table.slice(0, 500))
        result = model.detect(table.slice(500, None))
        elapsed = time.perf_counter() - started
    except Exception as exc:  # noqa: BLE001
        return Check("End-to-end test", "fail", f"train + detect failed: {type(exc).__name__}: {str(exc).splitlines()[0][:100]}",
                     "Run `rcwe doctor --debug` and report the traceback at the project's issue tracker.")
    caught = result.metrics["incidents_detected"] if result.metrics else 0
    if caught < 3:
        return Check("End-to-end test", "warn", f"trained and scanned demo data in {elapsed:.1f}s but caught only {caught} of 5 planted faults",
                     "Results on this machine may be unreliable; try `rcwe quickstart` and compare with the documentation.")
    return Check("End-to-end test", "ok", f"trained a model and caught {caught} of 5 planted faults in {elapsed:.1f}s")


def run_checks(debug: bool = False) -> List[Check]:
    checks: List[Check] = []
    steps: List[Callable[[], Check]] = [_python, _torch, _numpy, _package, _accelerator, _optional, _output_dir, _smoke]
    blocked = False
    for step in steps:
        if blocked and step in (_accelerator, _smoke):
            checks.append(Check(step.__name__.strip("_").replace("_", " ").title(), "warn", "skipped (fix the failures above first)"))
            continue
        try:
            check = step()
        except Exception as exc:  # noqa: BLE001
            if debug:
                raise
            check = Check(step.__name__.strip("_"), "fail", f"check crashed: {type(exc).__name__}: {exc}")
        checks.append(check)
        if check.status == "fail" and step in (_python, _torch, _numpy, _package):
            blocked = True
    return checks


def format_checks(checks: List[Check]) -> Tuple[str, bool]:
    marks = {"ok": "[ ok ]", "warn": "[warn]", "fail": "[FAIL]"}
    lines = ["RCWE doctor", "-----------"]
    for check in checks:
        lines.append(f"{marks[check.status]} {check.name}: {check.detail}")
        if check.fix and check.status != "ok":
            lines.append(f"       fix: {check.fix}")
        elif check.fix:
            lines.append(f"       {check.fix}")
    failed = [c for c in checks if c.status == "fail"]
    warned = [c for c in checks if c.status == "warn"]
    lines.append("")
    if failed:
        lines.append(f"{len(failed)} problem(s) need fixing before RCWE will work.")
    elif warned:
        lines.append("RCWE will work; see the warning(s) above.")
    else:
        lines.append("Everything looks good. Try: rcwe quickstart")
    return "\n".join(lines), not failed
