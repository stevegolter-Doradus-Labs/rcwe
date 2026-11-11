"""
Where generated files go.

Benchmarks, dashboards and checkpoints used to be written into the current directory,
which meant running anything inside a checkout left it full of (and overwrote) tracked
files. Everything RCWE generates now lands in one folder:

    ./rcwe_output/            (default)
    $RCWE_OUTPUT_DIR          (if set)

Functions that take an explicit path still write exactly where they are told.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

__all__ = ["output_dir", "output_path", "resolve_output", "reference_path"]

ENV_VAR = "RCWE_OUTPUT_DIR"
DEFAULT_DIR = "rcwe_output"


def output_dir(create: bool = True) -> Path:
    """The folder generated artefacts are written to."""
    folder = Path(os.environ.get(ENV_VAR) or DEFAULT_DIR).expanduser()
    if create:
        folder.mkdir(parents=True, exist_ok=True)
    return folder


def output_path(name: Union[str, Path]) -> str:
    """Path of a generated file called `name` inside the output folder."""
    return str(output_dir() / name)


def resolve_output(explicit: Optional[Union[str, Path]], default_name: str) -> str:
    """`explicit` if the caller gave one, otherwise `default_name` inside the output folder."""
    if explicit is None or explicit == "":
        return output_path(default_name)
    path = Path(explicit).expanduser()
    if path.parent != Path(""):
        path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def reference_path(name: str) -> Optional[Path]:
    """
    A previously generated report to read back: looked up in the output folder first, then in
    the reference results shipped with a source checkout (`docs/reference_results/`).
    """
    candidates = [
        output_dir(create=False) / name,
        Path(__file__).resolve().parent.parent / "docs" / "reference_results" / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None
