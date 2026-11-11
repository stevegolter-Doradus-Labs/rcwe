import os

import torch
from typing import Optional


def select_best_device(prefer_device: Optional[str] = None) -> torch.device:
    """
    Pick where to run: an explicit `prefer_device`, else the `RCWE_DEVICE` environment
    variable (e.g. `RCWE_DEVICE=cpu` to rule out GPU quirks), else CUDA, then Apple MPS, then CPU.
    """
    asked_by = "the device setting"
    if prefer_device is None:
        prefer_device = os.environ.get("RCWE_DEVICE") or None
        asked_by = "RCWE_DEVICE"
    if prefer_device is not None:
        return _requested_device(str(prefer_device), asked_by)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _requested_device(name: str, asked_by: str) -> torch.device:
    """The device someone asked for by name, or a plain-language error when this machine cannot provide it."""
    way_out = "Unset RCWE_DEVICE or set it to cpu." if asked_by == "RCWE_DEVICE" else "Use cpu, or leave the choice to RCWE."
    try:
        device = torch.device(name)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"{asked_by} asks for '{name}', which is not a device PyTorch knows (cpu, cuda, mps). {way_out}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"{asked_by} asks for CUDA, but PyTorch cannot use an NVIDIA GPU on this machine. {way_out}")
    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise ValueError(f"{asked_by} asks for Apple MPS, which is not available on this machine. {way_out}")
    return device


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def get_device_memory_kb(device: torch.device) -> float:
    if device.type == "cuda":
        return float(torch.cuda.memory_allocated(device)) / 1024.0
    if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
        return float(torch.mps.current_allocated_memory()) / 1024.0
    return 0.0


def describe_accelerator(device: torch.device) -> str:
    if device.type == "cuda":
        return f"NVIDIA CUDA ({torch.cuda.get_device_name(device)})"
    if device.type == "mps":
        return "Apple Silicon Metal Performance Shaders (mps)"
    return "CPU (Vectorized PyTorch)"
