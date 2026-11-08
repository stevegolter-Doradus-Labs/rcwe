import torch
from typing import Optional


def select_best_device(prefer_device: Optional[str] = None) -> torch.device:
    if prefer_device is not None:
        return torch.device(prefer_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
