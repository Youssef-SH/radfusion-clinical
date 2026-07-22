"""Resolve CPU/CUDA policy for neural experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torchvision
import torchxrayvision


@dataclass(frozen=True)
class ResolvedDevice:
    """Requested and effective device-dependent runtime settings."""

    device: torch.device
    requested_device: str
    resolved_device: str
    cuda_available: bool
    mixed_precision_requested: bool
    mixed_precision_effective: bool
    pin_memory_requested: str
    pin_memory_effective: bool
    torch_version: str
    torchvision_version: str
    torchxrayvision_version: str

    def provenance(self) -> dict[str, Any]:
        """Return compact serializable runtime provenance."""
        return {
            "requested_device": self.requested_device,
            "resolved_device": self.resolved_device,
            "cuda_available": self.cuda_available,
            "mixed_precision_requested": self.mixed_precision_requested,
            "mixed_precision_effective": self.mixed_precision_effective,
            "pin_memory_requested": self.pin_memory_requested,
            "pin_memory_effective": self.pin_memory_effective,
            "torch_version": self.torch_version,
            "torchvision_version": self.torchvision_version,
            "torchxrayvision_version": self.torchxrayvision_version,
        }


def resolve_device(
    requested_device: str,
    *,
    mixed_precision: bool,
    pin_memory_policy: str,
) -> ResolvedDevice:
    """Resolve automatic device, mixed precision, and pinned-memory policy."""
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("requested_device must be 'auto', 'cpu', or 'cuda'")
    if not isinstance(mixed_precision, bool):
        raise TypeError("mixed_precision must be Boolean")
    if pin_memory_policy not in {"auto", "enabled", "disabled"}:
        raise ValueError("pin_memory_policy must be 'auto', 'enabled', or 'disabled'")

    cuda_available = torch.cuda.is_available()
    if requested_device == "cuda" and not cuda_available:
        raise ValueError("CUDA was requested but is not available")
    resolved = (
        "cuda"
        if requested_device == "cuda" or (requested_device == "auto" and cuda_available)
        else "cpu"
    )
    use_cuda = resolved == "cuda"
    pin_requested = pin_memory_policy in {"auto", "enabled"}
    return ResolvedDevice(
        device=torch.device(resolved),
        requested_device=requested_device,
        resolved_device=resolved,
        cuda_available=cuda_available,
        mixed_precision_requested=mixed_precision,
        mixed_precision_effective=mixed_precision and use_cuda,
        pin_memory_requested=pin_memory_policy,
        pin_memory_effective=pin_requested and use_cuda,
        torch_version=torch.__version__,
        torchvision_version=torchvision.__version__,
        torchxrayvision_version=torchxrayvision.__version__,
    )
