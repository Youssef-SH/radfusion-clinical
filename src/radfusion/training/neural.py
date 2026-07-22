"""Provide deterministic neural-runtime seeding primitives."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_neural_runtime(seed: int) -> None:
    """Seed Python, NumPy, PyTorch CPU/CUDA, and deterministic kernels."""
    _validate_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def dataloader_generator(seed: int) -> torch.Generator:
    """Return a deterministically seeded DataLoader shuffle generator."""
    _validate_seed(seed)
    return torch.Generator().manual_seed(seed)


def seed_dataloader_worker(worker_id: int) -> None:
    """Seed one worker from its PyTorch-assigned initial seed."""
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _validate_seed(seed: object) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**31 - 1:
        raise ValueError("Neural seed must be an integer between 0 and 2147483647")
