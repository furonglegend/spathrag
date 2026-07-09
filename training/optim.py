# src/training/optim.py
"""
Optimizer and scheduler utilities.
Provides:
  - get_optimizer: constructs an AdamW optimizer for a given module or parameter list
  - get_scheduler: builds a learning rate scheduler
"""

from typing import Iterable, Optional
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR


def get_optimizer(model_or_params, lr: float = 1e-4, weight_decay: float = 0.01):
    """
    Create AdamW optimizer. Accepts either a nn.Module or an Iterable of parameters.
    """
    if hasattr(model_or_params, "parameters"):
        params = model_or_params.parameters()
    else:
        params = model_or_params
    return AdamW(params, lr=lr, weight_decay=weight_decay)


def get_scheduler(optimizer: torch.optim.Optimizer, scheduler_type: str = "cosine", **kwargs):
    """
    Return a scheduler. Supported: 'cosine', 'step'.
    kwargs passed to the underlying scheduler constructors.
    """
    if scheduler_type == "cosine":
        T_max = kwargs.get("T_max", 100)
        return CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_type == "step":
        step_size = kwargs.get("step_size", 10)
        gamma = kwargs.get("gamma", 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        raise ValueError(f"Unsupported scheduler_type: {scheduler_type}")
