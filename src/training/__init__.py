"""Training modules for ZIZOGPT."""

from .pretrain import main as pretrain_main
from .sft import main as sft_main
from .rl_training import main as rl_main

__all__ = [
    "pretrain_main",
    "sft_main",
    "rl_main",
]
