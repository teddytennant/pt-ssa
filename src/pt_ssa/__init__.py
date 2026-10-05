import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["JAX_ENABLE_X64"] = "1"

from pt_ssa.core import (
    attention_output,
    firing_indicators,
    log_threshold,
    pt_ssa,
    threshold,
)

__all__ = [
    "attention_output",
    "firing_indicators",
    "log_threshold",
    "pt_ssa",
    "threshold",
]
