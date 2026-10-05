"""Shared imports / helpers for the lane's definitions (no definitions here; pure, verity.primitives.circuits.ir-only)."""
from __future__ import annotations

from verity_vllm.program.registry import prims as P
from verity_vllm.program.registry.b1 import ZERO32, const, f32c
from verity.primitives.circuits.ir.types import Array, Tuple, Value

BF16, F32 = Value(16), Value(32)

__all__ = ["P", "ZERO32", "const", "f32c", "Array", "Tuple", "BF16", "F32"]
