"""circuit-check's bindings of `verity.ml.definitions.registry.boolean_tanh`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """RN32 tanh on one f32 word, and `gelu_tanh_and_mul` on bits at I = 8, each computing its table's entries."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import boolean_tanh as BT

    return [bind(BT.TanhF32Rn), bind(BT.GeluTanhMul, I=8)]
