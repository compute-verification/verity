"""circuit-check's bindings of `verity.ml.circuits.ops.rows`' Definitions, in `circuit_check.targets`' forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _core_roots() -> list[Any]:
    """The residual add over the e2e demo's rows (H = 128, and 16,384, the largest class's 262,144 bits) and the argmax over
    its byte vocabulary (V = 256)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops import rows as R

    return [bind(R.Bf16AddRow, N=128), bind(R.Bf16AddRow, N=16384), bind(R.Argmax, V=256)]
