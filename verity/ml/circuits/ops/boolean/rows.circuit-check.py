"""circuit-check's bindings of `verity.ml.circuits.ops.boolean.rows`' Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """The residual add and the argmax on bits at `ops.rows`' bindings: the e2e demo's rows (N = 128 and 16,384) and its
    byte vocabulary (V = 256)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import rows as R

    return [bind(R.Bf16AddRow, N=128), bind(R.Bf16AddRow, N=16384), bind(R.Argmax, V=256)]
