"""circuit-check's bindings of `verity.ml.circuits.ops.boolean.rows`' Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """The residual add and the argmax on bits at `ops.rows`' bindings; the embedding at a public position at its smallest
    binding (two one-word rows) and at the e2e demo's first phase (V = 256, H = 128); and its selection at the second phase's
    half rows and whole rows (H = 8,192 and 16,384), whose fill and token comparison those don't reach. The second phase's
    unit repeats the first's parts at its widths (the row digest's segments, the same 8-level read), each checked here or in
    its own module's bindings (`verity.ml.commitments.gates`)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import rows as R

    return [bind(R.Bf16AddRow, N=128), bind(R.Bf16AddRow, N=16384), bind(R.Argmax, V=256),
            bind(R.EmbeddingPositionSelect, V=256, H=128), bind(R.EmbeddingPositionSelect, V=256, H=8192),
            bind(R.EmbeddingPositionSelect, V=256, H=16384), R.position(2, 1), R.position(256, 128)]
