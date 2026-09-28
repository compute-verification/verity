"""circuit-check's bindings of `verity.ml.commitments.gates.rowleaf`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _merkle_roots() -> list[Any]:
    """The sha512/row-seg/v1 digest (`verity.ml.commitments.gates.rowleaf`) of a 13-bit row: one short segment with its
    zero fill and the outer hash, two SHA-512 compressions (a longer row repeats the segment part and lengthens the outer
    hash)."""
    from verity.ml.commitments.gates.rowleaf import digest

    return [digest(13)]
