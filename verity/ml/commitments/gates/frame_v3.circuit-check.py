"""circuit-check's bindings of `verity.ml.commitments.gates.frame_v3`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _merkle_roots() -> list[Any]:
    """The frame-v3-sha512 hidden-position read (`verity.ml.commitments.gates.frame_v3`) at one level over
    ``hm96-sha512/row/v2`` leaves: the leaf, one node and the comparison, five SHA-512 compressions (deeper reads repeat the
    node, at most three widths of its index). Each part over the suite's gate budget is checked at its smallest binding, which
    this one reaches."""
    from verity.ml.commitments.gates.frame_v3 import read

    return [read(1, "hm96-sha512/row/v2")]
