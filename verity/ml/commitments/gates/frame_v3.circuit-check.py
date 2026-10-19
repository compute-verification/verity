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


def _merkle_read_at_roots() -> list[Any]:
    """The read at a rank, a four-byte position and a depth (`FrameV3ReadAt`), at one level over ``hm96-sha512/row/v2``
    leaves: its leaf laid out at every position length, one node and the pick at depth 0 or 1."""
    from verity.ml.commitments.gates.frame_v3 import read_at

    return [read_at(1, 4, "hm96-sha512/row/v2")]
