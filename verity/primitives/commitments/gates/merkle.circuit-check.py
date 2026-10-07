"""circuit-check's bindings of `verity.primitives.commitments.gates.merkle`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _merkle_roots() -> list[Any]:
    """The hidden-index Merkle reads (`verity.primitives.commitments.gates.merkle`). The read, binary and 4-ary, at one level (a level is two or
    three SHA-512 compressions: deeper reads repeat it); about two minutes each; the 4-ary one checks the slot multiplexers among
    three siblings that the binary swap does not have. The subtree read binary over a run of two cells one level below the root,
    the smallest with every part (the cells' hm96 leaves, a run node over them in cell order, a level above it: eight
    compressions), and 4-ary over one cell (its leaf under a 4-ary level); its 4-ary run node is the 4-ary level's node without
    the slots, which the unit tests check against the reference at J = 1."""
    from verity.primitives.circuits.defs import bind
    from verity.primitives.commitments.gates.merkle import MerkleRead, MerkleSubtreeRead

    return [
        bind(MerkleRead, ARITY=2, D=1),
        bind(MerkleRead, ARITY=4, D=1),
        bind(MerkleSubtreeRead, ARITY=2, D=2, J=1),
        bind(MerkleSubtreeRead, ARITY=4, D=1, J=0),
    ]
