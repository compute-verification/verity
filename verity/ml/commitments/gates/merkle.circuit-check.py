"""circuit-check's bindings of `verity.ml.commitments.gates.merkle`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _merkle_roots() -> list[Any]:
    """The hidden-index Merkle read (`verity.ml.commitments.gates.merkle`), binary and 4-ary, at one level (a level is two or three
    SHA-512 compressions: deeper reads repeat it). About two minutes each; the 4-ary one checks the slot multiplexers among
    three siblings that the binary swap does not have."""
    from verity.ml.circuits.defs import bind
    from verity.ml.commitments.gates.merkle import MerkleRead

    return [
        bind(MerkleRead, ARITY=2, D=1),
        bind(MerkleRead, ARITY=4, D=1),
    ]
