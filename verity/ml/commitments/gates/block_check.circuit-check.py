"""circuit-check's bindings of `verity.ml.commitments.gates.block_check`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _block_check_roots() -> list[Any]:
    """The opening unit at a toy binding over `block_check.TOY` (p = 2^32 - 5, one schoolbook square a round), masked, with
    a 64-byte block: 16 words of two rounds, the payload's row digest and binding hash, two sign rows into a leaf of three
    under a plan of four, the leaf's and payload's hm96 leaves, and reads of depths 1, 2 and 1. What the toy catches that
    the parts' own checks can't is the wiring: each word's tweak chain from t0 and the binding, its rounds' order and masks,
    the payload's bytes into its digest, the binding's message and its hm96 leaf, the sketch's bytes into the receipt and
    its pick, each read's rank, position and depth, and the verdict's flags. The deployed unit (about 30 M ANDs at p512/k2)
    is past any whole lowering here, so each part is also bound at a deployed field, among them the 512- and 1,024-bit
    Karatsuba squares and their folds by C, which no toy word reaches. The toy's whole lowering (about 2.5 M ANDs) is over
    the default budget: its correspondence and pin are checked with ``--lower-max-ands 3000000``."""
    from verity.ml.circuits.defs import bind
    from verity.ml.commitments.gates.block_check import (BlockOpeningCheck, BlockVerdict, CodecBinding, CodecLess, CodecMatch,
                                                         CodecModAdd, CodecRound, CodecWord, LeafMatch, SketchRow)

    return [bind(BlockOpeningCheck, N=32, C=5, K=2, MASKED=1, BYTES=64, ROWS=2, LEAF=3, P=4, DR=1, DP=2, DV=1),
            bind(CodecRound, N=512, C=569, MASK=0), bind(CodecRound, N=1024, C=105, MASK=1),
            bind(CodecWord, N=512, C=569, K=2, MASKED=0), bind(CodecBinding, N=512, C=569), bind(CodecModAdd, N=512, C=569),
            bind(CodecLess, N=512, C=569), bind(CodecMatch, N=512), bind(SketchRow, BYTES=4096), bind(LeafMatch, LEAF=16, P=48),
            bind(BlockVerdict, W=64)]
