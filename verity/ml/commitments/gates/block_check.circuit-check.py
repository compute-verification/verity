"""circuit-check's bindings of `verity.ml.commitments.gates.block_check`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _block_check_roots() -> list[Any]:
    """Both units at toy bindings over `block_check.TOY` (p = 2^32 - 5, one schoolbook square a round) and a 64-byte block:
    the decode bound and masked (16 words of two rounds, the payload's row digest and the binding hash, three compressions)
    and unbound, and the receipt of two sign rows into a leaf of three under a plan of four (one compression). What the toys
    catch that the parts' own checks can't is the wiring: each word's tweak chain from t0 and the binding, its rounds' order
    and masks, the payload's bytes into its digest and the binding's message, the sketch's bytes into the receipt and its
    picks. The deployed units (about 15.6 M ANDs at p512/k2, 32 M at p1024/k3/masked) are past any whole lowering here, so
    each part is also bound at a deployed field, among them the 512- and 1,024-bit Karatsuba squares and their folds by C,
    which no toy word reaches. A toy decode's whole lowering (about 175,000 ANDs) is over the default budget: its
    correspondence and pin are checked with ``--lower-max-ands 200000``."""
    from verity.ml.circuits.defs import bind
    from verity.ml.commitments.gates.block_check import (BlockDecodeCheck, BlockReceiptCheck, CodecBinding, CodecLess,
                                                         CodecMatch, CodecModAdd, CodecRound, CodecWord, LeafMatch,
                                                         SketchRow)

    n, c = 32, 5
    return [bind(BlockDecodeCheck, N=n, C=c, K=2, MASKED=0, BYTES=64), bind(BlockDecodeCheck, N=n, C=c, K=2, MASKED=1, BYTES=64),
            bind(BlockReceiptCheck, ROWS=2, LEAF=3, P=4, BYTES=64),
            bind(CodecRound, N=512, C=569, MASK=0), bind(CodecRound, N=1024, C=105, MASK=1),
            bind(CodecWord, N=512, C=569, K=2, MASKED=0), bind(CodecBinding, N=512, C=569), bind(CodecModAdd, N=512, C=569),
            bind(CodecLess, N=512, C=569), bind(CodecMatch, N=512), bind(SketchRow, BYTES=4096), bind(LeafMatch, LEAF=16, P=8)]
