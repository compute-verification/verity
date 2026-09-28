"""`p3/v1`, research: two layers over the band graph `B(n, k)` with a transpose and a tweakable `P` (Lean `p3Scheme`),
encoded and decoded by the reference codec `verity_pous.scheme`.

It has no certificate for the implemented `H`.  Lean proves `p3_meets_64_D10_final` and `p3_meets_multi_64_D10_final`
(POUS `lean/submissions/p3-meets/P3MeetsFinal.lean`: `k = 117`, `D = 10`, one 64 KB segment and 971 of them) with `H`
a random oracle and an ideal tweakable `P`.  The code's `H` is the overwrite chain over `Π₂`, and carrying those
theorems to it is not done.
"""

from __future__ import annotations

from collections.abc import Sequence

from .. import primitives
from .. import scheme as p3_codec
from ..params import OPERATING_POINT, Params
from ..protocol import Scheme


class P3(Scheme):
    """P3 with the instantiation's `H` (the overwrite chain) and tweakable `P`."""

    name = p3_codec.P3.name
    backend = staticmethod(primitives.reference)

    def __init__(self, params: Params = OPERATING_POINT, prims: primitives.Primitives | None = None) -> None:
        if not isinstance(params, Params):
            raise TypeError("P3 takes params.Params")
        super().__init__(params, prims)

    def encode_segment(self, salt: bytes, s: int, w_seg: bytes) -> list[bytes]:
        return p3_codec.encode_segment(self.params, self.prims, salt, s, w_seg)

    def decode_segment(self, salt: bytes, s: int, blocks: Sequence[bytes]) -> bytes:
        return p3_codec.decode_segment(self.params, self.prims, salt, s, blocks)


__all__ = ["P3"]
