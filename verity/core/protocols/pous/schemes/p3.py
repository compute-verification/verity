"""`p3/v1`, research: two layers over the band graph `B(n, k)` with a transpose and a tweakable `P` (Lean `p3Scheme`),
encoded and decoded by the reference codec `verity.core.protocols.pous.scheme`.

It has no certificate.  Lean proves P3 with the overwrite-chain `H`, in the ideal two-permutation model, with no
hypothesis: `Pous.SecurityProofs.P3ConcreteMeets64` (`p3SchemeC 2384 B(220, 12)`, one segment, `D = 10`, `Q = 2^20`,
`k = 106`), `P3ConcreteMeets14GB` (the same at 971 segments) and `P3ConcreteMeets64D12` (`D = 12` on `B(220, 16)`).
`OPERATING_POINT` has their parameters (`n = 220`, `wc = 2384`, `B(220, 12)`, a 24-byte salt, `tag_s = salt ‖ u64 s`),
but not their `H`: Lean's chain starts from `iv(domP3 tag L v)`, zeros and then `tag_s ‖ u256(2v + L)` in the 512-bit
capacity part, and the code's from the instantiation's 48-byte header `"pous-p3-v1-H" ‖ tag_s ‖ u8(L) ‖ u16(v) ‖
u8(npar)` in the low bytes, with the capacity part zero (`primitives.OverwriteChain.initial`).  The pins are about
Lean's `H`, so P3 gets a certificate once the code starts from Lean's `h_0` (a new byte format) or Lean carries the
pins to the header.
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
