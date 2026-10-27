"""`dense-chain/v1`, selectable: the band scheme's construction on the complete DAG (Lean `chainDense B ℓ 512 (dom
tag_s)` per segment), encoded and decoded by the reference codec `verity.core.protocols.pous.dense`.

Its certificate at `dense.POINT_64K` is `Pous.SecurityProofs.ChainDenseMeets64`, with no hypothesis (in the
ideal-permutation model): the band's terms, with no cliff, for one segment. No guarantee covers the dense chain scheme
at more segments, so a profile of more carries that gap.
"""

from __future__ import annotations

from collections.abc import Sequence

from .. import dense as dense_codec
from ..primitives import Primitives
from ..protocol import Certificate, Scheme
from .band import POINT, chain_certificate


class Dense(Scheme):
    """The dense scheme: every earlier label of the segment is a parent."""

    name = dense_codec.DENSE.name
    backend = staticmethod(dense_codec.reference)

    def __init__(self, params: dense_codec.DenseParams = dense_codec.POINT_64K, prims: Primitives | None = None) -> None:
        if not isinstance(params, dense_codec.DenseParams):
            raise TypeError("the dense scheme takes dense.DenseParams")
        super().__init__(params, prims)

    @property
    def certificate(self) -> Certificate | None:
        if (self.params.blocks_per_segment, self.params.m) != POINT:
            return None
        return chain_certificate("Pous.SecurityProofs.ChainDenseMeets64", None, 1)

    def encode_segment(self, salt: bytes, s: int, w_seg: bytes) -> list[bytes]:
        return dense_codec.encode_segment(self.params, self.prims, salt, s, w_seg)

    def decode_segment(self, salt: bytes, s: int, blocks: Sequence[bytes]) -> bytes:
        return dense_codec.decode_segment(self.params, self.prims, salt, s, blocks)


__all__ = ["Dense"]
