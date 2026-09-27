"""Which GPU twin computes a reference codec: `engine/pous/codec.py` for `scheme.P3`, `engine/pous/dense.py` for
`dense.DENSE`, and the geometry each is served at.  Any other codec fails closed."""
from __future__ import annotations

from verity_pous import dense as dense_ref
from verity_pous.codec import Codec, Geometry
from verity_pous.params import OPERATING_POINT
from verity_pous.scheme import P3

from verity_vllm.engine.pous import codec, dense

def reference(name: str) -> tuple[Codec, Geometry]:
    """The reference codec called `name` and the geometry it is served at."""
    if name == P3.name:
        return P3, OPERATING_POINT
    if name == dense_ref.DENSE.name:
        return dense_ref.DENSE, dense_ref.SECURE_FIRST
    raise NotImplementedError(f"no reference codec {name!r} (have {P3.name!r}, {dense_ref.DENSE.name!r})")


def twin(ref: Codec):
    """The GPU `(encode, decode)` of the reference codec `ref`, on `[G, blocks per segment, block bytes]` segments and
    `[G, 32]` tags."""
    if ref.name == P3.name:
        return codec.encode, codec.decode
    if ref.name == dense_ref.DENSE.name:
        return dense.encode, dense.decode
    raise NotImplementedError(f"no GPU twin of the codec {ref.name!r} (have {P3.name!r}, {dense_ref.DENSE.name!r})")
