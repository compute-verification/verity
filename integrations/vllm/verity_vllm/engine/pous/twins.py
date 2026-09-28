"""Which GPU twin computes a reference codec: `engine/pous/codec.py` for `scheme.P3`, `engine/pous/dense.py` for
`dense.DENSE` and for `band.BAND12` (the same chain over the band of in-degree 12), and the geometry each is served at.  Any
other codec fails closed.  `reference` imports no torch, so the verifier's roles use it too; `twin` imports the kernels."""
from __future__ import annotations

from verity_pous import band as band_ref
from verity_pous import dense as dense_ref
from verity_pous.codec import Codec, Geometry
from verity_pous.params import OPERATING_POINT
from verity_pous.scheme import P3


def reference(name: str) -> tuple[Codec, Geometry]:
    """The reference codec called `name` and the geometry it is served at."""
    if name == P3.name:
        return P3, OPERATING_POINT
    if name == dense_ref.DENSE.name:
        return dense_ref.DENSE, dense_ref.POINT_64K
    if name == band_ref.BAND12.name:
        return band_ref.BAND12, band_ref.SECURE_FIRST
    raise NotImplementedError(f"no reference codec {name!r} (have {P3.name!r}, {dense_ref.DENSE.name!r}, {band_ref.BAND12.name!r})")


def primitives(ref: Codec, params: Geometry):
    """The reference primitives of the codec `ref` at `params`: P3's overwrite chain and `P_T`, or the band's and dense's
    chain (`dense.DenseChain`), which absorbs a block's parents.  Encoding one with the other's primitives gives another
    codeword of the same `W`, which decodes, but is not the one the GPU twin computes."""
    from verity_pous import primitives as p3_prims

    if ref.name == P3.name:
        return p3_prims.reference(params)
    if ref.name == dense_ref.DENSE.name:
        return dense_ref.reference(params)
    if ref.name == band_ref.BAND12.name:
        return band_ref.reference(params)
    raise NotImplementedError(f"no reference primitives of the codec {ref.name!r}")


def twin(ref: Codec):
    """The GPU `(encode, decode)` of the reference codec `ref`, on `[G, blocks per segment, block bytes]` segments and
    `[G, 32]` tags."""
    from verity_vllm.engine.pous import codec, dense

    if ref.name == P3.name:
        return codec.encode, codec.decode
    if ref.name == dense_ref.DENSE.name:
        return dense.encode, dense.decode
    if ref.name == band_ref.BAND12.name:
        d = band_ref.SECURE_FIRST.d
        return (lambda w, tags: dense.encode(w, tags, d)), (lambda c, tags: dense.decode_segments(c, tags, d))
    raise NotImplementedError(f"no GPU twin of the codec {ref.name!r} (have {P3.name!r}, {dense_ref.DENSE.name!r}, "
                              f"{band_ref.BAND12.name!r})")
