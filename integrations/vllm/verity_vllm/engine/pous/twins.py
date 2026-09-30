"""The GPU twins, keyed by the protocol's scheme id (`verity_pous.schemes`), each gated by that scheme's pinned vector.

A twin is the GPU `(encode, decode)` of a scheme's reference codec, on `[G, blocks per segment, block bytes]` segments and
`[G, 32]` tags (`codec.segment_tags`): `engine/pous/codec.py` for ``p3/v1``, and `engine/pous/dense.py` for
``dense-chain/v1`` and for ``band-chain/d12/v1`` (the same chain over the band of in-degree 12).  Any other scheme fails
closed.

Before a twin serves, `gate` encodes the input of the scheme's pinned vector (`protocols/pous/tests/fixtures/vectors.json`,
the file `verity_pous`'s own tests check the reference against) on the device and compares the digest: a whole segment at the
served geometry for ``p3/v1`` (``segment_64kb``) and for ``band-chain/d12/v1`` (``band/64kb/segment``), the first 16 blocks
of one for ``dense-chain/v1`` (``dense/64kb/prefix16``, which depend on those 16 blocks only).  A twin that does not reproduce
its vector refuses to serve.

`set_kernels` sets the kernels' dispatch switches (`KERNEL_FLAGS`, `codec` and `dense`'s); every setting computes the same
bits, and the gate runs after it.

`reference` and `primitives` import no torch, so the verifier's roles use them too; `twin` and `gate` import the kernels.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import verity_pous
from verity_pous import band as band_ref
from verity_pous import dense as dense_ref
from verity_pous.codec import Codec, Geometry
from verity_pous.params import OPERATING_POINT
from verity_pous.scheme import P3

#: the pinned vectors of `verity_pous` (`tests/pous_vectors.py` writes them)
VECTORS = Path(verity_pous.__file__).resolve().parents[1] / "tests" / "fixtures" / "vectors.json"


class GateError(RuntimeError):
    """A GPU twin did not reproduce its scheme's pinned vector."""


def reference(name: str) -> tuple[Codec, Geometry]:
    """The reference codec of the scheme `name` and the geometry it is served at (the scheme's registered one)."""
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


def _p3() -> tuple[Callable, Callable]:
    from verity_vllm.engine.pous import codec
    return codec.encode, codec.decode


def _dense() -> tuple[Callable, Callable]:
    from verity_vllm.engine.pous import dense
    return dense.encode, dense.decode


def _band() -> tuple[Callable, Callable]:
    from verity_vllm.engine.pous import dense
    d = band_ref.SECURE_FIRST.d
    return (lambda w, tags: dense.encode(w, tags, d)), (lambda c, tags: dense.decode_segments(c, tags, d))


@dataclass(frozen=True)
class Twin:
    """A scheme's GPU twin: its kernels (`load`), the pinned vector it must reproduce, how many leading blocks of a segment
    the vector pins (None: all of them), and the label its `W` is drawn under (`vector_w`)."""

    scheme: str
    load: Callable[[], tuple[Callable, Callable]]
    vector: str
    blocks: int | None
    w_label: Callable[[Geometry], str]


TWINS: dict[str, Twin] = {
    P3.name: Twin(P3.name, _p3, "segment_64kb", None, lambda p: f"w/{p.n}/{p.wc}/{p.k}/{p.segments}"),
    dense_ref.DENSE.name: Twin(dense_ref.DENSE.name, _dense, "dense/64kb/prefix16", 16, lambda p: "dense-w"),
    band_ref.BAND12.name: Twin(band_ref.BAND12.name, _band, "band/64kb/segment", None, lambda p: "band-segment-w"),
}

#: setting -> (module, switch, type): the kernels' dispatch switches a serving run may set (#188's serve-spec keys)
KERNEL_FLAGS: dict[str, tuple[str, str, type]] = {
    "thread_rows": ("codec", "THREAD_ROWS", int), "word_io": ("codec", "WORD_IO", bool), "fused": ("codec", "FUSED", bool),
    "pair": ("codec", "PAIR", bool), "pair_max_rows": ("codec", "PAIR_MAX_ROWS", int), "coalesced": ("codec", "COALESCED", bool),
    "chunked": ("dense", "CHUNKED", bool), "chunked_threads": ("dense", "CHUNKED_THREADS", int),
    "split": ("dense", "CHUNKED_SPLIT", bool), "fused_x": ("dense", "CHUNKED_FUSED_X", bool),
    "whole_chain": ("dense", "CHUNKED_WHOLE", bool),
}


def set_kernels(flags: dict[str, Any]) -> dict[str, Any]:
    """Set the switches `flags` names (`KERNEL_FLAGS`); every switch as it now is."""
    from verity_vllm.engine.pous import codec, dense

    unknown = sorted(set(flags) - set(KERNEL_FLAGS))
    if unknown:
        raise ValueError(f"unknown kernel settings {unknown} (known: {', '.join(KERNEL_FLAGS)})")
    mods = {"codec": codec, "dense": dense}
    for key, value in flags.items():
        mod, switch, typ = KERNEL_FLAGS[key]
        getattr(mods[mod], switch)[0] = typ(value)
    return {key: getattr(mods[mod], switch)[0] for key, (mod, switch, _) in KERNEL_FLAGS.items()}


def of(scheme: str) -> Twin:
    if scheme not in TWINS:
        raise NotImplementedError(f"no GPU twin of the scheme {scheme!r} (have {', '.join(sorted(TWINS))})")
    return TWINS[scheme]


def twin(ref: Codec) -> tuple[Callable, Callable]:
    """The GPU `(encode, decode)` of the reference codec `ref` (its scheme's twin)."""
    return of(ref.name).load()


def vector_bytes(label: str, n: int) -> bytes:
    """The pinned vectors' input stream (`tests/pous_vectors.py`'s `_bytes`)."""
    return hashlib.shake_256(b"verity-pous/vectors/" + label.encode()).digest(n)


def vector_input(scheme: str, params: Geometry | None = None) -> tuple[bytes, bytes, int]:
    """One segment's `W` for the scheme's pinned vector (the pinned blocks, then zeros), its salt, and the pinned block count."""
    t = of(scheme)
    params = params or reference(scheme)[1]
    n = params.blocks_per_segment if t.blocks is None else t.blocks
    pinned = vector_bytes(t.w_label(params), n * params.block_bytes)
    return pinned + bytes((params.blocks_per_segment - n) * params.block_bytes), vector_bytes("salt", 24), n


def gate(scheme: str, device: Any, *, encode: Callable | None = None, tags: Callable | None = None,
         params: Geometry | None = None, vectors: dict[str, str] | None = None) -> dict[str, Any]:
    """Encode the scheme's pinned vector with its twin on `device` and compare its digest; `GateError` when it differs.
    `encode`, `tags`, `params` and `vectors` replace the kernels, `codec.segment_tags`, the served geometry and the pinned file."""
    import torch

    t = of(scheme)
    params = params or reference(scheme)[1]
    if tags is None:
        from verity_vllm.engine.pous import codec
        tags = codec.segment_tags
    encode = encode or t.load()[0]
    pinned = (vectors or json.loads(VECTORS.read_text()))[t.vector]
    w, salt, n = vector_input(scheme, params)
    t0 = time.perf_counter()
    seg = torch.frombuffer(bytearray(w), dtype=torch.uint8).to(device).view(1, params.blocks_per_segment, params.block_bytes)
    c = encode(seg, tags(salt, [0], device))
    got = hashlib.sha256(c[0, :n].contiguous().cpu().numpy().tobytes()).hexdigest()
    if got != pinned:
        raise GateError(f"the GPU twin of {scheme} does not reproduce its pinned vector {t.vector!r}: {got} != {pinned}")
    return {"scheme": scheme, "vector": t.vector, "blocks": n, "digest": got, "seconds": time.perf_counter() - t0}
