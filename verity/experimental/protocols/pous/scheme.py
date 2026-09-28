"""The P3 scheme: Lean's `p3Scheme` for any `Params`, with a tweakable `P`.

Per segment `s` with tag `t = tag(salt, s)`, over the band graph `G`:

    c¹_v = P_(t,0,v)(W_v ⊕ H(t, 0, v, c¹_par(v)))                 (Lean `c1'`)
    c²_v = P_(t,1,v)(transpose(c¹)_v ⊕ H(t, 1, v, c²_par(v)))     (Lean `c2'`, `topLabel'`)

`C` is `c²` of every segment.  The decoder is Lean's `decodeW`: key every top label and invert `P` (the keys' rounds and
one inverse round, side by side), transpose back, then the same for the lower labels.

Segments encode and decode independently.  `encode_segment` and `decode_segment` work on one segment's bytes, and
`P3` is them as a `codec.Codec`; `encode` and `decode` run them segment by segment, so no integer is wider than a label.
The `*_prog` functions are the same computations as oracle programs, answered and measured by `oracle.run`, on labels
of any bit width.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .codec import Codec, encode_segments
from .labels import salt_value, tag, to_bytes, transpose, w_labels, w_value
from .oracle import PinvQuery, PQuery, Prog, Tweak, parallel, run
from .params import SALT_BITS, Params
from .primitives import KeyDerivation, Primitives, key_program, key_query


@dataclass(frozen=True, slots=True)
class Encoding:
    """`(pp, C)`: the salt and the `B` blocks, block `s * n + v` being segment `s`'s top label `v`."""

    params: Params
    salt: bytes
    blocks: tuple[int, ...]

    def block(self, i: int) -> bytes:
        return to_bytes(self.blocks[i], self.params.m)


def encode_segment_prog(params: Params, h: KeyDerivation, t: int,
                        w: Sequence[int]) -> Prog[tuple[list[int], list[int]]]:
    """One segment's lower and top labels, node by node."""
    c1: list[int] = []
    for v in range(params.n):
        key = yield from key_program(h, key_query(params, t, 0, v, c1))
        [label] = yield [PQuery(Tweak(t, 0, v), w[v] ^ key)]
        c1.append(label)
    x2 = transpose(c1, params.n, params.wc)
    c2: list[int] = []
    for v in range(params.n):
        key = yield from key_program(h, key_query(params, t, 1, v, c2))
        [label] = yield [PQuery(Tweak(t, 1, v), x2[v] ^ key)]
        c2.append(label)
    return c1, c2


def _inverse(t: Tweak, y: int) -> Prog[int]:
    [x] = yield [PinvQuery(t, y)]
    return x


def _unlabel(params: Params, h: KeyDerivation, t: int, layer: int, labels: Sequence[int]) -> Prog[list[int]]:
    """`P_(t,layer,v)⁻¹(labels_v) ⊕ H(t, layer, v, labels_par(v))` for every `v`, in parallel."""
    n = params.n
    out = yield from parallel([key_program(h, key_query(params, t, layer, v, labels)) for v in range(n)]
                              + [_inverse(Tweak(t, layer, v), labels[v]) for v in range(n)])
    return [out[n + v] ^ out[v] for v in range(n)]


def decode_segment_prog(params: Params, h: KeyDerivation, t: int, c2: Sequence[int]) -> Prog[list[int]]:
    """Lean `decodeW`: one segment's lower inputs from its top labels."""
    x2 = yield from _unlabel(params, h, t, 1, c2)
    return (yield from _unlabel(params, h, t, 0, transpose(x2, params.n, params.wc)))


def encode_prog(params: Params, h: KeyDerivation, w: int, salt: int) -> Prog[list[tuple[list[int], list[int]]]]:
    """Every segment's `(c¹, c²)`, the segments in parallel."""
    segments = w_labels(params, w)
    return (yield from parallel(encode_segment_prog(params, h, tag(salt, s), segments[s])
                                for s in range(params.segments)))


def decode_prog(params: Params, h: KeyDerivation, salt: int, blocks: Sequence[int]) -> Prog[int]:
    n = params.n
    if len(blocks) != params.blocks:
        raise ValueError(f"expected {params.blocks} blocks, got {len(blocks)}")
    lowers = yield from parallel(decode_segment_prog(params, h, tag(salt, s), blocks[s * n:(s + 1) * n])
                                 for s in range(params.segments))
    return w_value(params, lowers)


def _check_salt(salt: bytes) -> int:
    if type(salt) is not bytes or len(salt) != SALT_BITS // 8:
        raise ValueError(f"the salt is {SALT_BITS // 8} bytes, drawn by the verifier after W is committed")
    return salt_value(salt)


def _label_bytes(params: Params) -> int:
    if params.m % 8:
        raise ValueError("the byte API needs whole-byte labels; use the *_prog functions for other widths")
    return params.m // 8


def encode_segment(params: Params, prims: Primitives, salt: bytes, s: int, w_seg: bytes) -> list[bytes]:
    """Segment `s` of `C`: its `n` top labels (blocks `s * n .. s * n + n - 1`) from its `segment_bytes` of `W`."""
    mb = _label_bytes(params)
    if not 0 <= s < params.segments or len(w_seg) != params.segment_bytes:
        raise ValueError(f"segment {s} of W is {params.segment_bytes} bytes")
    labels = [int.from_bytes(w_seg[u * mb:(u + 1) * mb], "little") for u in range(params.n)]
    (_, c2), _ = run(encode_segment_prog(params, prims.h, tag(_check_salt(salt), s), labels), prims)
    return [c.to_bytes(mb, "little") for c in c2]


def decode_segment(params: Params, prims: Primitives, salt: bytes, s: int, blocks: Sequence[bytes]) -> bytes:
    """Segment `s` of `W` (`params.segment_bytes` bytes) from its `n` blocks `s * n .. s * n + n - 1`."""
    mb = _label_bytes(params)
    if not 0 <= s < params.segments or len(blocks) != params.n or any(len(b) != mb for b in blocks):
        raise ValueError(f"segment {s} needs its {params.n} blocks of {mb} bytes")
    values = [int.from_bytes(b, "little") for b in blocks]
    lowers, _ = run(decode_segment_prog(params, prims.h, tag(_check_salt(salt), s), values), prims)
    return b"".join(x.to_bytes(mb, "little") for x in lowers)


P3 = Codec("p3/v1", encode_segment, decode_segment)
"""The two-layer scheme as a codec."""


def encode(params: Params, prims: Primitives, w: bytes | Iterable[bytes], salt: bytes) -> Encoding:
    """`Enc(W, salt) = C`, segment by segment; `w` is `W` or its segments.  Deterministic and public given `W`, the salt
    and the primitives, so the server and the verifier compute the same `C` (`setup`).  For large `W`, iterate
    `codec.encode_segments` instead of holding every block."""
    _check_salt(salt)
    blocks = [int.from_bytes(b, "little") for seg in encode_segments(P3, params, prims, w, salt) for b in seg]
    return Encoding(params, salt, tuple(blocks))


def decode(params: Params, prims: Primitives, salt: bytes, blocks: Sequence[bytes]) -> bytes:
    """`Dec(pp, C) = W`, bit-exact, segment by segment."""
    n = params.n
    if len(blocks) != params.blocks:
        raise ValueError(f"expected {params.blocks} blocks, got {len(blocks)}")
    return b"".join(decode_segment(params, prims, salt, s, blocks[s * n:(s + 1) * n]) for s in range(params.segments))
