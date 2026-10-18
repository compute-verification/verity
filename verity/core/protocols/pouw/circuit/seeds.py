"""The seed and the noise a PoUW unit derives inside its own proof (decisions 105 and 106, `PROTOCOL.md` §`compile`).

A drawn unit's noise is neither a public nor a coin input: the unit computes its seed from the epoch's salt (the one coin),
its position and its operands' roots, all inputs of the unit bound before the draw, and its noise from the seed, in gates.

    seed_j  = SHA-512(u64(j) ‖ (SEED_TAG ‖ salt ‖ root(A_j) ‖ root(B_j)) ⊕ u64(j)*)        `PouwSeed`
    block_t = SHA-512(NOISE_TAG ‖ seed_j ‖ u64(t))                                          `PouwNoiseBlock`

``⊕ u64(j)*`` XORs every 64-bit word of the message after the first with the position: an injective encoding of (tag, salt,
position, roots), which makes every gate of the seed read the unit's own position, so no two units compute a common value
(the gate-level cut) although they share the salt and a site's units share its roots. The noise stream is the blocks
``block_0, block_1, ...`` in order, each as 16 big-endian 32-bit halves, bit β the bit β mod 32 (least significant first) of
half (β mod 512) // 32 of block β // 512 (`stream`); a Boolean unit reads it as the leaves of its blocks.

Each Definition has a word form (`_v1`, over `Value<32>` halves, with `hashes.Sha512Compress`) and a Boolean twin (`_v2`, over
32-bit elements, with `commitments.gates.sha512.compress_gadget`) whose word view it is. The salt port is `In(kind="coin")`
where port types (`ml.circuits.ports`) are on the tree.
"""
from __future__ import annotations

import hashlib
from typing import Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import const_bits
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.refs import Coll, Explicit, array_of, concat
from verity.ml.circuits.types import Array
from verity.ml.commitments.gates.sha512 import IV, compress_gadget, pad_tail, words_be

from . import hashes as H
from . import words as W

try:
    from verity.ml.circuits import ports as _ports
except ImportError:  # port types (circuits PROTOCOL.md §12) are not on this tree
    _ports = None

SEED_TAG = b"verity/pouw/seed/v2".ljust(24, b"\0")
NOISE_TAG = b"verity/pouw/noise/v1".ljust(24, b"\0")
M32 = W.M32
#: a digest (the salt, a root, a seed, a noise block) as 16 big-endian halves; a position or a block counter as 2
DIGEST = Array(16, W.U32)
POS = Array(2, W.U32)
DIGEST_BITS = BL.bit_type(DIGEST)
POS_BITS = BL.bit_type(POS)
BLOCK_BITS = 512


def types(boolean: bool) -> tuple[Array, Array]:
    """A digest's type and a position's at a level: words, or their bits (`boolean`)."""
    return (DIGEST_BITS, POS_BITS) if boolean else (DIGEST, POS)


# -- the reference -----------------------------------------------------------------------------------------------------------

def _u64(x: int) -> bytes:
    if not 0 <= x < 1 << 64:
        raise ValueError(f"a position or a counter is a u64, not {x!r}")
    return x.to_bytes(8, "big")


def seed(salt: bytes, position: int, root_a: bytes, root_b: bytes) -> bytes:
    """seed_j of the unit at `position` whose operands commit to `root_a` and `root_b`, under the epoch's `salt`."""
    for what, v in (("salt", salt), ("root_a", root_a), ("root_b", root_b)):
        if len(v) != 64:
            raise ValueError(f"{what} is a 64-byte digest, not {len(v)} bytes")
    p = _u64(position)
    data = SEED_TAG + bytes(salt) + bytes(root_a) + bytes(root_b)
    return hashlib.sha512(p + bytes(x ^ p[i % 8] for i, x in enumerate(data))).digest()


def noise_block(seed_j: bytes, t: int) -> bytes:
    """Block t of the noise stream of seed_j."""
    if len(seed_j) != 64:
        raise ValueError(f"a seed is a 64-byte digest, not {len(seed_j)} bytes")
    return hashlib.sha512(NOISE_TAG + bytes(seed_j) + _u64(t)).digest()


def halves(data: bytes) -> list[int]:
    """Bytes (a multiple of 4) as big-endian 32-bit halves."""
    if len(data) % 4:
        raise ValueError(f"{len(data)} bytes are no whole halves")
    return [int.from_bytes(data[i:i + 4], "big") for i in range(0, len(data), 4)]


def leaves(data: bytes, boolean: bool) -> list[int]:
    """`data`'s leaves as a unit's input reads them: its halves, or their bits (least significant first)."""
    hs = halves(data)
    return [(h >> j) & 1 for h in hs for j in range(32)] if boolean else hs


def position_leaves(position: int, boolean: bool) -> list[int]:
    return leaves(_u64(position), boolean)


def stream(seed_j: bytes, n_bits: int) -> list[int]:
    """The first `n_bits` bits of seed_j's noise stream."""
    out: list[int] = []
    t = 0
    while len(out) < n_bits:
        out += leaves(noise_block(seed_j, t), True)
        t += 1
    return out[:n_bits]


# -- in gates ----------------------------------------------------------------------------------------------------------------

def _pairs(c: Coll, n: int) -> list[tuple[Coll, Coll]]:
    return [(c[2 * i], c[2 * i + 1]) for i in range(n)]


def _seed_body(B, S, salt, position, roots):
    e = H.Emit(B)
    p = (position[0], position[1])
    data = H.sha_words(SEED_TAG) + _pairs(salt, 8) + _pairs(roots[0], 8) + _pairs(roots[1], 8)
    return e.array(H.sha512(e, [p] + [(e.call(W.PouwXor32, hi, p[0]), e.call(W.PouwXor32, lo, p[1])) for hi, lo in data]))


def _block_body(B, S, seed_j, t):
    e = H.Emit(B)
    return e.array(H.sha512(e, H.sha_words(NOISE_TAG) + _pairs(seed_j, 8) + [(t[0], t[1])]))


def _coin_salt(S):
    return _ports.Ports(params={"salt": _ports.In(kind="coin")})


def _word_def(name: str, params, body, doc: str, ports=None) -> CompositeDefinition:
    kw = {"ports": ports} if _ports is not None and ports is not None else {}
    return CompositeDefinition(name, 1, (), lambda S: (params, DIGEST), body, doc=doc, conformance="protocol-defined",
                               register=False, **kw)


_SEED_PARAMS = (("salt", DIGEST), ("position", POS), ("roots", Array(2, DIGEST)))
_BLOCK_PARAMS = (("seed", DIGEST), ("t", POS))
PouwSeed = _word_def("PouwSeed", _SEED_PARAMS, _seed_body, "seed_j from the salt (a coin), the unit's position and its "
                     "operands' roots (A's, then B's): `seeds.seed`, the message words after the first XORed with the position.",
                     ports=_coin_salt)
PouwNoiseBlock = _word_def("PouwNoiseBlock", _BLOCK_PARAMS, _block_body, "Block t of seed_j's noise stream: "
                           "SHA-512(NOISE_TAG ‖ seed ‖ u64(t)), `seeds.noise_block`.")


def _w64(hi: Sequence, lo: Sequence) -> list:
    return list(lo) + list(hi)


def _out(st: list[list]) -> list[list]:
    return [h for w in st for h in (w[32:], w[:32])]


def _sha512_bits(C, msg: list[list]) -> list[list]:
    st = [const_bits(v, 64) for v in IV]
    for i in range(0, len(msg), 16):
        st = compress_gadget(C, st, msg[i:i + 16])
    return st


def _seed_bits(C, *el):
    salt, pos, roots = el[:16], el[16:18], el[18:]
    p = _w64(*pos)
    data = ([const_bits(v, 64) for v in words_be(SEED_TAG)] + [_w64(salt[2 * i], salt[2 * i + 1]) for i in range(8)]
            + [_w64(roots[2 * i], roots[2 * i + 1]) for i in range(16)])
    n = 8 + 8 * len(data)
    msg = [p] + [[a ^ b for a, b in zip(w, p)] for w in data] + [const_bits(v, 64) for v in words_be(pad_tail(n))]
    return _out(_sha512_bits(C, msg))


def _block_bits(C, *el):
    seed_j, t = el[:16], el[16:18]
    data = [const_bits(v, 64) for v in words_be(NOISE_TAG)] + [_w64(seed_j[2 * i], seed_j[2 * i + 1]) for i in range(8)]
    data.append(_w64(*t))
    return _out(_sha512_bits(C, data + [const_bits(v, 64) for v in words_be(pad_tail(8 * len(data)))]))


PouwSeedBits = definition("PouwSeed", 2, _seed_bits, tuple((n, BL.bit_type(t)) for n, t in _SEED_PARAMS), DIGEST_BITS,
                          word=bind(PouwSeed), conformance="protocol-defined", register=False,
                          doc="seed_j on bits (two SHA-512 compressions): PouwSeed_v1")
PouwNoiseBlockBits = definition("PouwNoiseBlock", 2, _block_bits, tuple((n, BL.bit_type(t)) for n, t in _BLOCK_PARAMS),
                                DIGEST_BITS, word=bind(PouwNoiseBlock), conformance="protocol-defined", register=False,
                                doc="Block t of seed_j's noise stream on bits (one SHA-512 compression): PouwNoiseBlock_v1")
if _ports is not None:
    PouwSeedBits.ports = _coin_salt

SEEDS = (PouwSeed, PouwSeedBits)


def is_seed(fn) -> bool:
    """Whether `fn` (a node's Definition) is the seed, at either level."""
    return getattr(fn, "defn", None) in SEEDS


def seed_call(B, salt: Coll, position: Coll, roots: Coll, boolean: bool) -> Coll:
    """The unit's seed, in its body `B`: one call of `PouwSeed` (`_v2` on bits) on its salt, position and roots parameters."""
    return B.call(PouwSeedBits if boolean else PouwSeed, salt, position, roots)


def noise_blocks(B, seed_j: Coll, n: int, boolean: bool) -> list[Coll]:
    """Blocks 0 .. n - 1 of the seed's noise stream, in the body `B`; each counter is a constant of the body."""
    fn = PouwNoiseBlockBits if boolean else PouwNoiseBlock
    if boolean:
        bit: dict[int, tuple] = {}

        def const(v: int) -> tuple:
            if v not in bit:
                bit[v] = B.call(BL.constant(v)).refs[0]
            return bit[v]

        counters = [Coll(POS_BITS, Explicit(tuple(const(v) for v in position_leaves(t, True)))) for t in range(n)]
    else:
        e = H.Emit(B)
        counters = [array_of([e.c(t >> 32), e.c(t & M32)]) for t in range(n)]
    return [B.call(fn, seed_j, c) for c in counters]


def stream_bits(B, seed_j: Coll, n_bits: int) -> Coll:
    """The first `n_bits` bits of the seed's noise stream as one Boolean collection (a Boolean unit's)."""
    blocks = noise_blocks(B, seed_j, -(-n_bits // BLOCK_BITS), True)
    return Coll(Array(n_bits, BL.BIT), concat([b.refs for b in blocks]).slice(0, n_bits))


__all__ = ["BLOCK_BITS", "DIGEST", "DIGEST_BITS", "NOISE_TAG", "POS", "POS_BITS", "PouwNoiseBlock", "PouwNoiseBlockBits",
           "PouwSeed", "PouwSeedBits", "SEEDS", "SEED_TAG", "halves", "is_seed", "leaves", "noise_block", "noise_blocks",
           "position_leaves", "seed", "seed_call", "stream", "stream_bits"]
