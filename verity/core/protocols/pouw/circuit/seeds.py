"""The seed and the noise a PoUW unit derives inside its own proof (decisions 105 and 106, `PROTOCOL.md` §`compile`).

A drawn unit's noise is neither a public nor a coin input: the unit computes its seed from the window's salt
(`pouw.salt`, the one coin), its position and its operands' roots, all inputs of the unit bound before the draw, and its
noise from the seed, in gates.

    seed_j  = SHA-512(u64(j) ‖ (SEED_TAG ‖ salt ‖ root(A_j) ‖ root(B_j)) ⊕ u64(j)*)        `PouwSeed`
    σ_R     = SHA-512(u64(j) ‖ (RECEIPT_TAG ‖ salt ‖ root(A_j) ‖ root(B_j)) ⊕ u64(j)*)     `PouwUnitSeeds`, with seed_j
    block_t = SHA-512(NOISE_TAG ‖ seed_j ‖ u64(t))                                          `PouwNoiseBlock`

``⊕ u64(j)*`` XORs every 64-bit word of the message after the first with the position: an injective encoding of (tag, salt,
position, roots), which makes every gate of the seed read the unit's own position, so no two units compute a common value
(the gate-level cut) although they share the salt and a site's units share its roots. The noise stream is the blocks
``block_0, block_1, ...`` in order, each as 16 big-endian 32-bit halves, bit β the bit β mod 32 (least significant first) of
half (β mod 512) // 32 of block β // 512 (`stream`); a Boolean unit reads it as the leaves of its blocks.

σ_R keys a unit's receipt folds and starts their chains (decision 121): a hash of its own tag, never the noise seed, so no
fold key is a function of seed_j. A unit with receipt keys calls `PouwUnitSeeds` once, for both, rather than two seeds at
one position.

The one-stage verifier fills the salt, the positions and the roots itself, so the prover chooses none of them (`compile`'s
docstring, `sampled_proofs.one_stage.registered.supplied_rows`).

**J10-R**, a second noise form beside the
one above, which no unit takes by default (`h1tile.PouwH1Tile5`, `h1tile.PouwH1Tile6`, and under J10-R v2's longer
column streams `h1tile.PouwH1Tile7`, `h1tile.PouwH1Tile8`): one noise seed a call, one stream per call row and per call
column, and σ_R still the unit's own.

    seed_c  = SHA-512(u64(c) ‖ (CALL_SEED_TAG ‖ salt ‖ root(A) ‖ root(B)) ⊕ u64(c)*)       `PouwCallSeeds`, with σ_R
    block_t = SHA-512(NOISE_TAG ‖ seed_c ‖ u64(t))                                          `PouwNoiseBlock`, t from `PouwAdd64`

c is the call's first unit position and the roots its registered root(a/<K>@<s>) and root(b/<K>), so the call's
units share seed_c, two calls of one step and K (which share those roots) don't, and every column stream reads the A
root. σ_R is `receipt_seed` at the unit's own position, so the fold keys are the per-unit form's bit for bit. A unit
reads its rows' and columns' blocks from counters its public ``blocks`` gives (`h1tile.j10r_blocks`), each block's
counter that base plus a constant of the body (`PouwAdd64`). The gate-level cut above holds for σ_R and the fold keys,
not for the noise, which the call's units share on purpose.

Each Definition has a word form (`_v1`, over `Value<32>` halves, with `hashes.Sha512Compress`) and a Boolean twin (`_v2`, over
32-bit elements, with `commitments.gates.sha512.compress_gadget`) whose word view it is. The salt port is `In(kind="coin")`
where port types (`ml.circuits.ports`) are on the tree.
"""
from __future__ import annotations

import hashlib
from typing import Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import add, const_bits
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
RECEIPT_TAG = b"verity/pouw/receipt/v1".ljust(24, b"\0")
NOISE_TAG = b"verity/pouw/noise/v1".ljust(24, b"\0")
#: J10-R's call seed (`call_seed`)
CALL_SEED_TAG = b"verity/pouw/call-seed/v1".ljust(24, b"\0")
M32 = W.M32
#: a digest (the salt, a root, a seed, a noise block) as 16 big-endian halves; a position or a block counter as 2
DIGEST = Array(16, W.U32)
POS = Array(2, W.U32)
#: `PouwUnitSeeds`' return: seed_j, then σ_R
SEED_PAIR = Array(2, DIGEST)
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


def _tagged(tag: bytes, salt: bytes, position: int, root_a: bytes, root_b: bytes) -> bytes:
    for what, v in (("salt", salt), ("root_a", root_a), ("root_b", root_b)):
        if len(v) != 64:
            raise ValueError(f"{what} is a 64-byte digest, not {len(v)} bytes")
    p = _u64(position)
    data = tag + bytes(salt) + bytes(root_a) + bytes(root_b)
    return hashlib.sha512(p + bytes(x ^ p[i % 8] for i, x in enumerate(data))).digest()


def seed(salt: bytes, position: int, root_a: bytes, root_b: bytes) -> bytes:
    """seed_j of the unit at `position` whose operands commit to `root_a` and `root_b`, under the epoch's `salt`."""
    return _tagged(SEED_TAG, salt, position, root_a, root_b)


def receipt_seed(salt: bytes, position: int, root_a: bytes, root_b: bytes) -> bytes:
    """σ_R of the unit at `position` (decision 121): seed_j's message under `RECEIPT_TAG`, its receipt folds' keys."""
    return _tagged(RECEIPT_TAG, salt, position, root_a, root_b)


def call_seed(salt: bytes, call: int, root_a: bytes, root_b: bytes) -> bytes:
    """J10-R's seed_c of the call whose first unit is at position `call` and whose registered roots are `root_a`
    (root(a/<K>@<s>)) and `root_b` (root(b/<K>)): seed_j's message under `CALL_SEED_TAG` at the call's position."""
    return _tagged(CALL_SEED_TAG, salt, call, root_a, root_b)


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


def _unit_seeds_body(B, S, salt, position, roots):
    e = H.Emit(B)
    p = (position[0], position[1])

    def masked(words):
        return [(e.call(W.PouwXor32, hi, p[0]), e.call(W.PouwXor32, lo, p[1])) for hi, lo in words]

    rest = masked(_pairs(salt, 8) + _pairs(roots[0], 8) + _pairs(roots[1], 8))
    return array_of([e.array(H.sha512(e, [p] + masked(H.sha_words(tag)) + rest)) for tag in (SEED_TAG, RECEIPT_TAG)])


def _call_seeds_body(B, S, salt, position, call, roots):
    e = H.Emit(B)

    def tagged(tag, at):
        p = (at[0], at[1])
        data = H.sha_words(tag) + _pairs(salt, 8) + _pairs(roots[0], 8) + _pairs(roots[1], 8)
        return e.array(H.sha512(e, [p] + [(e.call(W.PouwXor32, hi, p[0]), e.call(W.PouwXor32, lo, p[1])) for hi, lo in data]))

    return array_of([tagged(CALL_SEED_TAG, call), tagged(RECEIPT_TAG, position)])


def _add64_body(B, S, a, b):
    e = H.Emit(B)
    return array_of([e.call(W.PouwAdd64Hi, a[0], a[1], b[0], b[1]), e.call(W.PouwAdd64Lo, a[1], b[1])])


def _block_body(B, S, seed_j, t):
    e = H.Emit(B)
    return e.array(H.sha512(e, H.sha_words(NOISE_TAG) + _pairs(seed_j, 8) + [(t[0], t[1])]))


def _coin_salt(S):
    position = {"position": _ports.In(kind="position")} if "position" in _ports.KINDS else {}
    return _ports.Ports(params={"salt": _ports.In(kind="coin"), **position})


def _word_def(name: str, params, body, doc: str, ports=None, ret=DIGEST) -> CompositeDefinition:
    kw = {"ports": ports} if _ports is not None and ports is not None else {}
    return CompositeDefinition(name, 1, (), lambda S: (params, ret), body, doc=doc, conformance="protocol-defined", **kw)


_SEED_PARAMS = (("salt", DIGEST), ("position", POS), ("roots", Array(2, DIGEST)))
_CALL_SEED_PARAMS = (("salt", DIGEST), ("position", POS), ("call", POS), ("roots", Array(2, DIGEST)))
_BLOCK_PARAMS = (("seed", DIGEST), ("t", POS))
_ADD64_PARAMS = (("a", POS), ("b", POS))
PouwSeed = _word_def("PouwSeed", _SEED_PARAMS, _seed_body, "seed_j from the salt (a coin), the unit's position and its "
                     "operands' roots (A's, then B's): `seeds.seed`, the message words after the first XORed with the position.",
                     ports=_coin_salt)
PouwUnitSeeds = _word_def("PouwUnitSeeds", _SEED_PARAMS, _unit_seeds_body, "A unit's two seeds, one seed role: [seed_j, "
                          "σ_R], `seeds.seed` and `seeds.receipt_seed` (decision 121: σ_R keys the receipt folds, under a "
                          "tag of its own), the masked salt and roots shared.", ports=_coin_salt, ret=SEED_PAIR)
PouwNoiseBlock = _word_def("PouwNoiseBlock", _BLOCK_PARAMS, _block_body, "Block t of seed_j's noise stream: "
                           "SHA-512(NOISE_TAG ‖ seed ‖ u64(t)), `seeds.noise_block`.")
PouwCallSeeds = _word_def("PouwCallSeeds", _CALL_SEED_PARAMS, _call_seeds_body, "J10-R's two seeds, one seed role: "
                          "[seed_c, σ_R], `seeds.call_seed` at the call's first position and `seeds.receipt_seed` at the "
                          "unit's own.", ports=_coin_salt, ret=SEED_PAIR)
PouwAdd64 = _word_def("PouwAdd64", _ADD64_PARAMS, _add64_body, "a + b mod 2^64 on two big-endian u32 halves each: J10-R's "
                      "block counters.", ret=POS)


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


def _unit_seeds_bits(C, *el):
    salt, pos, roots = el[:16], el[16:18], el[18:]
    p = _w64(*pos)

    def masked(words):
        return [[a ^ b for a, b in zip(w, p)] for w in words]

    rest = masked([_w64(salt[2 * i], salt[2 * i + 1]) for i in range(8)] + [_w64(roots[2 * i], roots[2 * i + 1])
                                                                             for i in range(16)])
    out = []
    for tag in (SEED_TAG, RECEIPT_TAG):
        data = masked([const_bits(v, 64) for v in words_be(tag)]) + rest
        out += _out(_sha512_bits(C, [p] + data + [const_bits(v, 64) for v in words_be(pad_tail(8 + 8 * len(data)))]))
    return out


def _call_seeds_bits(C, *el):
    salt, pos, call, roots = el[:16], el[16:18], el[18:20], el[20:]
    out = []
    for tag, at in ((CALL_SEED_TAG, call), (RECEIPT_TAG, pos)):
        p = _w64(*at)
        data = ([const_bits(v, 64) for v in words_be(tag)] + [_w64(salt[2 * i], salt[2 * i + 1]) for i in range(8)]
                + [_w64(roots[2 * i], roots[2 * i + 1]) for i in range(16)])
        msg = [p] + [[a ^ b for a, b in zip(w, p)] for w in data] + [const_bits(v, 64) for v in words_be(pad_tail(8 + 8 * len(data)))]
        out += _out(_sha512_bits(C, msg))
    return out


def _add64_bits(C, *el):
    s = add(C, _w64(*el[0:2]), _w64(*el[2:4]), width=64)
    return [s[32:], s[:32]]


def _block_bits(C, *el):
    seed_j, t = el[:16], el[16:18]
    data = [const_bits(v, 64) for v in words_be(NOISE_TAG)] + [_w64(seed_j[2 * i], seed_j[2 * i + 1]) for i in range(8)]
    data.append(_w64(*t))
    return _out(_sha512_bits(C, data + [const_bits(v, 64) for v in words_be(pad_tail(8 * len(data)))]))


PouwSeedBits = definition("PouwSeed", 2, _seed_bits, tuple((n, BL.bit_type(t)) for n, t in _SEED_PARAMS), DIGEST_BITS,
                          word=bind(PouwSeed), conformance="protocol-defined",
                          doc="seed_j on bits (two SHA-512 compressions): PouwSeed_v1")
PouwUnitSeedsBits = definition("PouwUnitSeeds", 2, _unit_seeds_bits, tuple((n, BL.bit_type(t)) for n, t in _SEED_PARAMS),
                               BL.bit_type(SEED_PAIR), word=bind(PouwUnitSeeds), conformance="protocol-defined",
                               doc="[seed_j, σ_R] on bits (four SHA-512 compressions): PouwUnitSeeds_v1")
PouwNoiseBlockBits = definition("PouwNoiseBlock", 2, _block_bits, tuple((n, BL.bit_type(t)) for n, t in _BLOCK_PARAMS),
                                DIGEST_BITS, word=bind(PouwNoiseBlock), conformance="protocol-defined",
                                doc="Block t of seed_j's noise stream on bits (one SHA-512 compression): PouwNoiseBlock_v1")
PouwCallSeedsBits = definition("PouwCallSeeds", 2, _call_seeds_bits, tuple((n, BL.bit_type(t)) for n, t in _CALL_SEED_PARAMS),
                               BL.bit_type(SEED_PAIR), word=bind(PouwCallSeeds), conformance="protocol-defined",
                               doc="[seed_c, σ_R] on bits (four SHA-512 compressions): PouwCallSeeds_v1")
PouwAdd64Bits = definition("PouwAdd64", 2, _add64_bits, tuple((n, BL.bit_type(t)) for n, t in _ADD64_PARAMS), POS_BITS,
                           word=bind(PouwAdd64), conformance="protocol-defined",
                           doc="a + b mod 2^64 on bits, one ripple carry: PouwAdd64_v1")
if _ports is not None:
    PouwSeedBits.ports = _coin_salt
    PouwUnitSeedsBits.ports = _coin_salt
    PouwCallSeedsBits.ports = _coin_salt

SEEDS = (PouwSeed, PouwSeedBits, PouwUnitSeeds, PouwUnitSeedsBits, PouwCallSeeds, PouwCallSeedsBits)


def is_seed(fn) -> bool:
    """Whether `fn` (a node's Definition) is a seed, `PouwSeed` or `PouwUnitSeeds`, at either level."""
    return getattr(fn, "defn", None) in SEEDS


def seed_call(B, salt: Coll, position: Coll, roots: Coll, boolean: bool) -> Coll:
    """The unit's seed, in its body `B`: one call of `PouwSeed` (`_v2` on bits) on its salt, position and roots parameters."""
    return B.call(PouwSeedBits if boolean else PouwSeed, salt, position, roots)


def unit_seeds_call(B, salt: Coll, position: Coll, roots: Coll, boolean: bool) -> tuple[Coll, Coll]:
    """(seed_j, σ_R) in the body `B`: one call of `PouwUnitSeeds` (`_v2` on bits), for a unit whose receipt folds take keys."""
    both = B.call(PouwUnitSeedsBits if boolean else PouwUnitSeeds, salt, position, roots)
    return both[0], both[1]


def call_seeds_call(B, salt: Coll, position: Coll, call: Coll, roots: Coll, boolean: bool) -> tuple[Coll, Coll]:
    """(seed_c, σ_R) in the body `B`: one call of `PouwCallSeeds` (`_v2` on bits), J10-R's one seed call a unit."""
    both = B.call(PouwCallSeedsBits if boolean else PouwCallSeeds, salt, position, call, roots)
    return both[0], both[1]


def noise_blocks_at(B, seed: Coll, groups: Sequence[tuple[Coll, Sequence[int]]], boolean: bool) -> list[list[Coll]]:
    """For each (base, offsets) of `groups`, blocks base + o of the seed's noise stream for each o of offsets, in the body
    `B`: each counter one `PouwAdd64` of the input base and the constant o, every counter before any block."""
    if boolean:
        bit: dict[int, tuple] = {}

        def const(v: int) -> tuple:
            if v not in bit:
                bit[v] = B.call(BL.constant(v)).refs[0]
            return bit[v]

        def offset(o: int) -> Coll:
            return Coll(POS_BITS, Explicit(tuple(const(v) for v in position_leaves(o, True))))

        add, block = PouwAdd64Bits, PouwNoiseBlockBits
    else:
        e = H.Emit(B)

        def offset(o: int) -> Coll:
            return array_of([e.c(o >> 32), e.c(o & M32)])

        add, block = PouwAdd64, PouwNoiseBlock
    counters = [[B.call(add, base, offset(o)) for o in offsets] for base, offsets in groups]
    return [[B.call(block, seed, c) for c in cs] for cs in counters]


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


__all__ = ["BLOCK_BITS", "CALL_SEED_TAG", "DIGEST", "DIGEST_BITS", "NOISE_TAG", "POS", "POS_BITS", "PouwAdd64",
           "PouwAdd64Bits", "PouwCallSeeds", "PouwCallSeedsBits", "PouwNoiseBlock", "PouwNoiseBlockBits", "PouwSeed",
           "PouwSeedBits", "PouwUnitSeeds", "PouwUnitSeedsBits", "RECEIPT_TAG", "SEEDS", "SEED_PAIR", "SEED_TAG",
           "call_seed", "call_seeds_call", "halves", "is_seed", "leaves", "noise_block", "noise_blocks", "noise_blocks_at",
           "position_leaves", "receipt_seed", "seed", "seed_call", "stream", "stream_bits", "unit_seeds_call"]
