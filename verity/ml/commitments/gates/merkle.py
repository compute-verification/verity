"""The hidden-index Merkle read: one public Boolean Definition per tree shape that opens a leaf at a position it takes as a value
(hiding model architectures: a unit reads its inputs from a committed row store at hidden positions).

    MerkleRead_v1{ARITY, D}
        (root: Array<8, Array<64, Value<1>>>, leaf: Array<8, Array<64, Value<1>>>, pos: Array<B D, Value<1>>,
         path: Array<D, Array<ARITY - 1, Array<8, Array<64, Value<1>>>>>) -> Value<1>
        ARITY 2 or 4, B = log2 ARITY bits per level, D >= 1 levels

**The tree.** Leaves and nodes are 64-byte digests, each as its eight SHA-512 words (bytes big-endian, `words_be`). A node is
SHA-512 over its ARITY children's bytes in slot order, ``SHA-512(c_0 || ... || c_{ARITY-1})`` (`node_ref`). Every leaf is at
depth D, which the Definition fixes, so a leaf is never taken for a node and nothing tags either; the tree holds ARITY^D leaves.

**The read.** ``pos`` is the leaf's position, an unsigned integer below ARITY^D, bits LSB first: its digit k (bits
``B k .. B k + B - 1``) is the slot, among its siblings, of the leaf's ancestor at level k (level 0 the leaf itself), and
``path[k]`` holds that ancestor's ARITY - 1 siblings in slot order. Level by level the current digest goes into the slot its digit
names (multiplexers over the children's bits) and the node is computed; the result is 1 iff the last node is ``root``
(`opens`). The circuit is a function of (ARITY, D) alone: the position, the path and the leaf are values, hidden like any other,
and where a unit uses the Definition only ``root`` is public.

**Cost per level.** The node's compressions (`sha512.compress_gadget`): binary, the message block from the constant IV, then
the padding block, whose schedule is free; 4-ary, two message blocks, then the padding block. The slots: binary, a conditional
swap (``t = d AND (cur XOR s)``, one AND per bit, 512); 4-ary, c_0 and c_3 each pick between the digest and one sibling (512
each), c_1 and c_2 among three (1,024 each), over the digit's one-hot ends ``d_0 = 0 = d_1`` and ``d_0 = 1 = d_1`` (2 ANDs): 3,074.
The comparison with ``root`` is 511 more. The tests pin each level's ANDs and the rows C-Flock commits for it (`forms` counts
the ANDs and the copies `sha512.commit` adds).

It is traced from `build` (`trace`); it names no word Definition, and `opens` (over ``hashlib``) is its reference.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from types import SimpleNamespace
from typing import Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition
from verity.ml.circuits.types import Array

from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.boolean.forms import and_reduce, const_bits
from .sha512 import IV, compress_gadget, pad_tail, words_be

__all__ = ["ARITIES", "MerkleRead", "arguments", "build", "node_ref", "opens", "root_ref", "traced"]

BIT = BL.BIT
ARITIES = (2, 4)
WORD = Array(64, BIT)
DIGEST = Array(8, WORD)


def _check(ARITY: int, D: int) -> None:
    if ARITY not in ARITIES or isinstance(ARITY, bool):
        raise ValueError(f"a Merkle read's ARITY is one of {ARITIES}, not {ARITY!r}")
    if isinstance(D, bool) or not isinstance(D, int) or D < 1:
        raise ValueError(f"a Merkle read's D is an int >= 1, not {D!r}")


def _b(ARITY: int) -> int:
    return ARITY.bit_length() - 1


# --- reference ---------------------------------------------------------------------------------------------------------------

def node_ref(children: Sequence[bytes]) -> bytes:
    return hashlib.sha512(b"".join(children)).digest()


def root_ref(ARITY: int, leaf: bytes, pos: int, path: Sequence[Sequence[bytes]]) -> bytes:
    """The root the path gives ``leaf`` at ``pos``: level k puts the current digest at slot ``(pos >> B k) mod ARITY`` among
    ``path[k]``'s siblings."""
    cur = leaf
    for k, sibs in enumerate(path):
        d = (pos >> (_b(ARITY) * k)) % ARITY
        cur = node_ref([*sibs[:d], cur, *sibs[d:]])
    return cur


def opens(ARITY: int, root: bytes, leaf: bytes, pos: int, path: Sequence[Sequence[bytes]]) -> int:
    """`MerkleRead`'s reference: 1 iff the path opens ``leaf`` at ``pos`` to ``root``."""
    return int(root_ref(ARITY, leaf, pos, path) == root)


def _digest_bits(d: bytes) -> list[int]:
    if len(d) != 64:
        raise ValueError(f"a digest is 64 bytes, not {len(d)}")
    return [(w >> j) & 1 for w in words_be(d) for j in range(64)]


def arguments(ARITY: int, root: bytes, leaf: bytes, pos: int, path: Sequence[Sequence[bytes]]) -> tuple[list[int], ...]:
    """``MerkleRead_v1{ARITY, D = len(path)}``'s four arguments, each a flat list of bits in leaf order
    (`verity.ml.circuits.evaluation.evaluate`'s form)."""
    D = len(path)
    _check(ARITY, D)
    if any(len(s) != ARITY - 1 for s in path) or not 0 <= pos < ARITY ** D:
        raise ValueError(f"a path of {D} levels holds {ARITY - 1} siblings a level, and the position is below {ARITY}^{D}")
    return (_digest_bits(root), _digest_bits(leaf), [(pos >> j) & 1 for j in range(_b(ARITY) * D)],
            [b for sibs in path for s in sibs for b in _digest_bits(s)])


# --- the circuit -------------------------------------------------------------------------------------------------------------

def _words(bits: list) -> list[list]:
    return [bits[64 * i:64 * i + 64] for i in range(len(bits) // 64)]


def _node(C, children: list[list]) -> list:
    """SHA-512 over the children (flat 512-bit digests in word order), as a flat 512-bit digest."""
    msg = [w for c in children for w in _words(c)]
    st = [const_bits(v, 64) for v in IV]
    for i in range(0, len(msg), 16):
        st = compress_gadget(C, st, msg[i:i + 16])
    st = compress_gadget(C, st, [const_bits(v, 64) for v in words_be(pad_tail(64 * len(children)))])
    return [b for w in st for b in w]


def _slots(C, digit: list, cur: list, sibs: list[list]) -> list[list]:
    """The children: ``cur`` at the slot ``digit`` names, ``sibs`` in the others in order."""
    if len(sibs) == 1:
        (s,) = sibs
        t = [C.AND(digit[0], x ^ y) for x, y in zip(cur, s)]
        return [[x ^ u for x, u in zip(cur, t)], [y ^ u for y, u in zip(s, t)]]
    d0, d1 = digit
    s0, s1, s2 = sibs
    e3 = C.AND(d0, d1)
    e0 = C.AND(d0.inv(), d1.inv())
    e1, e2, n1 = d0 ^ e3, d1 ^ e3, d1.inv()
    return [[b ^ C.AND(e0, x ^ b) for x, b in zip(cur, s0)],
            [b ^ C.AND(e0, a ^ b) ^ C.AND(e1, x ^ b) for x, a, b in zip(cur, s0, s1)],
            [c ^ C.AND(n1, b ^ c) ^ C.AND(e2, x ^ c) for x, b, c in zip(cur, s1, s2)],
            [c ^ C.AND(e3, x ^ c) for x, c in zip(cur, s2)]]


def build(C, ARITY: int, D: int, *elements):
    """The read over `forms`: ``elements`` are the root's 8 words, the leaf's 8, the position, and the path's D (ARITY - 1) 8
    words, each a list of forms LSB first."""
    b = _b(ARITY)
    flat = [[x for w in elements[8 * i:8 * i + 8] for x in w] for i in range(2)]
    root, cur = flat
    pos, words = elements[16], elements[17:]
    for k in range(D):
        sibs = [[x for w in words[8 * j:8 * j + 8] for x in w] for j in range((ARITY - 1) * k, (ARITY - 1) * (k + 1))]
        cur = _node(C, _slots(C, pos[b * k:b * k + b], cur, sibs))
    return [and_reduce(C, [(x ^ y).inv() for x, y in zip(cur, root)])]


def _signature(S):
    _check(S.ARITY, S.D)
    return ((("root", DIGEST), ("leaf", DIGEST), ("pos", Array(_b(S.ARITY) * S.D, BIT)),
             ("path", Array(S.D, Array(S.ARITY - 1, DIGEST)))), BIT)


def _types(ARITY: int, D: int):
    params, ret = _signature(SimpleNamespace(ARITY=ARITY, D=D))
    return [t for _, t in params], ret


@lru_cache(maxsize=None)
def traced(ARITY: int, D: int) -> TR.Traced:
    """The shape's gate circuit (`trace.trace`)."""
    ps, ret = _types(ARITY, D)
    return TR.trace(lambda C, *e: build(C, ARITY, D, *e), ps, ret)


def _body(B, S, *colls):
    ps, ret = _types(S.ARITY, S.D)
    return TR.emit(traced(S.ARITY, S.D), ps, ret, B, *colls)


MerkleRead = CompositeDefinition(
    "MerkleRead", 1, ("ARITY", "D"), _signature, _body, conformance="exact-model-tested",
    doc="1 iff path opens leaf at the position pos to root in the depth-D ARITY-ary SHA-512 tree: each level puts the digest at "
        "the slot pos's digit names among its ARITY - 1 siblings, a node SHA-512 over its children's bytes")
