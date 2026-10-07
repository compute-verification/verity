"""Hidden-index Merkle reads: public Boolean Definitions, one per tree shape, that open a leaf, or an aligned run of cells, at a
position they take as a value (a piece of a hidden circuit reads its inputs from committed scratch storage at hidden positions).

    MerkleRead_v1{ARITY, D}
        (root: Array<8, Array<64, Value<1>>>, leaf: Array<8, Array<64, Value<1>>>, pos: Array<B D, Value<1>>,
         path: Array<D, Array<ARITY - 1, Array<8, Array<64, Value<1>>>>>) -> Value<1>
    MerkleSubtreeRead_v1{ARITY, D, J}
        (root: Array<8, Array<64, Value<1>>>, bc: Array<ARITY^J, Array<16, Array<64, Value<1>>>>, pos: Array<B (D - J), Value<1>>,
         path: Array<D - J, Array<ARITY - 1, Array<8, Array<64, Value<1>>>>>) -> Value<1>
        ARITY 2 or 4, B = log2 ARITY bits per level, D >= 1 levels, 0 <= J < D

**The tree.** Leaves and nodes are 64-byte digests, each as its eight SHA-512 words (bytes big-endian, `words_be`). A node is
SHA-512 over its ARITY children's bytes in slot order, ``SHA-512(c_0 || ... || c_{ARITY-1})`` (`node_ref`). Every leaf is at
depth D, which the Definition fixes, so a leaf is never taken for a node and nothing tags either; the tree holds ARITY^D leaves.
Binary, it is C-Flock's Merkle tree (verifier PROTOCOL.md section 14: ``SHA-512(left || right)``, the even index on the left).

**The read.** ``pos`` is the leaf's position, an unsigned integer below ARITY^D, bits LSB first: its digit k (bits
``B k .. B k + B - 1``) is the slot, among its siblings, of the leaf's ancestor at level k (level 0 the leaf itself), and
``path[k]`` holds that ancestor's ARITY - 1 siblings in slot order. Level by level the current digest goes into the slot its digit
names (multiplexers over the children's bits) and the node is computed; the result is 1 iff the last node is ``root``
(`opens`). The circuit is a function of (ARITY, D) alone: the position, the path and the leaf are values, hidden like any other,
and where a unit uses the Definition only ``root`` is public.

**The subtree read** opens the aligned run of R = ARITY^J cells under one node at height J, cells ``pos R .. pos R + R - 1``, with
one path of D - J levels. A cell enters as its 128-byte ``hm96-sha512/v1`` commit string ``b || c`` (16 SHA-512 words, b's then
c's), and its leaf is hm96's tree leaf ``SHA-512(LEAF_PREFIX || b || c)`` (`leaf_ref`; the prefix is one block, a constant
midstate). The R leaves are hashed into the run's node level by level in cell order, with no multiplexer; then ``pos``, the run's
index among the ARITY^(D - J) runs, and ``path``, its siblings from height J up, climb as in the read (`opens_run`). J = 0 is
`MerkleRead` over the cell's leaf.

**Cost per level.** The node's compressions (`sha512.compress_gadget`): binary, the message block from the constant IV, then
the padding block, whose schedule is free; 4-ary, two message blocks, then the padding block. The slots: binary, a conditional
swap (``t = d AND (cur XOR s)``, one AND per bit, 512); 4-ary, c_0 and c_3 each pick between the digest and one sibling (512
each), c_1 and c_2 among three (1,024 each), over the digit's one-hot ends ``d_0 = 0 = d_1`` and ``d_0 = 1 = d_1`` (2 ANDs): 3,074.
The comparison with ``root`` is 511 more. A cell's leaf is a binary node's two compressions from another constant midstate,
and a run's node is a level without its slots. The tests pin each level's, leaf's and run node's ANDs and the rows C-Flock
commits for it (`forms` counts the ANDs and the copies `sha512.commit` adds).

They are traced from `build` and `build_subtree` (`trace`); they name no word Definition, and `opens` and `opens_run` (over
``hashlib``) are their references.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from types import SimpleNamespace
from typing import Sequence

from verity.primitives.circuits import boolean as BL
from verity.primitives.circuits.defs import CompositeDefinition
from verity.primitives.circuits.types import Array

from verity.primitives.circuits.boolean import trace as TR
from verity.primitives.circuits.boolean.forms import and_reduce, const_bits
from .sha512 import IV, compress_gadget, midstate, pad_tail, words_be

__all__ = ["ARITIES", "LEAF_PREFIX", "MerkleRead", "MerkleSubtreeRead", "arguments", "build", "build_subtree", "leaf_ref",
           "node_ref", "opens", "opens_run", "root_ref", "run_arguments", "run_root_ref", "traced", "traced_subtree"]

BIT = BL.BIT
ARITIES = (2, 4)
WORD = Array(64, BIT)
DIGEST = Array(8, WORD)
CELL = Array(16, WORD)

#: ``hm96-sha512/v1``'s tree-leaf prefix under its pinned key, ``verity.primitives.commitments.hm96.SHA512.leaf_prefix(default_key)``:
#: the leaf tag, the key's SHA-512, zero-padded to one block. Core's ``ml`` imports no ``commitments``; the tests pin the two
#: equal.
LEAF_PREFIX = (b"verity/hm96-sha512/leaf/v1\0" + bytes.fromhex(
    "afc5a60c85d8aa75cb4c7f726588d299a1712e1c6a621f09ca64809a5fedd744"
    "a0f6b0201d05e82b29cd5b274e6c29670cd4e4ab10831911fb1f9e0c7019df22"
)).ljust(128, b"\0")


def _check(ARITY: int, D: int) -> None:
    if ARITY not in ARITIES or isinstance(ARITY, bool):
        raise ValueError(f"a Merkle read's ARITY is one of {ARITIES}, not {ARITY!r}")
    if isinstance(D, bool) or not isinstance(D, int) or D < 1:
        raise ValueError(f"a Merkle read's D is an int >= 1, not {D!r}")


def _check_run(ARITY: int, D: int, J: int) -> None:
    _check(ARITY, D)
    if isinstance(J, bool) or not isinstance(J, int) or not 0 <= J < D:
        raise ValueError(f"a subtree read's J is an int with 0 <= J < D = {D}, not {J!r}")


def _b(ARITY: int) -> int:
    return ARITY.bit_length() - 1


# --- reference ---------------------------------------------------------------------------------------------------------------

def node_ref(children: Sequence[bytes]) -> bytes:
    return hashlib.sha512(b"".join(children)).digest()


def leaf_ref(bc: bytes) -> bytes:
    """A cell's leaf, ``SHA-512(LEAF_PREFIX || b || c)`` from its 128-byte commit string (hm96's ``tree_leaf``)."""
    if len(bc) != 128:
        raise ValueError(f"a cell's commit string b || c is 128 bytes, not {len(bc)}")
    return hashlib.sha512(LEAF_PREFIX + bc).digest()


def root_ref(ARITY: int, leaf: bytes, pos: int, path: Sequence[Sequence[bytes]]) -> bytes:
    """The root the path gives ``leaf`` at ``pos``: level k puts the current digest at slot ``(pos >> B k) mod ARITY`` among
    ``path[k]``'s siblings."""
    cur = leaf
    for k, sibs in enumerate(path):
        d = (pos >> (_b(ARITY) * k)) % ARITY
        cur = node_ref([*sibs[:d], cur, *sibs[d:]])
    return cur


def run_root_ref(ARITY: int, bcs: Sequence[bytes]) -> bytes:
    """The node of an aligned run: its cells' leaves hashed ARITY at a time, in cell order, up to one digest."""
    level = [leaf_ref(bc) for bc in bcs]
    while len(level) > 1:
        level = [node_ref(level[i:i + ARITY]) for i in range(0, len(level), ARITY)]
    return level[0]


def opens(ARITY: int, root: bytes, leaf: bytes, pos: int, path: Sequence[Sequence[bytes]]) -> int:
    """`MerkleRead`'s reference: 1 iff the path opens ``leaf`` at ``pos`` to ``root``."""
    return int(root_ref(ARITY, leaf, pos, path) == root)


def opens_run(ARITY: int, root: bytes, bcs: Sequence[bytes], pos: int, path: Sequence[Sequence[bytes]]) -> int:
    """`MerkleSubtreeRead`'s reference: 1 iff the path opens the run of cells with commit strings ``bcs`` at run position ``pos``
    to ``root``."""
    return opens(ARITY, root, run_root_ref(ARITY, bcs), pos, path)


def _digest_bits(d: bytes) -> list[int]:
    if len(d) != 64:
        raise ValueError(f"a digest is 64 bytes, not {len(d)}")
    return [(w >> j) & 1 for w in words_be(d) for j in range(64)]


def _path_bits(ARITY: int, pos: int, path: Sequence[Sequence[bytes]]) -> tuple[list[int], list[int]]:
    n = len(path)
    if any(len(s) != ARITY - 1 for s in path) or not 0 <= pos < ARITY ** n:
        raise ValueError(f"a path of {n} levels holds {ARITY - 1} siblings a level, and the position is below {ARITY}^{n}")
    return [(pos >> j) & 1 for j in range(_b(ARITY) * n)], [b for sibs in path for s in sibs for b in _digest_bits(s)]


def arguments(ARITY: int, root: bytes, leaf: bytes, pos: int, path: Sequence[Sequence[bytes]]) -> tuple[list[int], ...]:
    """``MerkleRead_v1{ARITY, D = len(path)}``'s four arguments, each a flat list of bits in leaf order
    (`verity.primitives.circuits.evaluation.evaluate`'s form)."""
    _check(ARITY, len(path))
    return (_digest_bits(root), _digest_bits(leaf), *_path_bits(ARITY, pos, path))


def run_arguments(ARITY: int, root: bytes, bcs: Sequence[bytes], pos: int,
                  path: Sequence[Sequence[bytes]]) -> tuple[list[int], ...]:
    """``MerkleSubtreeRead_v1{ARITY, D = J + len(path), J}``'s four arguments for a run of ``len(bcs) = ARITY^J`` cells, each a
    flat list of bits in leaf order."""
    _check(ARITY, 1)
    J = (len(bcs).bit_length() - 1) // _b(ARITY)
    if len(bcs) != ARITY ** J:
        raise ValueError(f"a run holds a power of {ARITY} cells, not {len(bcs)}")
    _check_run(ARITY, J + len(path), J)
    cells = []
    for bc in bcs:
        if len(bc) != 128:
            raise ValueError(f"a cell's commit string b || c is 128 bytes, not {len(bc)}")
        cells += [(w >> j) & 1 for w in words_be(bc) for j in range(64)]
    return (_digest_bits(root), cells, *_path_bits(ARITY, pos, path))


# --- the circuit -------------------------------------------------------------------------------------------------------------

def _words(bits: list) -> list[list]:
    return [bits[64 * i:64 * i + 64] for i in range(len(bits) // 64)]


def _sha(C, cv: list[int], msg: list[list], total: int) -> list:
    """SHA-512 from the chaining value ``cv`` (constant words) over ``msg`` (whole blocks of 64-bit words), ``total`` bytes hashed
    in all, as a flat 512-bit digest."""
    st = [const_bits(v, 64) for v in cv]
    for i in range(0, len(msg), 16):
        st = compress_gadget(C, st, msg[i:i + 16])
    st = compress_gadget(C, st, [const_bits(v, 64) for v in words_be(pad_tail(total))])
    return [b for w in st for b in w]


def _node(C, children: list[list]) -> list:
    """SHA-512 over the children (flat 512-bit digests in word order), as a flat 512-bit digest."""
    return _sha(C, IV, [w for c in children for w in _words(c)], 64 * len(children))


@lru_cache(maxsize=None)
def _leaf_cv() -> tuple[int, ...]:
    return tuple(midstate(LEAF_PREFIX))


def _leaf(C, bc: list) -> list:
    """A cell's leaf from its commit string (a flat 1024-bit b || c in word order), as a flat 512-bit digest."""
    return _sha(C, list(_leaf_cv()), _words(bc), len(LEAF_PREFIX) + 128)


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


def _opens(C, ARITY: int, levels: int, root: list, cur: list, pos: list, words: list) -> list:
    """Climb ``levels`` levels from ``cur`` by ``pos``'s digits and ``words``' siblings (8 words a digest), then compare with
    ``root``: one bit."""
    b = _b(ARITY)
    for k in range(levels):
        sibs = [[x for w in words[8 * j:8 * j + 8] for x in w] for j in range((ARITY - 1) * k, (ARITY - 1) * (k + 1))]
        cur = _node(C, _slots(C, pos[b * k:b * k + b], cur, sibs))
    return [and_reduce(C, [(x ^ y).inv() for x, y in zip(cur, root)])]


def build(C, ARITY: int, D: int, *elements):
    """The read over `forms`: ``elements`` are the root's 8 words, the leaf's 8, the position, and the path's D (ARITY - 1) 8
    words, each a list of forms LSB first."""
    root, cur = ([x for w in elements[8 * i:8 * i + 8] for x in w] for i in range(2))
    return _opens(C, ARITY, D, root, cur, elements[16], elements[17:])


def build_subtree(C, ARITY: int, D: int, J: int, *elements):
    """The subtree read over `forms`: ``elements`` are the root's 8 words, each cell's 16 (its b || c), the run's position, and
    the path's (D - J) (ARITY - 1) 8 words, each a list of forms LSB first."""
    R = ARITY ** J
    root = [x for w in elements[:8] for x in w]
    level = [_leaf(C, [x for w in elements[8 + 16 * i:24 + 16 * i] for x in w]) for i in range(R)]
    while len(level) > 1:
        level = [_node(C, level[i:i + ARITY]) for i in range(0, len(level), ARITY)]
    return _opens(C, ARITY, D - J, root, level[0], elements[8 + 16 * R], elements[9 + 16 * R:])


def _signature(S):
    _check(S.ARITY, S.D)
    return ((("root", DIGEST), ("leaf", DIGEST), ("pos", Array(_b(S.ARITY) * S.D, BIT)),
             ("path", Array(S.D, Array(S.ARITY - 1, DIGEST)))), BIT)


def _subtree_signature(S):
    _check_run(S.ARITY, S.D, S.J)
    n = S.D - S.J
    return ((("root", DIGEST), ("bc", Array(S.ARITY ** S.J, CELL)), ("pos", Array(_b(S.ARITY) * n, BIT)),
             ("path", Array(n, Array(S.ARITY - 1, DIGEST)))), BIT)


def _types(ARITY: int, D: int, J: int | None = None):
    S = SimpleNamespace(ARITY=ARITY, D=D, J=J)
    params, ret = _signature(S) if J is None else _subtree_signature(S)
    return [t for _, t in params], ret


@lru_cache(maxsize=None)
def traced(ARITY: int, D: int) -> TR.Traced:
    """The read's gate circuit (`trace.trace`)."""
    ps, ret = _types(ARITY, D)
    return TR.trace(lambda C, *e: build(C, ARITY, D, *e), ps, ret)


@lru_cache(maxsize=None)
def traced_subtree(ARITY: int, D: int, J: int) -> TR.Traced:
    """The subtree read's gate circuit (`trace.trace`)."""
    ps, ret = _types(ARITY, D, J)
    return TR.trace(lambda C, *e: build_subtree(C, ARITY, D, J, *e), ps, ret)


def _body(B, S, *colls):
    ps, ret = _types(S.ARITY, S.D)
    return TR.emit(traced(S.ARITY, S.D), ps, ret, B, *colls)


def _subtree_body(B, S, *colls):
    ps, ret = _types(S.ARITY, S.D, S.J)
    return TR.emit(traced_subtree(S.ARITY, S.D, S.J), ps, ret, B, *colls)


MerkleRead = CompositeDefinition(
    "MerkleRead", 1, ("ARITY", "D"), _signature, _body, conformance="exact-model-tested",
    doc="1 iff path opens leaf at the position pos to root in the depth-D ARITY-ary SHA-512 tree: each level puts the digest at "
        "the slot pos's digit names among its ARITY - 1 siblings, a node SHA-512 over its children's bytes")

MerkleSubtreeRead = CompositeDefinition(
    "MerkleSubtreeRead", 1, ("ARITY", "D", "J"), _subtree_signature, _subtree_body, conformance="exact-model-tested",
    doc="1 iff path opens the aligned run of ARITY^J cells with commit strings bc at the run position pos to root in the depth-D "
        "ARITY-ary SHA-512 tree: each cell's leaf is hm96-sha512's tree leaf of its b || c, the run's nodes are hashed in cell "
        "order, and each of the D - J levels above puts the digest at the slot pos's digit names among its ARITY - 1 siblings")
