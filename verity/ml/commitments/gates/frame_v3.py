"""frame-v3-sha512's hidden-position read: public Boolean Definitions that open a row leaf of a dense frame-v3-sha512 tree at a
position they take as a value (a unit reads a committed row at a hidden position, as `merkle.MerkleRead` does in an untagged
tree).

    FrameV3Read_v1{D, SCHEMA}
        (root: Digest, domain: Digest, value: Digest, pos: Array<D, Value<1>>, path: Array<D, Digest>) -> Value<1>
        Digest = Array<8, Array<64, Value<1>>>; 1 <= D <= 64 levels; SCHEMA the leaves' schema (``hm96-sha512/row/v2``, ...)

**The tree** is `verity.ml.commitments.merkle`'s framing through SHA-512 (``frame_v3/PROTOCOL.md`` §6) over a dense
domain: its positions are 0 ... count - 1 (`RangeIndexedDomain`), so a leaf's rank is its position, and D = merkle_depth(count)
(the slots from count up to 2^D hold pads, which no leaf opens to under ``cr/sha-512``: their tags differ). Every digest is 64
bytes, as its eight SHA-512 words (bytes big-endian, `sha512.words_be`). With F the frame ``veritor/protocol/merkle/frame/v3\\0``
and <p> = u64be(len p) || p:

    leaf(x)          = SHA-512( F || u32be(4) || "leaf" || <domain> || <uint(x)> || <uint(x)> || <SCHEMA> || <value> )
    node(k, i, l, r) = SHA-512( F || u32be(4) || "node" || <domain> || <uint(k)> || <uint(i)> || <l> || <r> )

``domain`` is the domain id, ``value`` the 64 bytes the leaf binds (a row digest, or a hiding row's hm96 leaf, `gates.hm96`),
and uint(x) is x big-endian in the fewest bytes, one for 0 (`merkle._uint`).

**The read.** ``pos`` is the leaf's position, D bits LSB first; ``path[k]`` is the sibling at level k (level 0 the leaf's).
Level k puts the current digest on the right iff bit k of pos is set (a conditional swap, 512 ANDs) and hashes the node at
index pos >> (k + 1); the result is 1 iff the last node is ``root`` (`opens`). The circuit is a function of (D, SCHEMA)
alone: the root, the domain id, the value, the position and the path are values, and where a unit uses the read the root and
the domain id are public.

**The parts,** each its own traced Definition, as `gates.lms` cuts its verifier, so a unit holds at most one node:

    FrameV3Leaf_v1{B, SCHEMA}   (domain, value, x: Array<8 B, Value<1>>) -> Digest                    leaf(x), 2 compressions
    FrameV3Node_v1{B}           (domain, k: Array<8, Value<1>>, i: Array<8 B, Value<1>>, d: Value<1>, cur, sib) -> Digest
                                node(k, i, l, r) with (l, r) = (cur, sib), swapped if d; 3 compressions
    LmsEqual_v1{N = 64}         `gates.lms`' comparison of the last node with the root, 511 ANDs

**uint's length.** uint(x) takes as many bytes L as x needs, so every byte after it moves with the hidden value. A part lays its
message out once for each L in 1 ... B and selects bit by bit with z_j = "x has a bit at or above 8 j", which nest:
m = m_1 + z_1 (m_2 + m_1) + z_2 (m_3 + m_2) + ..., an AND per bit where two layouts differ in a value's bit and none where they
differ in a constant. Every layout of a message pads to the same number of blocks (`_check_schema` refuses a binding where
they don't). The read passes a level's index pos >> (k + 1) zero-extended to whole bytes, so its node is ``FrameV3Node_v1{B}`` with
B = max(1, ceil((D - k - 1) / 8)): at most three node Definitions for D <= 24.

**Cost.** The domain id is a value, so every message's first block is a compression in gates: a leaf (209 + 2 L + |SCHEMA|
bytes) is two, a node (274 + L bytes; its level byte in the first block too) three. The layouts' selection is an AND per moved
value bit: about 1,150 ANDs for a two-byte node, 800 for a two-byte leaf and 1,550 for a three-byte one. ``PARTS`` and ``READS``
in the tests pin the gates.

**A read at a rank and a position, to a depth.** A domain whose positions don't start at 0 (an append's store indices
first ... first + count - 1, `RangeIndexedDomain(first, ...)`) has rank = position - first, and a unit that opens leaves of
trees of several sizes takes each tree's depth as a value:

    FrameV3ReadAt_v1{D, B, SCHEMA}
        (root: Digest, domain: Digest, value: Digest, rank: Array<D, Value<1>>, position: Array<8 B, Value<1>>,
         depth: Array<16, Value<1>>, path: Array<D, Digest>) -> Value<1>
    FrameV3LeafAt_v1{R, B, SCHEMA}  (domain, value, rank: Array<8 R, Value<1>>, position: Array<8 B, Value<1>>) -> Digest
    FrameV3Pick_v1{D}               (depth: Array<16, Value<1>>, levels: Array<D + 1, Digest>, root) -> Value<1>

The leaf binds uint(rank) and uint(position), each laid out at every length (R = index_bytes(D) by B layouts, all of one
block count); the climb is the read's on the rank's bits, and the result is 1 iff depth <= D and the digest after the first
depth levels is ``root`` (a depth-0 tree's root is its one leaf; the levels past depth hash whatever ``path`` holds there, and
nothing reads them). `opens_at` is its reference.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Callable, Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.boolean.forms import OR, ZERO, and_reduce, const_bits, or_reduce
from verity.ml.circuits.boolean.trace import part as _part
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.refs import Coll
from verity.ml.circuits.types import Array

from ..merkle import _FRAME, _LEAF, _NODE, _hash, _uint
from .lms import LmsEqual
from .sha512 import IV, compress_gadget, pad_tail, words_be

__all__ = ["FrameV3Leaf", "FrameV3LeafAt", "FrameV3Node", "FrameV3Pick", "FrameV3Read", "FrameV3ReadAt", "MAX_D", "arguments",
           "arguments_at", "index_bytes", "leaf_at_ref", "leaf_ref", "node_ref", "opens", "opens_at", "read", "read_at",
           "root_at_ref", "root_ref"]

BIT = BL.BIT
WORD = Array(64, BIT)
DIGEST = Array(8, WORD)
LEVEL = Array(8, BIT)
#: the deepest tree a read takes: a position is a u64
MAX_D = 64


# --- reference ---------------------------------------------------------------------------------------------------------------

def leaf_ref(domain_id: bytes, schema: str, value: bytes, x: int) -> bytes:
    """The frame-v3-sha512 leaf of ``value`` at rank = position = ``x`` (`CommitmentDomain.leaf` over a domain id)."""
    return _hash(_LEAF, domain_id, _uint(x), _uint(x), schema.encode("utf-8"), value, hash="sha512")


def node_ref(domain_id: bytes, level: int, index: int, left: bytes, right: bytes) -> bytes:
    """The frame-v3-sha512 node at ``level`` and ``index`` (`CommitmentDomain.node` over a domain id)."""
    return _hash(_NODE, domain_id, _uint(level), _uint(index), left, right, hash="sha512")


def root_ref(domain_id: bytes, schema: str, value: bytes, pos: int, path: Sequence[bytes]) -> bytes:
    """The root the path gives ``value`` at ``pos`` (`FrameV3.verify_path`'s climb)."""
    cur = leaf_ref(domain_id, schema, value, pos)
    for k, sib in enumerate(path):
        cur = node_ref(domain_id, k, pos >> (k + 1), *((sib, cur) if pos >> k & 1 else (cur, sib)))
    return cur


def opens(schema: str, root: bytes, domain_id: bytes, value: bytes, pos: int, path: Sequence[bytes]) -> int:
    """`FrameV3Read`'s reference: 1 iff the path opens ``value`` at ``pos`` to ``root``."""
    return int(root_ref(domain_id, schema, value, pos, path) == root)


def leaf_at_ref(domain_id: bytes, schema: str, value: bytes, rank: int, position: int) -> bytes:
    """The frame-v3-sha512 leaf of ``value`` at ``rank`` and ``position`` (`CommitmentDomain.leaf` over a domain id)."""
    return _hash(_LEAF, domain_id, _uint(rank), _uint(position), schema.encode("utf-8"), value, hash="sha512")


def root_at_ref(domain_id: bytes, schema: str, value: bytes, rank: int, position: int, path: Sequence[bytes]) -> bytes:
    """The root the path gives ``value`` at ``rank`` and ``position`` (`merkle.verify_opening`'s climb)."""
    cur = leaf_at_ref(domain_id, schema, value, rank, position)
    for k, sib in enumerate(path):
        cur = node_ref(domain_id, k, rank >> (k + 1), *((sib, cur) if rank >> k & 1 else (cur, sib)))
    return cur


def opens_at(schema: str, root: bytes, domain_id: bytes, value: bytes, rank: int, position: int, depth: int,
             path: Sequence[bytes]) -> int:
    """`FrameV3ReadAt`'s reference: 1 iff depth <= len(path) and the path's first ``depth`` siblings open ``value`` at
    ``rank`` and ``position`` to ``root``."""
    return int(depth <= len(path) and root_at_ref(domain_id, schema, value, rank, position, path[:depth]) == root)


def _digest_bits(d: bytes) -> list[int]:
    if len(d) != 64:
        raise ValueError(f"a digest is 64 bytes, not {len(d)}")
    return [(w >> j) & 1 for w in words_be(d) for j in range(64)]


def arguments(root: bytes, domain_id: bytes, value: bytes, pos: int, path: Sequence[bytes]) -> tuple[list[int], ...]:
    """``FrameV3Read_v1{D = len(path), SCHEMA}``'s five arguments, each a flat list of bits in leaf order."""
    D = len(path)
    _check_depth(D)
    if not 0 <= pos < 2 ** D:
        raise ValueError(f"a position in a depth-{D} tree is below 2^{D}")
    return (_digest_bits(root), _digest_bits(domain_id), _digest_bits(value), [(pos >> j) & 1 for j in range(D)],
            [b for s in path for b in _digest_bits(s)])


def arguments_at(B: int, root: bytes, domain_id: bytes, value: bytes, rank: int, position: int, depth: int,
                 path: Sequence[bytes]) -> tuple[list[int], ...]:
    """``FrameV3ReadAt_v1{D = len(path), B, SCHEMA}``'s seven arguments, each a flat list of bits in leaf order."""
    D = len(path)
    _check_depth(D)
    _check_bytes(B)
    if not 0 <= rank < 2 ** D or not 0 <= position < 2 ** (8 * B) or not 0 <= depth < 1 << 16:
        raise ValueError(f"a rank below 2^{D}, a position below 2^{8 * B} and a u16 depth")
    return (_digest_bits(root), _digest_bits(domain_id), _digest_bits(value), [(rank >> j) & 1 for j in range(D)],
            [(position >> j) & 1 for j in range(8 * B)], [(depth >> j) & 1 for j in range(16)],
            [b for s in path for b in _digest_bits(s)])


# --- bytes and words ---------------------------------------------------------------------------------------------------------
# A byte is 8 bits LSB first; a word is 64 bits LSB first, its bytes big-endian (`gates.lms`' layout).

def _bytes(words: Sequence[Sequence]) -> list[list]:
    return [list(w[8 * (7 - b):8 * (8 - b)]) for w in words for b in range(8)]


def _words(data: Sequence[Sequence]) -> list[list]:
    return [[x for b in range(8) for x in data[8 * k + 7 - b]] for k in range(len(data) // 8)]


def _const_bytes(data: bytes) -> list[list]:
    return [const_bits(b, 8) for b in data]


def _uint_bytes(bits: Sequence, n: int) -> list[list]:
    """The low 8 n bits of ``bits`` (LSB first) as n bytes big-endian."""
    padded = list(bits[:8 * n]) + [ZERO] * (8 * n - min(len(bits), 8 * n))
    return [padded[8 * (n - 1 - b):8 * (n - b)] for b in range(n)]


def _message(tag: bytes, parts: Sequence[Sequence]) -> list[list]:
    """The framed message (`merkle._message`) over parts of byte forms, padded, as SHA-512 words."""
    data = _const_bytes(_FRAME + len(tag).to_bytes(4, "big") + tag)
    for p in parts:
        data += _const_bytes(len(p).to_bytes(8, "big")) + list(p)
    return _words(data + _const_bytes(pad_tail(len(data))))


def _sha512(C, words: list[list]) -> list[list]:
    st = [const_bits(v, 64) for v in IV]
    for i in range(0, len(words), 16):
        st = compress_gadget(C, st, words[i:i + 16])
    return st


def _flags(C, bits: Sequence, n: int) -> list:
    """z_j for j = 1 ... n - 1: 1 iff ``bits`` (8 n of them) has a bit at or above 8 j set."""
    out, acc = {}, None
    for j in range(8 * n - 1, 7, -1):
        acc = bits[j] if acc is None else OR(C, acc, bits[j])
        if j % 8 == 0:
            out[j // 8] = acc
    return [out[j] for j in range(1, n)]


def _select(C, flags: Sequence, layouts: Sequence[list[list]]) -> list[list]:
    """Bit by bit, ``layouts[m]`` where the first m flags are set and the next is not (the flags nest)."""
    out = []
    for word in zip(*layouts):
        bits = []
        for col in zip(*word):
            m = col[0]
            for z, a, b in zip(flags, col, col[1:]):
                d = a ^ b
                if d.is_const():
                    m = m ^ z if d.c else m
                else:
                    m = m ^ C.AND(z, d)
            bits.append(m)
        out.append(bits)
    return out


def _hash_uint(C, tag: bytes, head: Sequence[Sequence], x: Sequence, n: int, tail: Callable[[list], Sequence[Sequence]]) -> list:
    """SHA-512 of the message whose parts are ``head``, then ``tail(uint(x))``, over every length of uint(x) up to n bytes."""
    layouts = [_message(tag, [*head, *tail(_uint_bytes(x, L))]) for L in range(1, n + 1)]
    return _sha512(C, _select(C, _flags(C, x, n), layouts))


# --- the parts over forms ----------------------------------------------------------------------------------------------------

def leaf_hash(C, schema: str, domain, value, x, n: int) -> list:
    """leaf(x) of ``value`` (words of forms; x 8 n bits LSB first)."""
    tail = lambda u: [u, u, _const_bytes(schema.encode("utf-8")), _bytes(value)]  # noqa: E731
    return _hash_uint(C, _LEAF, [_bytes(domain)], x, n, tail)


def leaf_at_hash(C, schema: str, domain, value, rank, position, nr: int, npos: int) -> list:
    """leaf(rank, position) of ``value`` (words of forms; rank 8 nr bits and position 8 npos bits, LSB first): the position's
    layouts selected under each length of uint(rank), then those under the rank's flags."""
    s = _const_bytes(schema.encode("utf-8"))
    flags = _flags(C, position, npos)
    rows = [_select(C, flags, [_message(_LEAF, [_bytes(domain), _uint_bytes(rank, lr), _uint_bytes(position, lp), s, _bytes(value)])
                               for lp in range(1, npos + 1)]) for lr in range(1, nr + 1)]
    return _sha512(C, _select(C, _flags(C, rank, nr), rows))


def _eqc(C, bits: Sequence, k: int):
    return and_reduce(C, [x if (k >> i) & 1 else x.inv() for i, x in enumerate(bits)]) if k < 1 << len(bits) else ZERO


def pick_root(C, depth, levels: Sequence[Sequence], root) -> list:
    """1 iff ``levels[depth]`` (each 8 words of forms) is ``root``, depth (16 bits, LSB first) at most len(levels) - 1."""
    flat = [x for w in root for x in w]
    hits = [C.AND(_eqc(C, depth, k), and_reduce(C, [(x ^ y).inv() for x, y in zip([x for w in lv for x in w], flat, strict=True)]))
            for k, lv in enumerate(levels)]
    return [or_reduce(C, hits)]


def node_hash(C, domain, level, index, d, cur, sib, n: int) -> list:
    """node(level, index, l, r), (l, r) = (cur, sib) swapped if ``d`` (level 8 bits, index 8 n bits, LSB first)."""
    t = [[C.AND(d, x ^ y) for x, y in zip(cw, sw)] for cw, sw in zip(cur, sib)]
    left = [[x ^ u for x, u in zip(cw, tw)] for cw, tw in zip(cur, t)]
    right = [[y ^ u for y, u in zip(sw, tw)] for sw, tw in zip(sib, t)]
    return _hash_uint(C, _NODE, [_bytes(domain), [list(level)]], index, n, lambda u: [u, _bytes(left), _bytes(right)])


# --- traced Definitions ------------------------------------------------------------------------------------------------------

def _leaf_blocks(schema: str, n: int, m: int | None = None) -> set[int]:
    """The block counts of the leaf messages under ``schema`` as uint(rank) takes 1 ... n bytes and uint(position) 1 ... m
    (the same length as the rank's, when m is None)."""
    size = 209 + len(schema.encode("utf-8"))
    if m is None:
        return {-(-(size + 2 * L + 17) // 128) for L in range(1, n + 1)}
    return {-(-(size + a + b + 17) // 128) for a in range(1, n + 1) for b in range(1, m + 1)}


def _check_bytes(B: int) -> None:
    if isinstance(B, bool) or not isinstance(B, int) or not 1 <= B <= MAX_D // 8:
        raise ValueError(f"a frame-v3 field's B is an int in 1 ... {MAX_D // 8} (bytes of uint), not {B!r}")


def _check_schema(schema: str, B: int) -> None:
    if not isinstance(schema, str) or not schema:
        raise ValueError(f"a frame-v3 leaf's SCHEMA is a nonempty str, not {schema!r}")
    if len(_leaf_blocks(schema, B)) != 1:
        raise ValueError(f"leaf messages under {schema!r} take {sorted(_leaf_blocks(schema, B))} blocks as uint(x) takes 1 ... {B} "
                         "bytes: one block count per binding")


def _check_depth(D: int) -> None:
    if isinstance(D, bool) or not isinstance(D, int) or not 1 <= D <= MAX_D:
        raise ValueError(f"a frame-v3 read's D is an int in 1 ... {MAX_D}, not {D!r}")


def index_bytes(bits: int) -> int:
    """B of a field of ``bits`` bits: its uint takes 1 ... B bytes."""
    return max(1, -(-bits // 8))


def _sig_leaf(S):
    _check_bytes(S.B)
    _check_schema(S.SCHEMA, S.B)
    return (("domain", DIGEST), ("value", DIGEST), ("x", Array(8 * S.B, BIT))), DIGEST


FrameV3Leaf = _part(
    "FrameV3Leaf", ("B", "SCHEMA"), _sig_leaf, lambda C, S, *e: leaf_hash(C, S.SCHEMA, e[0:8], e[8:16], e[16], S.B),
    "the frame-v3-sha512 leaf of value at rank = position = x under the domain id domain and SCHEMA; x LSB first, its "
    "uint 1 ... B bytes")


def _sig_node(S):
    _check_bytes(S.B)
    return (("domain", DIGEST), ("k", LEVEL), ("i", Array(8 * S.B, BIT)), ("d", BIT), ("cur", DIGEST), ("sib", DIGEST)), DIGEST


FrameV3Node = _part(
    "FrameV3Node", ("B",), _sig_node,
    lambda C, S, *e: node_hash(C, e[0:8], e[8], e[9], e[10][0], e[11:19], e[19:27], S.B),
    "the frame-v3-sha512 node at level k and index i (LSB first, its uint 1 ... B bytes) under the domain id domain over cur "
    "and sib, cur the right child iff d")


def _sig_leaf_at(S):
    _check_bytes(S.R)
    _check_bytes(S.B)
    if not isinstance(S.SCHEMA, str) or not S.SCHEMA:
        raise ValueError(f"a frame-v3 leaf's SCHEMA is a nonempty str, not {S.SCHEMA!r}")
    if len(_leaf_blocks(S.SCHEMA, S.R, S.B)) != 1:
        raise ValueError(f"leaf messages under {S.SCHEMA!r} take {sorted(_leaf_blocks(S.SCHEMA, S.R, S.B))} blocks as uint(rank) "
                         f"takes 1 ... {S.R} bytes and uint(position) 1 ... {S.B}: one block count per binding")
    return (("domain", DIGEST), ("value", DIGEST), ("rank", Array(8 * S.R, BIT)), ("position", Array(8 * S.B, BIT))), DIGEST


FrameV3LeafAt = _part(
    "FrameV3LeafAt", ("R", "B", "SCHEMA"), _sig_leaf_at,
    lambda C, S, *e: leaf_at_hash(C, S.SCHEMA, e[0:8], e[8:16], e[16], e[17], S.R, S.B),
    "the frame-v3-sha512 leaf of value at rank and position under the domain id domain and SCHEMA; each LSB first, uint(rank) "
    "1 ... R bytes and uint(position) 1 ... B")


def _sig_pick(S):
    _check_depth(S.D)
    return (("depth", Array(16, BIT)), ("levels", Array(S.D + 1, DIGEST)), ("root", DIGEST)), BIT


FrameV3Pick = _part(
    "FrameV3Pick", ("D",), _sig_pick,
    lambda C, S, depth, *e: pick_root(C, depth, [e[8 * k:8 * k + 8] for k in range(S.D + 1)], e[8 * (S.D + 1):]),
    "1 iff depth <= D and levels[depth] is root: a read's climb compared at the tree's depth")


# --- the read ----------------------------------------------------------------------------------------------------------------

def _signature(S):
    _check_depth(S.D)
    _check_schema(S.SCHEMA, index_bytes(S.D))
    return ((("root", DIGEST), ("domain", DIGEST), ("value", DIGEST), ("pos", Array(S.D, BIT)), ("path", Array(S.D, DIGEST))),
            BIT)


def _refs(c: Coll) -> list:
    return [c.refs[k] for k in range(c.refs.count)]


def _body(B, S, root, domain, value, pos, path):
    bits_of: dict[int, object] = {}

    def const(v: int, w: int) -> list:
        for b in sorted({v >> j & 1 for j in range(w)} - bits_of.keys()):
            bits_of[b] = B.call(BL.constant(b)).refs[0]
        return [bits_of[v >> j & 1] for j in range(w)]

    bits = _refs(pos)
    n = index_bytes(S.D)
    cur = B.call(bind(FrameV3Leaf, B=n, SCHEMA=S.SCHEMA), domain, value, Coll(Array(8 * n, BIT), TR._refs(bits + const(0, 8 * n - S.D))))
    for k in range(S.D):
        hi = bits[k + 1:]
        n = index_bytes(len(hi))
        i = Coll(Array(8 * n, BIT), TR._refs(hi + const(0, 8 * n - len(hi))))
        cur = B.call(bind(FrameV3Node, B=n), domain, Coll(LEVEL, TR._refs(const(k, 8))), i, pos[k], cur, path[k])
    return B.call(bind(LmsEqual, N=64), cur, root)


FrameV3Read = CompositeDefinition(
    "FrameV3Read", 1, ("D", "SCHEMA"), _signature, _body, conformance="exact-model-tested",
    doc="1 iff path opens value at position pos to root in the depth-D dense frame-v3-sha512 tree of domain id domain: the leaf "
        "binds the domain id, pos as rank and position, SCHEMA and value; level k swaps by pos's bit k and hashes the node with "
        "the domain id, k and the index pos >> (k + 1)")


def read(D: int, schema: str):
    return bind(FrameV3Read, D=D, SCHEMA=schema)


def _signature_at(S):
    _check_depth(S.D)
    _sig_leaf_at(SimpleNamespace(R=index_bytes(S.D), B=S.B, SCHEMA=S.SCHEMA))
    return ((("root", DIGEST), ("domain", DIGEST), ("value", DIGEST), ("rank", Array(S.D, BIT)), ("position", Array(8 * S.B, BIT)),
             ("depth", Array(16, BIT)), ("path", Array(S.D, DIGEST))), BIT)


def _body_at(B, S, root, domain, value, rank, position, depth, path):
    zero = B.call(BL.constant(0)).refs[0]
    one = None
    bits = _refs(rank)
    n = index_bytes(S.D)
    cur = B.call(bind(FrameV3LeafAt, R=n, B=S.B, SCHEMA=S.SCHEMA), domain, value,
                 Coll(Array(8 * n, BIT), TR._refs(bits + [zero] * (8 * n - S.D))), position)
    levels = _refs(cur)
    for k in range(S.D):
        if one is None and k:
            one = B.call(BL.constant(1)).refs[0]
        hi = bits[k + 1:]
        n = index_bytes(len(hi))
        i = Coll(Array(8 * n, BIT), TR._refs(hi + [zero] * (8 * n - len(hi))))
        level = Coll(LEVEL, TR._refs([one if k >> j & 1 else zero for j in range(8)]))
        cur = B.call(bind(FrameV3Node, B=n), domain, level, i, rank[k], cur, path[k])
        levels += _refs(cur)
    return B.call(bind(FrameV3Pick, D=S.D), depth, Coll(Array(S.D + 1, DIGEST), TR._refs(levels)), root)


FrameV3ReadAt = CompositeDefinition(
    "FrameV3ReadAt", 1, ("D", "B", "SCHEMA"), _signature_at, _body_at, conformance="exact-model-tested",
    doc="1 iff depth <= D and the first depth siblings of path open value at rank and position to root in a frame-v3-sha512 "
        "tree of domain id domain: the leaf binds the domain id, rank, position, SCHEMA and value; level k swaps by rank's bit k "
        "and hashes the node with the domain id, k and the index rank >> (k + 1)")


def read_at(D: int, B: int, schema: str):
    return bind(FrameV3ReadAt, D=D, B=B, SCHEMA=schema)
