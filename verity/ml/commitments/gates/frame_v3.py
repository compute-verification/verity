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
"""
from __future__ import annotations

from functools import lru_cache
from types import SimpleNamespace
from typing import Callable, Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.boolean.forms import OR, ZERO, const_bits
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.refs import Coll
from verity.ml.circuits.types import Array

from ..merkle import _FRAME, _LEAF, _NODE, _hash, _uint
from .lms import LmsEqual
from .sha512 import IV, compress_gadget, pad_tail, words_be

__all__ = ["FrameV3Leaf", "FrameV3Node", "FrameV3Read", "MAX_D", "arguments", "index_bytes", "leaf_ref", "node_ref", "opens",
           "read", "root_ref"]

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


def node_hash(C, domain, level, index, d, cur, sib, n: int) -> list:
    """node(level, index, l, r), (l, r) = (cur, sib) swapped if ``d`` (level 8 bits, index 8 n bits, LSB first)."""
    t = [[C.AND(d, x ^ y) for x, y in zip(cw, sw)] for cw, sw in zip(cur, sib)]
    left = [[x ^ u for x, u in zip(cw, tw)] for cw, tw in zip(cur, t)]
    right = [[y ^ u for y, u in zip(sw, tw)] for sw, tw in zip(sib, t)]
    return _hash_uint(C, _NODE, [_bytes(domain), [list(level)]], index, n, lambda u: [u, _bytes(left), _bytes(right)])


# --- traced Definitions ------------------------------------------------------------------------------------------------------

def _part(name: str, statics: tuple[str, ...], signature: Callable, build: Callable, doc: str) -> CompositeDefinition:
    """A Definition traced from ``build(C, S, *elements)`` once per binding of ``statics``."""

    @lru_cache(maxsize=None)
    def traced(*key) -> TR.Traced:
        S = SimpleNamespace(**dict(zip(statics, key)))
        params, ret = signature(S)
        return TR.trace(lambda C, *e: build(C, S, *e), [t for _, t in params], ret)

    def body(B, S, *colls):
        params, ret = signature(S)
        return TR.emit(traced(*(getattr(S, k) for k in statics)), [t for _, t in params], ret, B, *colls)

    d = CompositeDefinition(name, 1, statics, signature, body, conformance="exact-model-tested", doc=doc)
    d.traced = traced  # type: ignore[attr-defined]
    return d


def _leaf_blocks(schema: str, n: int) -> set[int]:
    size = 209 + len(schema.encode("utf-8"))
    return {-(-(size + 2 * L + 17) // 128) for L in range(1, n + 1)}


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
