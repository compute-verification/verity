"""V*'s link check: one row of a recursive session's link table (`verity_flock.rec_links`) holds, as a Boolean Definition, the
M0 unit that proves it, and their plain reference.

    U16 = Array<16, Value<1>>
    RecLink_v1{UB, PB}(trow: Array<64, U16>, xr: Array<32, U16>, yr: Array<96, U16>, raux: Array<A(D), U16>,
                       xw: Array<32, U16>, yw: Array<96, U16>, waux: Array<A(D), U16>) -> (rin: Array<64, U16>, rout: Array<64, U16>)

with ``D = UB + PB``. Every port is u16 words in byte-string order (bit ``t`` of a port is bit ``t & 7`` of its byte string's byte
``t >> 3``), as `verity_flock.rec_open`'s.

* ``trow``: the table row, 128 bytes (`rec_links.encode`): its kind, reader unit, reader port, writer unit and writer row, each a
  u32 little-endian, then zeros; a filler row is all zero. The statement reads it as a registered row at the instance's own
  position in the table, so instance ``i`` checks row ``i`` and the ``grid(L)`` instances check the whole table, in an order
  that names no link.
* ``xr``, ``yr``: the reader row's unsalted digest (``sha512/row/v2``, 64 bytes) and its hm96 salt (192 bytes); ``xw``, ``yw``
  the writer row's.
* ``raux``, ``waux``: the reader's and the writer's opening in the inner rows' trees, D siblings bottom up (64 bytes each, 32
  words from word ``32 h``), then zero words up to ``A(D) = 64 max(1, ⌈D / 2⌉)``.
* ``rin``, ``rout``: the root the reader's opening climbs to in the input rows' tree and the writer's in the output rows' tree
  (32 words, then 32 zero words, one M0 row each), or all zero when the row is malformed or its link does not hold.

**The trees.** The inner statement publishes each unit's rows' commit strings ``b ‖ c`` (hm96-sha512, fresh salts). The
verifier builds two trees of depth D from them itself: input row ``p`` of unit ``u`` is leaf ``u · 2^PB + p`` of the input rows'
tree, output row ``o`` of unit ``u`` leaf ``u · 2^PB + o`` of the output rows' tree; a leaf is the row's hm96 tree leaf
``SHA-512(leaf_prefix(key) ‖ b ‖ c)``, a leaf with no row 64 zero bytes (whose preimage nobody holds), and a node
``SHA-512(left ‖ right)``. It registers both roots, and the statement reads every instance's ``rin`` and ``rout`` against
them. A position comes from the table row, so which leaves an instance opens is private, and only the table's padded
count is public.

**The check.** The unit recomputes each row's hm96 leaf from its digest and salt (`gates.hm96.Hm96Sha512Leaf`) and climbs it by
the position's bits, the port's ``PB`` then the unit's ``UB``. The row is well formed when its kind is 0 or 1, its bytes past
the fields are zero, every field fits its bits (a unit ``UB``, a port or row ``PB``), and a filler's fields are zero. A
link holds when ``xr = xw``: the reader's row and the writer's are commitments to one digest, so to one row of one width
(``sha512/row/v2`` binds the bit length), except for a SHA-512 collision. A filler opens leaf 0 of each tree and claims
nothing of them. Each root is ANDed with ``ok`` (well formed, and a link's ``xr = xw``), so a refused row gives a zero root,
which no registered root is.

Each table row is checked by exactly one instance. Whether the table lists every cut wire, and each input row once, is the
partition's own predicate, not this unit's.

`_steps` states the computation once over two interpreters: `_Forms` for the M0 unit (`unit_circuit`) and `_Ir` for the
Definition (`RecLink`'s body: the leaves a Call of ``Hm96Sha512Leaf_v1``, each compression a Call of
``RecSha512Compress_v1``). `link_ref` (over ``hashlib`` and `hm96`) is the reference of both.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.boolean.forms import ONE, ZERO
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.refs import Coll, tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.commitments import hm96
from verity.ml.commitments.gates.hm96 import DIGEST as HM_DIGEST
from verity.ml.commitments.gates.hm96 import SALT as HM_SALT
from verity.ml.commitments.gates.hm96 import Hm96Sha512Leaf, leaf_gadget
from verity.ml.commitments.gates.sha512 import IV

from . import rec_links as RL
from .rec_open import BLOCK, DIGEST, NODE_WORDS, RecSha512Compress, _sha, bytes_of, words_of
from .sha512_circuit import word_bit

__all__ = ["OUT_WORDS", "PORTS", "RecLink", "Tree", "aux_words", "definition", "depth", "expected", "index", "link_ref",
           "port_words", "unit_circuit"]

BIT = BL.BIT
U16 = Array(16, BIT)
INSTANCE = hm96.SHA512
ROW_WORDS = RL.ROW_BYTES // 2
X_WORDS = INSTANCE.digest_bytes // 2
Y_WORDS = INSTANCE.salt_bytes // 2
OUT_WORDS = 64
ROW_BLOCK = 64
FIELD_BITS = 32
UB_MAX, PB_MAX, D_MAX = 30, 16, 32
EMPTY = bytes(64)
PORTS = ("trow", "xr", "yr", "raux", "xw", "yw", "waux")


def _check(UB: int, PB: int) -> None:
    for n, v in (("UB", UB), ("PB", PB)):
        if isinstance(v, bool) or not isinstance(v, int):
            raise TypeError(f"RecLink's {n} is an int, not {v!r}")
    if not (0 <= UB <= UB_MAX and 0 <= PB <= PB_MAX and UB + PB <= D_MAX):
        raise ValueError(f"RecLink's UB is 0 to {UB_MAX} and PB 0 to {PB_MAX}, D = UB + PB at most {D_MAX}: not UB {UB}, PB {PB}")


def depth(UB: int, PB: int) -> int:
    return UB + PB


def aux_words(D: int) -> int:
    return max(1, -(-NODE_WORDS * D // ROW_BLOCK)) * ROW_BLOCK


def port_words(UB: int, PB: int) -> tuple[int, ...]:
    a = aux_words(depth(UB, PB))
    return ROW_WORDS, X_WORDS, Y_WORDS, a, X_WORDS, Y_WORDS, a


def index(unit: int, row: int, PB: int) -> int:
    """A row's leaf in its tree: unit ``unit``'s row ``row`` (an input port or an output row)."""
    if not 0 <= row < 1 << PB:
        raise ValueError(f"row {row} does not fit {PB} bits")
    return unit << PB | row


# --- the trees, as the verifier builds them and the prover opens them --------------------------------------------------------

def leaf(bc: bytes) -> bytes:
    """A row's leaf: its hm96 tree leaf under the pinned key."""
    return INSTANCE.tree_leaf(INSTANCE.default_key, bc)


class Tree:
    """A rows tree of depth ``D`` over ``leaves`` (leaf index to 64 bytes; absent leaves `EMPTY`)."""

    def __init__(self, leaves: Mapping[int, bytes], D: int):
        if any(not 0 <= i < 1 << D for i in leaves):
            raise ValueError(f"a leaf index outside a tree of depth {D}")
        self.D = D
        layer = {i: v for i, v in leaves.items()}
        self.layers = [layer]
        empty = EMPTY
        self.empty = [empty]
        for _ in range(D):
            up: dict[int, bytes] = {}
            for i in {j >> 1 for j in layer}:
                up[i] = hashlib.sha512(layer.get(2 * i, empty) + layer.get(2 * i + 1, empty)).digest()
            empty = hashlib.sha512(empty + empty).digest()
            self.empty.append(empty)
            layer = up
            self.layers.append(layer)

    @property
    def root(self) -> bytes:
        return self.layers[self.D].get(0, self.empty[self.D])

    def path(self, pos: int) -> list[bytes]:
        """Leaf ``pos``'s siblings, bottom up."""
        return [self.layers[h].get((pos >> h) ^ 1, self.empty[h]) for h in range(self.D)]


def aux(sibs: Sequence[bytes], D: int) -> list[int]:
    ws = [w for s in sibs for w in words_of(s)]
    return ws + [0] * (aux_words(D) - len(ws))


def expected(root: bytes) -> list[int]:
    """The result an instance must give: a registered root, as a registered row."""
    return words_of(root) + [0] * (OUT_WORDS - NODE_WORDS)


# --- reference ---------------------------------------------------------------------------------------------------------------

def _field(row: bytes, f: int) -> int:
    return int.from_bytes(row[4 * f:4 * f + 4], "little")


def ok_ref(UB: int, PB: int, row: bytes, xr: bytes, xw: bytes) -> bool:
    """Whether the table row is well formed and, a link, holds (``xr == xw``)."""
    kind, reader, port, writer, out = (_field(row, f) for f in range(5))
    if kind not in (0, RL.KIND_LINK) or any(row[RL.FIELD_BYTES:]):
        return False
    if reader >> UB or writer >> UB or port >> PB or out >> PB:
        return False
    if kind == 0:
        return not (reader or port or writer or out)
    return xr == xw


def _climb_ref(start: bytes, sibs: Sequence[bytes], pos: int) -> bytes:
    cur = start
    for h, sib in enumerate(sibs):
        cur = hashlib.sha512(sib + cur if (pos >> h) & 1 else cur + sib).digest()
    return cur


def link_ref(UB: int, PB: int, trow: Sequence[int], xr: Sequence[int], yr: Sequence[int], raux: Sequence[int],
             xw: Sequence[int], yw: Sequence[int], waux: Sequence[int]) -> tuple[list[int], list[int]]:
    """`RecLink`'s reference on words: ``(rin, rout)``."""
    _check(UB, PB)
    D = depth(UB, PB)
    got = tuple(len(p) for p in (trow, xr, yr, raux, xw, yw, waux))
    if got != port_words(UB, PB):
        raise ValueError(f"RecLink{{{UB}, {PB}}} takes {port_words(UB, PB)} words a port, not {got}")
    row = bytes_of(trow)
    pr = _field(row, 2) | _field(row, 1) << PB
    pw = _field(row, 4) | _field(row, 3) << PB
    def sib(a):
        return [bytes_of(a[NODE_WORDS * h:NODE_WORDS * (h + 1)]) for h in range(D)]
    key = INSTANCE.default_key
    lr = leaf(INSTANCE.commit_string(key, bytes_of(xr), bytes_of(yr)))
    lw = leaf(INSTANCE.commit_string(key, bytes_of(xw), bytes_of(yw)))
    ok = ok_ref(UB, PB, row, bytes_of(xr), bytes_of(xw))
    def mask(r):
        return expected(r) if ok else [0] * OUT_WORDS
    return mask(_climb_ref(lr, sib(raux), pr & ((1 << D) - 1))), mask(_climb_ref(lw, sib(waux), pw & ((1 << D) - 1)))


# --- the computation, over an interpreter ------------------------------------------------------------------------------------

def _to_words(bits: list) -> list:
    """Byte-string-order bits as SHA-512 word order (`word_bit`)."""
    out = [None] * len(bits)
    for t, b in enumerate(bits):
        out[word_bit(t)] = b
    return out


def _from_words(ws: list) -> list:
    return [ws[word_bit(t)] for t in range(len(ws))]


def _all(ops, bits: list):
    """1 when every bit is 1 (an AND tree); 1 for no bits."""
    if not bits:
        return ops.const(1)
    while len(bits) > 1:
        half = len(bits) // 2
        bits = ops.and_(bits[:half], bits[half:2 * half]) + bits[2 * half:]
    return bits[0]


def _none(ops, bits: list):
    """1 when every bit is 0."""
    return _all(ops, ops.xor(bits, [ops.const(1)] * len(bits))) if bits else ops.const(1)


def _ok(ops, UB: int, PB: int, row: list, xr: list, xw: list):
    one = ops.const(1)
    def field(f):
        return row[FIELD_BITS * f:FIELD_BITS * (f + 1)]
    kind = field(0)
    rest_zero = _none(ops, kind[1:])
    is_fill = ops.and_([rest_zero], ops.xor([kind[0]], [one]))[0]
    is_link = ops.and_([rest_zero], [kind[0]])[0]
    tail = _none(ops, row[8 * RL.FIELD_BYTES:])
    fits = _none(ops, field(1)[UB:] + field(3)[UB:] + field(2)[PB:] + field(4)[PB:])
    fz = _none(ops, field(1)[:UB] + field(3)[:UB] + field(2)[:PB] + field(4)[:PB])
    eq = _none(ops, ops.xor(xr, xw))
    def not_(a):
        return ops.xor([a], [one])[0]
    def or_(a, b):
        return not_(ops.and_([not_(a)], [not_(b)])[0])
    kinds = or_(is_fill, is_link)
    return _all(ops, [kinds, tail, fits, or_(not_(is_fill), fz), or_(not_(is_link), eq)])


def _climb(ops, start: list, aux: list, dirs: list) -> list:
    cur = start
    for h, d in enumerate(dirs):
        sib = aux[16 * NODE_WORDS * h:16 * NODE_WORDS * (h + 1)]
        left, right = ops.swap(d, cur, sib)
        cur = _sha(ops, IV, left + right, 0)
    return cur


def _leaf(ops, x: list, y: list) -> list:
    """The row's hm96 tree leaf (512 bits, byte-string order) from its digest's and salt's bits (byte-string order)."""
    return _from_words(ops.hm96(_to_words(x), _to_words(y))[1024:1536])


def _steps(ops, UB: int, PB: int, trow: list, xr: list, yr: list, raux: list, xw: list, yw: list, waux: list) -> tuple[list, list]:
    """``rin``'s and ``rout``'s bits (byte-string order) from the ports' (`RecLink`'s, flattened)."""
    def field(f):
        return trow[FIELD_BITS * f:FIELD_BITS * (f + 1)]
    dr = field(2)[:PB] + field(1)[:UB]
    dw = field(4)[:PB] + field(3)[:UB]
    ok = _ok(ops, UB, PB, trow, xr, xw)
    zeros = [ops.const(0)] * (16 * (OUT_WORDS - NODE_WORDS))
    rin = ops.and_([ok] * 512, _climb(ops, _leaf(ops, xr, yr), raux, dr))
    rout = ops.and_([ok] * 512, _climb(ops, _leaf(ops, xw, yw), waux, dw))
    return rin + zeros, rout + zeros


class _Forms:
    def __init__(self, C):
        self.C = C

    def const(self, b: int):
        return ONE if b else ZERO

    def compress(self, cv: list, msg: list) -> list:
        from verity.ml.commitments.gates.sha512 import compress_gadget

        out = compress_gadget(self.C, [cv[64 * i:64 * i + 64] for i in range(8)], [msg[64 * i:64 * i + 64] for i in range(16)])
        return [x for w in out for x in w]

    def swap(self, d, a: list, s: list) -> tuple[list, list]:
        t = [self.C.AND(d, p ^ q) for p, q in zip(a, s)]
        return [p ^ u for p, u in zip(a, t)], [q ^ u for q, u in zip(s, t)]

    def and_(self, a: list, b: list) -> list:
        return [self.C.AND(p, q) for p, q in zip(a, b, strict=True)]

    def xor(self, a: list, b: list) -> list:
        return [p ^ q for p, q in zip(a, b, strict=True)]

    def hm96(self, x: list, y: list) -> list:
        out = leaf_gadget(self.C, [x[64 * i:64 * i + 64] for i in range(8)], [y[64 * i:64 * i + 64] for i in range(24)])
        return [b for w in out for b in w]


class _Ir:
    def __init__(self, B):
        self.B = B
        self._c: dict[int, tuple] = {}

    def const(self, b: int):
        if b not in self._c:
            self._c[b] = self.B.call(BL.constant(b)).refs[0]
        return self._c[b]

    @staticmethod
    def _list(c: Coll) -> list:
        return [c.refs[i] for i in range(len(c.refs))]

    def compress(self, cv: list, msg: list) -> list:
        return self._list(self.B.call(RecSha512Compress, Coll(DIGEST, TR._refs(cv)), Coll(BLOCK, TR._refs(msg))))

    def _pairs(self, fn, a: list, b: list) -> list:
        if len(a) != len(b):
            raise ValueError(f"{len(a)} and {len(b)} bits")
        return self._list(self.B.batch(fn, Coll(Array(2, Array(len(a), BIT)), TR._refs(a + b)), axes=(1,)))

    def swap(self, d, a: list, s: list) -> tuple[list, list]:
        t = self._pairs(BL.AND, [d] * len(a), self._pairs(BL.XOR, a, s))
        return self._pairs(BL.XOR, a, t), self._pairs(BL.XOR, s, t)

    def and_(self, a: list, b: list) -> list:
        return self._pairs(BL.AND, a, b)

    def xor(self, a: list, b: list) -> list:
        return self._pairs(BL.XOR, a, b)

    def hm96(self, x: list, y: list) -> list:
        return self._list(self.B.call(Hm96Sha512Leaf, Coll(HM_DIGEST, TR._refs(x)), Coll(HM_SALT, TR._refs(y))))


# --- the Definition ----------------------------------------------------------------------------------------------------------

def _signature(S):
    _check(S.UB, S.PB)
    params = tuple((nm, Array(n, U16)) for nm, n in zip(PORTS, port_words(S.UB, S.PB)))
    return params, Tuple(Array(OUT_WORDS, U16), Array(OUT_WORDS, U16))


def _body(B, S, trow, xr, yr, raux, xw, yw, waux):
    flat = _Ir._list
    rin, rout = _steps(_Ir(B), S.UB, S.PB, *(flat(p) for p in (trow, xr, yr, raux, xw, yw, waux)))
    return tuple_of(Coll(Array(OUT_WORDS, U16), TR._refs(rin)), Coll(Array(OUT_WORDS, U16), TR._refs(rout)))


RecLink = CompositeDefinition(
    "RecLink", 1, ("UB", "PB"), _signature, _body, conformance="exact-model-tested",
    doc="One row of the link table: the reader's and the writer's hm96 leaves (digest and salt each) climb the input and output "
        "rows' trees by the positions the row names (port or row in the low PB bits, unit in the UB above), and each root is "
        "a result, ANDed with ok: the row's kind 0 or 1, its tail zero, its fields in range and a filler's zero, and a "
        "link's two digests equal")


# --- the M0 unit -------------------------------------------------------------------------------------------------------------

def unit_circuit(UB: int, PB: int):
    """The unit over `forms`: inputs ``in0 ..`` (``trow``'s words, then ``xr``'s, ``yr``'s, ``raux``'s, ``xw``'s, ``yw``'s and
    ``waux``', 16 bits each), outputs ``rin0 .. rin63`` then ``rout0 .. rout63``."""
    from verity.ml.circuits.boolean.forms import Circuit

    _check(UB, PB)
    C = Circuit()
    pw = port_words(UB, PB)
    ins = [C.input(f"in{j}", 16) for j in range(sum(pw))]
    flat = [b for w in ins for b in w]
    cuts = [16 * sum(pw[:i]) for i in range(len(pw) + 1)]
    rin, rout = _steps(_Forms(C), UB, PB, *(flat[a:b] for a, b in zip(cuts, cuts[1:])))
    outs = [f"rin{j}" for j in range(OUT_WORDS)] + [f"rout{j}" for j in range(OUT_WORDS)]
    bits = rin + rout
    for j, nm in enumerate(outs):
        C.output(nm, bits[16 * j:16 * j + 16])
    return C, [f"in{j}" for j in range(len(ins))], outs


def definition(UB: int, PB: int):
    return bind(RecLink, UB=UB, PB=PB)
