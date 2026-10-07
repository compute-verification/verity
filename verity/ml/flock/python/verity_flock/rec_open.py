"""V*'s opening check (hiding model architectures, lane rec-v0): one inner Ligerito opening climbs to the cap node its position
names, as a Boolean Definition, the M0 unit that proves it, and their plain-integer reference.

    RecOpen_v1{LANES, H}(row: Array<8 LANES, Array<16, Value<1>>>, aux: Array<A(H), Array<16, Value<1>>>)
        -> Array<64, Array<16, Value<1>>>

Every port is u16 words (a Boolean element of 16 bits, `verity.ml.circuits.boolean`'s word view): a word's bits LSB first, a byte
string's bytes two to a word, little-endian, so leaf ``t`` of a port is bit ``t & 7`` of its byte string's byte ``t >> 3``.

* ``row``: the opened row's LANES elements of GF(2^128), each 16 bytes ``lo`` LE then ``hi`` LE (8 words): the row's bytes as
  the inner prover hashes them (`verity_flock.rec_vstar.row_bytes`).
* ``aux``: the opening's salt (192 bytes, words 0 .. 95), its H siblings bottom up (64 bytes each, 32 words from word 96), and
  the position's low H bits (word ``96 + 32 H``, bit ``h`` the side at level ``h``); zero words up to ``A(H)``, a whole number
  of 64-word M0 rows.
* The result: the node H levels up (32 words), the position's low H bits (one word, its other bits zero), and 31 zero words:
  one M0 row. The verifier registers what it must be (`expected`: the cap node at ``pos >> H`` and ``pos mod 2^H``), so one
  registered output row checks both the climb and the position (PROTOCOL.md §16.13 "Registered reads").

The leaf is ``hm96-sha512/v1`` (PROTOCOL.md §16.10 "Proof"): ``x = SHA-512(row)``, ``b || c`` its hiding commit string under
the salt (``hm96.SHA512``, ``default_key``: ``b = x XOR M y`` is linear, ``c = SHA-512(salt_prefix || y)`` two compressions
from the prefix's midstate), the leaf ``SHA-512(leaf_prefix || b || c)`` two compressions from the prefix's midstate. A node
is ``SHA-512(left || right)`` (§14), the current digest on the side its bit names (``t = d AND (cur XOR sib)`` swaps them).
Compressions: ``LANES / 8 + 1`` for ``x``, 2, 2, and 2 a level.

`_steps` states the computation once over two interpreters: `forms` for the M0 unit (`unit_circuit`: SHA-512's committed copies
keep its rows short, constant blocks fold) and the IR for the Definition (`RecOpen`'s body: each compression a Call of
`RecSha512Compress_v1`, traced from `sha512.compress_gadget`, so the descriptor holds one compression's gates). `open_ref`
(over ``hashlib`` and ``hm96``) is the reference of both.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from functools import cache, reduce
from types import SimpleNamespace

from verity.ml.commitments import hm96
from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.refs import Coll
from verity.ml.circuits.types import Array
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.boolean.forms import ONE, ZERO
from verity.ml.commitments.gates.sha512 import IV, commit, compress_gadget, midstate, pad_tail

from .sha512_circuit import word_bit

__all__ = ["OUT_WORDS", "RecOpen", "RecSha512Compress", "aux_words", "definition", "expected", "open_ref", "pack", "row_words",
           "unit_circuit"]

BIT = BL.BIT
U16 = Array(16, BIT)
WORD = Array(64, BIT)
DIGEST = Array(8, WORD)
BLOCK = Array(16, WORD)
INST = hm96.SHA512
KEY = INST.default_key
SALT_WORDS = INST.salt_bytes // 2
NODE_WORDS = 32
OUT_WORDS = 64
ROW_BLOCK = 64


def _check(LANES: int, H: int) -> None:
    for n, v in (("LANES", LANES), ("H", H)):
        if isinstance(v, bool) or not isinstance(v, int):
            raise TypeError(f"RecOpen's {n} is an int, not {v!r}")
    if LANES < 8 or LANES % 8:
        raise ValueError(f"RecOpen's LANES is a positive multiple of 8 (whole SHA-512 blocks of row), not {LANES}")
    if not 1 <= H <= 16:
        raise ValueError(f"RecOpen's H is 1 to 16 (the position's bits are one word), not {H}")


def row_words(LANES: int) -> int:
    return 8 * LANES


def dirs_at(H: int) -> int:
    return SALT_WORDS + NODE_WORDS * H


def aux_words(H: int) -> int:
    return -(-(dirs_at(H) + 1) // ROW_BLOCK) * ROW_BLOCK


# --- reference ---------------------------------------------------------------------------------------------------------------

def words_of(b: bytes) -> list[int]:
    return [b[i] | (b[i + 1] << 8) for i in range(0, len(b), 2)]


def bytes_of(words: Sequence[int]) -> bytes:
    return b"".join(int(w).to_bytes(2, "little") for w in words)


def pack(row: Sequence[int], salt: bytes, sibs: Sequence[bytes], low: int) -> tuple[list[int], list[int]]:
    """``(row, aux)`` words of one opening: the row's GF(2^128) elements, its salt, its siblings bottom up and the position's
    low ``len(sibs)`` bits."""
    H = len(sibs)
    rw = [(v >> (16 * k)) & 0xFFFF for v in row for k in range(8)]
    aux = words_of(salt) + [w for s in sibs for w in words_of(s)] + [low & ((1 << H) - 1)]
    return rw, aux + [0] * (aux_words(H) - len(aux))


def expected(node: bytes, low: int, H: int) -> list[int]:
    """The result an opening must give: the cap node at its position and the position's low H bits."""
    return words_of(node) + [low & ((1 << H) - 1)] + [0] * (OUT_WORDS - NODE_WORDS - 1)


def open_ref(LANES: int, H: int, row: Sequence[int], aux: Sequence[int]) -> list[int]:
    """`RecOpen`'s reference on words."""
    _check(LANES, H)
    if len(row) != row_words(LANES) or len(aux) != aux_words(H):
        raise ValueError(f"RecOpen{{{LANES}, {H}}} takes {row_words(LANES)} row words and {aux_words(H)} aux words")
    salt = bytes_of(aux[:SALT_WORDS])
    low = aux[dirs_at(H)] & ((1 << H) - 1)
    cur = INST.tree_leaf(KEY, INST.commit_string(KEY, hashlib.sha512(bytes_of(row)).digest(), salt))
    for h in range(H):
        sib = bytes_of(aux[SALT_WORDS + NODE_WORDS * h:SALT_WORDS + NODE_WORDS * (h + 1)])
        cur = hashlib.sha512(sib + cur if (low >> h) & 1 else cur + sib).digest()
    return expected(cur, low, H)


# --- the computation, over an interpreter ------------------------------------------------------------------------------------

@cache
def _consts() -> SimpleNamespace:
    return SimpleNamespace(rows=INST.rows_of(KEY), salt_mid=midstate(INST.salt_prefix), leaf_mid=midstate(INST.leaf_prefix(KEY)))


def _bits(data: bytes) -> list[int]:
    return [(data[t >> 3] >> (t & 7)) & 1 for t in range(8 * len(data))]


def _sha(ops, cv: Sequence[int], bits: list, prefix_len: int) -> list:
    """SHA-512 of a message whose first ``prefix_len`` bytes left the constant chaining value ``cv`` and whose rest is ``bits``
    (byte-string order): the digest in byte-string order."""
    total = prefix_len + len(bits) // 8
    msg = bits + [ops.const(b) for b in _bits(pad_tail(total))]
    st = [ops.const((v >> j) & 1) for v in cv for j in range(64)]
    for at in range(0, len(msg), 1024):
        blk = [None] * 1024
        for t in range(1024):
            blk[word_bit(t)] = msg[at + t]
        st = ops.compress(st, blk)
    return [st[word_bit(t)] for t in range(512)]


def _steps(ops, LANES: int, H: int, row: list, aux: list) -> list:
    """The result's bits (byte-string order) from ``row``'s and ``aux``'s (`RecOpen`'s ports, flattened)."""
    k = _consts()
    y = aux[:8 * INST.salt_bytes]
    x = _sha(ops, IV, row, 0)
    b = ops.commit([ops.xor([x[i]] + [y[j] for j in range(len(y)) if (k.rows[i] >> j) & 1]) for i in range(512)])
    c = _sha(ops, k.salt_mid, y, len(INST.salt_prefix))
    cur = _sha(ops, k.leaf_mid, b + c, 128)
    d = aux[16 * dirs_at(H):16 * dirs_at(H) + 16]
    for h in range(H):
        sib = aux[16 * (SALT_WORDS + NODE_WORDS * h):16 * (SALT_WORDS + NODE_WORDS * (h + 1))]
        left, right = ops.swap(d[h], cur, sib)
        cur = _sha(ops, IV, left + right, 0)
    zero = ops.const(0)
    return cur + d[:H] + [zero] * (16 - H) + [zero] * (16 * (OUT_WORDS - NODE_WORDS - 1))


class _Forms:
    def __init__(self, C):
        self.C = C

    def const(self, b: int):
        return ONE if b else ZERO

    def compress(self, cv: list, msg: list) -> list:
        out = compress_gadget(self.C, [cv[64 * i:64 * i + 64] for i in range(8)], [msg[64 * i:64 * i + 64] for i in range(16)])
        return [x for w in out for x in w]

    def xor(self, xs: list):
        return reduce(lambda a, b: a ^ b, xs)

    def commit(self, bits: list) -> list:
        return commit(self.C, bits)

    def swap(self, d, a: list, s: list) -> tuple[list, list]:
        t = [self.C.AND(d, p ^ q) for p, q in zip(a, s)]
        return [p ^ u for p, u in zip(a, t)], [q ^ u for q, u in zip(s, t)]


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

    def xor(self, xs: list):
        if len(xs) == 1:
            return xs[0]
        return self.B.call(BL.xor_gate(len(xs)), Coll(Array(len(xs), BIT), TR._refs(xs))).refs[0]

    def commit(self, bits: list) -> list:
        return bits

    def _pairs(self, fn, a: list, b: list) -> list:
        return self._list(self.B.batch(fn, Coll(Array(2, Array(len(a), BIT)), TR._refs(a + b)), axes=(1,)))

    def swap(self, d, a: list, s: list) -> tuple[list, list]:
        t = self._pairs(BL.AND, [d] * len(a), self._pairs(BL.XOR, a, s))
        return self._pairs(BL.XOR, a, t), self._pairs(BL.XOR, s, t)


# --- the Definitions ---------------------------------------------------------------------------------------------------------

RecSha512Compress = TR.definition(
    "RecSha512Compress", 1, lambda C, *e: compress_gadget(C, list(e[:8]), list(e[8:])), (("cv", DIGEST), ("msg", BLOCK)), DIGEST,
    doc="One SHA-512 compression (FIPS 180-4 §6.4.2) with the feed-forward, words LSB first: sha512.compress_gadget")


def _signature(S):
    _check(S.LANES, S.H)
    return (("row", Array(row_words(S.LANES), U16)), ("aux", Array(aux_words(S.H), U16))), Array(OUT_WORDS, U16)


def _body(B, S, row, aux):
    ops = _Ir(B)
    flat = _Ir._list
    return Coll(_signature(S)[1], TR._refs(_steps(ops, S.LANES, S.H, flat(row), flat(aux))))


RecOpen = CompositeDefinition(
    "RecOpen", 1, ("LANES", "H"), _signature, _body, conformance="exact-model-tested",
    doc="One inner Ligerito opening: the hm96-sha512/v1 leaf of the row under the salt climbs the H siblings by the position's "
        "low bits; the result is the node it reaches, then those bits")


# --- the M0 unit -------------------------------------------------------------------------------------------------------------

def unit_circuit(LANES: int, H: int):
    """The unit over `forms`: inputs ``in0 ..`` (``row``'s words, then ``aux``'s, 16 bits each), outputs ``out0 .. out63``."""
    from verity.ml.circuits.boolean.forms import Circuit

    _check(LANES, H)
    C = Circuit()
    ins = [C.input(f"in{j}", 16) for j in range(row_words(LANES) + aux_words(H))]
    flat = [b for w in ins for b in w]
    out = _steps(_Forms(C), LANES, H, flat[:16 * row_words(LANES)], flat[16 * row_words(LANES):])
    outs = [f"out{j}" for j in range(OUT_WORDS)]
    for j, nm in enumerate(outs):
        C.output(nm, out[16 * j:16 * j + 16])
    return C, [f"in{j}" for j in range(len(ins))], outs


def definition(LANES: int, H: int):
    return bind(RecOpen, LANES=LANES, H=H)
