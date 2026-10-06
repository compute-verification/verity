"""V*'s opening check (hiding model architectures, lane rec-v0): one inner Ligerito opening climbs to the cap node its position
names and adds its share of the level's consistency sums to a running sum, as a Boolean Definition, the M0 unit that proves
it, and their plain-integer reference.

    U16 = Array<16, Value<1>>
    RecOpen_v2{LANES, H}(row: Array<8 LANES, U16>, aux: Array<A(H), U16>, acc: Array<64, U16>, coef: Array<16 LANES, U16>)
        -> (out: Array<64, U16>, acc_out: Array<64, U16>)

Every port is u16 words (a Boolean element of 16 bits, `verity.primitives.circuits.boolean`'s word view): a word's bits LSB first, a byte
string's bytes two to a word, little-endian, so leaf ``t`` of a port is bit ``t & 7`` of its byte string's byte ``t >> 3``. An
element of F = GF(2^128) is 8 words, its 16 bytes ``lo`` LE then ``hi`` LE, so its bit k is bit ``k % 16`` of word ``k // 16``.

* ``row``: the opened row's LANES elements of F: the row's bytes as the inner prover hashes them
  (`verity_flock.rec_vstar.row_bytes`).
* ``aux``: the opening's salt (192 bytes, words 0 .. 95), its H siblings bottom up (64 bytes each, 32 words from word 96), and
  the position's low H bits (word ``96 + 32 H``, bit ``h`` the side at level ``h``); zero words up to ``A(H)``, a whole number
  of 64-word M0 rows.
* ``acc``: the running sums before this opening, two elements of F (words 0 .. 15), and 48 words carried as they are.
* ``coef``: the verifier's coefficients, ``c0[l]`` for each lane then ``c1[l]`` (8 words each): a public input.
* ``out``: the node H levels up (32 words), the position's low H bits (one word, its other bits zero), and 31 zero words:
  one M0 row. The verifier registers what it must be (`expected`: the cap node at ``pos >> H`` and ``pos mod 2^H``), so one
  registered output row checks both the climb and the position (PROTOCOL.md §16.13 "Registered reads").
* ``acc_out``: ``acc`` with ``Σ_l c_r[l] · row[l]`` added to its element r (`cons_structure`, a ``GfResiduals_v1`` of two
  residuals), its other words as they were.

**The consistency sums.** The inner verifier's final check is linear in each level's sums ``Σ_j eq(α)[j] · row_j[l]`` over the
level's opened rows (`verity_flock.rec_algebra`), with coin-derived coefficients ``κ_{r, l}`` in its two components r. The
verifier gives opening j of the level ``c_r[l] = κ_{r, l} · eq(α)[j]``, so the openings' terms ``Σ_l c_r[l] · row_j[l]`` sum to
the final check's share of that level's sums. The openings of a rep chain their running sums through one registered value
(`verity_flock.rec_outer`): each one's ``acc_out`` is the next one's ``acc``, and V*'s algebra reads the rep's first and last.

The leaf is ``hm96-sha512/v1`` (PROTOCOL.md §16.10 "Proof"): ``x = SHA-512(row)``, ``b || c`` its hiding commit string under
the salt (``hm96.SHA512``, ``default_key``: ``b = x XOR M y`` is linear, ``c = SHA-512(salt_prefix || y)`` two compressions
from the prefix's midstate), the leaf ``SHA-512(leaf_prefix || b || c)`` two compressions from the prefix's midstate. A node
is ``SHA-512(left || right)`` (§14), the current digest on the side its bit names (``t = d AND (cur XOR sib)`` swaps them).
Compressions: ``LANES / 8 + 1`` for ``x``, 2, 2, and 2 a level.

`_steps` states the computation once over two interpreters: `forms` for the M0 unit (`unit_circuit`: SHA-512's committed copies
keep its rows short, constant blocks fold, the products are summed leaf by leaf as `rec_residuals.residual_forms` sums them) and
the IR for the Definition (`RecOpen`'s body: each compression a Call of `RecSha512Compress_v1`, traced from
`sha512.compress_gadget`, so the descriptor holds one compression's gates, and the consistency term a Call of
``GfResiduals_v1{cons_structure(LANES)}``). `open_ref` (over ``hashlib``, ``hm96`` and `gf2k.mul`) is the reference of both.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from functools import cache, reduce
from types import SimpleNamespace

from verity.primitives.commitments import hm96
from verity.primitives.circuits import boolean as BL
from verity.primitives.circuits.defs import CompositeDefinition, bind
from verity.primitives.circuits.refs import Coll, tuple_of
from verity.primitives.circuits.types import Array, Tuple
from verity.primitives.circuits.boolean import trace as TR
from verity.primitives.circuits.boolean.forms import ONE, ZERO
from verity.primitives.commitments.gates.sha512 import IV, commit, compress_gadget, midstate, pad_tail
from verity_catalog.definitions.boolean import gf2k as G

from .rec_residuals import residual_forms
from .sha512_circuit import word_bit

__all__ = ["ACC_WORDS", "OUT_WORDS", "RecOpen", "RecSha512Compress", "acc_ref", "aux_words", "coef_words", "cons_structure",
           "definition", "elements", "expected", "open_ref", "pack", "row_words", "unit_circuit", "words"]

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
ACC_WORDS = 64
ROW_BLOCK = 64
F_WORDS = 8


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


def coef_words(LANES: int) -> int:
    return 2 * F_WORDS * LANES


def cons_structure(LANES: int) -> dict:
    """``GfResiduals_v1``'s S for the consistency term: w is the row's LANES elements then acc's two, v the coefficients
    (``c0[l]``, then ``c1[l]``), and residual r is ``Σ_l c_r[l] · row[l] + acc_r``."""
    ops = [[["w", l]] for l in range(LANES)] + [[["v", i]] for i in range(2 * LANES)]
    res = [{"mul": [[LANES + r * LANES + l, l] for l in range(LANES)], "add": [["w", LANES + r]]} for r in range(2)]
    return {"nw": LANES + 2, "nv": 2 * LANES, "ops": ops, "res": res}


# --- reference ---------------------------------------------------------------------------------------------------------------

def words(elems: Sequence[int], n_words: int | None = None) -> list[int]:
    """F elements as u16 words (8 each, ``lo`` LE then ``hi`` LE), zero words up to ``n_words``."""
    out = [(x >> (16 * k)) & 0xFFFF for x in elems for k in range(F_WORDS)]
    return out + [0] * ((n_words or len(out)) - len(out))


def elements(ws: Sequence[int]) -> list[int]:
    """`words`' inverse: the F elements of whole 8-word groups."""
    return [sum(int(ws[F_WORDS * i + k]) << (16 * k) for k in range(F_WORDS)) for i in range(len(ws) // F_WORDS)]


def acc_ref(acc: Sequence[int], coef: Sequence[int], row: Sequence[int]) -> list[int]:
    """``acc_out``'s words from ``acc``'s, ``coef``'s and ``row``'s."""
    lanes = elements(row)
    c = elements(coef)
    a = elements(acc[:2 * F_WORDS])
    t = [a[r] ^ reduce(lambda x, y: x ^ y, (G.mul(c[r * len(lanes) + l], v) for l, v in enumerate(lanes)), 0) for r in range(2)]
    return words(t) + list(acc[2 * F_WORDS:])


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


def open_ref(LANES: int, H: int, row: Sequence[int], aux: Sequence[int], acc: Sequence[int],
             coef: Sequence[int]) -> tuple[list[int], list[int]]:
    """`RecOpen`'s reference on words: ``(out, acc_out)``."""
    _check(LANES, H)
    if (len(row), len(aux), len(acc), len(coef)) != (row_words(LANES), aux_words(H), ACC_WORDS, coef_words(LANES)):
        raise ValueError(f"RecOpen{{{LANES}, {H}}} takes {row_words(LANES)} row, {aux_words(H)} aux, {ACC_WORDS} acc and "
                         f"{coef_words(LANES)} coef words")
    salt = bytes_of(aux[:SALT_WORDS])
    low = aux[dirs_at(H)] & ((1 << H) - 1)
    cur = INST.tree_leaf(KEY, INST.commit_string(KEY, hashlib.sha512(bytes_of(row)).digest(), salt))
    for h in range(H):
        sib = bytes_of(aux[SALT_WORDS + NODE_WORDS * h:SALT_WORDS + NODE_WORDS * (h + 1)])
        cur = hashlib.sha512(sib + cur if (low >> h) & 1 else cur + sib).digest()
    return expected(cur, low, H), acc_ref(acc, coef, row)


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


def _steps(ops, LANES: int, H: int, row: list, aux: list, acc: list, coef: list) -> tuple[list, list]:
    """``out``'s and ``acc_out``'s bits (byte-string order) from the ports' (`RecOpen`'s, flattened)."""
    t = ops.residuals(cons_structure(LANES), row + acc[:256], coef)
    return _climb(ops, H, row, aux), t + acc[256:]


def _climb(ops, H: int, row: list, aux: list) -> list:
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

    def residuals(self, S: dict, w: list, v: list) -> list:
        return residual_forms(self.C, S, w, v)


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

    def residuals(self, S: dict, w: list, v: list) -> list:
        return self._list(self.B.call(bind(G.GfResiduals, S=S), Coll(Array(S["nw"], G.F), TR._refs(w)),
                                      Coll(Array(S["nv"], G.F), TR._refs(v))))


# --- the Definitions ---------------------------------------------------------------------------------------------------------

RecSha512Compress = TR.definition(
    "RecSha512Compress", 1, lambda C, *e: compress_gadget(C, list(e[:8]), list(e[8:])), (("cv", DIGEST), ("msg", BLOCK)), DIGEST,
    doc="One SHA-512 compression (FIPS 180-4 §6.4.2) with the feed-forward, words LSB first: sha512.compress_gadget")


def _port_words(LANES: int, H: int) -> tuple[int, ...]:
    return row_words(LANES), aux_words(H), ACC_WORDS, coef_words(LANES)


def _signature(S):
    _check(S.LANES, S.H)
    params = tuple((nm, Array(n, U16)) for nm, n in zip(("row", "aux", "acc", "coef"), _port_words(S.LANES, S.H)))
    return params, Tuple(Array(OUT_WORDS, U16), Array(ACC_WORDS, U16))


def _body(B, S, row, aux, acc, coef):
    flat = _Ir._list
    out, acc_out = _steps(_Ir(B), S.LANES, S.H, flat(row), flat(aux), flat(acc), flat(coef))
    return tuple_of(Coll(Array(OUT_WORDS, U16), TR._refs(out)), Coll(Array(ACC_WORDS, U16), TR._refs(acc_out)))


RecOpen = CompositeDefinition(
    "RecOpen", 2, ("LANES", "H"), _signature, _body, conformance="exact-model-tested",
    doc="One inner Ligerito opening: the hm96-sha512/v1 leaf of the row under the salt climbs the H siblings by the position's "
        "low bits, the node it reaches and those bits the first result; the second is acc with the row's consistency terms "
        "Σ_l coef_r[l] · row[l] added to its two F elements")


# --- the M0 unit -------------------------------------------------------------------------------------------------------------

def unit_circuit(LANES: int, H: int):
    """The unit over `forms`: inputs ``in0 ..`` (``row``'s words, then ``aux``'s, ``acc``'s and ``coef``'s, 16 bits each),
    outputs ``out0 .. out63`` (``out``) then ``acc0 .. acc63`` (``acc_out``)."""
    from verity.primitives.circuits.boolean.forms import Circuit

    _check(LANES, H)
    C = Circuit()
    pw = _port_words(LANES, H)
    ins = [C.input(f"in{j}", 16) for j in range(sum(pw))]
    flat = [b for w in ins for b in w]
    cuts = [16 * sum(pw[:i]) for i in range(len(pw) + 1)]
    out, acc_out = _steps(_Forms(C), LANES, H, *(flat[a:b] for a, b in zip(cuts, cuts[1:])))
    outs = [f"out{j}" for j in range(OUT_WORDS)] + [f"acc{j}" for j in range(ACC_WORDS)]
    bits = out + acc_out
    for j, nm in enumerate(outs):
        C.output(nm, bits[16 * j:16 * j + 16])
    return C, [f"in{j}" for j in range(len(ins))], outs


def definition(LANES: int, H: int):
    return bind(RecOpen, LANES=LANES, H=H)
