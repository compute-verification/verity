"""V*'s opening check (hiding model architectures, lane rec-v0): one inner Ligerito opening climbs to the top node its position
names and adds its share of the level's consistency sums to a running sum, as a Boolean Definition, the M0 unit that proves
it, and their plain-integer reference.

    U16 = Array<16, Value<1>>
    RecOpen_v4{LANES, H}(row: Array<8 LANES, U16>, aux: Array<A(H), U16>, acc: Array<64, U16>, coef: Array<16 LANES, U16>,
                         dirs: Array<64, U16>) -> (out: Array<64, U16>, acc_out: Array<64, U16>)

Every port is u16 words (a Boolean element of 16 bits, `verity.ml.circuits.boolean`'s word view): a word's bits LSB first, a byte
string's bytes two to a word, little-endian, so leaf ``t`` of a port is bit ``t & 7`` of its byte string's byte ``t >> 3``. An
element of F = GF(2^128) is 8 words, its 16 bytes ``lo`` LE then ``hi`` LE, so its bit k is bit ``k % 16`` of word ``k // 16``.

* ``row``: the opened row's LANES elements of F: the row's bytes as the inner prover hashes them
  (`verity_flock.rec_vstar.row_bytes`).
* ``aux``: the opening's H siblings bottom up (64 bytes each, 32 words from word ``32 h``), then zero words up to
  ``A(H) = 64 max(1, ⌈H / 2⌉)``, a whole number of 64-word M0 rows and at least one (a circuit file's row is whole SHA-512
  blocks, so ``H = 0``'s ``aux`` is one row of zero words, which the unit does not read).
* ``acc``: the running sums before this opening, two elements of F (words 0 .. 15), and 48 words carried as they are.
* ``coef``: the verifier's coefficients, ``c0[l]`` for each lane then ``c1[l]`` (8 words each).
* ``dirs``: the position's low H bits (bit ``h`` the side at level ``h``: bit ``h % 16`` of word ``h // 16``, the other bits
  of words 0 and 1 zero) and 62 zero words, one M0 row (the unit reads the low H of words 0 and 1's 32 bits).
* ``out``: the node H levels up (32 words) and 32 zero words: one M0 row, the top node's own registered row
  (`expected`, `verity_flock.rec_live.top_rows`). The statement reads it from the prover's registration of the top the firewall
  committed under salts (`verity_flock.rec_outer`), so one registered output row checks the climb against the top
  (PROTOCOL.md §16.13 "Registered reads") and ``dirs`` the side at every level.

The statement reads ``coef`` and ``dirs`` from values the verifier registers itself (`verity_flock.rec_outer`), so it has no
public input.
* ``acc_out``: ``acc`` with ``Σ_l c_r[l] · row[l]`` added to its element r (`cons_structure`, a ``GfResiduals_v1`` of two
  residuals), its other words as they were.

**The consistency sums.** The inner verifier's final check is linear in each level's sums ``Σ_j eq(α)[j] · row_j[l]`` over the
level's opened rows (`verity_flock.rec_algebra`), with coin-derived coefficients ``κ_{r, l}`` in its two components r. The
verifier gives opening j of the level ``c_r[l] = κ_{r, l} · eq(α)[j]``, so the openings' terms ``Σ_l c_r[l] · row_j[l]`` sum to
the final check's share of that level's sums. The openings of a rep chain their running sums through one registered value
(`verity_flock.rec_outer`): each one's ``acc_out`` is the next one's ``acc``, and V*'s algebra reads the rep's first and last.

The leaf is plain SHA-512 of the row (the inner statement's ``flock-leaf/sha512-unsalted``,
`verity_flock.circuit.LEAF_SCHEME_PLAIN`): the tops are what the firewall salts, so a leaf needs none. A node is
``SHA-512(left || right)`` (§14), the current digest on the side its bit names (``t = d AND (cur XOR sib)`` swaps them).
Compressions: ``LANES / 8 + 1`` for the leaf and 2 a level.

``H`` is 0 to 32. At ``H = 0`` the leaf is the top: ``out`` is the leaf itself, which the statement reads against the
registered top as it reads any climb's node, and neither ``aux`` nor ``dirs`` is read. Above 16 the position's bits run into ``dirs``' word 1;
an opening of ``H ≤ 16`` reads word 0 alone, so its gates are the same at either bound.

`_steps` states the computation once over two interpreters: `forms` for the M0 unit (`unit_circuit`: constant blocks fold, the
products are summed leaf by leaf as `rec_residuals.residual_forms` sums them) and the IR for the Definition (`RecOpen`'s body:
each compression a Call of `RecSha512Compress_v1`, traced from `sha512.compress_gadget`, so the descriptor holds one
compression's gates, and the consistency term a Call of ``GfResiduals_v1{cons_structure(LANES)}``). `open_ref` (over
``hashlib`` and `gf2k.mul`) is the reference of both.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from functools import reduce

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.refs import Coll, tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.boolean.forms import ONE, ZERO
from verity.ml.commitments.gates.sha512 import IV, compress_gadget, pad_tail
from verity.ml.circuits.ops.boolean import gf2k as G

from .rec_residuals import residual_forms
from .sha512_circuit import word_bit

__all__ = ["ACC_WORDS", "DIRS_BITS", "DIRS_WORDS", "OUT_WORDS", "PORTS", "RecOpen", "RecSha512Compress", "acc_ref", "aux_words", "coef_words",
           "compressions", "cons_structure", "definition", "dirs_words", "elements", "expected", "open_ref", "pack", "port_words",
           "row_words", "unit_circuit", "words"]

BIT = BL.BIT
U16 = Array(16, BIT)
WORD = Array(64, BIT)
DIGEST = Array(8, WORD)
BLOCK = Array(16, WORD)
NODE_WORDS = 32
OUT_WORDS = 64
ACC_WORDS = 64
DIRS_WORDS = 64
DIRS_BITS = 32
ROW_BLOCK = 64
F_WORDS = 8


def _check(LANES: int, H: int) -> None:
    for n, v in (("LANES", LANES), ("H", H)):
        if isinstance(v, bool) or not isinstance(v, int):
            raise TypeError(f"RecOpen's {n} is an int, not {v!r}")
    if LANES < 8 or LANES % 8:
        raise ValueError(f"RecOpen's LANES is a positive multiple of 8 (whole SHA-512 blocks of row), not {LANES}")
    if not 0 <= H <= DIRS_BITS:
        raise ValueError(f"RecOpen's H is 0 to {DIRS_BITS} (the position's bits are dirs' first two words), not {H}")


def row_words(LANES: int) -> int:
    return 8 * LANES


def aux_words(H: int) -> int:
    return max(1, -(-NODE_WORDS * H // ROW_BLOCK)) * ROW_BLOCK


def coef_words(LANES: int) -> int:
    return 2 * F_WORDS * LANES


def compressions(LANES: int, H: int) -> int:
    """SHA-512 compressions an opening costs: the leaf's (its row's blocks and the padding block), then two a level."""
    return LANES // 8 + 1 + 2 * H


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


def pack(row: Sequence[int], sibs: Sequence[bytes]) -> tuple[list[int], list[int]]:
    """``(row, aux)`` words of one opening: the row's GF(2^128) elements and its siblings bottom up."""
    rw = [(v >> (16 * k)) & 0xFFFF for v in row for k in range(8)]
    aux = [w for s in sibs for w in words_of(s)]
    return rw, aux + [0] * (aux_words(len(sibs)) - len(aux))


def dirs_words(pos: int, H: int) -> list[int]:
    """``dirs``: the position's low H bits, 16 a word, then zero words."""
    low = pos & ((1 << H) - 1)
    return [low & 0xFFFF, low >> 16] + [0] * (DIRS_WORDS - 2)


def expected(node: bytes) -> list[int]:
    """The result an opening must give: the top node it climbs to, as that node's registered row."""
    return words_of(node) + [0] * (OUT_WORDS - NODE_WORDS)


def open_ref(LANES: int, H: int, row: Sequence[int], aux: Sequence[int], acc: Sequence[int], coef: Sequence[int],
             dirs: Sequence[int]) -> tuple[list[int], list[int]]:
    """`RecOpen`'s reference on words: ``(out, acc_out)``."""
    _check(LANES, H)
    if (len(row), len(aux), len(acc), len(coef), len(dirs)) != port_words(LANES, H):
        raise ValueError(f"RecOpen{{{LANES}, {H}}} takes {row_words(LANES)} row, {aux_words(H)} aux, {ACC_WORDS} acc, "
                         f"{coef_words(LANES)} coef and {DIRS_WORDS} dirs words")
    low = (dirs[0] | dirs[1] << 16) & ((1 << H) - 1)
    cur = hashlib.sha512(bytes_of(row)).digest()
    for h in range(H):
        sib = bytes_of(aux[NODE_WORDS * h:NODE_WORDS * (h + 1)])
        cur = hashlib.sha512(sib + cur if (low >> h) & 1 else cur + sib).digest()
    return expected(cur), acc_ref(acc, coef, row)


# --- the computation, over an interpreter ------------------------------------------------------------------------------------

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


def _steps(ops, LANES: int, H: int, row: list, aux: list, acc: list, coef: list, dirs: list) -> tuple[list, list]:
    """``out``'s and ``acc_out``'s bits (byte-string order) from the ports' (`RecOpen`'s, flattened)."""
    t = ops.residuals(cons_structure(LANES), row + acc[:256], coef)
    return _climb(ops, H, row, aux, dirs[:DIRS_BITS]), t + acc[256:]


def _climb(ops, H: int, row: list, aux: list, d: list) -> list:
    cur = _sha(ops, IV, row, 0)
    for h in range(H):
        sib = aux[16 * NODE_WORDS * h:16 * NODE_WORDS * (h + 1)]
        left, right = ops.swap(d[h], cur, sib)
        cur = _sha(ops, IV, left + right, 0)
    return cur + [ops.const(0)] * (16 * (OUT_WORDS - NODE_WORDS))


class _Forms:
    def __init__(self, C):
        self.C = C

    def const(self, b: int):
        return ONE if b else ZERO

    def compress(self, cv: list, msg: list) -> list:
        out = compress_gadget(self.C, [cv[64 * i:64 * i + 64] for i in range(8)], [msg[64 * i:64 * i + 64] for i in range(16)])
        return [x for w in out for x in w]

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

PORTS = ("row", "aux", "acc", "coef", "dirs")


def port_words(LANES: int, H: int) -> tuple[int, ...]:
    return row_words(LANES), aux_words(H), ACC_WORDS, coef_words(LANES), DIRS_WORDS


def _signature(S):
    _check(S.LANES, S.H)
    params = tuple((nm, Array(n, U16)) for nm, n in zip(PORTS, port_words(S.LANES, S.H)))
    return params, Tuple(Array(OUT_WORDS, U16), Array(ACC_WORDS, U16))


def _body(B, S, row, aux, acc, coef, dirs):
    flat = _Ir._list
    out, acc_out = _steps(_Ir(B), S.LANES, S.H, flat(row), flat(aux), flat(acc), flat(coef), flat(dirs))
    return tuple_of(Coll(Array(OUT_WORDS, U16), TR._refs(out)), Coll(Array(ACC_WORDS, U16), TR._refs(acc_out)))


RecOpen = CompositeDefinition(
    "RecOpen", 4, ("LANES", "H"), _signature, _body, conformance="exact-model-tested",
    doc="One inner Ligerito opening: the plain SHA-512 leaf of the row climbs the H siblings by the low bits dirs holds, and "
        "the node it reaches is the first result, as a registered row; the second is acc with the row's consistency terms "
        "Σ_l coef_r[l] · row[l] added to its two F elements")


# --- the M0 unit -------------------------------------------------------------------------------------------------------------

def unit_circuit(LANES: int, H: int):
    """The unit over `forms`: inputs ``in0 ..`` (``row``'s words, then ``aux``'s, ``acc``'s, ``coef``'s and ``dirs``', 16 bits
    each), outputs ``out0 .. out63`` (``out``) then ``acc0 .. acc63`` (``acc_out``)."""
    from verity.ml.circuits.boolean.forms import Circuit

    _check(LANES, H)
    C = Circuit()
    pw = port_words(LANES, H)
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
