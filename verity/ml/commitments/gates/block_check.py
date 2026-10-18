"""The remote bandwidth audit's check 3 (`verity.core.protocols.remote_bandwidth`, PROTOCOL.md §8) as Boolean Definitions:
what a proof runs on a drawn receipt's block in place of opening it, so the verifier holds the block, its payload and its
receipts only as hiding commitments, and a unit's verdict.

    BlockDecodeCheck_v1{N, C, K, MASKED, BYTES}
        (j: Array<4, U16>, bound: U16, seed: Array<16, U16>, coin: Array<16, U16>, tau: F, t0: F, c: Array<K, F>,
         [mask: F,] block: Array<BYTES / 2, U16>, payload: Array<BYTES / 2, U16>) -> U16
    BlockReceiptCheck_v1{ROWS, LEAF, P, BYTES}
        (mid: Array<32, U16>, j: Array<4, U16>, k: U16, pos: U16, signs: Array<ROWS, Array<BYTES / 16, U16>>,
         block: Array<BYTES / 2, U16>, leaf: Array<LEAF, Array<32, U16>>, plan: Array<P, U16>) -> U16
        U16 = Array<16, Value<1>>, F = Array<N / 16, U16>: an integer little-endian, a byte string in order (bit t is bit
        t & 7 of byte t >> 3); each returns one ok word, bit 0 the verdict, bits 1-15 zero in gates

**The decode.** ``BlockDecodeCheck`` is 1 iff every word of ``block`` (word w its bytes [S w, S (w + 1)), S = N / 8, of
the W = BYTES / S) is below p = 2^N - C and decodes to ``payload``'s word w (`codec.decode_word`): x -> rho^-1(x) - c_r -
t_w mod p for r = K - 1 ... 0, and, ``MASKED``, x -> sigma_K(x) after each round r >= 1, with t_w = t0 + w tau + b mod p and
b = 0 unless ``bound``'s bit 0, then b_j(c_a), derived in the unit from the payload it holds: LE(SHA-512(DOMAIN || u8
len(name) || name || seed || u32be 2^32 - 4 || u64be j || d || coin || u32be 0)) mod p, d = sha512/row/v2 of the payload
(`codec.binding` with its coin, `codec.digest`). So at ``bound`` it is ``codec.decode(block, j, key, d, coin) == payload``,
d the payload's own digest, and without it ``codec.decode(block, j, key) == payload`` (a registered block, under the seed's
key). The public row (j, bound, seed, coin, tau, t0, c, mask) is the verifier's own: the codec key of the registration's
seed (`codec.key`), t0 = tau W j mod p, the append's coin (zero bytes under the seed's key). A field's name is kv/v0's
(p512 = 2^512 - 569, p1024 = 2^1024 - 105); ``toy32`` (2^32 - 5) is circuit-check's and the tests' binding with a short
block (``BYTES`` below 4,096), the same circuit on small words, never deployed.

**The receipt.** ``BlockReceiptCheck`` is 1 iff the sketch-sha512 receipt of ``block`` (`receipts.SketchReceipter`): y =
SHA-512(prefix || u64be j || sigma), its chaining value after the one-block prefix ``mid`` (`receipts.sketch_prefix` of the
pass's seed) and sigma ROWS int32 little-endian, sigma_c = sum_u (-1)^(signs[c][u]) int8(block[u]), is ``leaf[k]`` (k <
LEAF), and the plan row holds block j at the position drawn: ``plan[pos] = j`` (pos < P, j < 2^16). Public: mid, j, k, pos and
the pass's sign bits (`receipts.sign_bits`, bit u of row c at bit u). Witness: the block, the receipt leaf (the receipts of
position pos's leaf, LEAF of them, the receipt tree's leaf under its salt) and the plan (the step's certified plan,
`requirement_units.PlanMatch`'s row).

**The links.** A unit's rows are hm96-sha512/row/v2 rows, and a row under the salt of another commitment of the same bytes
has that commitment's ``b || c`` (`receipts.hidden`), so the verifier holds every witness row to its commitment by
``b || c`` alone: the two units' block rows to each other; the payload row's to the canonical payload row the memory-map unit
reads (written blocks) and, through ``HM96.tree_leaf``, to leaf j of the append's tree or the registration's payload tree;
the receipt leaf's to its leaf of the pass's receipt tree; the plan row's to the plan unit's. The decode's ``d`` is
recomputed from the payload row: the statement's hm96 layer hashes the same bytes, but a unit reads none of it.

**The parts**, each its own traced Definition lowered by its builder (``build``), so a C-Flock unit gets the builder's
commits and the Definition the trace's ANDs:

    RowCompress_v1                  (cv: Array<8, Word>, block: Array<16, Word>) -> Array<8, Word>, one SHA-512 compression
    CodecBinding_v1{N, C}           (bound, seed: Array<256>, j: Array<64>, d: Array<8, Word>, coin: Array<256>) -> Array<N>
    CodecModAdd_v1{N, C}            (a, b) -> a + b mod p
    CodecLess_v1{N, C}              (y) -> y < p
    CodecRound_v1{N, C, MASK}       (x, c, t[, mask]) -> rho^-1(x) - c - t mod p, then sigma_K if MASK
    CodecMatch_v1{N}                (x, y, a) -> a and x = y
    CodecWord_v1{N, C, K, MASKED}   (y, t, c: Array<K, Array<N>>[, mask], want) -> y < p and its decode is want
    BlockOk_v1{W}                   (oks: Array<W>) -> U16
    SketchRow_v1{BYTES}             (block: Array<8 BYTES>, signs: Array<BYTES>) -> Array<32>, one row of sigma
    LeafMatch_v1{LEAF, P}           (y: Array<8, Word>, leaf, k, plan, pos, j) -> U16
    Word = Array<64, Value<1>>, a SHA-512 word LSB first; Array<n> = Array<n, Value<1>>

**Cost.** A squaring mod p is a Karatsuba square down to 32-bit schoolbook squares (about 96,000 ANDs at N = 512) and two
folds of the high half by C; a round adds about 9 N for its additions, negation and selections. ``BlockDecodeCheck`` is W K
rounds and 2 + ceil((BYTES + 17) / 128) compressions: about 15.6 M ANDs at p512/k2, about 32 M at p1024/k3/masked.
``BlockReceiptCheck`` is ROWS sketch rows of about 9 BYTES ANDs each (a carry-save count of the block's sign-flipped bits)
and one compression at 16 rows: about 0.66 M ANDs.

`decode_check` and `receipt_check` are the references, over each instance's u16 words; `decode_args` and `receipt_args` an
instance's arguments, `decode_rows` and `receipt_rows` its rows (C-Flock's ``in_rows`` params), `row_words` its rows' words.
"""
from __future__ import annotations

import hashlib
import struct
from functools import lru_cache
from types import SimpleNamespace
from typing import Callable, Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.boolean.forms import ONE, ZERO, OR, and_reduce, const_bits, maj, mux, or_reduce
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.refs import Coll
from verity.ml.circuits.types import Array
from verity.ml.commitments.rowleaf import sha512_row_v2_digest, sha512_row_v2_prefix

from .sha512 import IV, commit as _commit, compress_gadget, compress_ref, midstate, pad_tail, words_be

__all__ = ["BlockDecodeCheck", "BlockReceiptCheck", "FIELDS", "TOY", "decode_args", "decode_check", "decode_rows", "key",
           "receipt", "receipt_args", "receipt_check", "receipt_rows", "row_words", "sketch"]

BIT = BL.BIT
U16 = Array(16, BIT)
WORD = Array(64, BIT)
DIGEST = Array(8, WORD)
#: (N, C) -> the field's name in the codec's derivations (`codec.Field.name`)
FIELDS = {(1024, 105): "p1024", (512, 569): "p512", (32, 5): "toy32"}
TOY = (32, 5)
BLOCK_BYTES = 4096
DOMAIN = b"verity/word-codec/kv/v0\0"
TWEAK_INDEX, MASK_INDEX, COIN_INDEX = (1 << 32) - 1, (1 << 32) - 2, (1 << 32) - 4
SEED_BYTES = COIN_BYTES = 32
#: a carry-save tree commits a form longer than this; a ripple carry, one longer than CARRY
LIMIT, CARRY = 6, 4
#: Karatsuba squares down to schoolbook squares of at most this many bits
SCHOOL = 32


def _check_field(S) -> None:
    if (S.N, S.C) not in FIELDS:
        raise ValueError(f"(N, C) is a codec field's, one of {sorted(FIELDS)}, not ({S.N}, {S.C})")


def _check_bytes(S, field: bool) -> None:
    b = S.BYTES
    if isinstance(b, bool) or not isinstance(b, int) or b < 64 or b > BLOCK_BYTES or b & (b - 1):
        raise ValueError(f"BYTES is a power of two from 64 to {BLOCK_BYTES}, not {b!r}")
    if field and (S.N, S.C) != TOY and b != BLOCK_BYTES:
        raise ValueError(f"a block of the codec's {FIELDS[S.N, S.C]} is {BLOCK_BYTES} bytes; a shorter one is the toy's")


def _check_decode(S) -> None:
    _check_field(S)
    _check_bytes(S, True)
    if isinstance(S.K, bool) or not isinstance(S.K, int) or not 2 <= S.K <= 8 or S.MASKED not in (0, 1):
        raise ValueError(f"K is 2 to 8 rounds and MASKED 0 or 1, not K = {S.K!r}, MASKED = {S.MASKED!r}")


def _check_receipt(S) -> None:
    _check_bytes(S, False)
    _check_ints(S, ("ROWS", 64), ("LEAF", 16), ("P", (1 << 16) - 1))


def _check_ints(S, *bounds: tuple[str, int]) -> None:
    for k, top in bounds:
        v = getattr(S, k)
        if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= top:
            raise ValueError(f"{k} is 1 to {top}, not {v!r}")


def _p(S) -> int:
    return (1 << S.N) - S.C


def _words_of(S) -> int:
    return S.BYTES // (S.N // 8)


# --- bits, bytes and words ---------------------------------------------------------------------------------------------------
# A byte is 8 bits LSB first; a SHA-512 word 64 bits LSB first, its bytes big-endian. These rearrange bits of any kind.

def _bytes(words: Sequence[Sequence]) -> list[list]:
    return [list(w[8 * (7 - b):8 * (8 - b)]) for w in words for b in range(8)]


def _words(data: Sequence[Sequence]) -> list[list]:
    assert len(data) % 8 == 0
    return [[x for b in range(8) for x in data[8 * k + 7 - b]] for k in range(len(data) // 8)]


def _octets(bits: Sequence) -> list[list]:
    return [list(bits[8 * k:8 * k + 8]) for k in range(len(bits) // 8)]


def _u64be(bits: Sequence) -> list[list]:
    return [list(bits[8 * (7 - b):8 * (8 - b)]) for b in range(8)]


def _join(ws: Sequence[Sequence]) -> list:
    return [x for w in ws for x in w]


# --- arithmetic over forms ---------------------------------------------------------------------------------------------------
# Integers are bit lists LSB first. A long form is committed (`sha512.commit`): a row of its own in C-Flock's lowering, and in a
# trace a copy with no gate, written once as its XORs, so neither a row nor a traced form compounds.

def _short(C, x, limit: int = LIMIT):
    return x if len(x.s) <= limit else _commit(C, [x])[0]


def _pad(a: Sequence, n: int) -> list:
    return list(a[:n]) + [ZERO] * (n - len(a))


def _add(C, a, b, cin=ZERO, width: int | None = None) -> list:
    """(a + b + cin) mod 2^width (default: wide enough for the carry)."""
    n = width if width is not None else max(len(a), len(b)) + 1
    a, b = _pad(a, n), _pad(b, n)
    out, c = [], cin
    for i in range(n):
        out.append(a[i] ^ b[i] ^ c)
        if i < n - 1:
            c = _short(C, maj(C, a[i], b[i], c), CARRY)
    return out


def _sub(C, a, b, width: int) -> tuple[list, object]:
    """(a - b) mod 2^width and the borrow."""
    s = _add(C, _pad(a, width), [x.inv() for x in _pad(b, width)], ONE, width + 1)
    return s[:width], s[width].inv()


def _columns(C, cols: Sequence[Sequence], width: int) -> list:
    """The sum of weighted columns (column i's bits each worth 2^i) mod 2^width: full adders until no column holds three, then
    one ripple add; a long sum or carry is committed (`LIMIT`)."""
    cols = [list(c) for c in cols[:width]] + [[] for _ in range(width - len(cols))]
    for i in range(width):
        k = sum(x.c for x in cols[i] if x.is_const())
        cols[i] = [x for x in cols[i] if not x.is_const()]
        for e in range(width - i):
            if (k >> e) & 1:
                cols[i + e].append(ONE)
    while any(len(c) > 2 for c in cols):
        new: list[list] = [[] for _ in range(width)]
        for i, col in enumerate(cols):
            while len(col) >= 3:
                x, y, z = col.pop(), col.pop(), col.pop()
                new[i].append(_short(C, x ^ y ^ z))
                if i + 1 < width:
                    new[i + 1].append(_short(C, maj(C, x, y, z)))
            new[i].extend(col)
        cols = new
    return _add(C, [c[0] if c else ZERO for c in cols], [c[1] if len(c) > 1 else ZERO for c in cols], width=width)


def _square_school(C, x) -> list:
    n = len(x)
    cols: list[list] = [[] for _ in range(2 * n)]
    for i in range(n):
        cols[2 * i].append(x[i])
        for k in range(i + 1, n):
            cols[i + k + 1].append(C.AND(x[i], x[k]))
    return _columns(C, cols, 2 * n)


def _square(C, x) -> list:
    """x^2 (2 len(x) bits): Karatsuba's three half squares, x0^2, x1^2 and (x0 + x1)^2, down to `SCHOOL` bits."""
    n = len(x)
    if n <= SCHOOL:
        return _square_school(C, x)
    h = n // 2
    x0, x1 = x[:h], x[h:]
    s = _commit(C, _add(C, x0, x1))
    a, b, m = _commit(C, _square(C, x0)), _commit(C, _square(C, x1)), _square(C, s)
    mid, _ = _sub(C, m, _commit(C, _add(C, a, b)), len(m))
    whole = _pad(a, 2 * h) + b
    return _commit(C, whole[:h] + _add(C, whole[h:], _commit(C, mid), width=2 * n - h))


def _reduce(C, z, bound: int, N: int, Cc: int) -> list:
    """z mod p (N bits) for z <= bound: the high part folded in by 2^N = C (mod p) until z < 2 p, then one conditional
    subtraction."""
    p = (1 << N) - Cc
    z = list(z[:bound.bit_length()])
    while bound >= 2 * p:
        lo, hi = z[:N], z[N:]
        hb = bound >> N
        bound = (1 << N) - 1 + hb * Cc
        w = bound.bit_length()
        cols: list[list] = [[] for _ in range(w)]
        for i, x in enumerate(lo):
            cols[i].append(x)
        for e in range(Cc.bit_length()):
            if (Cc >> e) & 1:
                for i, x in enumerate(hi):
                    cols[i + e].append(x)
        z = _commit(C, _columns(C, cols, w))
    if bound < p:
        return _pad(z, N)
    z = _pad(z, N + 1)
    u = _add(C, z, const_bits(Cc, N + 1), width=N + 1)
    return mux(C, u[N], u[:N], z[:N])


def _less_p(C, y, N: int, Cc: int):
    """y < p: no carry out of y + C."""
    return _add(C, list(y), const_bits(Cc, N), width=N + 1)[N].inv()


def _mod_add(C, a, b, N: int, Cc: int) -> list:
    return _reduce(C, _add(C, a, b), 2 * ((1 << N) - Cc - 1), N, Cc)


def _mod_sub(C, a, b, N: int, Cc: int) -> list:
    d, borrow = _sub(C, a, b, N)
    e, _ = _sub(C, d, const_bits(Cc, N), N)
    return mux(C, borrow, e, d)


def _round(C, N: int, Cc: int, x, c, t, mask=None) -> list:
    p = (1 << N) - Cc
    q = _reduce(C, _square(C, list(x)), (p - 1) ** 2, N, Cc)
    q = _commit(C, q)
    r = mux(C, x[0], _mod_sub(C, [ZERO] * N, q, N, Cc), q)
    out = _commit(C, _mod_sub(C, r, _commit(C, _mod_add(C, c, t, N, Cc)), N, Cc))
    if mask is not None:
        v = [a ^ m for a, m in zip(out, mask)]
        out = _commit(C, mux(C, _less_p(C, v, N, Cc), v, out))
    return out


def _eq(C, a, b):
    return and_reduce(C, [(x ^ y).inv() for x, y in zip(a, b, strict=True)])


def _eqc(C, a, k: int):
    return and_reduce(C, [x if (k >> i) & 1 else x.inv() for i, x in enumerate(a)]) if k < 1 << len(a) else ZERO


def _sha(C, data: list[list]) -> list[list]:
    """SHA-512 over ``data``'s bytes from the IV: its eight words."""
    msg = _words(data + [const_bits(b, 8) for b in pad_tail(len(data))])
    st = [const_bits(v, 64) for v in IV]
    for i in range(0, len(msg), 16):
        st = compress_gadget(C, st, msg[i:i + 16])
    return st


# --- the parts ---------------------------------------------------------------------------------------------------------------

def _part(name: str, statics: tuple[str, ...], signature: Callable, build: Callable, doc: str) -> CompositeDefinition:
    """A Definition traced from ``build(C, S, *elements)`` once per binding of ``statics``, which C-Flock lowers by ``build``."""

    @lru_cache(maxsize=None)
    def traced(*key) -> TR.Traced:
        S = SimpleNamespace(**dict(zip(statics, key)))
        params, ret = signature(S)
        return TR.trace(lambda C, *e: build(C, S, *e), [t for _, t in params], ret)

    def body(B, S, *colls):
        params, ret = signature(S)
        return TR.emit(traced(*(getattr(S, k) for k in statics)), [t for _, t in params], ret, B, *colls)

    d = CompositeDefinition(name, 1, statics, signature, body, conformance="exact-model-tested", doc=doc)
    d.build = build  # type: ignore[attr-defined]
    d.traced = traced  # type: ignore[attr-defined]
    return d


def _f(S):
    return Array(S.N, BIT)


RowCompress = _part(
    "RowCompress", (), lambda S: ((("cv", DIGEST), ("block", Array(16, WORD))), DIGEST),
    lambda C, S, *e: compress_gadget(C, list(e[:8]), list(e[8:24])),
    "One SHA-512 compression (FIPS 180-4 §6.4.2) with its feed-forward, on 64-bit words LSB first, lowered with its schedule "
    "and state words committed")


def _sig_binding(S):
    _check_field(S)
    return ((("bound", BIT), ("seed", Array(8 * SEED_BYTES, BIT)), ("j", Array(64, BIT)), ("d", DIGEST),
             ("coin", Array(8 * COIN_BYTES, BIT))), _f(S))


def _build_binding(C, S, bound, seed, j, *e):
    d, coin = e[:8], e[8]
    name = FIELDS[S.N, S.C].encode()
    head = DOMAIN + bytes([len(name)]) + name
    data = ([const_bits(b, 8) for b in head] + _octets(seed) + [const_bits(b, 8) for b in struct.pack(">I", COIN_INDEX)]
            + _u64be(j) + _bytes(d) + _octets(coin) + [const_bits(b, 8) for b in bytes(4)])
    z = _join(_bytes(_sha(C, data)))
    b = _reduce(C, z, (1 << 512) - 1, S.N, S.C)
    return _commit(C, [C.AND(bound[0], x) for x in b])


CodecBinding = _part(
    "CodecBinding", ("N", "C"), _sig_binding, _build_binding,
    "b_j(c_a) if bound, else 0: LE(SHA-512(DOMAIN || u8 len(name) || name || seed || u32be 2^32 - 4 || u64be j || d || coin || "
    "u32be 0)) mod p, the offset of block j's bound key under its append's coin (codec.binding)")


def _sig_pair(S):
    _check_field(S)
    return (("a", _f(S)), ("b", _f(S))), _f(S)


CodecModAdd = _part("CodecModAdd", ("N", "C"), _sig_pair,
                    lambda C, S, a, b: _commit(C, _mod_add(C, a, b, S.N, S.C)), "a + b mod p, for a and b below p")


def _sig_less(S):
    _check_field(S)
    return (("y", _f(S)),), BIT


CodecLess = _part("CodecLess", ("N", "C"), _sig_less, lambda C, S, y: [_less_p(C, y, S.N, S.C)], "1 iff y < p, canonical")


def _sig_round(S):
    _check_field(S)
    if S.MASK not in (0, 1):
        raise ValueError(f"MASK is 0 or 1, not {S.MASK!r}")
    return (("x", _f(S)), ("c", _f(S)), ("t", _f(S))) + ((("mask", _f(S)),) if S.MASK else ()), _f(S)


CodecRound = _part(
    "CodecRound", ("N", "C", "MASK"), _sig_round, lambda C, S, x, c, t, *m: _round(C, S.N, S.C, x, c, t, *m),
    "One decode round of the codec: rho^-1(x) - c - t mod p (rho^-1(x) = x^2, or p - x^2 for x odd), then, if MASK, sigma_K "
    "(the masked value if below p)")


def _sig_match(S):
    return (("x", Array(S.N, BIT)), ("y", Array(S.N, BIT)), ("a", BIT)), BIT


CodecMatch = _part("CodecMatch", ("N",), _sig_match, lambda C, S, x, y, a: [C.AND(a[0], _eq(C, x, y))], "a and x = y")


def _sig_word(S):
    _check_field(S)
    F = _f(S)
    return ((("y", F), ("t", F), ("c", Array(S.K, F))) + ((("mask", F),) if S.MASKED else ()) + (("want", F),)), BIT


def _body_word(B, S, y, t, c, *rest):
    mask, want = (rest[0] if S.MASKED else None), rest[-1]
    ok = B.call(bind(CodecLess, N=S.N, C=S.C), y)
    x = y
    for r in reversed(range(S.K)):
        m = int(bool(S.MASKED) and r > 0)
        x = B.call(bind(CodecRound, N=S.N, C=S.C, MASK=m), x, c[r], t, *([mask] if m else []))
    return B.call(bind(CodecMatch, N=S.N), x, want, ok)


CodecWord = CompositeDefinition(
    "CodecWord", 1, ("N", "C", "K", "MASKED"), _sig_word, _body_word, conformance="exact-model-tested",
    doc="1 iff the encoded word y is below p and decodes to want: K rounds from r = K - 1, each CodecRound_v1 with c_r and the "
        "word's tweak t, sigma_K after each round r >= 1 if MASKED (codec.decode_word)")


def _sig_ok(S):
    _check_ints(S, ("W", BLOCK_BYTES))
    return (("oks", Array(S.W, BIT)),), U16


BlockOk = _part("BlockOk", ("W",), _sig_ok, lambda C, S, oks: [and_reduce(C, list(oks))] + [ZERO] * 15,
                "The ok word: bit 0 the AND of W verdicts, bits 1-15 zero")


def _sig_sketch(S):
    _check_bytes(S, False)
    return (("block", Array(8 * S.BYTES, BIT)), ("signs", Array(S.BYTES, BIT))), Array(32, BIT)


def _build_sketch(C, S, block, signs):
    """sum_u (-1)^s_u int8(e_u) mod 2^32 = sum_u int8(e_u xor s_u (8 times)) + s_u: a flip is the one's complement plus one,
    and int8(v) = v_0..6 + 128 (1 - v_7) - 128."""
    cols: list[list] = [[] for _ in range(32)]
    for u in range(S.BYTES):
        s = signs[u]
        for k in range(7):
            cols[k].append(block[8 * u + k] ^ s)
        cols[7].append((block[8 * u + 7] ^ s).inv())
        cols[0].append(s)
    neg = (-128 * S.BYTES) % (1 << 32)
    for k in range(32):
        if (neg >> k) & 1:
            cols[k].append(ONE)
    return _commit(C, _columns(C, cols, 32))


SketchRow = _part(
    "SketchRow", ("BYTES",), _sig_sketch, _build_sketch,
    "One row of a block's sketch: sum over its bytes u of (-1)^signs[u] int8(block[u]), as an int32 (receipts.SketchReceipter)")


def _sig_leaf(S):
    _check_ints(S, ("LEAF", 16), ("P", (1 << 16) - 1))
    return ((("y", DIGEST), ("leaf", Array(S.LEAF, Array(512, BIT))), ("k", U16), ("plan", Array(S.P, U16)), ("pos", U16),
             ("j", Array(64, BIT))), U16)


def _build_leaf(C, S, *e):
    y, leaf, k = _join(_bytes(e[:8])), e[8:8 + S.LEAF], e[8 + S.LEAF]
    plan, pos, j = e[9 + S.LEAF:9 + S.LEAF + S.P], e[9 + S.LEAF + S.P], e[10 + S.LEAF + S.P]
    hot = [_eqc(C, k, i) for i in range(S.LEAF)]
    picked = [ZERO] * 512
    for h, w in zip(hot, leaf):
        picked = [o ^ C.AND(h, x) for o, x in zip(picked, w)]
    at = [_eqc(C, pos, q) for q in range(S.P)]
    entry = [ZERO] * 16
    for h, w in zip(at, plan):
        entry = [o ^ C.AND(h, x) for o, x in zip(entry, w)]
    oks = [or_reduce(C, hot), _eq(C, picked, y), or_reduce(C, at), _eq(C, entry, j[:16]), or_reduce(C, j[16:]).inv()]
    return [and_reduce(C, oks)] + [ZERO] * 15


LeafMatch = _part(
    "LeafMatch", ("LEAF", "P"), _sig_leaf, _build_leaf,
    "The receipt check's ok word: y = leaf[k] (k < LEAF) and plan[pos] = j (pos < P, j below 2^16)")


# --- the units ---------------------------------------------------------------------------------------------------------------

class _Wires:
    """Bit references in a body: the constant bits, and Colls of any references."""

    def __init__(self, B) -> None:
        self.bit = [B.call(BL.constant(b)).refs[0] for b in (0, 1)]

    def const(self, value: int, width: int) -> list:
        return [self.bit[(value >> k) & 1] for k in range(width)]

    @staticmethod
    def refs(c: Coll) -> list:
        return [c.refs[k] for k in range(c.refs.count)]

    @staticmethod
    def coll(t, refs: list) -> Coll:
        return Coll(t, TR._refs(list(refs)))


def _uwords(n_bits: int):
    return Array(n_bits // 16, U16)


def _decode_signature(S):
    _check_decode(S)
    F = _uwords(S.N)
    return (((("j", Array(4, U16)), ("bound", U16), ("seed", Array(16, U16)), ("coin", Array(16, U16)), ("tau", F), ("t0", F),
              ("c", Array(S.K, F))) + ((("mask", F),) if S.MASKED else ())
             + (("block", _uwords(8 * S.BYTES)), ("payload", _uwords(8 * S.BYTES)))), U16)


def _compress(B, X: _Wires, cv: Coll, data: list[list]) -> Coll:
    words = _words(data)
    for i in range(0, len(words), 16):
        cv = B.call(RowCompress, cv, X.coll(Array(16, WORD), _join(words[i:i + 16])))
    return cv


def _decode_body(B, S, *colls):
    params, _ret = _decode_signature(S)
    a = dict(zip([n for n, _ in params], colls))
    X = _Wires(B)
    F, W = _f(S), _words_of(S)
    pay = X.refs(a["payload"])
    prefix = sha512_row_v2_prefix(8 * S.BYTES)
    cv = X.coll(DIGEST, [b for v in midstate(prefix) for b in X.const(v, 64)])
    d = _compress(B, X, cv, _octets(pay) + [X.const(b, 8) for b in pad_tail(len(prefix) + S.BYTES)])
    b = B.call(bind(CodecBinding, N=S.N, C=S.C), a["bound"][0], X.coll(Array(256, BIT), X.refs(a["seed"])),
               X.coll(Array(64, BIT), X.refs(a["j"])), d, X.coll(Array(256, BIT), X.refs(a["coin"])))
    add = bind(CodecModAdd, N=S.N, C=S.C)
    tau = X.coll(F, X.refs(a["tau"]))
    t = B.call(add, X.coll(F, X.refs(a["t0"])), b)
    ts = [t]
    for _w in range(1, W):
        t = B.call(add, t, tau)
        ts.append(t)
    args = [X.coll(Array(W, F), X.refs(a["block"])), X.coll(Array(W, F), _join(X.refs(x) for x in ts)),
            X.coll(Array(S.K, F), X.refs(a["c"]))]
    axes: tuple = (0, 0, None)
    if S.MASKED:
        args.append(X.coll(F, X.refs(a["mask"])))
        axes += (None,)
    oks = B.batch(bind(CodecWord, N=S.N, C=S.C, K=S.K, MASKED=S.MASKED), *args, X.coll(Array(W, F), pay), axes=axes + (0,))
    return B.call(bind(BlockOk, W=W), oks)


BlockDecodeCheck = CompositeDefinition(
    "BlockDecodeCheck", 1, ("N", "C", "K", "MASKED", "BYTES"), _decode_signature, _decode_body, conformance="exact-model-tested",
    doc="1 iff every word of block is below p and the block decodes to payload under the codec key (tau, c, mask) at block j "
        "(t0 = tau W j mod p), with b_j(c_a) of the payload's own digest and coin if bound (codec.decode)")


def _receipt_signature(S):
    _check_receipt(S)
    return ((("mid", Array(32, U16)), ("j", Array(4, U16)), ("k", U16), ("pos", U16), ("signs", Array(S.ROWS, _uwords(S.BYTES))),
             ("block", _uwords(8 * S.BYTES)), ("leaf", Array(S.LEAF, Array(32, U16))), ("plan", Array(S.P, U16))), U16)


def _receipt_body(B, S, mid, j, k, pos, signs, block, leaf, plan):
    X = _Wires(B)
    sig = X.refs(B.batch(bind(SketchRow, BYTES=S.BYTES), X.coll(Array(8 * S.BYTES, BIT), X.refs(block)),
                         X.coll(Array(S.ROWS, Array(S.BYTES, BIT)), X.refs(signs)), axes=(None, 0)))
    jb = X.refs(j)
    data = _u64be(jb) + _octets(sig)
    data += [X.const(b, 8) for b in pad_tail(128 + len(data))]
    y = _compress(B, X, X.coll(DIGEST, X.refs(mid)), data)
    return B.call(bind(LeafMatch, LEAF=S.LEAF, P=S.P), y, X.coll(Array(S.LEAF, Array(512, BIT)), X.refs(leaf)), k,
                  X.coll(Array(S.P, U16), X.refs(plan)), pos, X.coll(Array(64, BIT), jb))


BlockReceiptCheck = CompositeDefinition(
    "BlockReceiptCheck", 1, ("ROWS", "LEAF", "P", "BYTES"), _receipt_signature, _receipt_body, conformance="exact-model-tested",
    doc="1 iff block's sketch-sha512 receipt at j, from the prefix's chaining value mid under the pass's sign rows, is "
        "leaf[k], and plan[pos] = j (receipts.SketchReceipter)")


# --- references --------------------------------------------------------------------------------------------------------------

def _xof(message: bytes, length: int) -> bytes:
    out, ctr = b"", 0
    while len(out) < length:
        out += hashlib.sha512(message + struct.pack(">I", ctr)).digest()
        ctr += 1
    return out[:length]


def _message(S, seed: bytes, r: int) -> bytes:
    name = FIELDS[S.N, S.C].encode()
    return DOMAIN + bytes([len(name)]) + name + seed + struct.pack(">I", r)


def key(S, seed: bytes) -> tuple[tuple[int, ...], int, int | None]:
    """The codec key of ``seed`` at (N, C, K, MASKED): its round constants c_0 .. c_(K-1), its tweak tau and its mask
    (`codec.key`'s, over any field of `FIELDS`)."""
    _check_decode(S)
    if type(seed) is not bytes or len(seed) != SEED_BYTES:
        raise ValueError(f"a codec seed is {SEED_BYTES} bytes")
    p, w = _p(S), S.N // 8

    def element(r: int) -> int:
        return int.from_bytes(_xof(_message(S, seed, r), w + 8), "little") % p

    mask = int.from_bytes(_xof(_message(S, seed, MASK_INDEX), w), "little") if S.MASKED else None
    return tuple(element(r) for r in range(S.K)), element(TWEAK_INDEX) or 1, mask


def _binding(S, seed: bytes, j: int, d: bytes, coin: bytes) -> int:
    m = _message(S, seed, COIN_INDEX) + j.to_bytes(8, "big") + d + coin
    return int.from_bytes(_xof(m, 64), "little") % _p(S)


def _decode_word(S, y: int, t: int, c: Sequence[int], mask: int | None) -> int:
    p, x = _p(S), y
    for r in reversed(range(S.K)):
        q = x * x % p
        x = ((q if x & 1 == 0 else (p - q) % p) - c[r] - t) % p
        if r and mask is not None:
            v = x ^ mask
            x = v if v < p else x
    return x


def _int(words: Sequence[int]) -> int:
    return sum(w << (16 * i) for i, w in enumerate(words))


def _bytes_of(words: Sequence[int]) -> bytes:
    return b"".join(struct.pack("<H", w) for w in words)


def _u16s(x: int | bytes, n: int | None = None) -> list[int]:
    if isinstance(x, bytes):
        return list(struct.unpack(f"<{len(x) // 2}H", x))
    return [(x >> (16 * i)) & 0xFFFF for i in range(n)]


def decode_check(S, a: dict) -> int:
    """`BlockDecodeCheck`'s reference on an instance's u16 words (``a[param]``, as `decode_args` gives them), its public
    field elements below p."""
    _check_decode(S)
    p, w, fw = _p(S), S.N // 8, S.N // 16
    block, payload = _bytes_of(a["block"]), _bytes_of(a["payload"])
    ys = [int.from_bytes(block[w * i:w * (i + 1)], "little") for i in range(_words_of(S))]
    ms = [int.from_bytes(payload[w * i:w * (i + 1)], "little") for i in range(_words_of(S))]
    tau, t0 = _int(a["tau"]), _int(a["t0"])
    c = [_int(a["c"][fw * r:fw * (r + 1)]) for r in range(S.K)]
    mask = _int(a["mask"]) if S.MASKED else None
    b = 0
    if a["bound"][0] & 1:
        d = sha512_row_v2_digest(payload, 8 * S.BYTES)
        b = _binding(S, _bytes_of(a["seed"]), _int(a["j"]), d, _bytes_of(a["coin"]))
    return int(all(y < p and _decode_word(S, y, (t0 + i * tau + b) % p, c, mask) == m for i, (y, m) in enumerate(zip(ys, ms))))


def decode_args(S, seed: bytes, j: int, block: bytes, payload: bytes, coin: bytes | None = None) -> dict[str, list[int]]:
    """An instance's u16 words: block j's encoding and its payload, under the codec key of ``seed``, bound under its append's
    ``coin`` if one is given (an appended block), else the seed's key (a registered block)."""
    _check_decode(S)
    if len(block) != S.BYTES or len(payload) != S.BYTES or not 0 <= j < 1 << 64:
        raise ValueError(f"a block and its payload are {S.BYTES} bytes and j a u64")
    if coin is not None and (type(coin) is not bytes or len(coin) != COIN_BYTES):
        raise ValueError(f"an append's coin is {COIN_BYTES} bytes")
    c, tau, mask = key(S, seed)
    fw = S.N // 16
    out = {"j": _u16s(j, 4), "bound": [int(coin is not None)], "seed": _u16s(seed), "coin": _u16s(coin or bytes(COIN_BYTES)),
           "tau": _u16s(tau, fw), "t0": _u16s(tau * _words_of(S) * j % _p(S), fw), "c": [x for r in c for x in _u16s(r, fw)]}
    if S.MASKED:
        out["mask"] = _u16s(mask, fw)
    out.update(block=_u16s(block), payload=_u16s(payload))
    return out


def sketch(block: bytes, signs: Sequence[bytes]) -> bytes:
    """sigma: per sign row, sum_u (-1)^(its bit u) int8(block[u]), as int32 little-endian (`receipts.SketchReceipter.sketch`)."""
    e = [b - 256 if b > 127 else b for b in block]
    out = []
    for row in signs:
        bits = int.from_bytes(row, "little")
        out.append(sum(-x if (bits >> u) & 1 else x for u, x in enumerate(e)))
    return struct.pack(f"<{len(out)}i", *out)


def receipt(prefix: bytes, j: int, block: bytes, signs: Sequence[bytes]) -> bytes:
    """y = SHA-512(prefix || u64be j || sigma) (`receipts.SketchReceipter`, ``prefix`` its `sketch_prefix`)."""
    return hashlib.sha512(prefix + j.to_bytes(8, "big") + sketch(block, signs)).digest()


def _digest_from(mid: Sequence[int], tail: bytes) -> bytes:
    msg = tail + pad_tail(128 + len(tail))
    cv = list(mid)
    for i in range(0, len(msg), 128):
        cv = compress_ref(cv, words_be(msg[i:i + 128]))
    return b"".join(v.to_bytes(8, "big") for v in cv)


def receipt_check(S, a: dict) -> int:
    """`BlockReceiptCheck`'s reference on an instance's u16 words (``a[param]``, as `receipt_args` gives them)."""
    _check_receipt(S)
    mid = [_int(a["mid"][4 * k:4 * (k + 1)]) for k in range(8)]
    j, k, pos = _int(a["j"]), a["k"][0], a["pos"][0]
    rw = S.BYTES // 16
    signs = [_bytes_of(a["signs"][rw * c:rw * (c + 1)]) for c in range(S.ROWS)]
    y = _digest_from(mid, j.to_bytes(8, "big") + sketch(_bytes_of(a["block"]), signs))
    leaf = _bytes_of(a["leaf"])
    return int(k < S.LEAF and leaf[64 * k:64 * (k + 1)] == y and pos < S.P and j < 1 << 16 and a["plan"][pos] == j)


def receipt_args(S, prefix: bytes, signs: Sequence[bytes], j: int, k: int, pos: int, block: bytes, leaf: bytes,
                 plan: Sequence[int]) -> dict[str, list[int]]:
    """An instance's u16 words: the pass's one-block receipt ``prefix`` (`receipts.sketch_prefix`) and sign rows
    (`receipts.sign_bits`), block j at position ``pos`` of the plan, ``k`` its place in its receipt leaf ``leaf`` (LEAF
    receipts), the block's encoding and the step's plan row (P entries)."""
    _check_receipt(S)
    if len(prefix) != 128 or len(signs) != S.ROWS or any(len(r) != S.BYTES // 8 for r in signs):
        raise ValueError(f"a prefix is one block and the signs {S.ROWS} rows of {S.BYTES // 8} bytes")
    if len(block) != S.BYTES or len(leaf) != 64 * S.LEAF or len(plan) != S.P:
        raise ValueError(f"a block is {S.BYTES} bytes, its leaf {S.LEAF} receipts and the plan {S.P} entries")
    if not (0 <= j < 1 << 64 and 0 <= k < 1 << 16 and 0 <= pos < 1 << 16):
        raise ValueError("j is a u64, k and pos u16s")
    return {"mid": [x for v in midstate(prefix) for x in _u16s(v, 4)], "j": _u16s(j, 4), "k": [k], "pos": [pos],
            "signs": [x for r in signs for x in _u16s(r)], "block": _u16s(block), "leaf": _u16s(leaf), "plan": list(plan)}


# --- each instance's rows ----------------------------------------------------------------------------------------------------

def decode_rows(S) -> list[tuple[str, int, list[str]]]:
    """BlockDecodeCheck's rows: (row, bits, params), the public row first."""
    _check_decode(S)
    pub = ["j", "bound", "seed", "coin", "tau", "t0", "c"] + (["mask"] if S.MASKED else [])
    bits = 16 * (4 + 1 + 16 + 16) + S.N * (2 + S.K + int(bool(S.MASKED)))
    return [("pub", bits, pub), ("block", 8 * S.BYTES, ["block"]), ("payload", 8 * S.BYTES, ["payload"])]


def receipt_rows(S) -> list[tuple[str, int, list[str]]]:
    """BlockReceiptCheck's rows: (row, bits, params), the public rows (the pass's chaining value and indices, its sign rows)
    first."""
    _check_receipt(S)
    return [("pub", 16 * (32 + 4 + 2), ["mid", "j", "k", "pos"]), ("signs", S.ROWS * S.BYTES, ["signs"]),
            ("block", 8 * S.BYTES, ["block"]), ("leaf", 512 * S.LEAF, ["leaf"]), ("plan", 16 * S.P, ["plan"])]


def row_words(rows: Sequence[tuple[str, int, list[str]]], args: dict) -> dict[str, list[int]]:
    """Each row's u16 words from an instance's arguments, its params' words in order."""
    out = {r: [w for p in ps for w in args[p]] for r, _bits, ps in rows}
    if any(16 * len(out[r]) != bits for r, bits, _ in rows):
        raise AssertionError("a row's words are not its bits")
    return out
