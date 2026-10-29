"""The remote bandwidth audit's opening of a drawn receipt (`verity.core.protocols.remote_bandwidth`, PROTOCOL.md §6's checks
1-3 and 5, as §8 deploys them) as one Boolean Definition: what a proof runs on a drawn receipt in place of opening it, so the
block, its payload, its receipt leaf and every salt and path stay the unit's witness and the verifier reads the unit's
verdict.

    BlockOpeningCheck_v1{N, C, K, MASKED, BYTES, ROWS, LEAF, P, DR, DP, DV}
        (j: Array<4, U16>, bound: U16, seed: Array<16, U16>, coin: Array<16, U16>, tau: F, t0: F, c: Array<K, F>, [mask: F,]
         mid: Array<32, U16>, pos: U16, rroot: H, rdom: H, rdepth: U16, proot: H, pdom: H, prank: Array<2, U16>, pdepth: U16,
         vroot: H, vdom: H, vrank: Array<2, U16>, vdepth: U16,
         signs: Array<ROWS, Array<BYTES / 16, U16>>, block: Array<BYTES / 2, U16>, payload: Array<BYTES / 2, U16>,
         leaf: Array<LEAF, Array<32, U16>>, plan: Array<P, U16>, salts: Array<2, Array<96, U16>>,
         rpath: Array<DR, H>, ppath: Array<DP, H>, vpath: Array<DV, H>) -> U16
        U16 = Array<16, Value<1>>, F = Array<N / 16, U16> and H = Array<32, U16> (a 64-byte digest): an integer
        little-endian, a byte string in order (bit t is bit t & 7 of byte t >> 3); ``mid`` is SHA-512's eight words, each
        64 bits LSB first

**Public** (the verifier's own, `opening_public`), each of the four kinds the rules allow: the sample (block j, its position
pos in the pass, whose leaf is pos >> 4 and whose receipt is entry pos & 15 of it) and the registration's codec key at j
(the seed's codec key, `codec.key`: c, tau and mask; t0 = tau W j mod p for the block's W words; the seed itself), the
append's coin c_a (``bound``, an appended block; zero bytes and unbound for a registered one), the pass's receipt prefix
``mid`` (the chaining value after `receipts.sketch_prefix` of its seed) and sign rows ``signs`` (`receipts.sign_bits`), and
three roots, each with its tree's domain id and depth (`merkle.merkle_depth` of its count): the pass's receipt root
(``rroot``), the payload's (``proot``: the registration's payload root, or the root of the append holding j, with rank
``prank`` = j - first), and, for an appended block, the value commitment's (``vroot``, its rank and position ``vrank``).
**Witness:** the block, its payload, the receipt leaf (LEAF receipts), the step's certified plan row (P entries), the receipt
leaf's and the payload's salts (192 bytes each) and their paths, each padded with zero digests to its tree's D.

**What it proves**, each check a flag of the verdict:

- *receipt-path* (check 2): the receipt leaf, hidden under ``salts[0]`` (`gates.hm96` over its sha512/row/v2 digest,
  `receipts.hidden`), opens at rank and position pos >> 4 to ``rroot`` in the tree of domain id ``rdom`` (the pass's seed
  and plan, `receipts.domain`) at depth ``rdepth``; and the plan row holds j at pos: plan[pos] = j (pos < P, j < 2^16).
- *codec* (check 3): every word w of ``block`` (bytes [S w, S (w + 1)), S = N / 8, of its W = BYTES / S) is below
  p = 2^N - C and decodes to ``payload``'s word w (`codec.decode_word`): x -> rho^-1(x) - c_r - t_w mod p for r = K - 1
  ... 0, and, ``MASKED``, x -> sigma_K(x) after each round r >= 1, with t_w = t0 + w tau + b mod p, b = 0 unless
  ``bound``'s bit 0, then b_j(c_a): LE(SHA-512(DOMAIN || u8 len(name) || name || seed || u32be 2^32 - 4 || u64be j || d ||
  coin || u32be 0)) mod p, d = sha512/row/v2 of the payload (`codec.binding`, `codec.digest`). So the block is
  ``codec.decode(block, j, key, d, coin) == payload`` when bound, and ``codec.decode(block, j, key) == payload`` when not.
- *payload-path* (check 3): the payload, hidden under ``salts[1]`` over the same d, opens at rank ``prank`` and position j
  to ``proot`` in the tree of domain id ``pdom`` at depth ``pdepth``; and, when bound, at rank and position ``vrank`` to the
  value commitment ``vroot`` in ``vdom`` at ``vdepth``: the payload is the committed KV the circuit wrote to block j, the
  payload row the memory-map unit holds to the step's values under the layout (its ``pay`` row at the place
  `requirement_units.map_value_leaves` names block j).
- *receipt* (check 5): the sketch-sha512 receipt of the block (`receipts.SketchReceipter`): y = SHA-512(prefix || u64be j ||
  sigma), its chaining value from ``mid``, sigma ROWS int32 little-endian, sigma_r = sum_u (-1)^(signs[r][u]) int8(block[u]),
  is ``leaf[pos & 15]`` (pos & 15 < LEAF).

Check 1 is the unit's shape: every port has its fixed size, and the public row is the verifier's. The verdict word is bit 0
the four checks' AND (accepted) and bits 1-4 each check's failure in that order (`VERDICT`), bits 5-15 zero, within the 16-bit
size class, the least public one; `verdict` reads it as §6's names, None for accepted or the first check that failed. A field's name is kv/v0's
(p512 = 2^512 - 569, p1024 = 2^1024 - 105); ``toy32`` (2^32 - 5) is circuit-check's and the tests' binding with a short block
(``BYTES`` below 4,096), the same circuit on small words, never deployed.

**The plan link.** The plan row is the one witness row the verifier holds outside the unit: a row's commitment is
hm96-sha512/row/v2, so under the salt of another commitment of the same words it has that commitment's ``b || c``, and the
verifier holds the plan row's to the certified plan row of the pass's step (the plan unit's) by ``b || c`` alone. The other
rows' ``b || c`` are hiding and nothing reads them.

**The parts**, each its own traced Definition lowered by its builder (``build``), so a Flock unit gets the builder's
commits and the Definition the trace's ANDs:

    RowCompress_v1                  (cv: Array<8, Word>, block: Array<16, Word>) -> Array<8, Word>, one SHA-512 compression
    CodecBinding_v1{N, C}           (bound, seed: Array<256>, j: Array<64>, d: Array<8, Word>, coin: Array<256>) -> Array<N>
    CodecModAdd_v1{N, C}            (a, b) -> a + b mod p
    CodecLess_v1{N, C}              (y) -> y < p
    CodecRound_v1{N, C, MASK}       (x, c, t[, mask]) -> rho^-1(x) - c - t mod p, then sigma_K if MASK
    CodecMatch_v1{N}                (x, y, a) -> a and x = y
    CodecWord_v1{N, C, K, MASKED}   (y, t, c: Array<K, Array<N>>[, mask], want) -> y < p and its decode is want
    SketchRow_v1{BYTES}             (block: Array<8 BYTES>, signs: Array<BYTES>) -> Array<32>, one row of sigma
    LeafMatch_v1{LEAF, P}           (y: Array<8, Word>, leaf, k: U16, plan, pos: U16, j: Array<64>) -> Array<2>
    BlockVerdict_v1{W}              (oks: Array<W>, match: Array<2>, reads: Array<3>, bound) -> U16
    Hm96Sha512Leaf_v1, FrameV3ReadAt_v1{D, 4, hm96-sha512/row/v2}   `gates.hm96`, `gates.frame_v3`
    Word = Array<64, Value<1>>, a SHA-512 word LSB first; Array<n> = Array<n, Value<1>>

**Cost.** A squaring mod p is a Karatsuba square down to 32-bit schoolbook squares (about 96,000 ANDs at N = 512) and two
folds of the high half by C; a round adds about 9 N for its additions, negation and selections. The decode is W K rounds and
2 + ceil((BYTES + 17) / 128) compressions (about 15.6 M ANDs at p512/k2, 13.4 M of rounds; about 32 M at p1024/k3/masked);
the receipt ROWS sketch rows of about 9 BYTES ANDs each and one compression (0.66 M at 16 rows); the receipt leaf's digest
ceil((64 LEAF + 17) / 128) compressions, each hm96 leaf four, each read's leaf two and each level three, so DR + DP + DV
levels. The Flock lowering adds about half again, ANDs on identical operands (circuit-check's redundant-gates/boolean).
At A7's binding (p512/k2 unmasked, 4,096-byte blocks, 16 rows, LEAF 16, P 48, DR 2, DP 5, DV 6) the unit lowers to
29,142,550 ANDs, k_log 26, within Flock's K_MAX of 27: checks 1-3 and 5 are one unit. p1024/k3/masked would be past K_MAX.

`opening_check` is the reference, over an instance's u16 words; `opening_args` an instance's arguments, `opening_public` its
public rows' (the verifier's own), `opening_rows` its rows (Flock's ``in_rows`` params), `row_words` its rows' words and
`verdict` the name its verdict word says.
"""
from __future__ import annotations

import hashlib
import struct
from typing import Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.boolean.forms import ONE, ZERO, OR, and_reduce, const_bits, maj, mux, or_reduce
from verity.ml.circuits.boolean.trace import part as _part
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.refs import Coll
from verity.ml.circuits.types import Array
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2, sha512_row_v2_digest, sha512_row_v2_prefix

from .frame_v3 import opens_at, read_at
from .hm96 import Hm96Sha512Leaf, reference as hm96_leaf
from .sha512 import IV, commit as _commit, compress_gadget, compress_ref, midstate, pad_tail, words_be

__all__ = ["BlockOpeningCheck", "FIELDS", "TOY", "VERDICT", "key", "opening_args", "opening_check", "opening_public",
           "opening_rows", "receipt", "row_words", "sketch", "verdict"]

BIT = BL.BIT
U16 = Array(16, BIT)
WORD = Array(64, BIT)
DIGEST = Array(8, WORD)
H = Array(32, U16)
#: (N, C) -> the field's name in the codec's derivations (`codec.Field.name`)
FIELDS = {(1024, 105): "p1024", (512, 569): "p512", (32, 5): "toy32"}
TOY = (32, 5)
BLOCK_BYTES = 4096
DOMAIN = b"verity/word-codec/kv/v0\0"
TWEAK_INDEX, MASK_INDEX, COIN_INDEX = (1 << 32) - 1, (1 << 32) - 2, (1 << 32) - 4
SEED_BYTES = COIN_BYTES = 32
SALT_BYTES = 192
#: a receipt leaf holds this many receipts: position pos is entry pos & 15 of leaf pos >> 4 (`receipts.PER_LEAF`)
PER_LEAF = 16
#: a read's position is a u32 (uint 1 ... 4 bytes)
POSITION_BYTES = 4
#: the verdict word's bits 1 ... 4: each check's failure, in the order §6 checks them (`audit.FAILURES`' names)
VERDICT = ("receipt-path", "codec", "payload-path", "receipt")
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


def _check_opening(S) -> None:
    _check_field(S)
    _check_bytes(S, True)
    if isinstance(S.K, bool) or not isinstance(S.K, int) or not 2 <= S.K <= 8 or S.MASKED not in (0, 1):
        raise ValueError(f"K is 2 to 8 rounds and MASKED 0 or 1, not K = {S.K!r}, MASKED = {S.MASKED!r}")
    # the receipt leaf's rank is pos >> 4 of a u16 pos; the payload's and value's ranks are u32s whose uint is at most 3 bytes
    _check_ints(S, ("ROWS", 64), ("LEAF", PER_LEAF), ("P", (1 << 16) - 1), ("DR", 12), ("DP", 24), ("DV", 24))


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
# Integers are bit lists LSB first. A long form is committed (`sha512.commit`): a row of its own in Flock's lowering, and in a
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
    _check_ints(S, ("LEAF", PER_LEAF), ("P", (1 << 16) - 1))
    return ((("y", DIGEST), ("leaf", Array(S.LEAF, Array(512, BIT))), ("k", U16), ("plan", Array(S.P, U16)), ("pos", U16),
             ("j", Array(64, BIT))), Array(2, BIT))


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
    return [C.AND(or_reduce(C, hot), _eq(C, picked, y)),
            and_reduce(C, [or_reduce(C, at), _eq(C, entry, j[:16]), or_reduce(C, j[16:]).inv()])]


LeafMatch = _part(
    "LeafMatch", ("LEAF", "P"), _sig_leaf, _build_leaf,
    "The receipt's two matches: bit 0 y = leaf[k] (k < LEAF), bit 1 plan[pos] = j (pos < P, j below 2^16)")


def _sig_verdict(S):
    _check_ints(S, ("W", BLOCK_BYTES))
    return (("oks", Array(S.W, BIT)), ("match", Array(2, BIT)), ("reads", Array(3, BIT)), ("bound", BIT)), U16


def _build_verdict(C, S, oks, match, reads, bound):
    ok = {"receipt-path": C.AND(reads[0], match[1]), "codec": and_reduce(C, list(oks)),
          "payload-path": C.AND(reads[1], OR(C, reads[2], bound[0].inv())), "receipt": match[0]}
    flags = [ok[n] for n in VERDICT]
    return [and_reduce(C, flags)] + [f.inv() for f in flags] + [ZERO] * (15 - len(flags))


BlockVerdict = _part(
    "BlockVerdict", ("W",), _sig_verdict, _build_verdict,
    "The opening's verdict word: bit 0 the AND of its four checks, bits 1-4 each one's failure (receipt-path: the receipt "
    "leaf's read and plan[pos] = j; codec: the W words' decodes; payload-path: the payload's read and, if bound, the value "
    "commitment's; receipt: y = leaf[k]), bits 5-15 zero")


# --- the unit ----------------------------------------------------------------------------------------------------------------

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


PUBLIC = ("j", "bound", "seed", "coin", "tau", "t0", "c", "mask", "mid", "pos", "rroot", "rdom", "rdepth", "proot", "pdom", "prank",
          "pdepth", "vroot", "vdom", "vrank", "vdepth")
AUX = ("salts", "rpath", "ppath", "vpath")


def _opening_signature(S):
    _check_opening(S)
    F = _uwords(S.N)
    pub = (("j", Array(4, U16)), ("bound", U16), ("seed", Array(16, U16)), ("coin", Array(16, U16)), ("tau", F), ("t0", F),
           ("c", Array(S.K, F))) + ((("mask", F),) if S.MASKED else ()) + (
        ("mid", Array(32, U16)), ("pos", U16), ("rroot", H), ("rdom", H), ("rdepth", U16), ("proot", H), ("pdom", H),
        ("prank", Array(2, U16)), ("pdepth", U16), ("vroot", H), ("vdom", H), ("vrank", Array(2, U16)), ("vdepth", U16))
    wit = (("signs", Array(S.ROWS, _uwords(S.BYTES))), ("block", _uwords(8 * S.BYTES)), ("payload", _uwords(8 * S.BYTES)),
           ("leaf", Array(S.LEAF, H)), ("plan", Array(S.P, U16)), ("salts", Array(2, _uwords(8 * SALT_BYTES))),
           ("rpath", Array(S.DR, H)), ("ppath", Array(S.DP, H)), ("vpath", Array(S.DV, H)))
    return pub + wit, U16


def _compress(B, X: _Wires, cv: Coll, data: list[list]) -> Coll:
    words = _words(data)
    for i in range(0, len(words), 16):
        cv = B.call(RowCompress, cv, X.coll(Array(16, WORD), _join(words[i:i + 16])))
    return cv


def _row_digest(B, X: _Wires, bits: list) -> Coll:
    """sha512/row/v2 of a row of whole bytes, from its one-block prefix's constant chaining value."""
    prefix = sha512_row_v2_prefix(len(bits))
    cv = X.coll(DIGEST, [b for v in midstate(prefix) for b in X.const(v, 64)])
    return _compress(B, X, cv, _octets(bits) + [X.const(b, 8) for b in pad_tail(len(prefix) + len(bits) // 8)])


def _digests(bits: list, n: int) -> list:
    """n byte strings of 64 bytes each (bits in byte order) as SHA-512 words."""
    return [x for i in range(n) for x in _join(_words(_octets(bits[512 * i:512 * (i + 1)])))]


def _salt(X: _Wires, bits: list) -> Coll:
    return X.coll(Array(SALT_BYTES // 8, WORD), _join(_words(_octets(bits))))


def _opening_body(B, S, *colls):
    params, _ret = _opening_signature(S)
    a = dict(zip([n for n, _ in params], colls))
    X = _Wires(B)
    F, W = _f(S), _words_of(S)
    pay, jb, pos = X.refs(a["payload"]), X.refs(a["j"]), X.refs(a["pos"])
    j64, zero = X.coll(Array(64, BIT), jb), X.bit[0]

    # check 3: the payload's digest, its block's binding offset and every word's decode
    d = _row_digest(B, X, pay)
    b = B.call(bind(CodecBinding, N=S.N, C=S.C), a["bound"][0], X.coll(Array(256, BIT), X.refs(a["seed"])), j64, d,
               X.coll(Array(256, BIT), X.refs(a["coin"])))
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

    # check 5 and the plan entry: the block's sketch receipt from the pass's prefix, entry pos & 15 of the leaf; plan[pos] = j
    sig = X.refs(B.batch(bind(SketchRow, BYTES=S.BYTES), X.coll(Array(8 * S.BYTES, BIT), X.refs(a["block"])),
                         X.coll(Array(S.ROWS, Array(S.BYTES, BIT)), X.refs(a["signs"])), axes=(None, 0)))
    data = _u64be(jb) + _octets(sig)
    data += [X.const(b_, 8) for b_ in pad_tail(128 + len(data))]
    y = _compress(B, X, X.coll(DIGEST, X.refs(a["mid"])), data)
    leaf = X.refs(a["leaf"])
    match = B.call(bind(LeafMatch, LEAF=S.LEAF, P=S.P), y, X.coll(Array(S.LEAF, Array(512, BIT)), leaf),
                   X.coll(U16, pos[:4] + [zero] * 12), X.coll(Array(S.P, U16), X.refs(a["plan"])), a["pos"], j64)

    # check 2: the receipt leaf, hidden under its salt, on its path to the pass's root at rank and position pos >> 4
    salts = X.refs(a["salts"])
    hl = X.refs(B.call(Hm96Sha512Leaf, _row_digest(B, X, leaf), _salt(X, salts[:8 * SALT_BYTES])))

    def read(D, root, dom, value, rank, position, depth, path):
        nb = 8 * POSITION_BYTES
        return B.call(read_at(D, POSITION_BYTES, V2), X.coll(DIGEST, _digests(X.refs(a[root]), 1)),
                      X.coll(DIGEST, _digests(X.refs(a[dom]), 1)), value, X.coll(Array(D, BIT), (rank + [zero] * D)[:D]),
                      X.coll(Array(nb, BIT), (position + [zero] * nb)[:nb]), a[depth],
                      X.coll(Array(D, DIGEST), _digests(X.refs(a[path]), D)))

    r = read(S.DR, "rroot", "rdom", X.coll(DIGEST, hl[1024:]), pos[4:], pos[4:], "rdepth", "rpath")

    # check 3: the payload, hidden under its salt, on its path to the payload root (or its append's) at its rank and position
    # j, and, bound, to the value commitment at its rank and position
    hp = X.coll(DIGEST, X.refs(B.call(Hm96Sha512Leaf, d, _salt(X, salts[8 * SALT_BYTES:])))[1024:])
    p = read(S.DP, "proot", "pdom", hp, X.refs(a["prank"]), jb, "pdepth", "ppath")
    vr = X.refs(a["vrank"])
    v = read(S.DV, "vroot", "vdom", hp, vr, vr, "vdepth", "vpath")
    return B.call(bind(BlockVerdict, W=W), oks, match, X.coll(Array(3, BIT), X.refs(r) + X.refs(p) + X.refs(v)), a["bound"][0])


BlockOpeningCheck = CompositeDefinition(
    "BlockOpeningCheck", 1, ("N", "C", "K", "MASKED", "BYTES", "ROWS", "LEAF", "P", "DR", "DP", "DV"), _opening_signature,
    _opening_body, conformance="exact-model-tested",
    doc="A drawn receipt's opening (remote_bandwidth §6 checks 1-3 and 5): the receipt leaf on its path to the pass's root "
        "and plan[pos] = j (receipt-path); every word of block below p and decoding to payload under the codec key at j, with "
        "b_j(c_a) of the payload's own digest and coin if bound (codec); the payload on its path to the payload root or its "
        "append's and, if bound, to the value commitment (payload-path); the block's sketch receipt the leaf's entry pos & 15 "
        "(receipt). Returns bit 0 accepted, bits 1-4 each check's failure")


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
    _check_field(S)
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


def _paths(words: Sequence[int], n: int) -> list[bytes]:
    raw = _bytes_of(words)
    return [raw[64 * i:64 * (i + 1)] for i in range(n)]


def opening_check(S, a: dict) -> int:
    """`BlockOpeningCheck`'s reference on an instance's u16 words (``a[param]``, as `opening_args` gives them), its public
    field elements below p: the verdict word."""
    _check_opening(S)
    p, w, fw, W = _p(S), S.N // 8, S.N // 16, _words_of(S)
    block, payload = _bytes_of(a["block"]), _bytes_of(a["payload"])
    j, pos, bound = _int(a["j"]), a["pos"][0], a["bound"][0] & 1
    ys = [int.from_bytes(block[w * i:w * (i + 1)], "little") for i in range(W)]
    ms = [int.from_bytes(payload[w * i:w * (i + 1)], "little") for i in range(W)]
    tau, t0 = _int(a["tau"]), _int(a["t0"])
    c = [_int(a["c"][fw * r:fw * (r + 1)]) for r in range(S.K)]
    mask = _int(a["mask"]) if S.MASKED else None
    d = sha512_row_v2_digest(payload, 8 * S.BYTES)
    b = _binding(S, _bytes_of(a["seed"]), j, d, _bytes_of(a["coin"])) if bound else 0
    codec = all(y < p and _decode_word(S, y, (t0 + i * tau + b) % p, c, mask) == m for i, (y, m) in enumerate(zip(ys, ms)))

    mid = [_int(a["mid"][4 * k:4 * (k + 1)]) for k in range(8)]
    rw = S.BYTES // 16
    signs = [_bytes_of(a["signs"][rw * r:rw * (r + 1)]) for r in range(S.ROWS)]
    y = _digest_from(mid, j.to_bytes(8, "big") + sketch(block, signs))
    leaf, k = _bytes_of(a["leaf"]), pos % PER_LEAF
    rec = k < S.LEAF and leaf[64 * k:64 * (k + 1)] == y
    plan = pos < S.P and j < 1 << 16 and a["plan"][pos] == j

    salts = _bytes_of(a["salts"])
    aux = {"r": _paths(a["rpath"], S.DR), "p": _paths(a["ppath"], S.DP), "v": _paths(a["vpath"], S.DV)}
    lv = hm96_leaf(sha512_row_v2_digest(leaf, 512 * S.LEAF), salts[:SALT_BYTES])[2]
    pv = hm96_leaf(d, salts[SALT_BYTES:])[2]
    top = (1 << 8 * POSITION_BYTES) - 1

    def at(root: str, dom: str, value: bytes, rank: int, position: int, depth: str, path: list[bytes], D: int) -> bool:
        return bool(opens_at(V2, _bytes_of(a[root]), _bytes_of(a[dom]), value, rank % (1 << D), position & top, a[depth][0], path))

    r = at("rroot", "rdom", lv, pos >> 4, pos >> 4, "rdepth", aux["r"], S.DR)
    pp = at("proot", "pdom", pv, _int(a["prank"]), j, "pdepth", aux["p"], S.DP)
    vv = at("vroot", "vdom", pv, _int(a["vrank"]), _int(a["vrank"]), "vdepth", aux["v"], S.DV)
    ok = {"receipt-path": r and plan, "codec": codec, "payload-path": pp and (vv or not bound), "receipt": rec}
    flags = [ok[n] for n in VERDICT]
    return int(all(flags)) | sum((not f) << (1 + i) for i, f in enumerate(flags))


def verdict(word: int) -> str | None:
    """The name the verdict word says: None (accepted, word 1) or the first failed check's (`VERDICT`); "malformed" for a word
    no unit returns."""
    if word == 1:
        return None
    return next((n for i, n in enumerate(VERDICT) if (word >> (1 + i)) & 1), "malformed")


def _tree(t: tuple[bytes, bytes, int, int] | None, D: int, name: str) -> dict[str, list[int]]:
    root, dom, rank, depth = t if t is not None else (bytes(64), bytes(64), 0, 0)
    if len(root) != 64 or len(dom) != 64 or not 0 <= rank < 1 << 32 or not 0 <= depth <= D:
        raise ValueError(f"the {name} tree is a 64-byte root and domain id, a u32 rank and a depth of at most {D}")
    return {"root": _u16s(root), "dom": _u16s(dom), "rank": _u16s(rank, 2), "depth": [depth]}


def opening_public(S, seed: bytes, j: int, coin: bytes | None, prefix: bytes, signs: Sequence[bytes], pos: int,
                   receipts: tuple[bytes, bytes, int], payloads: tuple[bytes, bytes, int, int],
                   values: tuple[bytes, bytes, int, int] | None) -> dict[str, list[int]]:
    """The public rows' words (`opening_rows`' first two), which the verifier derives itself: block j's codec key under the
    registration's codec ``seed``, bound under its append's ``coin`` if one is given (an appended block), else the seed's
    key (a registered block); the pass's one-block receipt ``prefix`` (`receipts.sketch_prefix`) and sign rows
    (`receipts.sign_bits`), position ``pos`` of its plan; and the trees: ``receipts`` the pass's (root, domain id, depth),
    ``payloads`` the payload's (root, domain id, rank of j, depth), ``values`` the value commitment's (root, domain id, rank
    and position, depth; None, read as zeros, for a registered block)."""
    _check_opening(S)
    if not 0 <= j < 1 << (8 * POSITION_BYTES) or not 0 <= pos < 1 << 16:
        raise ValueError(f"j is below 2^{8 * POSITION_BYTES} and pos a u16")
    if coin is not None and (type(coin) is not bytes or len(coin) != COIN_BYTES):
        raise ValueError(f"an append's coin is {COIN_BYTES} bytes")
    if len(prefix) != 128 or len(signs) != S.ROWS or any(len(r) != S.BYTES // 8 for r in signs):
        raise ValueError(f"a prefix is one block and the signs {S.ROWS} rows of {S.BYTES // 8} bytes")
    c, tau, mask = key(S, seed)
    fw = S.N // 16
    out = {"j": _u16s(j, 4), "bound": [int(coin is not None)], "seed": _u16s(seed), "coin": _u16s(coin or bytes(COIN_BYTES)),
           "tau": _u16s(tau, fw), "t0": _u16s(tau * _words_of(S) * j % _p(S), fw), "c": [x for r in c for x in _u16s(r, fw)]}
    if S.MASKED:
        out["mask"] = _u16s(mask, fw)
    out["mid"] = [x for v in midstate(prefix) for x in _u16s(v, 4)]
    out["pos"] = [pos]
    r = _tree((receipts[0], receipts[1], pos >> 4, receipts[2]), S.DR, "receipt")
    out.update(rroot=r["root"], rdom=r["dom"], rdepth=r["depth"])
    t = _tree(payloads, S.DP, "payload")
    out.update(proot=t["root"], pdom=t["dom"], prank=t["rank"], pdepth=t["depth"])
    t = _tree(values, S.DV, "value")
    out.update(vroot=t["root"], vdom=t["dom"], vrank=t["rank"], vdepth=t["depth"])
    out["signs"] = [x for r_ in signs for x in _u16s(r_)]
    return out


def opening_args(S, public: dict[str, list[int]], block: bytes, payload: bytes, leaf: bytes, plan: Sequence[int],
                 leaf_salt: bytes, payload_salt: bytes, rpath: Sequence[bytes], ppath: Sequence[bytes],
                 vpath: Sequence[bytes]) -> dict[str, list[int]]:
    """An instance's u16 words: `opening_public`'s public rows, the block's encoding and its payload, its receipt leaf
    (LEAF receipts), the step's plan row (P entries), the receipt leaf's and the payload's salts, and the three paths (each
    at most its tree's D siblings, padded with zero digests)."""
    _check_opening(S)
    if len(block) != S.BYTES or len(payload) != S.BYTES or len(leaf) != 64 * S.LEAF or len(plan) != S.P:
        raise ValueError(f"a block and its payload are {S.BYTES} bytes, the leaf {S.LEAF} receipts and the plan {S.P} entries")
    if len(leaf_salt) != SALT_BYTES or len(payload_salt) != SALT_BYTES:
        raise ValueError(f"a salt is {SALT_BYTES} bytes")
    out = dict(public, block=_u16s(block), payload=_u16s(payload), leaf=_u16s(leaf), plan=list(plan),
               salts=_u16s(leaf_salt + payload_salt))
    for name, path, D in (("rpath", rpath, S.DR), ("ppath", ppath, S.DP), ("vpath", vpath, S.DV)):
        if len(path) > D or any(len(x) != 64 for x in path):
            raise ValueError(f"{name} is at most {D} digests of 64 bytes, not {len(path)}")
        out[name] = _u16s(b"".join(path) + bytes(64 * (D - len(path))))
    return out


# --- each instance's rows ----------------------------------------------------------------------------------------------------

def opening_rows(S) -> list[tuple[str, int, list[str]]]:
    """BlockOpeningCheck's rows: (row, bits, params), the public rows (its public data, the pass's sign rows) first."""
    params, _ = _opening_signature(S)
    bits = {n: t.leaves for n, t in params}
    pub = [n for n in PUBLIC if n in bits]
    return ([("pub", sum(bits[n] for n in pub), pub), ("signs", bits["signs"], ["signs"])]
            + [(n, bits[n], [n]) for n in ("block", "payload", "leaf", "plan")]
            + [("aux", sum(bits[n] for n in AUX), list(AUX))])


def row_words(rows: Sequence[tuple[str, int, list[str]]], args: dict) -> dict[str, list[int]]:
    """Each row's u16 words from an instance's arguments, its params' words in order."""
    out = {r: [w for p in ps for w in args[p]] for r, _bits, ps in rows}
    if any(16 * len(out[r]) != bits for r, bits, _ in rows):
        raise AssertionError("a row's words are not its bits")
    return out
