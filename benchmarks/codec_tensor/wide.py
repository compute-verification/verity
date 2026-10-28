"""Sloth over one wide word, the tensor-core codec's byte-exact reference.

~~~text
field      p = 2^n − c, p ≡ 3 (mod 4), c the smallest such that p is prime: n = 256U − 1 for U = 1 … 5 (a word fills U
           32-byte chunks, the tensor-core squaring's unit; 2^255 − 19 is 1 mod 4, so p255 = 2^255 − 765), m521, and
           the full-slot p512 = 2^512 − 569, p768 = 2^768 − 825, p1024 = 2^1024 − 105 (the word codec's p1024),
           p1536 = 2^1536 − 3453 and p2048 = 2^2048 − 1557
word       S = ⌈n/8⌉ bytes, little-endian, canonical (< p): 32U, 66 for m521, n/8 for the full-slot fields
payload    B = ⌊(n − 1)/8⌋ bytes, little-endian (m < 2^(8B) < p): 32U − 1, and 65; a full-slot field's payload is the
           whole word, any m < p, which every S FP8 weights are unless the last is 0xFF (an E4M3 NaN)
ρ(a)       Sloth's permutation: s = a^((p+1)/4); if s² = a the even one of {s, p − s}, else (s² = −a) the odd one
ρ⁻¹(y)     y² if y is even, p − y² if y is odd: one squaring and the sign rule
XOF(m)     SHA512(m ‖ u32be 0) ‖ SHA512(m ‖ u32be 1) ‖ …
c_j        LE(XOF(DOMAIN ‖ frame ‖ seed ‖ u32be j)[0 : S + 8]) mod p, j = 0 … k − 1, frame = u8 len(name) ‖ name
τ          the same at j = 2^32 − 1, and 1 in place of 0
t(i)       τ · i mod p, i the word's index in the store
round j    x ↦ ρ(x + c_j + t(i)); encode is rounds 0 … k − 1 from the payload, decode rounds k − 1 … 0 inverted,
           x ↦ ρ⁻¹(x) − c_j − t(i), and a word decodes only if the result is < 2^(8B) (every word, at full slot)
K          a masked key's (`key(..., mask=True)`): LE(XOF(DOMAIN ‖ frame ‖ seed ‖ u32be 2^32 − 2)[0 : S]) mod 2^n, not
           reduced mod p
σ_K(x)     x ⊕ K if that is below p, else x: an involution on [0, p), applied to x before each round j ≥ 1 of a masked
           word's encode and after each inverted round j ≥ 1 of its decode (`benchmarks/word_codec/kv.py`'s)
~~~

The seed is sampled after the weights are registered. Decoding a word costs k squarings; encoding it k roots, each
(p+1)/4's addition chain (n − 2 squarings for a Mersenne prime) and the square test.

The fused GEMM's layout (`encode_tasks`): an FP8 weight matrix W (N × K) in `codec.plain_fragments` order, one task (16
rows × 512 columns, 8,192 bytes) at a time; a task's bytes, zero-padded, are the payloads of L = `TASK_WORDS[name]`
consecutive words (L·S a multiple of 16 so a task is one bulk copy), word w of task q at store index L·q + w. The
task's stored bytes are the block the bandwidth tag (`ro-blake3/v1`) hashes.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

DOMAIN = b"verity/word-codec/wide/v0\0"
SEED_BYTES = 32
TWEAK_INDEX = (1 << 32) - 1
MASK_INDEX = (1 << 32) - 2
TASK_BYTES = 8192


@dataclass(frozen=True, slots=True)
class Field:
    name: str
    n: int
    c: int

    @property
    def p(self) -> int:
        return (1 << self.n) - self.c

    @property
    def word_bytes(self) -> int:
        return -(-self.n // 8)

    @property
    def full_slot(self) -> bool:
        return self.n % 32 == 0

    @property
    def payload_bytes(self) -> int:
        return self.n // 8 if self.full_slot else (self.n - 1) // 8

    @property
    def max_payload(self) -> int:
        return self.p - 1 if self.full_slot else (1 << (8 * self.payload_bytes)) - 1

    @property
    def root_exp(self) -> int:
        return (self.p + 1) // 4


FIELDS = {f.name: f for f in (Field("p255", 255, 765), Field("p511", 511, 481), Field("m521", 521, 1),
                               Field("p767", 767, 1081), Field("p1023", 1023, 361), Field("m1279", 1279, 1),
                               Field("p512", 512, 569), Field("p768", 768, 825), Field("p1024", 1024, 105),
                               Field("p1536", 1536, 3453), Field("p2048", 2048, 1557))}
TASK_WORDS = {"p255": 265, "p511": 131, "m521": 128, "p767": 87, "p1023": 65, "m1279": 52, "p512": 128, "p768": 86,
              "p1024": 64, "p1536": 43, "p2048": 32}


def field(name: str) -> Field:
    try:
        return FIELDS[name]
    except KeyError:
        raise ValueError(f"unknown field {name!r}; known: {sorted(FIELDS)}") from None


def check_fields() -> None:
    for f in FIELDS.values():
        assert f.p % 4 == 3 and f.p.bit_length() == f.n
        if f.full_slot:
            assert f.payload_bytes == f.word_bytes and (1 << (8 * f.word_bytes)) - (1 << (8 * f.word_bytes - 8)) <= f.p
        else:
            assert 1 << (8 * f.payload_bytes) < f.p
        lw = TASK_WORDS[f.name]
        assert (lw * f.word_bytes) % 16 == 0 and lw * f.payload_bytes >= TASK_BYTES


@dataclass(frozen=True, slots=True)
class Key:
    field: Field
    c: tuple[int, ...]
    tau: int
    mask: int | None = None

    @property
    def rounds(self) -> int:
        return len(self.c)

    def tweak(self, i: int) -> int:
        return self.tau * i % self.field.p

    def sigma(self, x: int) -> int:
        v = x ^ self.mask
        return v if v < self.field.p else x


def xof(message: bytes, length: int) -> bytes:
    out = b""
    ctr = 0
    while len(out) < length:
        out += hashlib.sha512(message + struct.pack(">I", ctr)).digest()
        ctr += 1
    return out[:length]


def _message(f: Field, seed: bytes, j: int) -> bytes:
    name = f.name.encode()
    return DOMAIN + bytes([len(name)]) + name + seed + struct.pack(">I", j)


def _element(f: Field, seed: bytes, j: int) -> int:
    return int.from_bytes(xof(_message(f, seed, j), f.word_bytes + 8), "little") % f.p


def key(name: str, seed: bytes, rounds: int, mask: bool = False) -> Key:
    f = field(name)
    if type(seed) is not bytes or len(seed) != SEED_BYTES:
        raise ValueError(f"a seed is {SEED_BYTES} bytes")
    if not 1 <= rounds < MASK_INDEX:
        raise ValueError("a word has at least one round")
    k = int.from_bytes(xof(_message(f, seed, MASK_INDEX), f.word_bytes), "little") % (1 << f.n) if mask else None
    return Key(f, tuple(_element(f, seed, j) for j in range(rounds)), _element(f, seed, TWEAK_INDEX) or 1, k)


def rho(a: int, f: Field) -> int:
    p = f.p
    if not 0 <= a < p:
        raise ValueError("ρ takes a canonical element")
    s = pow(a, f.root_exp, p)
    square = s * s % p == a
    if (s & 1) == (0 if square else 1):
        return s
    return p - s if s else 0


def rho_inv(y: int, f: Field) -> int:
    p = f.p
    if not 0 <= y < p:
        raise ValueError("ρ⁻¹ takes a canonical element")
    q = y * y % p
    return q if y & 1 == 0 else (p - q) % p


def encode_word(m: int, i: int, k: Key) -> int:
    f = k.field
    if not 0 <= m <= f.max_payload:
        raise ValueError(f"a payload is {f.payload_bytes} bytes" + (", below p" if f.full_slot else ""))
    t = k.tweak(i)
    x = m
    for j, c in enumerate(k.c):
        if j and k.mask is not None:
            x = k.sigma(x)
        x = rho((x + c + t) % f.p, f)
    return x


def decode_word(y: int, i: int, k: Key) -> int:
    """The payload of word `y` at index `i`; `ValueError` if `y` is not the encoding of one."""
    f = k.field
    if not 0 <= y < f.p:
        raise ValueError("a word is a canonical element")
    t = k.tweak(i)
    x = y
    for j in reversed(range(k.rounds)):
        x = (rho_inv(x, f) - k.c[j] - t) % f.p
        if j and k.mask is not None:
            x = k.sigma(x)
    if x > f.max_payload:
        raise ValueError(f"word {i} decodes to no payload")
    return x


def sigma_rare_payloads(k: Key, i: int) -> list[int]:
    """At full slot, payloads at index `i` whose first round's output x has x ⊕ K at or above p (σ_K leaves x), and next
    to them: σ_K's rare branch, which a random word takes with probability c / 2^n."""
    f, t = k.field, k.tweak(i)
    xs = [(f.p + r) ^ k.mask for r in (0, 1, f.c - 1)] + [(f.p - 1) ^ k.mask]
    return [(rho_inv(x, f) - k.c[0] - t) % f.p for x in xs if x < f.p]


def word_bytes(y: int, f: Field) -> bytes:
    return y.to_bytes(f.word_bytes, "little")


def bytes_word(b: bytes, f: Field) -> int:
    if len(b) != f.word_bytes:
        raise ValueError(f"a word is {f.word_bytes} bytes")
    return int.from_bytes(b, "little")


def task_payloads(task: bytes, f: Field) -> list[int]:
    """A task's 8,192 bytes as its L payloads, the last ones zero-padded."""
    if len(task) != TASK_BYTES:
        raise ValueError(f"a task is {TASK_BYTES} bytes")
    lw, pb = TASK_WORDS[f.name], f.payload_bytes
    padded = task + bytes(lw * pb - TASK_BYTES)
    return [int.from_bytes(padded[pb * w:pb * (w + 1)], "little") for w in range(lw)]


def payloads_task(payloads: list[int], f: Field) -> bytes:
    pb = f.payload_bytes
    return b"".join(m.to_bytes(pb, "little") for m in payloads)[:TASK_BYTES]


def encode_tasks(plain: bytes, k: Key, start_task: int = 0) -> bytes:
    """The store of a byte stream of whole tasks (a matrix's `codec.plain_fragments`), task q at `start_task + q`."""
    f = k.field
    if len(plain) % TASK_BYTES:
        raise ValueError("whole tasks")
    lw, out = TASK_WORDS[f.name], bytearray()
    for q in range(len(plain) // TASK_BYTES):
        ms = task_payloads(plain[TASK_BYTES * q:TASK_BYTES * (q + 1)], f)
        base = (start_task + q) * lw
        for w, m in enumerate(ms):
            out += word_bytes(encode_word(m, base + w, k), f)
    return bytes(out)


def decode_tasks(store: bytes, k: Key, start_task: int = 0) -> bytes:
    f = k.field
    lw, sb = TASK_WORDS[f.name], f.word_bytes
    tb = lw * sb
    if len(store) % tb:
        raise ValueError("whole tasks")
    out = bytearray()
    for q in range(len(store) // tb):
        base = (start_task + q) * lw
        ms = [decode_word(bytes_word(store[tb * q + sb * w:tb * q + sb * (w + 1)], f), base + w, k) for w in range(lw)]
        out += payloads_task(ms, f)
    return bytes(out)


# --- the tensor-core squaring's decomposition (`tc.cu`), on Python integers -----------------------------------------
#
# a has U·32 byte limbs a_i (zero past the word). Its square's byte coefficients c_m = Σ_{i+j=m} a_i a_j are the sums,
# over w = 0 … U, of the 32 × 32 × 32 products  C[r][s] += Σ_v A_w[r][v] B_w[v][s],  A_w[r][v] = a_(32w + r − v) (a
# Toeplitz block, two m16 row halves h = r // 16), B_w[v][s] = a_(32(s − w) + v), at coefficient m = 32 s + r: each pair
# (i, j) is counted once, at w = (i + v) // 32, v = j mod 32. Column s runs over [w, w + U); the accumulator columns are
# groups of 8, group g covering s ∈ [cb_g, cb_g + 8), and w takes the first group that holds [w, w + U). An MMA is one
# (w, h, group); its B operand depends only on δ = cb_g − w.


def tc_groups(u: int) -> list[int]:
    """Each group's first column cb: groups of 8 such that every [w, w + u), w = 0 … u, lies in one."""
    if not 1 <= u <= 8:
        raise ValueError("1 ≤ U ≤ 8 chunks")
    groups, w = [], 0
    while w <= u:
        cb = w
        groups.append(cb)
        while w <= u and w + u <= cb + 8:
            w += 1
    return groups


def tc_plan(u: int) -> list[tuple[int, int]]:
    """(w, group index) for every w = 0 … u: its MMAs (two, one a row half)."""
    groups = tc_groups(u)
    plan = []
    for w in range(u + 1):
        g = next(gi for gi, cb in enumerate(groups) if cb <= w and w + u <= cb + 8)
        plan.append((w, g))
    return plan


def tc_mmas(u: int) -> int:
    return 2 * len(tc_plan(u))


def tc_square_coeffs(limbs: list[int], u: int) -> list[int]:
    """The square's byte coefficients c_0 … c_(64u − 1) by the decomposition, accumulator by accumulator."""
    if len(limbs) > 32 * u or any(not 0 <= x < 256 for x in limbs):
        raise ValueError("at most 32U byte limbs")
    a = list(limbs) + [0] * (32 * u - len(limbs))

    def at(i: int) -> int:
        return a[i] if 0 <= i < 32 * u else 0

    groups = tc_groups(u)
    acc = [[[0] * 8 for _ in range(32)] for _ in groups]          # group, r, column
    for w, g in tc_plan(u):
        cb = groups[g]
        for r in range(32):
            for col in range(8):
                s = cb + col
                acc[g][r][col] += sum(at(32 * w + r - v) * at(32 * (s - w) + v) for v in range(32))
    out = [0] * (64 * u)
    for g, cb in enumerate(groups):
        for r in range(32):
            for col in range(8):
                m = 32 * (cb + col) + r
                if m < 64 * u:
                    out[m] += acc[g][r][col]
                elif acc[g][r][col]:
                    raise AssertionError("a product past the square")
    return out


def square_from_coeffs(coeffs: list[int]) -> int:
    return sum(c << (8 * m) for m, c in enumerate(coeffs))


# --- the tensor-core path's carries (`wt.cu` tc_square / tc_canon), lane by lane --------------------------------------
#
# Lane l of a warp holds digit l of a value, 8U bits (bytes U l … U l + U − 1), for a field with S = 32U. The square's
# coefficients folded at byte S (2^(8S) = 2c mod p) give each lane a 64-bit sum; shuffle rounds move each lane's carry to
# the next lane (lane 31's to lane 0, times 2c) until every carry is 0 or 1, and a carry-lookahead (two ballots: generate,
# propagate) finishes, lane 31's carry out wrapping (at most twice). The twin checks every bound the device relies on.

SHUF_ROUNDS = {1: 11, 2: 6, 3: 5, 4: 4, 5: 3}


def tc_ok(f: Field) -> bool:
    return f.word_bytes % 32 == 0 and f.n == 8 * f.word_bytes - 1


def _lookahead(v: list[int], db: int) -> tuple[list[int], int]:
    mask = (1 << db) - 1
    if any(x >> (db + 1) for x in v):
        raise AssertionError("a digit past 2^(DB + 1) at the lookahead")
    gen = sum(1 << i for i, x in enumerate(v) if x > mask)
    w = [x & mask for x in v]
    prop = sum(1 << i for i, x in enumerate(w) if x == mask)
    a = gen | prop
    sm = a + gen
    cin = sm ^ a ^ gen
    return [(x + (cin >> i & 1)) & mask for i, x in enumerate(w)], sm >> 32


def _add_low(v: list[int], db: int, k: int) -> list[int]:
    v = list(v)
    v[0] += k & ((1 << db) - 1)
    v[1] += k >> db
    return v


def _value(v: list[int], db: int) -> int:
    return sum(x << (db * i) for i, x in enumerate(v))


def tc_square_lanes(a: int, f: Field) -> int:
    """`wt.cu`'s tc_square of a < 2^(8S): digits < 2^(8U), a value < 2^(8S) congruent to a^2 mod p."""
    u = f.word_bytes // 32
    db, s, fw = 8 * u, f.word_bytes, 2 * f.c
    assert tc_ok(f) and (1 << (8 * s)) % f.p == fw and 0 <= a < 1 << (8 * s)
    coeffs = tc_square_coeffs(list(a.to_bytes(s, "little")), u)
    acc = []
    for lane in range(32):
        x = 0
        for b in range(u - 1, -1, -1):
            k = u * lane + b
            x = (x << 8) + coeffs[k] + fw * coeffs[k + s]
        assert x < 1 << 64
        acc.append(x)
    mask = (1 << db) - 1
    for _ in range(SHUF_ROUNDS[u]):
        c = [x >> db for x in acc]
        acc = [(x & mask) + (c[i - 1] if i else c[31] * fw) for i, x in enumerate(acc)]
        assert all(x < 1 << 64 for x in acc)
    acc, out = _lookahead(acc, db)
    if out:
        acc, out = _lookahead(_add_low(acc, db, fw), db)
        if out:
            acc, out = _lookahead(_add_low(acc, db, fw), db)
            assert not out
    v = _value(acc, db)
    assert v < 1 << (8 * s) and v % f.p == a * a % f.p
    return v


def tc_canon_lanes(v: int, f: Field) -> int:
    u = f.word_bytes // 32
    db = 8 * u
    digits = [(v >> (db * i)) & ((1 << db) - 1) for i in range(32)]
    hb = digits[31] >> (db - 1)
    digits[31] &= (1 << (db - 1)) - 1
    if hb:
        digits, out = _lookahead(_add_low(digits, db, f.c), db)
        assert not out
    w, out = _lookahead(_add_low(digits, db, f.c), db)
    assert not out
    if w[31] >> (db - 1):
        digits = w[:31] + [w[31] & ((1 << (db - 1)) - 1)]
    r = _value(digits, db)
    assert r == v % f.p
    return r
