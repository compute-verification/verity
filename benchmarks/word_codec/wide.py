"""The wide word codecs, byte-exact: Sloth over p = 2^127 − 1, 2^255 − 19, 2^521 − 1, 2^256 − 189 and 2^512 − 569, in
the fused GEMM's layout, with the encoded words tagged by the bandwidth audit's `ro-blake3/v1`
(`verity.core.protocols.bandwidth.tags`).

~~~text
field      p127 = 2^127 − 1, p521 = 2^521 − 1, p256 = 2^256 − 189, p512 = 2^512 − 569 (3 mod 4); p255 = 2^255 − 19
           (5 mod 8). p256 and p512 are full width: a word is all of its 32·limbs bits, and a payload is any m < p, which
           every m whose top byte is not 0xFF is (C < 2^(b − 8)): FP8 E4M3 weights always are, 0xFF being its NaN
ρ⁻¹(y)     3 mod 4: y² if y is even, p − y² if y is odd; p255: y² if y is even, 2·y² if y is odd (2 is a non-residue)
ρ(a)       the y with ρ⁻¹(y) = a: 3 mod 4: s = a^((p+1)/4), the even one of {s, p − s} if s² = a, else the odd one, the
           power p − 2 squarings for a Mersenne p, and for p = 2^b − C the chain (a^(2^n − 1))^(2^s) · a^(2^s − D),
           D = (C − 1)/4 < 2^s, n = b − 2 − s (`root_chain`); p255: if a is a square, the even root of a, else the odd
           root of a/2, from β = a^((p+3)/8) (251 squarings and 11 multiplications) times 1, √−1, √(1/(2√−1)) or
           √−1·√(1/(2√−1))
c_j, τ     SHA512(DOMAIN ‖ name ‖ 0 ‖ seed ‖ u32be j ‖ u8 h) for h = 0, 1, the 128 bytes little-endian, mod p; j = 0 … k − 1
           for the rounds, j = 2^32 − 1 for the tweak key τ (1 in place of 0)
t(i)       τ · i mod p, i the word's index in the store
encode     m (`payload` bits, m < min(2^payload, p)) through rounds x ↦ ρ(x + c_j + t(i)), j = 0 … k − 1
decode     x ↦ ρ⁻¹(x) − c_j − t(i), j = k − 1 … 0; a word decodes to a payload only if the result is < 2^payload
word       y < p as `limbs` little-endian u32 limbs (4, 8, 17, 8, 16: 16, 32, 68, 32, 64 bytes)
~~~

The fields, and ρ and ρ⁻¹ at p = 3 mod 4, are the store codec's (`verity.core.protocols.poub.codec`'s
`Field`, `rho` and `rho_inv`); p255's ρ and the keys are this module's.

The fused GEMM's layout is the word codec's (`codec.fragment_stream`): tasks of one 16-row block and 512 columns, in
order (row block, column chunk); in each task lane l's 256 bytes of its mma.m16n8k32 A fragments are the little-endian
integer S, and its word w carries bits w·b … w·b + b − 1 of S (b = `payload`), L = ⌈2048 / b⌉ words a lane
(17, 9, 4, 8 and 4 at the default payloads 126, 254, 520, 256 and 512). Word w of lane l of task q has index
i = 32·L·q + 32·w + l.

The store E is the tasks' records in order. Record q is, for w in 0 … L − 1, for limb e, for lane l, u32le(limb e of
word (q, l, w)): a warp's load of one limb reads 128 contiguous bytes. Block j = 32·q + l of the bandwidth audit is
lane l's L words of task q, each `word_bytes` little-endian, in order; its tag is `ro-blake3/v1`'s
BLAKE3-keyed(K, u64be j ‖ E_j), and the tags are committed 16 to a leaf under the audit's tree.
"""

from __future__ import annotations

import hashlib
import random
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

from verity.core.protocols.poub import codec as RB

sys.path.insert(0, str(Path(__file__).resolve().parent))

import codec  # noqa: E402

DOMAIN = b"verity/word-codec/wide/v0\0"
SEED_BYTES = 32
TWEAK_INDEX = (1 << 32) - 1
LANES, TASK_TILES, TASK_COLS, LANE_BYTES = codec.LANES, codec.TASK_TILES, codec.TASK_COLS, codec.LANE_BYTES


class Field(RB.Field):
    """The store codec's field (`RB.Field`, p = 2^n − c) in the fused GEMM's packing: `bits` = n, the default
    `payload` its `payload_bits`, and a word stored as its `limbs` u32 limbs."""

    __slots__ = ()

    @property
    def bits(self) -> int:
        return self.n

    @property
    def payload(self) -> int:
        return self.payload_bits

    @property
    def word_bytes(self) -> int:
        """A stored word's bytes, its limbs': 68 at p521, whose field's S is 66."""
        return 4 * self.limbs

    @property
    def three_mod_four(self) -> bool:
        return self.p % 4 == 3

    def lane_words(self, payload: int | None = None) -> int:
        return -(-8 * LANE_BYTES // self.check_payload(payload))

    def full_width(self, b: int) -> bool:
        """A payload of all the word's bits, p = 2^b − C with C < 2^(b − 8): every m whose top byte is not 0xFF is < p."""
        return b == 32 * self.limbs and (1 << b) - self.p < 1 << (b - 8)

    def check_payload(self, payload: int | None) -> int:
        b = self.payload if payload is None else payload
        if not 8 <= b or ((1 << b) > self.p and not self.full_width(b)):
            raise ValueError(f"{self.name}: a payload is under {self.p.bit_length() - 1} bits, or the word's full width")
        return b

    def payload_bound(self, payload: int | None = None) -> int:
        """The payloads are the m below this: 2^payload, or p at full width."""
        return min(1 << self.check_payload(payload), self.p)

    def record_bytes(self, payload: int | None = None) -> int:
        return LANES * self.lane_words(payload) * self.word_bytes

    def expansion(self, payload: int | None = None) -> float:
        """Stored bytes over FP8 weight bytes."""
        return self.record_bytes(payload) / (LANES * LANE_BYTES)


FIELDS = {f.bits: f for f in (Field("p127", 127, 1), Field("p255", 255, 19), Field("p521", 521, 1),
                               Field("p256", 256, 189), Field("p512", 512, 569))}


def field(bits: int) -> Field:
    try:
        return FIELDS[bits]
    except KeyError:
        raise ValueError(f"no wide field of {bits} bits; known: {sorted(FIELDS)}") from None


# --- p = 5 mod 8: √−1 and the non-residue branch's multipliers --------------------------------------------------------

def _check_5mod8(p: int) -> None:
    if p % 8 != 5:
        raise ValueError("the second rule takes p = 5 mod 8")


def sqrt_m1(p: int) -> int:
    _check_5mod8(p)
    return pow(2, (p - 1) // 4, p)


def _sqrt_5mod8(a: int, p: int) -> int:
    """A square root of the square a mod p (Atkin's candidate a^((p+3)/8), times √−1 if it squares to −a)."""
    b = pow(a, (p + 3) // 8, p)
    if b * b % p == a:
        return b
    b = b * sqrt_m1(p) % p
    if b * b % p != a:
        raise ValueError("not a square")
    return b


def c1(p: int) -> int:
    """c1 with c1² = 1/(2·√−1): β·c1 is a root of a/2 when β² = √−1·a, and β·c1·√−1 when β² = −√−1·a."""
    return _sqrt_5mod8(pow(2 * sqrt_m1(p), p - 2, p), p)


P255 = FIELDS[255].p
SQRT_M1 = sqrt_m1(P255)
C1 = c1(P255)
C1_SQRT_M1 = C1 * SQRT_M1 % P255


def legendre(a: int, p: int) -> int:
    return 0 if a % p == 0 else 1 if pow(a, (p - 1) // 2, p) == 1 else -1


def rho(a: int, f: Field) -> int:
    """Sloth's square-root permutation of Z_p: the store codec's (`RB.rho`) at p = 3 mod 4."""
    if f.three_mod_four:
        return RB.rho(a, f)
    p = f.p
    if not 0 <= a < p:
        raise ValueError("ρ takes a canonical element")
    if a == 0:
        return 0
    if legendre(a, p) == 1:
        s, want_odd = _sqrt_5mod8(a, p), False
    else:
        s, want_odd = _sqrt_5mod8(a * pow(2, p - 2, p) % p, p), True
    return s if (s & 1) == want_odd else p - s


def rho_fast_5mod8(a: int, p: int = P255) -> int:
    """`rho` for p = 5 mod 8 the way the kernel computes it: one exponentiation β = a^((p+3)/8), then β² against ±a and
    ±√−1·a."""
    if a == 0:
        return 0
    i, c = sqrt_m1(p), c1(p)
    b = pow(a, (p + 3) // 8, p)
    q = b * b % p
    if q == a:
        s, odd = b, False
    elif q == p - a:
        s, odd = b * i % p, False
    elif q == i * a % p:
        s, odd = b * c % p, True
    elif q == p - i * a % p:
        s, odd = b * c * i % p, True
    else:
        raise AssertionError("β² is ±a or ±√−1·a")
    return s if (s & 1) == odd else p - s


def rho_inv(y: int, f: Field) -> int:
    if f.three_mod_four:
        return RB.rho_inv(y, f)
    p = f.p
    if not 0 <= y < p:
        raise ValueError("ρ⁻¹ takes a canonical element")
    q = y * y % p
    return q if y & 1 == 0 else 2 * q % p


@dataclass(frozen=True, slots=True)
class Key:
    field: Field
    c: tuple[int, ...]
    tau: int

    @property
    def rounds(self) -> int:
        return len(self.c)

    def tweak(self, i: int) -> int:
        return self.tau * i % self.field.p


def _element(f: Field, seed: bytes, j: int) -> int:
    head = DOMAIN + f.name.encode() + b"\0" + seed + struct.pack(">I", j)
    x = hashlib.sha512(head + b"\0").digest() + hashlib.sha512(head + b"\1").digest()
    return int.from_bytes(x, "little") % f.p


def key(f: Field, seed: bytes, rounds: int) -> Key:
    if type(seed) is not bytes or len(seed) != SEED_BYTES:
        raise ValueError(f"a seed is {SEED_BYTES} bytes")
    if not 1 <= rounds < TWEAK_INDEX:
        raise ValueError("a word has at least one round")
    return Key(f, tuple(_element(f, seed, j) for j in range(rounds)), _element(f, seed, TWEAK_INDEX) or 1)


def encode_word(m: int, i: int, k: Key, payload: int | None = None) -> int:
    f = k.field
    if not 0 <= m < f.payload_bound(payload):
        raise ValueError("a payload is under 2^payload, and under p")
    t, x = k.tweak(i), m
    for c in k.c:
        x = rho((x + c + t) % f.p, f)
    return x


def decode_raw(y: int, i: int, k: Key) -> int:
    """The k inverse rounds of any canonical `y` at index `i`, the payload check left out (the kernel's output)."""
    f = k.field
    if not 0 <= y < f.p:
        raise ValueError("a word is a canonical element")
    t, x = k.tweak(i), y
    for c in reversed(k.c):
        x = (rho_inv(x, f) - c - t) % f.p
    return x


def decode_word(y: int, i: int, k: Key, payload: int | None = None) -> int:
    """The payload of word `y` at index `i`; `ValueError` if `y` is not the encoding of one."""
    x = decode_raw(y, i, k)
    if x >> k.field.check_payload(payload):
        raise ValueError(f"word {i} decodes to no payload")
    return x


def limbs(y: int, f: Field) -> list[int]:
    return [y >> (32 * e) & 0xFFFFFFFF for e in range(f.limbs)]


def word_bytes(y: int, f: Field) -> bytes:
    return y.to_bytes(f.word_bytes, "little")


# --- the fused GEMM's layout: codec.fragment_stream's lane streams, packed into wide words -----------------------------

def pack(stream: bytes, payload: int) -> list[int]:
    s, mask = int.from_bytes(stream, "little"), (1 << payload) - 1
    return [s >> (payload * i) & mask for i in range(-(-8 * len(stream) // payload))]


def unpack(payloads: list[int], length: int, payload: int) -> bytes:
    s = sum(m << (payload * i) for i, m in enumerate(payloads))
    return s.to_bytes(-(-payload * len(payloads) // 8), "little")[:length]


def word_index(q: int, lane: int, w: int, lw: int) -> int:
    return LANES * lw * q + LANES * w + lane


def fragment_payloads(w: bytes, n: int, kdim: int, f: Field, payload: int | None = None) -> list[list[int]]:
    """Each (task, lane)'s L payloads, (task, lane) in order."""
    b = f.check_payload(payload)
    return [pack(s, b) for s in codec.fragment_stream(w, n, kdim)]


def encode_store(lanes: list[list[int]], k: Key, payload: int | None = None, base: int = 0) -> list[list[int]]:
    """Each (task, lane)'s L encoded words; word indices from `base`."""
    out = []
    for idx, ms in enumerate(lanes):
        q, lane = divmod(idx, LANES)
        lw = len(ms)
        out.append([encode_word(m, base + word_index(q, lane, wi, lw), k, payload) for wi, m in enumerate(ms)])
    return out


def store_bytes(words: list[list[int]], f: Field) -> bytes:
    """E: the records of `words` (each (task, lane)'s L words) in the store's order."""
    lw = len(words[0])
    out = bytearray()
    for q in range(len(words) // LANES):
        for wi in range(lw):
            for e in range(f.limbs):
                for lane in range(LANES):
                    out += struct.pack("<I", words[q * LANES + lane][wi] >> (32 * e) & 0xFFFFFFFF)
    return bytes(out)


def block(words: list[list[int]], j: int, f: Field, tb: int = 1) -> bytes:
    """E_j: lane j mod 32's words of tasks tb·(j div 32) … tb·(j div 32) + tb − 1 (tb adjacent K-chunks of the same
    rows, one warp of the fused GEMM each)."""
    q, lane = divmod(j, LANES)
    return b"".join(word_bytes(y, f) for t in range(tb) for y in words[LANES * (tb * q + t) + lane])


# --- the bandwidth audit's tags over the encoded words ---------------------------------------------------------------

def tags_of(words: list[list[int]], f: Field, seed: bytes, base: int = 0, tb: int = 1) -> list[bytes]:
    from verity.core.protocols.bandwidth import tags

    assert len(words) % (LANES * tb) == 0
    keyed = tags.get("ro-blake3/v1").keyed(seed)
    return [keyed(base + j, block(words, j, f, tb)) for j in range(len(words) // tb)]


def leaves_of(ys: list[bytes], seed: bytes) -> list[bytes]:
    from verity.core.protocols.bandwidth import tags

    return tags.leaves(tags.get("ro-blake3/v1"), seed, ys)


def root_of(ys: list[bytes], seed: bytes) -> bytes:
    from verity.core.protocols.pous.verifier import MerkleTree

    return MerkleTree(leaves_of(ys, seed)).root


def random_weights(n: int, kdim: int, seed: int) -> bytes:
    rng = random.Random(seed)
    return bytes(rng.getrandbits(8) for _ in range(n * kdim))


# --- the frontier's quantities ---------------------------------------------------------------------------------------

def root_split(f: Field) -> tuple[int, int, int]:
    """(n, s, e) with (p + 1)/4 = (2^n − 1)·2^s + e for p = 2^b − C = 3 mod 4: D = (C − 1)/4, s the least with
    2^s > D, e = 2^s − D, n = b − 2 − s."""
    d = ((1 << f.bits) - f.p - 1) // 4
    s = d.bit_length()
    n, e = f.bits - 2 - s, (1 << s) - d
    assert (((1 << n) - 1) << s) + e == (f.p + 1) // 4
    return n, s, e


def _pow2m1_chain(n: int) -> tuple[int, int]:
    """Squarings and multiplications of a^(2^n − 1) by x_2j = x_j^(2^j)·x_j and x_(j+1) = x_j²·a (`wide.cu`'s `pow2m1`)."""
    if n == 1:
        return 0, 0
    s, m = _pow2m1_chain(n - 1 if n % 2 else n // 2)
    return s + (1 if n % 2 else n // 2), m + 1


def root_chain(f: Field) -> tuple[int, int]:
    """Squarings and multiplications in one ρ the kernel computes (the square check's squaring not counted)."""
    if not f.three_mod_four:
        return 251, 11
    if f.p == (1 << f.bits) - 1:
        return f.bits - 2, 0
    n, s, e = root_split(f)
    sq, mu = _pow2m1_chain(n)
    return sq + s + e.bit_length() - 1, mu + e.bit_count() - 1 + 1


def root_squarings(f: Field) -> int:
    return root_chain(f)[0]
