"""The wide word codecs, byte-exact: Sloth over p = 2^127 − 1, 2^255 − 19 and 2^521 − 1, in the fused GEMM's layout,
with the encoded words tagged by the bandwidth audit's `ro-blake3/v1` (`verity.core.protocols.bandwidth.tags`).

~~~text
field      p127 = 2^127 − 1, p521 = 2^521 − 1 (3 mod 4); p255 = 2^255 − 19 (5 mod 8)
ρ⁻¹(y)     3 mod 4: y² if y is even, p − y² if y is odd; p255: y² if y is even, 2·y² if y is odd (2 is a non-residue)
ρ(a)       the y with ρ⁻¹(y) = a: 3 mod 4: s = a^((p+1)/4) (p − 2 squarings), the even one of {s, p − s} if s² = a, else
           the odd one; p255: if a is a square, the even root of a, else the odd root of a/2, from β = a^((p+3)/8)
           (251 squarings and 11 multiplications) times 1, √−1, √(1/(2√−1)) or √−1·√(1/(2√−1))
c_j, τ     SHA512(DOMAIN ‖ name ‖ 0 ‖ seed ‖ u32be j ‖ u8 h) for h = 0, 1, the 128 bytes little-endian, mod p; j = 0 … k − 1
           for the rounds, j = 2^32 − 1 for the tweak key τ (1 in place of 0)
t(i)       τ · i mod p, i the word's index in the store
encode     m (`payload` bits, m < 2^payload) through rounds x ↦ ρ(x + c_j + t(i)), j = 0 … k − 1
decode     x ↦ ρ⁻¹(x) − c_j − t(i), j = k − 1 … 0; a word decodes to a payload only if the result is < 2^payload
word       y < p as `limbs` little-endian u32 limbs (4, 8, 17: 16, 32, 68 bytes)
~~~

The fused GEMM's layout is the word codec's (`codec.fragment_stream`): tasks of one 16-row block and 512 columns, in
order (row block, column chunk); in each task lane l's 256 bytes of its mma.m16n8k32 A fragments are the little-endian
integer S, and its word w carries bits w·b … w·b + b − 1 of S (b = `payload`), L = ⌈2048 / b⌉ words a lane
(17, 9 and 4 at the default payloads 126, 254 and 520). Word w of lane l of task q has index i = 32·L·q + 32·w + l.

The store E is the tasks' records in order. Record q is, for w in 0 … L − 1, for limb e, for lane l, u32le(limb e of
word (q, l, w)): a warp's load of one limb reads 128 contiguous bytes. Block j = 32·q + l of the bandwidth audit is
lane l's L words of task q, each `word_bytes` little-endian, in order; its tag is `ro-blake3/v1`'s
BLAKE3-keyed(K, u64be j ‖ E_j), and the tags are committed 16 to a leaf under the audit's tree.
"""

from __future__ import annotations

import hashlib
import random
import struct
from dataclasses import dataclass

DOMAIN = b"verity/word-codec/wide/v0\0"
SEED_BYTES = 32
TWEAK_INDEX = (1 << 32) - 1
LANES = 32
TASK_TILES = 16
TASK_COLS = 32 * TASK_TILES
LANE_BYTES = 16 * TASK_TILES
TILE_ROWS = 16


@dataclass(frozen=True, slots=True)
class Field:
    name: str
    bits: int
    p: int
    limbs: int
    payload: int

    @property
    def word_bytes(self) -> int:
        return 4 * self.limbs

    @property
    def three_mod_four(self) -> bool:
        return self.p % 4 == 3

    def lane_words(self, payload: int | None = None) -> int:
        return -(-8 * LANE_BYTES // self.check_payload(payload))

    def check_payload(self, payload: int | None) -> int:
        b = self.payload if payload is None else payload
        if not 8 <= b or (1 << b) > self.p:
            raise ValueError(f"{self.name}: a payload is under {self.p.bit_length() - 1} bits")
        return b

    def record_bytes(self, payload: int | None = None) -> int:
        return LANES * self.lane_words(payload) * self.word_bytes

    def expansion(self, payload: int | None = None) -> float:
        """Stored bytes over FP8 weight bytes."""
        return self.record_bytes(payload) / (LANES * LANE_BYTES)


FIELDS = {
    127: Field("p127", 127, (1 << 127) - 1, 4, 126),
    255: Field("p255", 255, (1 << 255) - 19, 8, 254),
    521: Field("p521", 521, (1 << 521) - 1, 17, 520),
}


def field(bits: int) -> Field:
    try:
        return FIELDS[bits]
    except KeyError:
        raise ValueError(f"no wide field of {bits} bits; known: {sorted(FIELDS)}") from None


# --- p255's constants: √−1 and the non-residue branch's multipliers ----------------------------------------------------

P255 = FIELDS[255].p
SQRT_M1 = pow(2, (P255 - 1) // 4, P255)
assert SQRT_M1 * SQRT_M1 % P255 == P255 - 1


def _sqrt_p255(a: int) -> int:
    """A square root of the square a mod p255 (Atkin's candidate a^((p+3)/8), times √−1 if it squares to −a)."""
    b = pow(a, (P255 + 3) // 8, P255)
    if b * b % P255 == a:
        return b
    b = b * SQRT_M1 % P255
    if b * b % P255 != a:
        raise ValueError("not a square")
    return b


# c1² = 1/(2·√−1): β·c1 is a root of a/2 when β² = √−1·a, and β·c1·√−1 when β² = −√−1·a.
C1 = _sqrt_p255(pow(2 * SQRT_M1, P255 - 2, P255))
C1_SQRT_M1 = C1 * SQRT_M1 % P255
assert pow(2, (P255 - 1) // 2, P255) == P255 - 1          # 2 is a non-residue (p = 5 mod 8)


def legendre(a: int, p: int) -> int:
    return 0 if a % p == 0 else 1 if pow(a, (p - 1) // 2, p) == 1 else -1


def rho(a: int, f: Field) -> int:
    """Sloth's square-root permutation of Z_p."""
    p = f.p
    if not 0 <= a < p:
        raise ValueError("ρ takes a canonical element")
    if a == 0:
        return 0
    if f.three_mod_four:
        s = pow(a, (p + 1) // 4, p)
        want_odd = s * s % p != a
        return s if (s & 1) == want_odd else p - s
    if legendre(a, p) == 1:
        s, want_odd = _sqrt_p255(a), False
    else:
        s, want_odd = _sqrt_p255(a * pow(2, p - 2, p) % p), True
    return s if (s & 1) == want_odd else p - s


def rho_fast_p255(a: int) -> int:
    """`rho` for p255 the way the kernel computes it: one exponentiation β = a^((p+3)/8), then β² against ±a and ±√−1·a."""
    p = P255
    if a == 0:
        return 0
    b = pow(a, (p + 3) // 8, p)
    q = b * b % p
    if q == a:
        s, odd = b, False
    elif q == p - a:
        s, odd = b * SQRT_M1 % p, False
    elif q == SQRT_M1 * a % p:
        s, odd = b * C1 % p, True
    elif q == p - SQRT_M1 * a % p:
        s, odd = b * C1_SQRT_M1 % p, True
    else:
        raise AssertionError("β² is ±a or ±√−1·a")
    return s if (s & 1) == odd else p - s


def rho_inv(y: int, f: Field) -> int:
    p = f.p
    if not 0 <= y < p:
        raise ValueError("ρ⁻¹ takes a canonical element")
    q = y * y % p
    if y & 1 == 0:
        return q
    return (p - q) % p if f.three_mod_four else 2 * q % p


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
    if not 0 <= m < 1 << f.check_payload(payload):
        raise ValueError("a payload is under 2^payload")
    t, x = k.tweak(i), m
    for c in k.c:
        x = rho((x + c + t) % f.p, f)
    return x


def decode_word(y: int, i: int, k: Key, payload: int | None = None) -> int:
    """The payload of word `y` at index `i`; `ValueError` if `y` is not the encoding of one."""
    f = k.field
    if not 0 <= y < f.p:
        raise ValueError("a word is a canonical element")
    t, x = k.tweak(i), y
    for c in reversed(k.c):
        x = (rho_inv(x, f) - c - t) % f.p
    if x >> f.check_payload(payload):
        raise ValueError(f"word {i} decodes to no payload")
    return x


def limbs(y: int, f: Field) -> list[int]:
    return [y >> (32 * e) & 0xFFFFFFFF for e in range(f.limbs)]


def word_bytes(y: int, f: Field) -> bytes:
    return y.to_bytes(f.word_bytes, "little")


# --- the fused GEMM's layout -----------------------------------------------------------------------------------------

def fragment_offset(lane: int, r: int, b: int) -> tuple[int, int]:
    """(row, column) in a 16 × 32 tile of byte `b` of register `r` of `lane`'s mma.m16n8k32 A fragment (8-bit types)."""
    g, q = lane >> 2, lane & 3
    return g + 8 * (r & 1), 4 * q + 16 * (r >> 1) + b


def check_shape(n: int, kdim: int) -> None:
    if n <= 0 or kdim <= 0 or n % TILE_ROWS or kdim % TASK_COLS:
        raise ValueError(f"the layout takes N a multiple of {TILE_ROWS} and K of {TASK_COLS}")


def lane_streams(w: bytes, n: int, kdim: int) -> list[bytes]:
    """W (n × kdim bytes, row-major) as each (task, lane)'s 256-byte stream, tasks in order (row block, column chunk)."""
    check_shape(n, kdim)
    if len(w) != n * kdim:
        raise ValueError("W is n × kdim bytes")
    out = []
    for nb in range(n // TILE_ROWS):
        for kc in range(kdim // TASK_COLS):
            for lane in range(LANES):
                s = bytearray(LANE_BYTES)
                for tt in range(TASK_TILES):
                    for r in range(4):
                        for b in range(4):
                            row, col = fragment_offset(lane, r, b)
                            s[16 * tt + 4 * r + b] = w[(TILE_ROWS * nb + row) * kdim + kc * TASK_COLS + 32 * tt + col]
                out.append(bytes(s))
    return out


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
    return [pack(s, b) for s in lane_streams(w, n, kdim)]


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


def block(words: list[list[int]], j: int, f: Field) -> bytes:
    """E_j: lane j mod 32's words of task j div 32."""
    return b"".join(word_bytes(y, f) for y in words[j])


# --- the bandwidth audit's tags over the encoded words ---------------------------------------------------------------

def tags_of(words: list[list[int]], f: Field, seed: bytes, base: int = 0) -> list[bytes]:
    from verity.core.protocols.bandwidth import tags

    keyed = tags.get("ro-blake3/v1").keyed(seed)
    return [keyed(base + j, block(words, j, f)) for j in range(len(words))]


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

def root_squarings(f: Field) -> int:
    """Squarings in one ρ the kernel computes (p255's 11 multiplications not counted)."""
    return f.bits - 2 if f.three_mod_four else 251
