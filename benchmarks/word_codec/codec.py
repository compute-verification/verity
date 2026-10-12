"""The word codec, byte-exact: Sloth in one 61-bit word (`note:console/20261012T0355Z-draft-rate-based-memory-audits`).

~~~text
p          2^61 − 1 (3 mod 4), so a square root is a^((p+1)/4) = a^(2^59): 59 squarings
ρ(a)       Sloth's permutation of Z_p (Lenstra–Wesolowski): s = a^(2^59); if s² = a (a is a square), the even one of
           {s, p − s}, else (s² = −a) the odd one
ρ⁻¹(y)     y² if y is even, p − y² if y is odd: one squaring and the sign rule
c_j        round j's constant: u64le(SHA512(DOMAIN ‖ seed ‖ u32be j)[0:8]) mod p, j = 0 … k − 1
τ          the tweak key: the same with j = 2^32 − 1, and 1 in place of 0
t(i)       τ · i mod p, i the word's index in the store
round j    x ↦ ρ(x + c_j + t(i))
encode     m (seven payload bytes, little-endian, m < 2^56) through rounds 0 … k − 1; the word is y < p, stored u64le
decode     rounds k − 1 … 0 inverted: x ↦ ρ⁻¹(x) − c_j − t(i); a word decodes to a payload only if the result is < 2^56
~~~

The seed is drawn after the weights are registered. Encoding a word costs k roots (59k squarings and a square test
each) and decoding it k squarings, whatever k is, and there is no keystream: the honest prover decodes at about
one squaring per round and word, and a prover missing the word rebuilds it at about 59.

Two layouts carry a byte stream into words, seven payload bytes each:
- `flat`: word i holds bytes 7i … 7i + 6 of the stream (the standalone decode's store);
- `fragment`: an FP8 weight matrix W (N × K, K contiguous) in the operand order of the fused GEMM (`fragment_stream`):
  tasks of one 16-row block and 512 columns, in each task lane l's 256 bytes (16 k-tiles of its mma.m16n8k32 A
  fragment) packed into 37 words, word w of lane l at index 37·32·task + 32·w + l.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

P = (1 << 61) - 1
ROOT_EXP = (P + 1) // 4
assert ROOT_EXP == 1 << 59
PAYLOAD_BYTES = 7
WORD_BYTES = 8
DOMAIN = b"verity/word-codec/v0\0"
SEED_BYTES = 32
TWEAK_INDEX = (1 << 32) - 1

# The fused GEMM's fragment layout (mma.sync.m16n8k32, 8-bit A operand: 16 rows × 32 columns, 16 bytes a lane).
TILE_ROWS = 16
TILE_COLS = 32
TASK_TILES = 16
TASK_COLS = TILE_COLS * TASK_TILES
LANES = 32
LANE_BYTES = 16 * TASK_TILES
LANE_WORDS = -(-LANE_BYTES // PAYLOAD_BYTES)
TASK_WORDS = LANE_WORDS * LANES


@dataclass(frozen=True, slots=True)
class Key:
    """The seed's constants: `c[j]` for every round and the tweak key `tau`."""

    c: tuple[int, ...]
    tau: int

    @property
    def rounds(self) -> int:
        return len(self.c)

    def tweak(self, i: int) -> int:
        return self.tau * i % P


def _field(seed: bytes, j: int) -> int:
    return int.from_bytes(hashlib.sha512(DOMAIN + seed + struct.pack(">I", j)).digest()[:8], "little") % P


def key(seed: bytes, rounds: int) -> Key:
    if type(seed) is not bytes or len(seed) != SEED_BYTES:
        raise ValueError(f"a seed is {SEED_BYTES} bytes")
    if not 1 <= rounds < TWEAK_INDEX:
        raise ValueError("a word has at least one round")
    return Key(tuple(_field(seed, j) for j in range(rounds)), _field(seed, TWEAK_INDEX) or 1)


def rho(a: int, p: int = P) -> int:
    """Sloth's square-root permutation of Z_p (any prime p = 3 mod 4; the codec's is `P`)."""
    if not 0 <= a < p:
        raise ValueError("ρ takes a canonical element")
    s = pow(a, (p + 1) // 4, p)
    square = s * s % p == a
    if (s & 1) == (0 if square else 1):
        return s
    return p - s if s else 0


def rho_inv(y: int, p: int = P) -> int:
    if not 0 <= y < p:
        raise ValueError("ρ⁻¹ takes a canonical element")
    q = y * y % p
    return q if y & 1 == 0 else (p - q) % p


def encode_word(m: int, i: int, k: Key) -> int:
    if not 0 <= m < 1 << (8 * PAYLOAD_BYTES):
        raise ValueError("a payload is seven bytes")
    t = k.tweak(i)
    x = m
    for c in k.c:
        x = rho((x + c + t) % P)
    return x


def decode_word(y: int, i: int, k: Key) -> int:
    """The payload of word `y` at index `i`; `ValueError` if `y` is not the encoding of one."""
    if not 0 <= y < P:
        raise ValueError("a word is a canonical element")
    t = k.tweak(i)
    x = y
    for c in reversed(k.c):
        x = (rho_inv(x) - c - t) % P
    if x >> (8 * PAYLOAD_BYTES):
        raise ValueError(f"word {i} decodes to no payload")
    return x


def pack(stream: bytes) -> list[int]:
    """The payloads of a byte stream, seven bytes a word, the last one zero-padded."""
    stream = bytes(stream) + bytes(-len(stream) % PAYLOAD_BYTES)
    return [int.from_bytes(stream[o:o + PAYLOAD_BYTES], "little") for o in range(0, len(stream), PAYLOAD_BYTES)]


def unpack(payloads: list[int], length: int) -> bytes:
    return b"".join(m.to_bytes(PAYLOAD_BYTES, "little") for m in payloads)[:length]


def words_bytes(words: list[int]) -> bytes:
    return b"".join(w.to_bytes(WORD_BYTES, "little") for w in words)


def bytes_words(data: bytes) -> list[int]:
    if len(data) % WORD_BYTES:
        raise ValueError("a store is whole words")
    return [int.from_bytes(data[o:o + WORD_BYTES], "little") for o in range(0, len(data), WORD_BYTES)]


def encode_payloads(payloads: list[int], k: Key, start: int = 0) -> list[int]:
    return [encode_word(m, start + n, k) for n, m in enumerate(payloads)]


def decode_words(words: list[int], k: Key, start: int = 0) -> list[int]:
    return [decode_word(y, start + n, k) for n, y in enumerate(words)]


def encode_flat(stream: bytes, k: Key) -> bytes:
    return words_bytes(encode_payloads(pack(stream), k))


def decode_flat(store: bytes, k: Key, length: int) -> bytes:
    return unpack(decode_words(bytes_words(store), k), length)


def fragment_offset(lane: int, r: int, b: int) -> tuple[int, int]:
    """(row, column) in a 16 × 32 tile of byte `b` of register `r` of `lane`'s mma.m16n8k32 A fragment (8-bit types)."""
    g, q = lane >> 2, lane & 3
    return g + 8 * (r & 1), 4 * q + 16 * (r >> 1) + b


def check_shape(n: int, kdim: int) -> None:
    if n <= 0 or kdim <= 0 or n % TILE_ROWS or kdim % TASK_COLS:
        raise ValueError(f"the fragment layout takes N a multiple of {TILE_ROWS} and K of {TASK_COLS}")


def fragment_stream(w: bytes, n: int, kdim: int) -> list[bytes]:
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
                            s[16 * tt + 4 * r + b] = w[(TILE_ROWS * nb + row) * kdim + kc * TASK_COLS + TILE_COLS * tt + col]
                out.append(bytes(s))
    return out


def fragment_payloads(w: bytes, n: int, kdim: int) -> list[int]:
    """The payloads of the fragment layout, in store order (word w of lane l of a task at 37·32·task + 32·w + l)."""
    lanes = fragment_stream(w, n, kdim)
    out = [0] * (len(lanes) * LANE_WORDS)
    for idx, s in enumerate(lanes):
        task, lane = divmod(idx, LANES)
        for wi, m in enumerate(pack(s)):
            out[task * TASK_WORDS + wi * LANES + lane] = m
    return out


def plain_fragments(w: bytes, n: int, kdim: int) -> bytes:
    """The plain FP8 GEMM's layout: each task's 16 k-tiles, each tile's 32 lanes × 16 bytes, lane-contiguous."""
    lanes = fragment_stream(w, n, kdim)
    out = bytearray()
    for task in range(len(lanes) // LANES):
        for tt in range(TASK_TILES):
            for lane in range(LANES):
                out += lanes[task * LANES + lane][16 * tt:16 * tt + 16]
    return bytes(out)


def bound(eps: float, r: float, tau: float, b_side: float, read_rate: float, theta: float = 0.0) -> dict[str, float]:
    """μ ≤ B_side/R + (σ + θ)/(ε(r − 1)), σ = 1 − ε − τ: the fraction of the store a prover can lack and still answer a
    rate-based audit at read rate R, with the encoding units' capacity counted as 1 and θ more of it from units the
    measurement left out (the tensor cores)."""
    if not 0 < eps <= 1 or r <= 1:
        raise ValueError("ε in (0, 1] and r > 1")
    sigma = 1 - eps - tau
    rebuild = (sigma + theta) / (eps * (r - 1))
    side = b_side / read_rate
    return {"sigma": sigma, "rebuild": rebuild, "side": side, "mu": side + rebuild}
