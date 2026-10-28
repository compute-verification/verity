"""P2, **experimental**: the squaring-based POUS scheme from the efficient-cryptography campaign (POUS
`internal/efficient-crypto/p2-verdict.md`; its spec is the POUS store's `docs/efficient-crypto/p2-spec-draft.md`, frozen
v1, amended to v1.1 (the RTT cap) and to v4 (one hash, SHA-512) in `../p2/PROTOCOL.md`, which this
module implements).  Every block is independent, and its decode is two big-integer squarings.  Two codecs differ only
in how a block's keys are expanded:

- `p2-16448/v4`: counter-mode SHA-512 with the spec's v4 framing, the expander its setup assumption P2-EXP-IO names
  (decision F4).
- `p2-16448/v2`: ChaCha8 from a SHAKE256 seed per segment.  It is **nonconforming** to F4, which adds a reduced-round
  assumption, but its keys cost about 1% of the decode's multiply work on the device.

~~~text
p          2^w − c0, prime, p ≡ 3 mod 4           registered: w = 16448, c0 = 21065 (`p2-p16448-c21065`, 2,056-byte blocks)
payload    m < 2^(w−16): 2,054 bytes of W per block, the spec's payload profile `p2-2054B` (rate 1.001)
integers   every block, payload and key is fixed-width, unsigned and big-endian, as the spec's wire format
sq(a)      a² mod p, negated (p − a² mod p) when a is odd
ρ(y)       the parity-canonical root of ±y, so sq(ρ(y)) = y: even iff y is a quadratic residue
σ_K(r)     r ⊕ K if r ⊕ K < p, else r             an involution on [0, p)
encode     c = ρ(σ_K(ρ(m + t mod p)))            two roots: slow (w − 2 squarings each)
decode     m = sq(σ_K(sq(c))) − t mod p           two squarings
keys       fresh per block j of segment s, drawn after W through the salt, and expanded where they are used (on the
           device), so nothing key-sized is stored.  K is the first w/8-byte big-endian word of a stream, and t the first
           later word below p (exact rejection sampling; a candidate is ≥ p with probability c0 / 2^w).
  v4       prefix = Z("verity/pous/p2/keys/v4\0" ‖ frame("p2/v4") ‖ frame(prime id) ‖ frame(salt)), Z zero-pads to
           whole 128-byte blocks;  base = prefix ‖ u64be s ‖ u64be j
           stream_P = SHA512(base ‖ P ‖ u32be 0) ‖ SHA512(base ‖ P ‖ u32be 1) ‖ …;  K from stream_K, t from stream_t
  v2       seed_s = SHAKE256("verity/pous/p2/seed/v1" ‖ tag_s)[0 : 32],  tag_s = salt ‖ u64 s
           one stream, ChaCha8(key = seed_s, nonce = u64 j, counter 0, 1, …): K, then t's candidates
~~~

`frame` is `verity.ml.randomness.frame`, and the prime id is `p2-p{w}-c{c0}`.  After the prefix's midstate, each
64-byte chunk is one compression (the base's last 16 bytes, P and the counter are 21 bytes).  Given the same prefix,
v4's keys are the spec's byte for byte (`tests/fixtures/p2-v4-keys.json`, from `benchmarks/pous/p2_v1/vectors.json`).
The prefix differs from the spec's in one place: where the spec frames the model commitment and its 32-byte setup salt,
v4 frames the protocol's 24-byte salt, which `setup.draw_salt` derives from the commitment to W.  So the salt still
binds the model, but the bytes are not the spec's vectors'; that goes with the rest of the spec's setup framing (its
leaves and challenge derivation), not with the expander.

**What is proved, and what is not.**
- The certificate is the pinned `Pous.SecurityProofs.P2SlackFamilyUncond`, its `λ = 128` row at the spec's k = 105
  (decision F6): for every block count `2^8 ≤ B ≤ 2^23` and every payload width `b` with `20·16448 ≤ 21·b ≤ 21·16447`,
  the profile's 16,432 among them, an adversary holding `⌊ρ·|C|⌋ + 128·B` bits, with sequential reveal, `2^20`
  forward queries per answer and any depth, is accepted with probability at most `1% + 2^-128`.  The 128 bits per block
  are a budget for structure an instantiation might leak.  That is **abstract M1** on P2's exact domain `[0, p)`:
  independent forward-only uniform permutations of `[0, p)`, with no hypothesis, and the same for both codecs.
  `P2.certificate` is it at the registered point alone (`2^19` blocks); `family_certificate` is it for any v4 store in
  that block range, which the run protocol takes where the registered one doesn't fit.
- Above `2^23` blocks the run protocol takes `wide_certificate`: the per-period guarantee
  `Pous.Period.SecurityProofs.periodHeldAt`, for every block count `2^19 ≤ N ≤ 2^27` at the registered width and prime,
  the same `⌊ρ·|C|⌋` at `1% + 2^-128` (`PeriodHeldDemo` at Demo B's `1038·2^16` blocks: 1,060,008,348,186 bits).  It
  is in the ring model, not abstract M1, and rests on the open conjecture `P2RingRoundSecureWide` (unchecked in `N`
  above `2^25`), the device model `RingDeviceAt` and the coin commitments' hiding at `N` (`WIDE_CLAIMS`).
- **The instantiation assumption is not accepted** (spec decision 8, Daniel's deferred default): that
  square–mask–square with fresh per-block keys instantiates M1's slow-inverse permutation is a novel, algebraic
  assumption, and it is outside Lean.
- So is SeqRoot: no 16,448-bit root, nor another computation with its effect on the challenged block, finishes within
  twice `Δ + RTT` on the allowed hardware, a CPU or GPU core with cooperation within one chiplet.  `call_ns` stands for
  it, half the fastest such root (`ROOT_CALL_NS`), and the certificate allows no such call within the deadline.  The
  spec caps the RTT allowance at `RTT_ALLOWANCE_CAP_NS`.
- A shared mask with linear tweaks is model M2, which is not covered; the keys here are per block.
- Neither codec is ever the default.  Its profile says so plainly (`EXPERIMENTAL`).
"""

from __future__ import annotations

import hashlib
import itertools
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cached_property

from verity.ml.randomness import frame

from ..codec import Codec
from ..labels import salt_value, tag
from ..params import SALT_BITS
from ..protocol import CapacityBound, Scheme

EXPERIMENTAL = ("abstract-M1 result (pinned P2SlackFamilyUncond); the P2-M1 instantiation assumption is not accepted, "
                "and the XOR-then-reduce compression attack on it is open")

CHACHA8_NONCONFORMING = "ChaCha8 key expander, nonconforming to the spec's F4 (counter-mode SHA-512)"

SHA512_CTR, CHACHA8 = "sha512-ctr-v4", "chacha8"
"""The two key expanders: v4's and v2's."""

KEYS_TAG = b"verity/pous/p2/keys/v4\0"
PROTOCOL_ID = "p2/v4"
SEED_DOMAIN = b"verity/pous/p2/seed/v1"

P2_CLAIMS = ("m1-ideal-permutations", "sms-m1-instantiation", "seq-root", "random-oracle", "perfect-isolation",
             "cr/sha-512", "live-verifier", "designated-verifier", "trusted-host")
"""What P2's certificate rests on:
- abstract M1 (the proof's model);
- the SMS-to-M1 instantiation (not accepted);
- SeqRoot;
- counter-mode SHA-512 as a random oracle for the keys (v4's, the spec's P2-EXP-IO), or SHAKE256 for v2's segment seeds
  (whose ChaCha8 streams P2-EXP-IO does not name);
- the help-accounted row (`perfect-isolation`, the earlier model's id): the certificate bounds what the audited location
  held plus the help it received, and nothing attests the prover's machine;
- SHA-512 collision resistance for the commitment to W and for `vk`;
- a live, designated verifier, whose memory challenger is the auditor's hardware, a VM under the auditor's account
  that the prover doesn't control and nothing attests (core's `trusted-host`)."""

WIDE_CLAIMS = (*P2_CLAIMS, "p2-ring-round-secure-wide", "pous-ring-device-at", "statistical-hiding")
"""What `wide_certificate` rests on: `P2_CLAIMS`, and one claim for each premise `periodHeldAt` takes at `N` blocks, the
conjecture `Pous.P2.Assumptions.P2RingRoundSecureWide`, the device model `Pous.Period.RingDeviceAt` and the coin
commitments' hiding `Pous.P2.Assumptions.CoinCommitHides (coinViewAt N)`."""

POINT = (16448, 21065, 1 << 16)
"""`w`, `c0` and blocks per segment at the registered point.  `2^19` blocks, the certificate's store (1.08 GB), are 8
segments."""

CERT_BLOCKS = 1 << 19

CERT_FAMILY_BLOCKS = (1 << 8, 1 << 23)
"""The block counts `P2SlackFamilyUncond` proves its `λ = 128` row at `k = 105` for, at the registered width and prime
(`family_certificate`)."""

CERT_WIDE_BLOCKS = (1 << 19, 1 << 27)
"""The block counts `Pous.Period.PeriodHeldAt` covers at the registered width and prime (`wide_certificate`), the range
of `P2RingRoundSecureWide`."""

CERT_K = 105
"""The spec's k (decision F6): `P2SlackFamilyUncond`'s `λ = 128` row, for every block count from 2^8 to 2^23."""

ROOT_CALL_NS = 854_029
"""Half the fastest 16,448-bit root on the allowed hardware, 8 cores of one chiplet cooperating, in ns rounded down:
the `Δ + RTT` that SeqRoot's 2× margin must stay under.  The certificate allows no such call within the deadline.  The
single-core floor is the timing red team's Zen 5 at 5.7 GHz with full-rate IFMA, 2,060.845 cycles = 361.55 ns per
squaring (`regate.py`) over the root's 16,486 squaring-equivalents, 5,960.54 µs; the measured 8-core speedup on one
Zen 4 chiplet is 2,716.7 / 778.5 = 3.48966× per squaring (run r20260929-060834-0ec4,
art:b92b6d56fa12cc8e04b3f6843991c2e76f49366adc89140a96c33842f70b41ed).  So the cooperating root is 1,708,058.35 ns
and half of it 854,029.18 ns (`note:20261004T2102Z-report-relay-p2-redteam-timing`,
art:23a323f313d2193873c59ee8ab59a0f1197167a0b4c6d008346d771c3e72de57; spec v1.1, `p2/PROTOCOL.md` §12)."""

DELTA_NS = 500_000
"""Δ at the spec's operating point, 0.5 ms, with the verifier off the responder's node (`p2/PROTOCOL.md` §6, decision F2)."""

RTT_ALLOWANCE_CAP_NS = 354_000
"""The cap on the measured RTT allowance in spec v1.1 (`p2/PROTOCOL.md` §12): the
largest whole µs with `Δ + cap < ROOT_CALL_NS`, so `Δ + RTT ≤ 854 µs`.  The certificate refuses every whole µs past
it, and the verifier any RTT past it or any Δ but `DELTA_NS` (`P2.deadline_refusals`), which the certificate's sum
alone admits up to `ROOT_CALL_NS − 1`."""


@dataclass(frozen=True, slots=True)
class P2Params:
    """`segments` segments of `blocks_per_segment` blocks; each block stores `w` bits and carries `w − 16` bits of W.
    `expander` is `SHA512_CTR` (v4) or `CHACHA8` (v2)."""

    w: int
    c0: int
    blocks_per_segment: int
    segments: int = 1
    expander: str = SHA512_CTR

    def __post_init__(self) -> None:
        for name in ("w", "c0", "blocks_per_segment", "segments"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.w % 8 or self.w <= 16:
            raise ValueError("w is a whole number of bytes above 16 bits")
        if self.c0 % 4 != 1 or self.c0 >= 1 << 16:
            raise ValueError("c0 ≡ 1 mod 4 (so p ≡ 3 mod 4) and below 2^16")
        if self.blocks_per_segment > 1 << 16:
            raise ValueError("block indices within a segment are 16 bits (the verifier's leaves)")
        if self.expander not in (SHA512_CTR, CHACHA8):
            raise ValueError(f"the expander is {SHA512_CTR!r} (v4) or {CHACHA8!r} (v2)")

    @property
    def p(self) -> int:
        return (1 << self.w) - self.c0

    @property
    def m(self) -> int:
        return self.w

    @property
    def block_bytes(self) -> int:
        return self.w // 8

    @property
    def payload_bytes(self) -> int:
        return (self.w - 16) // 8

    @property
    def blocks(self) -> int:
        return self.segments * self.blocks_per_segment

    @property
    def segment_bytes(self) -> int:
        return self.blocks_per_segment * self.payload_bytes

    @property
    def w_bytes(self) -> int:
        return self.segments * self.segment_bytes

    @property
    def prime_id(self) -> str:
        return f"p2-p{self.w}-c{self.c0}"

    @property
    def profile(self) -> str:
        """The payload profile: `p2-2054B` at the registered width, whose manifest (and so the commitment) names it."""
        return f"p2-{self.payload_bytes}B"

    @property
    def codec(self) -> Codec:
        return Codec(f"p2-{self.w}/{'v4' if self.expander == SHA512_CTR else 'v2'}", encode_segment, decode_segment)

    def with_segments(self, segments: int) -> P2Params:
        return P2Params(self.w, self.c0, self.blocks_per_segment, segments, self.expander)

    def manifest(self) -> dict[str, int | str]:
        """v4's names its expander, so the commitments of the two codecs to one W differ; v2's is as it was."""
        out: dict[str, int | str] = {"scheme": "p2", "w": self.w, "c0": self.c0, "profile": self.profile,
                                     "blocks_per_segment": self.blocks_per_segment, "segments": self.segments}
        if self.expander == SHA512_CTR:
            out["expander"] = SHA512_CTR
        return out


class _Field:
    """Arithmetic mod `p = 2^w − c0`: squaring with two folds, and the parity-canonical root."""

    def __init__(self, w: int, c0: int) -> None:
        self.w, self.c0, self.p, self.mask = w, c0, (1 << w) - c0, (1 << w) - 1
        self.small = (c0 - 1) // 4   # (p + 1)/4 = 2^(w−2) − small

    def square(self, a: int) -> int:
        x = a * a
        x = (x & self.mask) + self.c0 * (x >> self.w)
        x = (x & self.mask) + self.c0 * (x >> self.w)
        # Two folds leave x below 2^w + c0·(c0 + 1), under 2p only when c0² is small against 2^w (as at every registered
        # width); `%` is exact for the large c0 a small width accepts.
        return x % self.p if x >= self.p else x

    def sq(self, a: int) -> int:
        """`sq(a)`: a² mod p, negated when a is odd."""
        s = self.square(a)
        return (self.p - s) % self.p if a & 1 else s

    def _sqrt(self, y: int) -> int:
        """`y^((p+1)/4)`: `w − 2` squarings of `y`, times the inverse of `y^small`."""
        if y == 0:
            return 0
        x = y
        for _ in range(self.w - 2):
            x = self.square(x)
        return x * pow(pow(y, self.small, self.p), -1, self.p) % self.p

    def rho(self, y: int) -> int:
        """The parity-canonical root of ±y: `sq(ρ(y)) = y`."""
        y %= self.p
        x = self._sqrt(y)
        if x * x % self.p == y:
            return x if x % 2 == 0 else self.p - x
        x = self._sqrt((self.p - y) % self.p)
        if x * x % self.p != (self.p - y) % self.p:
            raise ArithmeticError("neither y nor −y is a square: p is not ≡ 3 mod 4 prime")
        return x if x % 2 == 1 else self.p - x

    def sigma(self, r: int, k: int) -> int:
        x = r ^ k
        return x if x < self.p else r


def _field(params: P2Params) -> _Field:
    return _FIELDS.setdefault((params.w, params.c0), _Field(params.w, params.c0))


_FIELDS: dict[tuple[int, int], _Field] = {}


_M32 = 0xFFFFFFFF
_SIGMA = (0x61707865, 0x3320646E, 0x79622D32, 0x6B206574)   # "expand 32-byte k"


def chacha_block(key: bytes, counter: int, nonce: int, rounds: int = 8) -> bytes:
    """One 64-byte block of Bernstein's ChaCha: 256-bit key, 64-bit block counter and 64-bit nonce (state words 12–15
    are the counter then the nonce, little endian), `rounds` rounds."""
    x = [*_SIGMA, *struct.unpack("<8I", key), counter & _M32, counter >> 32, nonce & _M32, nonce >> 32]
    s = list(x)

    def qr(a: int, b: int, c: int, d: int) -> None:
        x[a] = (x[a] + x[b]) & _M32; x[d] ^= x[a]; x[d] = ((x[d] << 16) | (x[d] >> 16)) & _M32
        x[c] = (x[c] + x[d]) & _M32; x[b] ^= x[c]; x[b] = ((x[b] << 12) | (x[b] >> 20)) & _M32
        x[a] = (x[a] + x[b]) & _M32; x[d] ^= x[a]; x[d] = ((x[d] << 8) | (x[d] >> 24)) & _M32
        x[c] = (x[c] + x[d]) & _M32; x[b] ^= x[c]; x[b] = ((x[b] << 7) | (x[b] >> 25)) & _M32

    for _ in range(rounds // 2):
        qr(0, 4, 8, 12); qr(1, 5, 9, 13); qr(2, 6, 10, 14); qr(3, 7, 11, 15)
        qr(0, 5, 10, 15); qr(1, 6, 11, 12); qr(2, 7, 8, 13); qr(3, 4, 9, 14)
    return struct.pack("<16I", *((a + b) & _M32 for a, b in zip(x, s)))


def _salt(salt: bytes) -> bytes:
    if type(salt) is not bytes or len(salt) != SALT_BITS // 8:
        raise ValueError(f"the salt is {SALT_BITS // 8} bytes, drawn by the verifier after W is committed")
    return salt


def segment_seed(salt: bytes, s: int) -> bytes:
    """v2: segment `s`'s 32-byte ChaCha8 key, from `tag_s = salt ‖ u64 s` (injective across segments)."""
    return hashlib.shake_256(SEED_DOMAIN + tag(salt_value(_salt(salt)), s).to_bytes(32, "little")).digest(32)


def expand_prefix(params: P2Params, salt: bytes) -> bytes:
    """v4: every chunk's SHA-512 input before its `u64be(segment) ‖ u64be(block) ‖ purpose ‖ u32be(counter)`, zero-padded
    to whole 128-byte blocks so that a chunk is one compression after the prefix's midstate."""
    prefix = KEYS_TAG + frame(PROTOCOL_ID) + frame(params.prime_id) + frame(_salt(salt))
    return prefix + bytes(-len(prefix) % 128)


def stream(base: bytes, purpose: bytes, nbytes: int) -> bytes:
    """The first `nbytes` of `SHA512(base ‖ purpose ‖ u32be(0)) ‖ SHA512(base ‖ purpose ‖ u32be(1)) ‖ …`."""
    h = hashlib.sha512(base + purpose)
    out = bytearray()
    for i in range(-(-nbytes // 64)):
        c = h.copy()
        c.update(i.to_bytes(4, "big"))
        out += c.digest()
    return bytes(out[:nbytes])


def below(p: int, candidates) -> int:
    """The first candidate below `p`: exact rejection sampling, never a reduction mod p."""
    return next(x for x in candidates if x < p)


def sha512_keys(params: P2Params, prefix: bytes, s: int, j: int) -> tuple[int, int]:
    """v4: block `j` of segment `s`'s `(K, t)` after `prefix` (`expand_prefix`), the spec's §4 streams: `K` is the first
    `w/8` bytes of `stream(base, "K")`, and `t` the first `w/8`-byte word of `stream(base, "t")` below `p`."""
    nb, base = params.block_bytes, prefix + s.to_bytes(8, "big") + j.to_bytes(8, "big")
    k = int.from_bytes(stream(base, b"K", nb), "big")
    return k, below(params.p, (int.from_bytes(stream(base, b"t", (i + 1) * nb)[i * nb:], "big") for i in itertools.count()))


def chacha_keys(params: P2Params, seed: bytes, j: int) -> tuple[int, int]:
    """v2: block `j`'s `(K, t)` from its segment's seed: the ChaCha8 stream with nonce `j`, read as consecutive
    big-endian `w/8`-byte words.  `K` is the first, and `t` the first later word below `p`."""
    nb, blocks = params.block_bytes, (chacha_block(seed, i, j) for i in itertools.count())
    buf = bytearray()

    def word() -> int:
        while len(buf) < nb:
            buf.extend(next(blocks))
        out = int.from_bytes(buf[:nb], "big")
        del buf[:nb]
        return out

    k = word()
    return k, below(params.p, iter(word, None))


def segment_keys(params: P2Params, salt: bytes, s: int) -> Callable[[int], tuple[int, int]]:
    """Segment `s`'s keys by block, `j ↦ (K, t)`, under the params' expander."""
    if params.expander == CHACHA8:
        seed = segment_seed(salt, s)
        return lambda j: chacha_keys(params, seed, j)
    prefix = expand_prefix(params, salt)
    return lambda j: sha512_keys(params, prefix, s, j)


def block_keys(params: P2Params, salt: bytes, s: int, j: int) -> tuple[int, int]:
    """Block `j` of segment `s`'s fresh keys `(K, t)`."""
    return segment_keys(params, salt, s)(j)


def _check(params: P2Params, s: int) -> None:
    if not isinstance(params, P2Params):
        raise TypeError("P2's codec takes P2Params")
    if not 0 <= s < params.segments:
        raise ValueError(f"segment {s} is outside 0..{params.segments - 1}")


def encode_blocks(params: P2Params, salt: bytes, s: int, j0: int, payloads: bytes) -> list[bytes]:
    """Blocks `j0, j0 + 1, …` of segment `s` of C from their payloads, `len(payloads)` a whole number of them: a slice of
    `encode_segment`, so a segment's blocks can be encoded apart (each block's keys are its own)."""
    _check(params, s)
    f, pb, nb = _field(params), params.payload_bytes, params.block_bytes
    n, rest = divmod(len(payloads), pb)
    if rest or not 0 <= j0 <= j0 + n <= params.blocks_per_segment:
        raise ValueError(f"blocks {j0}.. of {len(payloads)} payload bytes are not whole blocks of segment {s}")
    keys, out = segment_keys(params, salt, s), []
    for j in range(n):
        k, t = keys(j0 + j)
        m = int.from_bytes(payloads[j * pb:(j + 1) * pb], "big")
        out.append(f.rho(f.sigma(f.rho((m + t) % f.p), k)).to_bytes(nb, "big"))
    return out


def encode_segment(params: P2Params, prims, salt: bytes, s: int, w_seg: bytes) -> list[bytes]:
    """Segment `s` of C: `c_j = ρ(σ_K(ρ(m_j + t_j)))` per block (two roots each)."""
    _check(params, s)
    if len(w_seg) != params.segment_bytes:
        raise ValueError(f"a segment of W is {params.segment_bytes} bytes")
    return encode_blocks(params, salt, s, 0, w_seg)


def decode_segment(params: P2Params, prims, salt: bytes, s: int, blocks: Sequence[bytes]) -> bytes:
    """Segment `s` of W: `m_j = sq(σ_K(sq(c_j))) − t_j mod p` per block (two squarings each).  Raises `ValueError` on a
    block that is not a codeword (`c ≥ p`, or a result of more than `w − 16` bits)."""
    _check(params, s)
    if len(blocks) != params.blocks_per_segment:
        raise ValueError(f"a segment of C is {params.blocks_per_segment} blocks")
    f, pb, nb = _field(params), params.payload_bytes, params.block_bytes
    keys, out = segment_keys(params, salt, s), bytearray()
    for j, block in enumerate(blocks):
        if len(block) != nb:
            raise ValueError(f"block {j} is not {nb} bytes")
        c = int.from_bytes(block, "big")
        if c >= f.p:
            raise ValueError(f"block {j} is not below p")
        k, t = keys(j)
        m = (f.sq(f.sigma(f.sq(c), k)) - t) % f.p
        if m >> (8 * pb):
            raise ValueError(f"block {j} decodes to more than {8 * pb} bits: not a codeword")
        out += m.to_bytes(pb, "big")
    return bytes(out)


class P2(Scheme):
    """P2 v4 (counter-mode SHA-512 keys) at its parameters; experimental, never the default."""

    experimental = EXPERIMENTAL
    expander = SHA512_CTR
    backend = staticmethod(lambda params: None)

    def __init__(self, params: P2Params | None = None, prims=None) -> None:
        params = P2Params(*POINT, expander=self.expander) if params is None else params
        if not isinstance(params, P2Params):
            raise TypeError("P2 takes P2Params")
        if params.expander != self.expander:
            raise ValueError(f"{type(self).__name__} expands keys by {self.expander}, not {params.expander}")
        super().__init__(params, prims)

    @property
    def name(self) -> str:
        return self.params.codec.name

    @cached_property
    def certificate(self) -> CapacityBound | None:
        p = self.params
        if (p.w, p.c0, p.blocks_per_segment) != POINT:
            return None
        return CapacityBound(k=CERT_K, rounds=0, work=1 << 20, call_ns=ROOT_CALL_NS,
                             theorem="Pous.SecurityProofs.P2SlackFamilyUncond", segments=CERT_BLOCKS // POINT[2],
                             claims=P2_CLAIMS)

    def deadline_refusals(self, deadline_ns: int, rtt_allowance_ns: int) -> tuple[str, ...]:
        out = []
        if deadline_ns != DELTA_NS:
            out.append(f"the spec's operating point sets Δ = {DELTA_NS} ns, not {deadline_ns}")
        if rtt_allowance_ns > RTT_ALLOWANCE_CAP_NS:
            out.append(f"the RTT allowance of {rtt_allowance_ns} ns is past the spec's cap of {RTT_ALLOWANCE_CAP_NS} ns")
        return tuple(out)

    def encode_segment(self, salt: bytes, s: int, w_seg: bytes) -> list[bytes]:
        return encode_segment(self.params, self.prims, salt, s, w_seg)

    def decode_segment(self, salt: bytes, s: int, blocks: Sequence[bytes]) -> bytes:
        return decode_segment(self.params, self.prims, salt, s, blocks)


class P2ChaCha(P2):
    """P2 v2 (ChaCha8 keys): experimental, and nonconforming to the spec's expander."""

    experimental = f"{EXPERIMENTAL}; {CHACHA8_NONCONFORMING}"
    expander = CHACHA8


def family_certificate(params: P2Params) -> CapacityBound | None:
    """`P2SlackFamilyUncond`'s `λ = 128` row for a v4 store at the registered width and prime of any block count in
    `CERT_FAMILY_BLOCKS`, in any number of segments: the theorem is about `m1p B 16448 (2^16448 − 21065) b` for every such
    `B`, and segments only frame each block's keys.  `P2.certificate` stays the registered point's alone, the one the
    memory challenger's coverage and the timed audit's Lean properties read; the run protocol (`period.gaps`,
    `PousCheck.RunProtocol.gaps`) takes this one where the registered one doesn't fit."""
    lo, hi = CERT_FAMILY_BLOCKS
    if (params.w, params.c0) != POINT[:2] or params.expander != SHA512_CTR or not lo <= params.blocks <= hi:
        return None
    return CapacityBound(k=CERT_K, rounds=0, work=1 << 20, call_ns=ROOT_CALL_NS,
                         theorem="Pous.SecurityProofs.P2SlackFamilyUncond", segments=None, claims=P2_CLAIMS)


def wide_certificate(params: P2Params) -> CapacityBound | None:
    """`Pous.Period.SecurityProofs.periodHeldAt` for a v4 store at the registered width and prime of any block count in
    `CERT_WIDE_BLOCKS`, in any number of segments: at every point of the run's composed check and every `N` from `2^19`
    to `2^27`, an accepted period with no break while the node held fewer than `⌊ρ·N·16448⌋` bits has probability at
    most `δ + ε_max = 1% + 2^-128`, read at `N` = the store's blocks, under `WIDE_CLAIMS`.  The run protocol takes it
    above `family_certificate`'s range (`period.certificate`, `PousCheck.RunProtocol.certificate`)."""
    lo, hi = CERT_WIDE_BLOCKS
    if (params.w, params.c0) != POINT[:2] or params.expander != SHA512_CTR or not lo <= params.blocks <= hi:
        return None
    return CapacityBound(k=CERT_K, rounds=0, work=1 << 20, call_ns=ROOT_CALL_NS,
                         theorem="Pous.Period.SecurityProofs.periodHeldAt", segments=None, claims=WIDE_CLAIMS)


__all__ = ["CERT_BLOCKS", "CERT_FAMILY_BLOCKS", "CERT_K", "CERT_WIDE_BLOCKS", "CHACHA8", "CHACHA8_NONCONFORMING", "DELTA_NS", "EXPERIMENTAL", "KEYS_TAG", "P2_CLAIMS",
           "POINT", "PROTOCOL_ID", "ROOT_CALL_NS", "RTT_ALLOWANCE_CAP_NS", "SEED_DOMAIN", "SHA512_CTR", "WIDE_CLAIMS", "P2", "P2ChaCha", "P2Params",
           "below", "block_keys", "chacha_block", "chacha_keys", "decode_segment", "encode_blocks", "encode_segment",
           "expand_prefix", "family_certificate", "segment_keys", "segment_seed", "sha512_keys", "stream", "wide_certificate"]
