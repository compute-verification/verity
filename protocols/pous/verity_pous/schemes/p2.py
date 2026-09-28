"""`p2-16448/v0`, **experimental**: P2, the squaring-based POUS scheme from the efficient-cryptography campaign (POUS
`internal/efficient-crypto/p2-verdict.md`).  Every block is independent, and its decode is two big-integer squarings
with no hash call.

~~~text
p          2^w − c0, prime, p ≡ 3 mod 4           registered: w = 16448, c0 = 21065 (2,056-byte blocks)
payload    m < 2^(w−16): 2,054 bytes of W per block, little endian (rate 1.001)
sq(a)      a² mod p, negated (p − a² mod p) when a is odd
ρ(y)       the parity-canonical root of ±y, so sq(ρ(y)) = y: even iff y is a quadratic residue
σ_K(r)     r ⊕ K if r ⊕ K < p, else r             an involution on [0, p)
encode     c = ρ(σ_K(ρ(m + t mod p)))            two roots: slow (w − 2 squarings each)
decode     m = sq(σ_K(sq(c))) − t mod p           two squarings
keys       fresh per block j of segment s, drawn after W through the salt:
           K = SHAKE256("verity/pous/p2/K/v0" ‖ tag_s ‖ u64 j)[0 : w/8]
           t = SHAKE256("verity/pous/p2/t/v0" ‖ tag_s ‖ u64 j)[0 : w/8 + 8] mod p,  tag_s = salt ‖ u64 s
~~~

**What is proved, and what is not.**
- The certificate is `P2MeetsM1Uncond` (POUS `lean/pous/Pous/Pinned.lean`, graded PASS; statement reviewer
  bc-d1405c07, GO WITH CONDITIONS): `∀ D, Meets (m1 (2^23) 16448) .sequential D (2^20) 88 εMax`. That is **abstract
  M1**: 2^23 independent forward-only ideal permutations on 16,448 bits, with no hypothesis.
- **The instantiation assumption is pending Daniel.** That square–mask–square with fresh per-block keys instantiates
  M1's slow-inverse permutation is a novel, algebraic assumption, and it is outside Lean.
- So is SeqRoot: one 16,448-bit root takes longer than twice `Δ + RTT` on the fastest core, with cooperating cores
  excluded. It sets `call_ns`, and the certificate allows no root within the deadline.
- A shared mask with linear tweaks is model M2, which is not covered; the keys here are per block.
- The scheme is never the default. Its profile says, plainly, "abstract-M1 result; instantiation assumption pending
  Daniel".
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property

from ..codec import Codec
from ..labels import salt_value, tag
from ..params import SALT_BITS
from ..protocol import Certificate, Scheme

EXPERIMENTAL = "abstract-M1 result; instantiation assumption pending Daniel"

K_DOMAIN, T_DOMAIN = b"verity/pous/p2/K/v0", b"verity/pous/p2/t/v0"

P2_CLAIMS = ("m1-ideal-permutations", "sms-m1-instantiation", "seq-root", "random-oracle", "perfect-isolation",
             "cr/sha-256", "live-verifier", "designated-verifier")
"""What P2's certificate rests on:
- abstract M1 (the proof's model);
- the SMS-to-M1 instantiation (pending Daniel);
- SeqRoot;
- SHAKE256 as a random oracle for the per-block keys;
- perfect isolation;
- SHA-256 collision resistance for the commitment to W and for `vk`;
- a live, designated verifier."""

POINT = (16448, 21065, 1 << 16)
"""`w`, `c0` and blocks per segment at the registered point.  `2^23` blocks, the certificate's store (17.2 GB), are
128 segments."""

ROOT_CALL_NS = 2_900_000
"""Half the fastest core's time for one 16,448-bit root: the 2× margin of the timing red team, which sized w = 16,448
for `Δ + RTT` up to 2.9 ms (`internal/efficient-crypto/attacks/p2-redteam-timing.md`).  The certificate allows no such
call within the deadline."""


@dataclass(frozen=True, slots=True)
class P2Params:
    """`segments` segments of `blocks_per_segment` blocks; each block stores `w` bits and carries `w − 16` bits of W."""

    w: int
    c0: int
    blocks_per_segment: int
    segments: int = 1

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
    def codec(self) -> Codec:
        return Codec(f"p2-{self.w}/v0", encode_segment, decode_segment)

    def with_segments(self, segments: int) -> P2Params:
        return P2Params(self.w, self.c0, self.blocks_per_segment, segments)

    def manifest(self) -> dict[str, int | str]:
        return {"scheme": "p2", "w": self.w, "c0": self.c0, "blocks_per_segment": self.blocks_per_segment,
                "segments": self.segments}


class _Field:
    """Arithmetic mod `p = 2^w − c0`: squaring with two folds, and the parity-canonical root."""

    def __init__(self, w: int, c0: int) -> None:
        self.w, self.c0, self.p, self.mask = w, c0, (1 << w) - c0, (1 << w) - 1
        self.small = (c0 - 1) // 4   # (p + 1)/4 = 2^(w−2) − small

    def square(self, a: int) -> int:
        x = a * a
        x = (x & self.mask) + self.c0 * (x >> self.w)
        x = (x & self.mask) + self.c0 * (x >> self.w)
        return x - self.p if x >= self.p else x

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


def block_keys(params: P2Params, salt: bytes, s: int, j: int) -> tuple[int, int]:
    """Block `j` of segment `s`'s fresh keys `(K, t)`, keyed by `tag_s = salt ‖ u64 s` (injective across segments)."""
    if type(salt) is not bytes or len(salt) != SALT_BITS // 8:
        raise ValueError(f"the salt is {SALT_BITS // 8} bytes, drawn by the verifier after W is committed")
    ts = tag(salt_value(salt), s).to_bytes(32, "little") + j.to_bytes(8, "little")
    nb = params.block_bytes
    k = int.from_bytes(hashlib.shake_256(K_DOMAIN + ts).digest(nb), "little")
    t = int.from_bytes(hashlib.shake_256(T_DOMAIN + ts).digest(nb + 8), "little") % params.p
    return k, t


def _check(params: P2Params, s: int) -> None:
    if not isinstance(params, P2Params):
        raise TypeError("P2's codec takes P2Params")
    if not 0 <= s < params.segments:
        raise ValueError(f"segment {s} is outside 0..{params.segments - 1}")


def encode_segment(params: P2Params, prims, salt: bytes, s: int, w_seg: bytes) -> list[bytes]:
    """Segment `s` of C: `c_j = ρ(σ_K(ρ(m_j + t_j)))` per block (two roots each)."""
    _check(params, s)
    if len(w_seg) != params.segment_bytes:
        raise ValueError(f"a segment of W is {params.segment_bytes} bytes")
    f, pb, nb = _field(params), params.payload_bytes, params.block_bytes
    out = []
    for j in range(params.blocks_per_segment):
        k, t = block_keys(params, salt, s, j)
        m = int.from_bytes(w_seg[j * pb:(j + 1) * pb], "little")
        out.append(f.rho(f.sigma(f.rho((m + t) % f.p), k)).to_bytes(nb, "little"))
    return out


def decode_segment(params: P2Params, prims, salt: bytes, s: int, blocks: Sequence[bytes]) -> bytes:
    """Segment `s` of W: `m_j = sq(σ_K(sq(c_j))) − t_j mod p` per block (two squarings each).  Raises `ValueError` on a
    block that is not a codeword (`c ≥ p`, or a result of more than `w − 16` bits)."""
    _check(params, s)
    if len(blocks) != params.blocks_per_segment:
        raise ValueError(f"a segment of C is {params.blocks_per_segment} blocks")
    f, pb, nb = _field(params), params.payload_bytes, params.block_bytes
    out = bytearray()
    for j, block in enumerate(blocks):
        if len(block) != nb:
            raise ValueError(f"block {j} is not {nb} bytes")
        c = int.from_bytes(block, "little")
        if c >= f.p:
            raise ValueError(f"block {j} is not below p")
        k, t = block_keys(params, salt, s, j)
        m = (f.sq(f.sigma(f.sq(c), k)) - t) % f.p
        if m >> (8 * pb):
            raise ValueError(f"block {j} decodes to more than {8 * pb} bits: not a codeword")
        out += m.to_bytes(pb, "little")
    return bytes(out)


class P2(Scheme):
    """P2 at its parameters; experimental, never the default."""

    experimental = EXPERIMENTAL
    backend = staticmethod(lambda params: None)

    def __init__(self, params: P2Params | None = None, prims=None) -> None:
        params = P2Params(*POINT) if params is None else params
        if not isinstance(params, P2Params):
            raise TypeError("P2 takes P2Params")
        super().__init__(params, prims)

    @property
    def name(self) -> str:
        return self.params.codec.name

    @cached_property
    def certificate(self) -> Certificate | None:
        p = self.params
        if (p.w, p.c0, p.blocks_per_segment) != POINT:
            return None
        return Certificate(k=88, rounds=0, work=1 << 20, call_ns=ROOT_CALL_NS, theorem="P2MeetsM1Uncond",
                           segments=(1 << 23) // POINT[2], claims=P2_CLAIMS)

    def encode_segment(self, salt: bytes, s: int, w_seg: bytes) -> list[bytes]:
        return encode_segment(self.params, self.prims, salt, s, w_seg)

    def decode_segment(self, salt: bytes, s: int, blocks: Sequence[bytes]) -> bytes:
        return decode_segment(self.params, self.prims, salt, s, blocks)


__all__ = ["EXPERIMENTAL", "K_DOMAIN", "P2_CLAIMS", "POINT", "ROOT_CALL_NS", "T_DOMAIN", "P2", "P2Params", "block_keys",
           "decode_segment", "encode_segment"]
