"""A stateful hash-based signature: LMS with LM-OTS (RFC 8554) over SHA-512, in plain Python, the scheme of the network
certifier's certificates and the memory challenger's records, and the reference of their in-proof verifier
(`verity.primitives.commitments.gates.lms`).

**The hash.** ``H(x) = SHA-512(x)[:N]``, N = 32 (SHA-512 truncated, the default) or 64 (or `TOY_N` = 8, asked for by name,
for circuit-check's binding of the verifier). Every hash input is RFC 8554's, with
its domain separators: a chain step ``H(I || u32(q) || u16(i) || u8(j) || tmp)``, the message
``Q = H(I || u32(q) || u16(D_MESG) || C || message)``, the one-time public key ``K = H(I || u32(q) || u16(D_PBLC) || z_0 ||
... || z_{P-1})``, a leaf ``H(I || u32(r) || u16(D_LEAF) || K)`` and an interior node ``H(I || u32(r) || u16(D_INTR) || left
|| right)``, node r of the tree numbered from the root r = 1 (a leaf q is node 2^H + q). A message is any bytes; a
certifier signs a record's 64-byte SHA-512 digest, the message the in-proof verifier takes.

**Parameters.** (N, W, H): N the hash length, W in {1, 2, 4, 8} the Winternitz width, H the tree height (2^H one-time keys;
20 for a certifier). P = U + V chains, U = ceil(8N / W) message digits and V checksum digits, and the checksum shifted left
LS = 16 - V W bits (RFC 8554 §4.1, Appendix B).

**Differences from RFC 8554.** No type fields: a verifier holds one parameter set, so a signature is ``u32(q) || C || y_0 ||
... || y_{P-1} || path_0 || ... || path_{H-1}`` and a public key ``I || T[1]``. Keys are deterministic from a seed (the RFC's
Appendix A): ``x_q[i] = H(I || u32(q) || u16(i) || u8(0xff) || SEED)``, ``C = H(I || u32(q) || u16(0xfffd) || u8(0xff) ||
SEED)`` and ``I = SHA-512(b"verity/lms/v0/I" || SEED)[:16]``. The scheme is stateful: a key signs at most once per q,
and keeping q is the signer's (`sign` takes it). With the hash swapped for SHA-256 (`Params.hash`), the tests reproduce
RFC 8554's Test Case 2.

Standard library only, and slow: keygen hashes 2^H P (2^W - 1) times, so the tests use small H.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

__all__ = ["D_INTR", "D_LEAF", "D_MESG", "D_PBLC", "TOY_N", "Params", "PrivateKey", "Signature", "digits", "key", "keygen",
           "public_key", "root_of", "sign", "verify"]

D_PBLC, D_MESG, D_LEAF, D_INTR = 0x8080, 0x8181, 0x8282, 0x8383


def _u32(x: int) -> bytes:
    return x.to_bytes(4, "big")


def _u16(x: int) -> bytes:
    return x.to_bytes(2, "big")


#: The one N below 32 that `Params` admits, and only with ``toy=True``: one SHA-512 word, the smallest value of the same byte and
#: word layout, for circuit-check's binding of the in-proof verifier. Never a key's: 64 bits of hash is no signature.
TOY_N = 8


@dataclass(frozen=True)
class Params:
    N: int
    W: int
    H: int
    toy: bool = field(default=False, compare=False)

    def __post_init__(self) -> None:
        sizes = (TOY_N, 32, 64) if self.toy else (32, 64)
        if self.N not in sizes or self.W not in (1, 2, 4, 8) or isinstance(self.H, bool) or not 1 <= self.H <= 30:
            raise ValueError(f"LMS parameters are N in {sizes}, W in (1, 2, 4, 8), 1 <= H <= 30, not {self}")

    @property
    def U(self) -> int:
        return -(-8 * self.N // self.W)

    @property
    def V(self) -> int:
        return -(-(((2 ** self.W - 1) * self.U).bit_length()) // self.W)

    @property
    def P(self) -> int:
        return self.U + self.V

    @property
    def LS(self) -> int:
        return 16 - self.V * self.W

    def hash(self, data: bytes) -> bytes:
        return hashlib.sha512(data).digest()[:self.N]

    def signature_bytes(self) -> int:
        return 4 + self.N * (1 + self.P + self.H)


@dataclass(frozen=True)
class PrivateKey:
    params: Params
    I: bytes
    seed: bytes
    tree: tuple[bytes, ...]  # node r at index r, 1 <= r < 2^(H+1)

    @property
    def root(self) -> bytes:
        return self.tree[1]


@dataclass(frozen=True)
class Signature:
    q: int
    C: bytes
    y: tuple[bytes, ...]
    path: tuple[bytes, ...]

    def encode(self) -> bytes:
        return _u32(self.q) + self.C + b"".join(self.y) + b"".join(self.path)

    @classmethod
    def decode(cls, params: Params, data: bytes) -> "Signature":
        if len(data) != params.signature_bytes():
            raise ValueError(f"an LMS signature for {params} is {params.signature_bytes()} bytes, not {len(data)}")
        n, q = params.N, int.from_bytes(data[:4], "big")
        cut = [data[4 + n * k:4 + n * (k + 1)] for k in range(1 + params.P + params.H)]
        return cls(q, cut[0], tuple(cut[1:1 + params.P]), tuple(cut[1 + params.P:]))


def _coef(s: bytes, i: int, w: int) -> int:
    return ((2 ** w - 1) & (s[i * w // 8] >> (8 - (w * (i % (8 // w)) + w))))


def digits(params: Params, Q: bytes) -> list[int]:
    """The P chain lengths a signature reveals: Q's U digits, then its checksum's V (RFC 8554 §4.4, ``coef(Q || Cksm(Q))``)."""
    w = params.W
    cksm = sum(2 ** w - 1 - _coef(Q, i, w) for i in range(params.U)) << params.LS
    s = Q + _u16(cksm)
    return [_coef(s, i, w) for i in range(params.P)]


def _chain(params: Params, I: bytes, q: int, i: int, tmp: bytes, start: int, stop: int) -> bytes:
    for j in range(start, stop):
        tmp = params.hash(I + _u32(q) + _u16(i) + bytes([j]) + tmp)
    return tmp


def _x(params: Params, I: bytes, seed: bytes, q: int, i: int) -> bytes:
    return params.hash(I + _u32(q) + _u16(i) + b"\xff" + seed)


def _ots_public(params: Params, I: bytes, q: int, z: list[bytes]) -> bytes:
    return params.hash(I + _u32(q) + _u16(D_PBLC) + b"".join(z))


def _leaf(params: Params, I: bytes, q: int, K: bytes) -> bytes:
    return params.hash(I + _u32(2 ** params.H + q) + _u16(D_LEAF) + K)


def _interior(params: Params, I: bytes, r: int, left: bytes, right: bytes) -> bytes:
    return params.hash(I + _u32(r) + _u16(D_INTR) + left + right)


def keygen(params: Params, seed: bytes) -> PrivateKey:
    """The key with 2^H one-time keys that ``seed`` (N bytes) determines, with its whole tree."""
    if len(seed) != params.N:
        raise ValueError(f"an LMS seed is {params.N} bytes, not {len(seed)}")
    return key(params, hashlib.sha512(b"verity/lms/v0/I" + seed).digest()[:16], seed)


def key(params: Params, I: bytes, seed: bytes) -> PrivateKey:
    """The key of identifier ``I`` (16 bytes) whose one-time keys ``seed`` derives (RFC 8554 Appendix A)."""
    if len(I) != 16:
        raise ValueError(f"an LMS identifier is 16 bytes, not {len(I)}")
    top = 2 ** params.H
    tree: list[bytes] = [b""] * (2 * top)
    e = 2 ** params.W - 1
    for q in range(top):
        z = [_chain(params, I, q, i, _x(params, I, seed, q, i), 0, e) for i in range(params.P)]
        tree[top + q] = _leaf(params, I, q, _ots_public(params, I, q, z))
    for r in range(top - 1, 0, -1):
        tree[r] = _interior(params, I, r, tree[2 * r], tree[2 * r + 1])
    return PrivateKey(params, I, seed, tuple(tree))


def public_key(key: PrivateKey) -> bytes:
    """``I || T[1]``."""
    return key.I + key.root


def _message_hash(params: Params, I: bytes, q: int, C: bytes, message: bytes) -> bytes:
    return params.hash(I + _u32(q) + _u16(D_MESG) + C + message)


def sign(key: PrivateKey, q: int, message: bytes) -> Signature:
    """The signature of ``message`` with one-time key q, which the caller never uses again."""
    p = key.params
    if not 0 <= q < 2 ** p.H:
        raise ValueError(f"q is below 2^{p.H}, not {q}")
    C = p.hash(key.I + _u32(q) + _u16(0xFFFD) + b"\xff" + key.seed)
    a = digits(p, _message_hash(p, key.I, q, C, message))
    y = tuple(_chain(p, key.I, q, i, _x(p, key.I, key.seed, q, i), 0, a[i]) for i in range(p.P))
    r, path = 2 ** p.H + q, []
    while r > 1:
        path.append(key.tree[r ^ 1])
        r //= 2
    return Signature(q, C, y, tuple(path))


def root_of(params: Params, I: bytes, message: bytes, sig: Signature) -> bytes:
    """The root the signature's one-time key and path give (RFC 8554 Algorithms 4b and 6a)."""
    if not 0 <= sig.q < 2 ** params.H or len(sig.y) != params.P or len(sig.path) != params.H:
        raise ValueError(f"a signature for {params} has q below 2^H, P chain values and H path nodes")
    e = 2 ** params.W - 1
    a = digits(params, _message_hash(params, I, sig.q, sig.C, message))
    z = [_chain(params, I, sig.q, i, sig.y[i], a[i], e) for i in range(params.P)]
    tmp, r = _leaf(params, I, sig.q, _ots_public(params, I, sig.q, z)), 2 ** params.H + sig.q
    for sib in sig.path:
        tmp = _interior(params, I, r // 2, sib, tmp) if r & 1 else _interior(params, I, r // 2, tmp, sib)
        r //= 2
    return tmp


def verify(params: Params, public: bytes, message: bytes, sig: Signature) -> bool:
    """Whether ``sig`` signs ``message`` under ``public`` (``I || T[1]``)."""
    if len(public) != 16 + params.N:
        raise ValueError(f"an LMS public key is {16 + params.N} bytes, not {len(public)}")
    try:
        return root_of(params, public[:16], message, sig) == public[16:]
    except ValueError:
        return False
