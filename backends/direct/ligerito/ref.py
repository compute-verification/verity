"""Reference Ligerito polynomial commitment over BabyBear with degree-6 extension challenges (pure Python / numpy).

This file is the SPEC: the GPU prototype (``pcs.py``, lane ligerito-proto) and the Rust verifier (``backends/ligerito-verify``,
lane ligerito-verify-rs) are checked against it bit for bit (``proof_to_dict`` / fixtures).  Toy sizes only (N <= ~2^16);
nothing here is fast.

Conventions (agreed with ligerito-proto 17:45Z):

* Base field ``F_p``, ``p = 2^31 - 2^27 + 1``; challenge field ``F = F_p[x]/(x^6 - 31)``; an element is 6 base coordinates
  (coefficient of ``x^i`` at index ``i``), 24 B little-endian u32 on the wire.
* The committed vector ``f`` has ``N = prod(radices)`` base entries indexed in MIXED RADIX, digit 0 least significant
  (``x = d_0 + rad_0 (d_1 + rad_1 (...))``); every digit is binary except that digit 0 may be ternary (our batches are
  ``3 x 2^28`` cells: the factor 3 from K = 1536 units per VU stays in the TALL dimension of every level).  A *tensor* over
  a digit range is a list of ``(radix_j, 6)`` weight vectors ``w_j``; it denotes the vector ``x -> prod_j w_j[x_j]``.  The
  statement is a tensor claim ``<f, w> = v`` (``w`` over all digits); the multilinear evaluation ``f~(z) = v`` is the tensor
  ``w_j = (1 - z_j, z_j)`` (``eq_tensor``) with a Lagrange-3 factor on a ternary digit (``lagrange3``).
* Level ``i`` (``i = 1..L``) holds ``M_i`` = the current vector reshaped COLUMN-MAJOR into ``tall_i x wide_i = 2^{k'_i}``
  (row index = the low digits, column index = the high ``k'_i`` binary digits), so a tensor over the vector splits into a
  row tensor and a column tensor for free.  Every column is Reed-Solomon encoded in the COEFFICIENT basis on the subgroup
  ``<omega_n>`` of order ``n_i`` (natural order: codeword row ``s`` holds ``sum_r M[r, c] omega_n^{s r}``; no INTT), and the
  Merkle tree is over the ``n_i`` codeword rows (leaf = BLAKE3 of the row as little-endian u32: ``wide_i`` words for the base
  level 1, ``6 wide_i`` words for the extension levels; node = BLAKE3(left || right)).  The commitment is ``root_1``.
* A level's claims are a list of tensor terms ``(gamma_t, a_t (rows), b_t (cols))`` with ``sum_t gamma_t <M_i, a_t (x) b_t> =
  total``; the prover computes ``u_t = M_i^T a_t`` and runs a degree-2 sumcheck over the ``k'_i`` column digits on
  ``sum_t gamma_t b_t(c) u_t(c)`` (messages = 3 coefficients per round, low digit first), ending with challenges ``r`` and the
  folded vector ``v_i = M_i rbar`` (``rbar = eq_tensor(r)`` expanded); the terms carried to ``v_i`` are ``(gamma_t b_t~(r), a_t)``
  with total = the last sumcheck claim.  ``v_i`` reshaped is ``M_{i+1}``.
* After ``root_{i+1}`` is absorbed the verifier picks ``|S_i|`` rows of codeword ``i`` (``coins.indices``), the prover opens them
  (rows + batched Merkle siblings), the verifier computes ``y_s = <row_s, rbar_i>`` itself (so no "expected symbols" travel)
  and each opened row becomes the tensor claim ``<(omega_n^{s r})_r, v_i> = y_s`` (a tensor: ``(1, omega^s, omega^{2s})`` on a
  ternary digit, ``(1, omega^{s 2^j ...})`` on binary ones).  All claims on ``v_i`` are batched with fresh coefficients
  ``alpha`` (one per claim) into level ``i + 1``'s term list.  The last vector ``v_L`` is sent in the clear; its claims and the
  level-``L`` opened rows are checked directly.

Coin order (one ``coins`` round trip each; the transcript IS the protocol): per level ``i``: ``absorb(root_i)``; if ``i >= 2``:
``indices(S_{i-1})``, ``absorb(opened rows_{i-1} || siblings_{i-1})``, ``challenge(1 + |S_{i-1}|) = alpha``; then ``k'_i x``
(``absorb(3 coefficients)``, ``challenge(1) = r_j``).  After level ``L``: ``absorb(v_L)``, ``indices(S_L)``, ``absorb(opened
rows_L || siblings_L)``.  ``FSCoins`` is the Fiat-Shamir instance used for the fixtures (BLAKE3 chain + BLAKE3 XOF, u64 mod p
sampling); any object with the same four methods (the proto's ``LiveCoins``) can be passed instead.

Soundness (paper (17), unique decoding): per level ``((1 + rate_i)/2)^{|S_i|} + (k'_i n_i + 2 k'_i + |S_{i-1}| + 1)/|F|`` --
sized by ``params.py``; the toy query counts in the tests are NOT sound parameters.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

try:
    import blake3 as _blake3

    def H(data: bytes, length: int = 32) -> bytes:
        return _blake3.blake3(data).digest(length=length)

    HASH_NAME = "blake3"
except ImportError:                                   # pragma: no cover - the fixtures require blake3
    def H(data: bytes, length: int = 32) -> bytes:
        return hashlib.shake_256(data).digest(length)

    HASH_NAME = "shake256 (blake3 missing: fixtures NOT comparable)"

__all__ = ["P", "BETA", "Dims", "FSCoins", "Proof", "commit", "prove_eval", "verify", "eq_tensor", "lagrange3",
           "mle_eval", "tensor_eval", "proof_to_dict", "proof_from_dict", "encode", "ntt", "emul", "eadd", "epow"]

# ---------------------------------------------------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------------------------------------------------

P = 2**31 - 2**27 + 1
GEN = 31                    # multiplicative generator of F_p (TWO_ADICITY = 27)
BETA = 31                   # F = F_p[x] / (x^6 - BETA); irreducible since 6 | p - 1 and 31 is neither a square nor a cube
EXT = 6
INV2 = (P + 1) // 2
I64 = np.int64


def root_of_unity(n: int) -> int:
    assert n & (n - 1) == 0 and n <= 2**27
    return pow(GEN, (P - 1) // n, P)


def emul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Extension product, broadcasting over leading axes; inputs canonical in [0, p)."""
    a = np.asarray(a, dtype=I64)
    b = np.asarray(b, dtype=I64)
    shape = np.broadcast_shapes(a.shape[:-1], b.shape[:-1])
    conv = np.zeros(shape + (2 * EXT - 1,), dtype=I64)
    for i in range(EXT):
        for j in range(EXT):
            conv[..., i + j] += (a[..., i] * b[..., j]) % P      # each < p, at most 6 summands: < 2^34
    out = conv[..., :EXT].copy()
    out[..., :EXT - 1] += conv[..., EXT:] * BETA                 # x^{6+j} = BETA x^j, j < 5; < 2^34 + 31 * 2^34
    return out % P


def eadd(a, b):
    return (np.asarray(a, dtype=I64) + np.asarray(b, dtype=I64)) % P


def esub(a, b):
    return (np.asarray(a, dtype=I64) - np.asarray(b, dtype=I64)) % P


def esum(a: np.ndarray, axis=0) -> np.ndarray:
    return np.sum(np.asarray(a, dtype=I64), axis=axis) % P        # <= 2^32 terms < 2^31 each: fits int64


def escale(a: np.ndarray, s: int) -> np.ndarray:
    """Extension array times a base scalar."""
    return (np.asarray(a, dtype=I64) * (int(s) % P)) % P


def epow(a: np.ndarray, e: int) -> np.ndarray:
    r = ext_one()
    base = np.asarray(a, dtype=I64)
    while e:
        if e & 1:
            r = emul(r, base)
        base = emul(base, base)
        e >>= 1
    return r


def ext_one(shape=()) -> np.ndarray:
    o = np.zeros(tuple(shape) + (EXT,), dtype=I64)
    o[..., 0] = 1
    return o


def ext_from_base(x) -> np.ndarray:
    x = np.asarray(x, dtype=I64) % P
    o = np.zeros(x.shape + (EXT,), dtype=I64)
    o[..., 0] = x
    return o


def _limb_matvec(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``A @ x mod p`` for base entries in [0, p), exact in int64 (16-bit limbs of ``x``; rows of ``A`` <= 2^16 long)."""
    assert A.shape[1] <= 1 << 16
    lo = A @ (x & 0xFFFF)                       # products < 2^47, <= 2^16 of them
    hi = A @ (x >> 16)                          # products < 2^46
    return (lo % P + (hi % P) * 65536) % P


def matT_vec(M: np.ndarray, a: np.ndarray) -> np.ndarray:
    """``M^T a`` for ``M`` base (R, C) or ext (R, C, 6) and ``a`` ext (R, 6) -> ext (C, 6)."""
    a = np.asarray(a, dtype=I64)
    if M.ndim == 2:
        return np.stack([_limb_matvec(M.T, a[:, e]) for e in range(EXT)], axis=-1)
    conv = np.zeros((M.shape[1], 2 * EXT - 1), dtype=I64)
    for i in range(EXT):
        MT = np.ascontiguousarray(M[:, :, i].T)
        for j in range(EXT):
            conv[:, i + j] += _limb_matvec(MT, a[:, j])
    out = conv[:, :EXT].copy()
    out[:, :EXT - 1] += conv[:, EXT:] * BETA
    return out % P


def mat_vec(M: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``M x`` for ``M`` base (R, C) or ext (R, C, 6) and ``x`` ext (C, 6) -> ext (R, 6)."""
    if M.ndim == 2:
        return matT_vec(np.ascontiguousarray(M.T), x)
    return matT_vec(np.ascontiguousarray(M.transpose(1, 0, 2)), x)


# ---------------------------------------------------------------------------------------------------------------------
# NTT / Reed-Solomon (coefficient basis, natural order)
# ---------------------------------------------------------------------------------------------------------------------

def _bitrev(n: int) -> np.ndarray:
    bits = n.bit_length() - 1
    return np.array([int(f"{i:0{bits}b}"[::-1], 2) if bits else 0 for i in range(n)], dtype=np.int64)


def ntt(x: np.ndarray) -> np.ndarray:
    """``X[s] = sum_r x[r] omega_n^{s r}`` along axis 0 (``n = x.shape[0]`` a power of two), other axes carried along."""
    n = x.shape[0]
    x = np.asarray(x, dtype=I64)[_bitrev(n)] % P
    rest = x.shape[1:]
    w = root_of_unity(n)
    L = 2
    while L <= n:
        wL = pow(w, n // L, P)
        tw = np.array([pow(wL, j, P) for j in range(L // 2)], dtype=I64).reshape((1, L // 2) + (1,) * len(rest))
        y = x.reshape((n // L, L) + rest)
        a, b = y[:, :L // 2], (y[:, L // 2:] * tw) % P
        x = np.concatenate([(a + b) % P, (a - b) % P], axis=1).reshape((n,) + rest)
        L *= 2
    return x


def encode(M: np.ndarray, n: int) -> np.ndarray:
    """RS-encode every column of ``M`` (R, C[, 6]) to length ``n >= R``: zero-pad the coefficients, NTT along the rows."""
    R = M.shape[0]
    assert R <= n
    padded = np.zeros((n,) + M.shape[1:], dtype=I64)
    padded[:R] = M
    return ntt(padded)


# ---------------------------------------------------------------------------------------------------------------------
# Tensors (mixed radix, digit 0 least significant)
# ---------------------------------------------------------------------------------------------------------------------

Tensor = list  # of (radix_j, 6) int64 arrays


def expand(t: Tensor) -> np.ndarray:
    v = ext_one((1,))
    for w in t:
        v = emul(v[None, :, :], np.asarray(w, dtype=I64)[:, None, :]).reshape(-1, EXT)   # index = x + len(v) * d
    return v


def split(t: Tensor, rows: int) -> tuple[Tensor, Tensor]:
    """Row digits (product == ``rows``) and column digits."""
    prod, m = 1, 0
    while prod < rows:
        prod *= len(t[m])
        m += 1
    assert prod == rows, (prod, rows)
    return t[:m], t[m:]


def tensor_eval_binary(t: Tensor, r: np.ndarray) -> np.ndarray:
    """``prod_j (w_j[0] (1 - r_j) + w_j[1] r_j)`` for binary digits: the multilinear extension of the tensor at ``r``."""
    out = ext_one()
    for w, rj in zip(t, r):
        assert len(w) == 2
        out = emul(out, eadd(emul(w[0], esub(ext_one(), rj)), emul(w[1], rj)))
    return out


def eq_tensor(z: np.ndarray) -> Tensor:
    """``w_j = (1 - z_j, z_j)`` for binary digits (``z`` shape (k, 6))."""
    return [np.stack([esub(ext_one(), zj), zj]) for zj in np.asarray(z, dtype=I64)]


def lagrange3(z: np.ndarray) -> np.ndarray:
    """Lagrange basis on {0, 1, 2} at ``z`` (the ternary digit's multilinear analogue): shape (3, 6)."""
    z = np.asarray(z, dtype=I64)
    one, two = ext_one(), escale(ext_one(), 2)
    l0 = escale(emul(esub(z, one), esub(z, two)), INV2)             # (z-1)(z-2)/2
    l1 = esub(np.zeros(EXT, dtype=I64), emul(z, esub(z, two)))        # -z(z-2)
    l2 = escale(emul(z, esub(z, one)), INV2)                          # z(z-1)/2
    return np.stack([l0, l1, l2])


def rs_row_tensor(s: int, n: int, radices: list[int]) -> Tensor:
    """The generator row ``(omega_n^{s x})_x`` over the digits ``radices`` as a tensor."""
    w = pow(root_of_unity(n), s, P)
    t, stride = [], 1
    for rad in radices:
        base = pow(w, stride, P)
        t.append(ext_from_base(np.array([pow(base, d, P) for d in range(rad)], dtype=I64)))
        stride *= rad
    return t


def tensor_eval(f: np.ndarray, t: Tensor) -> np.ndarray:
    """``<f, expand(t)>`` for a base vector ``f``."""
    return esum(emul(ext_from_base(f), expand(t)), axis=0)


def mle_eval(f: np.ndarray, z: np.ndarray, radices: list[int]) -> np.ndarray:
    return tensor_eval(f, point_tensor(z, radices))


def point_tensor(z: np.ndarray, radices: list[int]) -> Tensor:
    """The evaluation point as a tensor: ``eq`` on binary digits, Lagrange-3 on a ternary digit 0."""
    z = np.asarray(z, dtype=I64)
    assert len(z) == len(radices)
    return [lagrange3(zj) if rad == 3 else np.stack([esub(ext_one(), zj), zj]) for zj, rad in zip(z, radices)]


# ---------------------------------------------------------------------------------------------------------------------
# Merkle (BLAKE3 rows, batched multi-opening)
# ---------------------------------------------------------------------------------------------------------------------

def row_bytes(row: np.ndarray) -> bytes:
    return np.ascontiguousarray(np.asarray(row, dtype=I64).reshape(-1) % P).astype("<u4").tobytes()


def merkle_tree(U: np.ndarray) -> list[list[bytes]]:
    """Levels from leaves (index 0) to the root; ``U`` is (n, C[, 6])."""
    n = U.shape[0]
    assert n & (n - 1) == 0
    levels = [[H(row_bytes(U[s])) for s in range(n)]]
    while len(levels[-1]) > 1:
        prev = levels[-1]
        levels.append([H(prev[2 * i] + prev[2 * i + 1]) for i in range(len(prev) // 2)])
    return levels


def merkle_multi_open(levels: list[list[bytes]], idxs: list[int]) -> list[bytes]:
    """Siblings needed to recompute the root from the leaves ``idxs`` (deduplicated; canonical order: level by level,
    known nodes in ascending index order, a sibling is emitted only when it is not itself known)."""
    sibs = []
    known = sorted(set(idxs))
    for lvl in levels[:-1]:
        nxt = []
        kset = set(known)
        for i in known:
            if i ^ 1 not in kset:
                sibs.append(lvl[i ^ 1])
            if i >> 1 not in nxt:
                nxt.append(i >> 1)
        known = nxt
    return sibs


def merkle_multi_verify(root: bytes, n: int, idxs: list[int], rows: list[np.ndarray], sibs: list[bytes]) -> bool:
    if len(set(idxs)) != len(idxs) or any(not 0 <= i < n for i in idxs):
        return False
    known = {i: H(row_bytes(r)) for i, r in zip(idxs, rows)}
    pos = 0
    while n > 1:
        nxt = {}
        for i in sorted(known):
            if i >> 1 in nxt:
                continue
            if i ^ 1 in known:
                sib = known[i ^ 1]
            else:
                if pos >= len(sibs):
                    return False
                sib = sibs[pos]
                pos += 1
            left, right = (known[i], sib) if i % 2 == 0 else (sib, known[i])
            nxt[i >> 1] = H(left + right)
        known = nxt
        n //= 2
    return pos == len(sibs) and known.get(0) == root


# ---------------------------------------------------------------------------------------------------------------------
# Coins (Fiat-Shamir instance of the proto's Coins protocol)
# ---------------------------------------------------------------------------------------------------------------------

class FSCoins:
    """``state <- BLAKE3(state || u32 |label| || label || u64 |data| || data)`` on absorb; a challenge of ``n`` elements is the
    BLAKE3 XOF of ``state || label || 0x01 || u32 n`` read as ``6 n`` little-endian u64 words each reduced mod p (bias
    2^-33); ``indices`` reads u64 words mod ``domain`` from the XOF of ``state || label || 0x03 || u32 domain || u32 count``,
    skipping repeats, until ``count`` distinct; every challenge/indices call then advances ``state <- BLAKE3(state || label
    || 0x02)`` and counts one round."""

    def __init__(self, seed: bytes = b"ligerito-ref/v1"):
        self.state = H(seed)
        self.rounds = 0

    def absorb(self, label: bytes, data: bytes) -> None:
        self.state = H(self.state + len(label).to_bytes(4, "little") + label + len(data).to_bytes(8, "little") + data)

    def _advance(self, label: bytes) -> None:
        self.state = H(self.state + label + b"\x02")
        self.rounds += 1

    def challenge(self, label: bytes, n: int) -> np.ndarray:
        raw = H(self.state + label + b"\x01" + n.to_bytes(4, "little"), length=8 * EXT * n)
        words = np.frombuffer(raw, dtype="<u8").astype(object)
        out = np.array([int(w) % P for w in words], dtype=I64).reshape(n, EXT)
        self._advance(label)
        return out

    def indices(self, label: bytes, domain: int, count: int) -> list[int]:
        assert count <= domain
        out: list[int] = []
        need = 2 * count + 8
        while len(out) < count:
            raw = H(self.state + label + b"\x03" + domain.to_bytes(4, "little") + count.to_bytes(4, "little"), length=8 * need)
            out = []
            for w in np.frombuffer(raw, dtype="<u8"):
                v = int(w) % domain
                if v not in out:
                    out.append(v)
                if len(out) == count:
                    break
            need *= 2
        self._advance(label)
        return out


# ---------------------------------------------------------------------------------------------------------------------
# Parameters, proof, prover state
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Dims:
    """``tall[i] x 2^wide_log2[i]`` matrices, codeword length ``2^n_log2[i]``, ``queries[i]`` opened rows per level;
    ``tall[i] == tall[i+1] * 2^wide_log2[i+1]``; ``tall[i]`` is ``2^k`` or ``3 * 2^k``."""
    tall: tuple[int, ...]
    wide_log2: tuple[int, ...]
    n_log2: tuple[int, ...]
    queries: tuple[int, ...]

    @property
    def L(self) -> int:
        return len(self.tall)

    @property
    def N(self) -> int:
        return self.tall[0] << self.wide_log2[0]

    def wide(self, i: int) -> int:
        return 1 << self.wide_log2[i]

    def n(self, i: int) -> int:
        return 1 << self.n_log2[i]

    def row_radices(self, i: int) -> list[int]:
        t = self.tall[i]
        odd = 3 if t % 3 == 0 else 1
        assert t == odd << (t // odd).bit_length() - 1, t
        return ([3] if odd == 3 else []) + [2] * ((t // odd).bit_length() - 1)

    @property
    def radices(self) -> list[int]:
        return self.row_radices(0) + [2] * self.wide_log2[0]

    def validate(self) -> None:
        assert len(self.wide_log2) == len(self.n_log2) == len(self.queries) == self.L >= 1
        for i in range(self.L):
            self.row_radices(i)
            assert self.n(i) >= self.tall[i] and self.queries[i] <= self.n(i)
            if i + 1 < self.L:
                assert self.tall[i] == self.tall[i + 1] << self.wide_log2[i + 1], (i, self.tall)

    @classmethod
    def from_params(cls, p, queries: tuple[int, ...] | None = None) -> "Dims":
        """From a ``params.LigeritoParams`` (its exact shape; pass smaller ``queries`` for toy runs)."""
        return cls(tuple(p.tall(i) for i in range(p.L)), tuple(p.wide_log2), tuple(p.n(i).bit_length() - 1 for i in range(p.L)),
                   tuple(queries or p.queries))

    @classmethod
    def toy(cls, log_n: int, L: int, queries: int = 12, rate_log2: int = 2, odd3: bool = False) -> "Dims":
        """A small shape: split ``log_n`` binary digits as evenly as the paper does (final vector gets the rest)."""
        wide = [max(2, log_n // (L + 1))] * L
        tall, n_log2 = [], []
        rest = log_n - sum(wide)
        assert rest >= 1, "too few digits for this many levels"
        for i in range(L):
            k = rest + sum(wide[i + 1:])
            tall.append((3 if odd3 else 1) << k)
            n_log2.append(k + rate_log2 + (2 if odd3 else 0))
        return cls(tuple(tall), tuple(wide), tuple(n_log2), tuple([queries] * L))


@dataclass
class LevelProof:
    sumcheck: np.ndarray                    # (k'_i, 3, 6) coefficients c0, c1, c2 per round
    opened_idx: list[int]                   # S_i
    opened_rows: list[np.ndarray]           # rows of codeword i (wide_i,) base or (wide_i, 6) ext
    siblings: list[bytes]


@dataclass
class Proof:
    dims: Dims
    roots: list[bytes]
    levels: list[LevelProof]
    final: np.ndarray                       # v_L, (tall_L, 6)
    value: np.ndarray                       # claimed v = <f, w>, (6,)


@dataclass
class ProverState:
    dims: Dims
    f: np.ndarray
    M1: np.ndarray
    U1: np.ndarray
    tree1: list[list[bytes]]

    @property
    def root(self) -> bytes:
        return self.tree1[-1][0]


# ---------------------------------------------------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------------------------------------------------

def _reshape(v: np.ndarray, tall: int, wide: int) -> np.ndarray:
    """Column-major: ``M[row, col] = v[row + tall * col]``; ``v`` is (N,) base or (N, 6) ext."""
    return np.ascontiguousarray(v.reshape((wide, tall) + v.shape[1:]).swapaxes(0, 1))


def commit(f: np.ndarray, dims: Dims) -> ProverState:
    dims.validate()
    f = np.asarray(f, dtype=I64) % P
    assert f.shape == (dims.N,)
    M1 = _reshape(f, dims.tall[0], dims.wide(0))
    U1 = encode(M1, dims.n(0))
    return ProverState(dims, f, M1, U1, merkle_tree(U1))


def _ext_bytes(a: np.ndarray) -> bytes:
    return row_bytes(a)


def _absorb_opening(coins, rows: list[np.ndarray], sibs: list[bytes]) -> None:
    coins.absorb(b"opened", b"".join(row_bytes(r) for r in rows) + b"".join(sibs))


def _batch_terms(coins, dims: Dims, i: int, carried, carried_total, rows: list[np.ndarray], idx: list[int], rbar_prev: np.ndarray):
    """Level ``i >= 1``: fold the carried claims and the opened rows of level ``i - 1`` into one term list with fresh alphas."""
    alpha = coins.challenge(b"batch", 1 + len(idx))
    radices = dims.row_radices(i) + [2] * dims.wide_log2[i]
    terms, total = [], emul(alpha[0], carried_total)
    for gamma, a in carried:
        ra, ca = split(a, dims.tall[i])
        terms.append((emul(alpha[0], gamma), ra, ca))
    for m, (s, row) in enumerate(zip(idx, rows)):
        y = esum(emul(ext_from_base(row) if row.ndim == 1 else row, rbar_prev), axis=0)
        total = eadd(total, emul(alpha[1 + m], y))
        ra, ca = split(rs_row_tensor(s, dims.n(i - 1), radices), dims.tall[i])
        terms.append((alpha[1 + m], ra, ca))
    return terms, total


def _sumcheck_prove(coins, M: np.ndarray, terms, k: int):
    B = [expand(b) for (_, _, b) in terms]
    U = [matT_vec(M, expand(a)) for (_, a, _) in terms]
    G = [g for (g, _, _) in terms]
    msgs, r = [], []
    for _ in range(k):
        g0 = g1 = g2 = np.zeros(EXT, dtype=I64)
        for gam, b, u in zip(G, B, U):
            b0, b1, u0, u1 = b[0::2], b[1::2], u[0::2], u[1::2]
            g0 = eadd(g0, emul(gam, esum(emul(b0, u0))))
            g1 = eadd(g1, emul(gam, esum(emul(b1, u1))))
            b2, u2 = esub(escale(b1, 2), b0), esub(escale(u1, 2), u0)          # value at X = 2
            g2 = eadd(g2, emul(gam, esum(emul(b2, u2))))
        c2 = escale(eadd(esub(g2, escale(g1, 2)), g0), INV2)
        c1 = esub(esub(g1, g0), c2)
        msg = np.stack([g0, c1, c2])
        coins.absorb(b"sumcheck", _ext_bytes(msg))
        rj = coins.challenge(b"r", 1)[0]
        msgs.append(msg)
        r.append(rj)
        B = [eadd(b[0::2], emul(esub(b[1::2], b[0::2]), rj)) for b in B]
        U = [eadd(u[0::2], emul(esub(u[1::2], u[0::2]), rj)) for u in U]
    r = np.stack(r) if r else np.zeros((0, EXT), dtype=I64)
    final = np.zeros(EXT, dtype=I64)
    for gam, b, u in zip(G, B, U):
        final = eadd(final, emul(gam, emul(b[0], u[0])))
    return np.stack(msgs) if msgs else np.zeros((0, 3, EXT), dtype=I64), r, final


def _sumcheck_verify(coins, msgs: np.ndarray, claim: np.ndarray, k: int):
    r = []
    for j in range(k):
        c0, c1, c2 = msgs[j]
        if not np.array_equal(eadd(escale(c0, 2), eadd(c1, c2)), claim):        # g(0) + g(1) == claim
            return None, None, f"sumcheck round {j}: g(0) + g(1) != claim"
        coins.absorb(b"sumcheck", _ext_bytes(msgs[j]))
        rj = coins.challenge(b"r", 1)[0]
        r.append(rj)
        claim = eadd(c0, emul(rj, eadd(c1, emul(c2, rj))))
    return (np.stack(r) if r else np.zeros((0, EXT), dtype=I64)), claim, None


def prove_eval(state: ProverState, w: Tensor, coins) -> Proof:
    """Prove ``<f, w> = v`` for the tensor ``w`` over ``dims.radices`` (``v`` is computed and included)."""
    dims = state.dims
    assert [len(t) for t in w] == dims.radices, "point tensor radices must match dims.radices"
    value = tensor_eval(state.f, w)
    ra, ca = split(w, dims.tall[0])
    terms, total = [(ext_one(), ra, ca)], value
    M, U, tree = state.M1, state.U1, state.tree1
    roots, levels = [], []
    prev_rows: list[np.ndarray] = []
    prev_idx: list[int] = []
    rbar = None
    for i in range(dims.L):
        roots.append(tree[-1][0])
        coins.absorb(b"root", roots[-1])
        if i >= 1:
            prev_idx = coins.indices(b"queries", dims.n(i - 1), dims.queries[i - 1])
            prev_rows = [prev_U[s] for s in prev_idx]
            sibs = merkle_multi_open(prev_tree, prev_idx)
            _absorb_opening(coins, prev_rows, sibs)
            levels[-1].opened_idx, levels[-1].opened_rows, levels[-1].siblings = prev_idx, prev_rows, sibs
            terms, total = _batch_terms(coins, dims, i, carried, total, prev_rows, prev_idx, rbar)
        msgs, r, total = _sumcheck_prove(coins, M, terms, dims.wide_log2[i])
        levels.append(LevelProof(msgs, [], [], []))
        rbar = expand(eq_tensor(r))
        v = mat_vec(M, rbar)                                                    # (tall_i, 6)
        carried = [(emul(g, tensor_eval_binary(b, r)), a) for (g, a, b) in terms]
        prev_U, prev_tree = U, tree
        if i + 1 < dims.L:
            M = _reshape(v, dims.tall[i + 1], dims.wide(i + 1))
            U = encode(M, dims.n(i + 1))
            tree = merkle_tree(U)
    coins.absorb(b"final", _ext_bytes(v))
    idx = coins.indices(b"queries", dims.n(dims.L - 1), dims.queries[dims.L - 1])
    rows = [prev_U[s] for s in idx]
    sibs = merkle_multi_open(prev_tree, idx)
    _absorb_opening(coins, rows, sibs)
    levels[-1].opened_idx, levels[-1].opened_rows, levels[-1].siblings = idx, rows, sibs
    return Proof(dims, roots, levels, v, value)


def verify(root: bytes, w: Tensor, value: np.ndarray, proof: Proof, coins) -> tuple[bool, str]:
    dims = proof.dims
    try:
        dims.validate()
    except AssertionError as e:
        return False, f"dims: {e}"
    if [len(t) for t in w] != dims.radices:
        return False, "point tensor radices do not match dims"
    if len(proof.roots) != dims.L or len(proof.levels) != dims.L or proof.roots[0] != root:
        return False, "roots / levels shape or commitment mismatch"
    if not np.array_equal(np.asarray(value, dtype=I64) % P, proof.value % P):
        return False, "claimed value differs from the proof's value"
    ra, ca = split(w, dims.tall[0])
    terms, total = [(ext_one(), ra, ca)], np.asarray(value, dtype=I64) % P
    rbar = None
    for i in range(dims.L):
        coins.absorb(b"root", proof.roots[i])
        if i >= 1:
            lp = proof.levels[i - 1]
            idx = coins.indices(b"queries", dims.n(i - 1), dims.queries[i - 1])
            if idx != list(lp.opened_idx) or len(lp.opened_rows) != len(idx):
                return False, f"level {i}: opened indices are not the verifier's"
            exp_shape = (dims.wide(i - 1),) if i - 1 == 0 else (dims.wide(i - 1), EXT)
            if any(np.asarray(r).shape != exp_shape for r in lp.opened_rows):
                return False, f"level {i}: opened row shape"
            if not merkle_multi_verify(proof.roots[i - 1], dims.n(i - 1), idx, lp.opened_rows, lp.siblings):
                return False, f"level {i}: Merkle multi-opening of level {i - 1} rejected"
            _absorb_opening(coins, lp.opened_rows, lp.siblings)
            terms, total = _batch_terms(coins, dims, i, carried, total, lp.opened_rows, idx, rbar)
        msgs = proof.levels[i].sumcheck
        if np.asarray(msgs).shape != (dims.wide_log2[i], 3, EXT):
            return False, f"level {i + 1}: sumcheck message shape"
        r, total, err = _sumcheck_verify(coins, msgs, total, dims.wide_log2[i])
        if err:
            return False, f"level {i + 1}: {err}"
        rbar = expand(eq_tensor(r))
        carried = [(emul(g, tensor_eval_binary(b, r)), a) for (g, a, b) in terms]
    v = np.asarray(proof.final, dtype=I64)
    if v.shape != (dims.tall[-1], EXT):
        return False, "final vector shape"
    coins.absorb(b"final", _ext_bytes(v))
    lp = proof.levels[-1]
    idx = coins.indices(b"queries", dims.n(dims.L - 1), dims.queries[dims.L - 1])
    if idx != list(lp.opened_idx) or len(lp.opened_rows) != len(idx):
        return False, "last level: opened indices are not the verifier's"
    exp_shape = (dims.wide(dims.L - 1),) if dims.L == 1 else (dims.wide(dims.L - 1), EXT)
    if any(np.asarray(rw).shape != exp_shape for rw in lp.opened_rows):
        return False, "last level: opened row shape"
    if not merkle_multi_verify(proof.roots[-1], dims.n(dims.L - 1), idx, lp.opened_rows, lp.siblings):
        return False, "last level: Merkle multi-opening rejected"
    _absorb_opening(coins, lp.opened_rows, lp.siblings)
    # carried claims on v_L, checked directly
    acc = np.zeros(EXT, dtype=I64)
    for g, a in carried:
        acc = eadd(acc, emul(g, esum(emul(expand(a), v))))
    if not np.array_equal(acc, total):
        return False, "final vector does not satisfy the carried claims"
    # opened rows of the last codeword against the RS encoding of v_L
    rad = dims.row_radices(dims.L - 1)
    for s, row in zip(idx, lp.opened_rows):
        y = esum(emul(ext_from_base(row) if row.ndim == 1 else row, rbar))
        if not np.array_equal(y, esum(emul(expand(rs_row_tensor(s, dims.n(dims.L - 1), rad)), v))):
            return False, f"last level: opened row {s} is not the encoding of the final vector"
    return True, "ok"


# ---------------------------------------------------------------------------------------------------------------------
# Fixtures (JSON) for the Rust verifier
# ---------------------------------------------------------------------------------------------------------------------

def proof_to_dict(proof: Proof) -> dict:
    d = proof.dims
    return {
        "format": "ligerito-ref/v1", "hash": HASH_NAME, "field": {"p": P, "ext": "x^6 - 31"},
        "dims": {"tall": list(d.tall), "wide_log2": list(d.wide_log2), "n_log2": list(d.n_log2), "queries": list(d.queries)},
        "roots": [r.hex() for r in proof.roots],
        "levels": [{"sumcheck": np.asarray(l.sumcheck).tolist(), "opened_idx": list(map(int, l.opened_idx)),
                    "opened_rows": [np.asarray(r).tolist() for r in l.opened_rows], "siblings": [s.hex() for s in l.siblings]}
                   for l in proof.levels],
        "final": np.asarray(proof.final).tolist(), "value": np.asarray(proof.value).tolist(),
    }


def proof_from_dict(d: dict) -> Proof:
    dims = Dims(tuple(d["dims"]["tall"]), tuple(d["dims"]["wide_log2"]), tuple(d["dims"]["n_log2"]), tuple(d["dims"]["queries"]))
    levels = [LevelProof(np.array(l["sumcheck"], dtype=I64).reshape(-1, 3, EXT), list(l["opened_idx"]),
                         [np.array(r, dtype=I64) for r in l["opened_rows"]], [bytes.fromhex(s) for s in l["siblings"]])
              for l in d["levels"]]
    return Proof(dims, [bytes.fromhex(r) for r in d["roots"]], levels, np.array(d["final"], dtype=I64), np.array(d["value"], dtype=I64))


def fixture(f: np.ndarray, dims: Dims, z: np.ndarray, seed: bytes = b"fixture") -> dict:
    """A complete statement + proof + expected verdict for the Rust verifier (``python -m backends.direct.ligerito.ref``)."""
    st = commit(f, dims)
    w = point_tensor(z, dims.radices)
    pr = prove_eval(st, w, FSCoins(seed))
    ok, why = verify(st.root, w, pr.value, pr, FSCoins(seed))
    return {"seed": seed.decode(), "f": f.tolist(), "z": np.asarray(z).tolist(), "commitment": st.root.hex(),
            "proof": proof_to_dict(pr), "expect": {"ok": ok, "reason": why}}


def main() -> None:                                                             # pragma: no cover
    import sys
    rng = np.random.default_rng(7)
    out = []
    for log_n, L, odd3 in [(8, 1, False), (10, 2, False), (12, 3, False), (10, 2, True)]:
        dims = Dims.toy(log_n, L, queries=8, odd3=odd3)
        f = rng.integers(0, P, dims.N, dtype=I64)
        z = rng.integers(0, P, (len(dims.radices), EXT), dtype=I64)
        out.append(fixture(f, dims, z))
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()
