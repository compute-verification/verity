"""``leaf/ajtai``: a Ring-SIS (Ajtai / SWIFFT-style) row digest over BabyBear, linear in the committed operand bits.

The Ligero chain applies ONE constraint system at every column, so a digest carried column to column must be a public
linear state transition ``acc_out = M acc_in + B s_j`` (``s_j``: the column's operand bits).  With ``M`` = multiplication
by X in ``R = F_p[X] / (X^n + 1)`` (a negacyclic shift) and ``B = [b_0 .. b_{m-1}]`` seeded ring elements, one per bit
position of a column, the VU's digest is

    h = sum_j X^(steps-1-j) B s_j = sum_i b_i(X) g_i(X),     g_i(X) = sum_j X^(steps-1-j) s_{j,i}   (deg < steps),

i.e. Ring-SIS with m = 256 public ring elements and short (binary, degree < steps) unknowns: a collision is a nonzero
ternary solution of ``sum_i b_i gamma_i = 0`` in R_p (Lyubashevsky-Micciancio-Peikert-Rosen 2008, SWIFFT).  A digest
carried as ``acc + A_blk s_j`` WITHOUT the shift would be ``A_blk sum_j s_j`` -- column-swap collisions -- which is why
the ring structure is not optional (lane note §1).  Unsalted and deterministic: hiding only up to preimage search on a
row (same caveat as Poseidon2 today).  ``role`` is not absorbed: the x and W trees are separate commitment domains.

Injectivity needs ``n >= steps`` (degree of g_i below the ring degree: with wrap-around, X^(n+k) = -X^k makes
``s_a - s_b`` a coefficient and (1, 1) collides with (0, 0)); :func:`compose` enforces it.

Parameters (lane ajtai-design FINAL 17:10Z, ``ajtai-babybear-n<N>-b1-r0/v1``): ``n`` = ring degree = digest elements,
a power of two -- 64 for the 48-step relations (fp8-ada, fp4), 128 for the 96-step bf16-hopper (Core-SVP classical
285 / 563 bits under their M1 estimator; n = 96 is reducible and excluded), ``beta = 1`` (bits), ``B = [b_0 .. b_{255}]``, ``b_i`` = the
SHAKE-256(b"verity/ajtai-babybear/B/v1" || LE32(i)) stream read as LE u32 words masked to 31 bits, values >= p rejected
(acceptance 15/16; their ``derive_ring_element``), prefix-consistent in n.  ``leaf_bytes`` = n x LE u32.  The design's SIS bound
carries over to this ring instance under the standard heuristic (Ring-SIS ~ SIS of the same lattice dimension n x 256
ring elements = 16384 >= 12288 columns); the ring structure is what makes identical per-column constraints position-aware.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from ..compile import P, Expr, Linear

__all__ = ["AjtaiLeaf", "B_TAG", "DEFAULT_N", "derive_B", "derive_matrix", "derive_ring_element", "digest_rows", "shift_x"]

DEFAULT_N = 64
B_TAG = b"verity/ajtai-babybear/B/v1"
VERSION = "v1"


def derive_ring_element(i: int, n: int) -> np.ndarray:
    """``b_i`` (n coefficients): the SHAKE-256 XOF of ``B_TAG || LE32(i)`` read as LE u32 words ``u``, ``v = u & 0x7fffffff``,
    accepted iff ``v < p`` (acceptance 15/16, no modular bias); coefficient k = the k-th accepted value.  Byte-identical to
    lane ajtai-design's ``ajtai_params.derive_ring_element`` (their regression vector is in the tests).  Prefix-consistent
    in n (an XOF): ``ajtai-n64`` and ``ajtai-n128`` are one family."""
    h = hashlib.shake_256(B_TAG + int(i).to_bytes(4, "little"))
    stream_len = 4 * (int(n * 1.2) + 64)
    while True:
        words = np.frombuffer(h.digest(stream_len), dtype="<u4").astype(np.int64) & 0x7FFFFFFF
        words = words[words < P]
        if len(words) >= n:
            return np.ascontiguousarray(words[:n])
        stream_len *= 2


def derive_B(m: int, n: int) -> list[np.ndarray]:
    """The m ring elements of the leaf, ``[b_0 .. b_{m-1}]``, one per operand bit position of a column."""
    return [derive_ring_element(i, n) for i in range(m)]


def derive_matrix(n: int, m: int) -> np.ndarray:
    """``B`` as the (n, m) int64 matrix the gadget and the native side apply: ``B[x, i]`` = coefficient x of ``b_i``
    (``B @ s`` = ``sum_i s_i b_i`` in R_p); the one derivation for every call site."""
    return np.stack(derive_B(m, n), axis=1)


def params_descriptor(n: int, m_col: int, salt_bits: int = 0) -> bytes:
    """What ``params_sha256`` hashes: everything the verifier needs to rebuild the scheme (the verifier regenerates B
    from the seed string; it never takes B from a statement)."""
    return (f"ajtai-babybear/{VERSION}|p={P}|n={n}|m_col={m_col}|beta=1|salt_bits={salt_bits}|ring=F_p[X]/(X^n+1)|M=X"
            f"|B={B_TAG.decode()}|b_i=SHAKE-256(tag||LE32(i)) LE u32 &0x7fffffff reject>=p|leaf=LE u32 x n").encode()


def shift_x(v):
    """Multiplication by X in F_p[X]/(X^n + 1) along the last axis: out[0] = -v[n-1], out[x] = v[x-1]."""
    if isinstance(v, np.ndarray):
        out = np.roll(v, 1, axis=-1)
        out[..., 0] = (-out[..., 0]) % P
        return out % P
    import torch
    out = torch.roll(v, 1, dims=-1)
    out[..., 0] = (-out[..., 0]) % P
    return out % P


def _bits_of(words, word_bits: int):
    """(..., k) words -> (..., k * word_bits) bits, word-major, little-endian within a word (compose's bit-row order)."""
    if isinstance(words, np.ndarray):
        sh = np.arange(word_bits, dtype=np.int64)
        b = (words[..., None] >> sh) & 1
        return b.reshape(*words.shape[:-1], words.shape[-1] * word_bits)
    import torch
    sh = torch.arange(word_bits, dtype=torch.int64, device=words.device)
    b = (words[..., None] >> sh) & 1
    return b.reshape(*words.shape[:-1], words.shape[-1] * word_bits)


def _apply_B(bits, B):
    """(..., m) bits (0/1) -> (..., n) = bits @ B^T mod p, exactly: float64 products of a 0/1 by a < 2^31 value, at most
    m = 256 terms (< 2^39 < 2^53), reduced once."""
    if isinstance(bits, np.ndarray):
        acc = bits.astype(np.float64) @ B.astype(np.float64).T
        return np.rint(acc).astype(np.int64) % P
    import torch
    Bt = torch.as_tensor(B, dtype=torch.float64, device=bits.device).T
    acc = bits.to(torch.float64) @ Bt
    return torch.round(acc).to(torch.int64) % P


def digest_rows(rows, word_bits: int, B: np.ndarray, m_col: int) -> tuple:
    """``rows`` (R, n_words) -> (acc_in (R, steps, n), acc_out (R, steps, n)): the carried state entering every column
    and leaving it; the digest is ``acc_out[:, -1]``.  numpy or torch, on the rows' device."""
    k = m_col // word_bits
    if rows.shape[-1] % k:
        raise ValueError(f"a row of {rows.shape[-1]} words is not a whole number of {k}-word columns")
    steps = rows.shape[-1] // k
    R = rows.shape[0]
    n = B.shape[0]
    bits = _bits_of(rows.reshape(R, steps, k), word_bits)                   # (R, steps, m)
    contrib = _apply_B(bits, B)                                              # (R, steps, n)
    if isinstance(rows, np.ndarray):
        acc_in = np.zeros((R, steps, n), dtype=np.int64)
        acc_out = np.zeros((R, steps, n), dtype=np.int64)
        acc = np.zeros((R, n), dtype=np.int64)
    else:
        import torch
        acc_in = torch.zeros((R, steps, n), dtype=torch.int64, device=rows.device)
        acc_out = torch.zeros((R, steps, n), dtype=torch.int64, device=rows.device)
        acc = torch.zeros((R, n), dtype=torch.int64, device=rows.device)
    for j in range(steps):
        acc_in[:, j] = acc
        acc = (shift_x(acc) + contrib[:, j]) % P
        acc_out[:, j] = acc
    return acc_in, acc_out


class AjtaiLeaf:
    """The scheme object (:class:`leaf.base.LeafScheme`).  ``m_col`` = operand bits per chain column (k words x word_bits;
    256 for every registered relation), fixed at construction so ``B`` is one matrix per (n, m_col)."""

    name = "ajtai"
    assumption = "Ring-SIS over F_p[X]/(X^n+1), p = BabyBear (Ajtai / SWIFFT collision resistance; lattice)"

    def __init__(self, n: int = DEFAULT_N, m_col: int = 256, salt_bits: int = 0):
        if n <= 0 or m_col <= 0:
            raise ValueError("n and m_col must be positive")
        if n & (n - 1):
            raise ValueError(f"the ring degree n = {n} must be a power of two (X^n + 1 irreducible; ajtai-design §8)")
        if salt_bits:
            raise NotImplementedError("the salted (hiding) variant is not built: v1 is the unsalted binding-only leaf (r0)")
        self.n, self.m_col, self.salt_bits = int(n), int(m_col), int(salt_bits)
        self.digest_elems = self.n
        self.tag = f"ajtai-n{self.n}"
        # `r0`: unsalted -- binding only, NOT hiding (the digest is linear in the row's bits; ajtai-design §3 / §8)
        self.hash_name = f"ajtai-babybear-n{self.n}-b1-r{self.salt_bits}/{VERSION}"
        self.schema = f"ajtai-babybear-n{self.n}-b1-r{self.salt_bits}/row/{VERSION}"
        self.params_sha256 = hashlib.sha256(params_descriptor(self.n, self.m_col, self.salt_bits)).hexdigest()
        self._B = None

    def __repr__(self):
        return f"AjtaiLeaf(n={self.n}, m_col={self.m_col})"

    @property
    def B(self) -> np.ndarray:
        if self._B is None:
            self._B = derive_matrix(self.n, self.m_col)
        return self._B

    # --- native side --------------------------------------------------------------------------------------------------
    def native(self, rows, *, word_bits: int, role: int = 0):
        """(R, n) digests of ``rows`` (R, n_words) -- numpy in, numpy out; torch in, torch out (same device)."""
        if isinstance(rows, np.ndarray):
            rows = np.ascontiguousarray(rows, dtype=np.int64)
        _, acc_out = digest_rows(rows, word_bits, self.B, self.m_col)
        return acc_out[:, -1]

    def step(self, acc, bits):
        """One column of the state transition on arbitrary (not necessarily binary) bit values: ``X acc + B bits`` mod p.
        ``acc`` (..., n), ``bits`` (..., m_col) int64 numpy or torch; exact for bit values < 2^22."""
        return (shift_x(acc) + _apply_B(bits, self.B)) % P

    def states(self, words, *, word_bits: int):
        """``words`` (cols, k) of consecutive columns (one VU's ``steps`` columns at a time, see ``hash_hints``): the
        (acc_in, acc_out) of :func:`digest_rows` for rows formed by ``cols // steps`` -- the caller reshapes."""
        return digest_rows(words, word_bits, self.B, self.m_col)

    def leaf_bytes(self, digest_row) -> bytes:
        """The n elements as little-endian u32 (4n bytes; the design note's framing)."""
        d = np.asarray(digest_row, dtype=np.int64).ravel()
        if d.shape != (self.n,) or ((d < 0) | (d >= P)).any():
            raise ValueError(f"an Ajtai row digest is {self.n} BabyBear elements")
        return d.astype("<u4").tobytes()

    # --- gadget side --------------------------------------------------------------------------------------------------
    def gadget(self, ctx, words: Sequence[Expr], *, word_bits: int, role: int, carry_in: Sequence[Expr] | None,
               first: bool = True, last: bool = True, bits: Sequence[Expr] | None = None,
               scope: str = "") -> tuple[list[Expr], list[Expr]]:
        """One chain column: ``carry_in`` = the n linked ``acc_in`` rows (the chain start constraint makes them 0 at a
        VU's first column); ``bits`` = the column's ``m_col`` committed bit rows, word-major (else the words are
        decomposed through ``ctx.bit_rows``, which reuses an existing decomposition).  Appends n hint rows ``acc_out``
        and n Linear constraints ``acc_out[x] - (X acc_in)[x] - sum_i B[x, i] bit_i = 0``; returns (digest exprs =
        acc_out, carry_out = acc_out).  ``first`` / ``last`` are both true of every column of a Ligero chain."""
        if carry_in is None or len(carry_in) != self.n:
            raise ValueError(f"the Ajtai gadget carries {self.n} linked rows")
        if bits is None:
            bits = []
            for i, w in enumerate(words):
                bits += [Expr(ctx, {r: 1}, 0) for r in ctx.bit_rows(w, word_bits, f"{scope}w{i}")]
        if len(bits) != self.m_col:
            raise ValueError(f"{len(bits)} bit rows, the leaf hashes {self.m_col} per column")
        bit_rows = []
        for b in bits:
            if b.c != 0 or len(b.lin) != 1 or next(iter(b.lin.values())) != 1:
                raise ValueError("operand bits must be single committed rows")
            bit_rows.append(next(iter(b.lin)))
        cin = []
        for a in carry_in:
            if a.c != 0 or len(a.lin) != 1 or next(iter(a.lin.values())) != 1:
                raise ValueError("carried state must be single committed rows")
            cin.append(next(iter(a.lin)))
        B = self.B
        out = []
        pre = f"{scope}." if scope else ""
        for x in range(self.n):
            o = ctx.hint(f"{pre}acc_out[{x}]")
            oi = next(iter(o.lin))
            lin = {oi: 1}
            if x == 0:
                lin[cin[self.n - 1]] = 1                     # -(-acc_in[n-1])
            else:
                lin[cin[x - 1]] = P - 1
            for i, r in enumerate(bit_rows):
                v = int(B[x, i])
                if v:
                    lin[r] = (lin.get(r, 0) + P - v) % P
            # a field identity by construction (the acc rows have no integer interval): appended without the audit
            ctx.sys.linear.append(Linear(f"hash.{pre}acc_out[{x}]", {k: v for k, v in lin.items() if v}, 0, None))
            out.append(o)
        return out, out

    def witness(self):
        return None
