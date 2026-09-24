"""BabyBear and BabyBear^6 arithmetic on torch tensors (int64 compute, int32 storage) plus the small-value
Python-int versions the transcript-side logic uses.

Base field ``p = 2^31 - 2^27 + 1``; challenge field ``F_{p^6} = F_p[X] / (X^6 - W)``, ``W = 22`` (the smallest
non-square non-cube, the modulus the a-packed kernels ``../packed/reference.py`` are generated for; the v1-v2e
prover used ``W = 31``, ``../tensor/reference.py``'s choice -- the same field up to isomorphism, the same soundness
bound, a different coordinate representation, hence a different transcript).  An extension element is a
tensor whose last dimension is 6 (planar coordinate order ``c0..c5``); a base element is any int64 tensor of
canonical values in ``[0, p)``.  Every product of two canonical values (< 2^62) fits int64, so one product is
reduced before it is accumulated; sums of up to 2^32 canonical values fit as well.

The heavy contractions (``Σ_g w_g X_g``, ``Σ_c eq_c V_c``, the Reed-Solomon row combinations) are done as exact
float64 matmuls on 16-bit limbs (``mm_mod``): with ``k`` terms the partial sums are below ``k · 2^32``, exact in
float64 for ``k ≤ 2^21``.  On the H100 those run on the FP64 tensor cores; on a laptop CPU the same code runs.

Hypercube conventions are the Rust crate's (``field.rs``): variables are bound from the top bit down (``fold``
halves a table by combining ``v[i]`` and ``v[i + half]``), a point is the list of challenges in that order, and
``eq_table(point)`` is indexed accordingly.
"""
from __future__ import annotations

import torch

P = 2**31 - 2**27 + 1
W = 22          # X^6 - 22 irreducible (22 is a non-square non-cube mod p); must equal packed.reference.W
DEG = 6
_R32 = (2**32) % P          # 2^32 mod p
_R16 = 2**16

Ext = tuple[int, int, int, int, int, int]
ZERO: Ext = (0,) * DEG
ONE: Ext = (1,) + (0,) * (DEG - 1)

# 36 -> 6 reduction matrix for the product coordinates (i, j) -> i + j (mod X^6 = W)
_RED_ROWS = []
for _i in range(DEG):
    for _j in range(DEG):
        row = [0] * DEG
        if _i + _j < DEG:
            row[_i + _j] = 1
        else:
            row[_i + _j - DEG] = W
        _RED_ROWS.append(row)
_RED_CACHE: dict[torch.device, torch.Tensor] = {}


def red_matrix(device) -> torch.Tensor:
    m = _RED_CACHE.get(device)
    if m is None:
        m = torch.tensor(_RED_ROWS, dtype=torch.float64, device=device)
        _RED_CACHE[device] = m
    return m


# ---------------------------------------------------------------------------------------------------------------------
# Python-int extension arithmetic (small values: challenges, claims, interpolation)
# ---------------------------------------------------------------------------------------------------------------------

def e_base(a: int) -> Ext:
    return (a % P,) + (0,) * (DEG - 1)


def e_add(a: Ext, b: Ext) -> Ext:
    return tuple((x + y) % P for x, y in zip(a, b))


def e_sub(a: Ext, b: Ext) -> Ext:
    return tuple((x - y) % P for x, y in zip(a, b))


def e_neg(a: Ext) -> Ext:
    return tuple((-x) % P for x in a)


def e_scale(a: Ext, c: int) -> Ext:
    c %= P
    return tuple(x * c % P for x in a)


def e_mul(a: Ext, b: Ext) -> Ext:
    acc = [0] * (2 * DEG - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                acc[i + j] += x * y
    return tuple((acc[d] + W * acc[d + DEG]) % P for d in range(DEG - 1)) + (acc[DEG - 1] % P,)


def e_pow(a: Ext, n: int) -> Ext:
    r, b = ONE, a
    while n:
        if n & 1:
            r = e_mul(r, b)
        b = e_mul(b, b)
        n >>= 1
    return r


def e_is_zero(a: Ext) -> bool:
    return all(x % P == 0 for x in a)


def e_inv(a: Ext) -> Ext:
    """Inverse by Fermat in F_{p^6} (only used in tests)."""
    return e_pow(a, P**DEG - 2)


def interpolate(vals: list[Ext], r: Ext) -> Ext:
    """``s(r)`` from ``s(0..d)`` by Lagrange interpolation (``field.rs::interpolate``)."""
    d = len(vals) - 1
    acc = ZERO
    for i in range(d + 1):
        num = ONE
        den = 1
        for j in range(d + 1):
            if j == i:
                continue
            num = e_mul(num, e_sub(r, e_base(j)))
            den = den * (i - j)
        acc = e_add(acc, e_scale(e_mul(vals[i], num), pow(den % P, P - 2, P)))
    return acc


def eq_point(a: list[Ext], b: list[Ext]) -> Ext:
    e = ONE
    for x, y in zip(a, b):
        e = e_mul(e, e_add(e_mul(x, y), e_mul(e_sub(ONE, x), e_sub(ONE, y))))
    return e


def to_bytes(x: Ext) -> bytes:
    return b"".join(int(c).to_bytes(4, "little") for c in x)


def from_bytes(b: bytes) -> Ext:
    return tuple(int.from_bytes(b[4 * i:4 * i + 4], "little") for i in range(DEG))


# ---------------------------------------------------------------------------------------------------------------------
# Tensor arithmetic
# ---------------------------------------------------------------------------------------------------------------------

def t_ext(x: Ext, device) -> torch.Tensor:
    return torch.tensor([int(c) % P for c in x], dtype=torch.int64, device=device)


def t_exts(xs: list[Ext], device) -> torch.Tensor:
    if not xs:
        return torch.zeros((0, DEG), dtype=torch.int64, device=device)
    return torch.tensor([[int(c) % P for c in x] for x in xs], dtype=torch.int64, device=device)


def py_ext(t: torch.Tensor) -> Ext:
    return tuple(int(v) for v in t.reshape(-1).tolist())


def py_exts(t: torch.Tensor) -> list[Ext]:
    return [tuple(int(v) for v in row) for row in t.reshape(-1, DEG).tolist()]


def ext_from_base(a: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(a.shape + (DEG,), dtype=torch.int64, device=a.device)
    out[..., 0] = a
    return out


def base_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b) % P


def ext_scale(a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Extension tensor times a base tensor (broadcast over the coordinate axis)."""
    return (a * c.unsqueeze(-1)) % P


def ext_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Coordinate-wise product of two extension tensors (broadcasting), reduced mod ``X^6 - W`` and ``p``.
    On CUDA the fused Triton kernel (``kernels.ext_mul``) is used when available; the torch path is the reference."""
    if _fast_ext_mul is not None and a.is_cuda:
        return _fast_ext_mul(a, b)
    return _ref_ext_mul(a, b)


def _ref_ext_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.to(torch.int64)
    b = b.to(torch.int64)
    prod = (a.unsqueeze(-1) * b.unsqueeze(-2)) % P           # (..., 6, 6): [i, j] = a_i b_j
    shape = torch.broadcast_shapes(a.shape[:-1], b.shape[:-1])
    flat = prod.reshape(-1, DEG * DEG).to(torch.float64)
    out = flat @ red_matrix(a.device)                          # exact: < 36 * 31 * 2^31 < 2^53
    return (out.to(torch.int64) % P).reshape(*shape, DEG)


def ext_mul_chunked(a: torch.Tensor, b: torch.Tensor, chunk_elems: int = 1 << 22) -> torch.Tensor:
    """``ext_mul`` for large operands of equal leading shape, bounded temporaries."""
    shape = torch.broadcast_shapes(a.shape[:-1], b.shape[:-1])
    n = 1
    for s in shape:
        n *= s
    if n <= chunk_elems:
        return ext_mul(a, b)
    a = a.expand(*shape, DEG).reshape(n, DEG)
    b = b.expand(*shape, DEG).reshape(n, DEG)
    out = torch.empty((n, DEG), dtype=torch.int64, device=a.device)
    for lo in range(0, n, chunk_elems):
        hi = min(n, lo + chunk_elems)
        out[lo:hi] = ext_mul(a[lo:hi], b[lo:hi])
    return out.reshape(*shape, DEG)


def ext_sum(a: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return a.to(torch.int64).sum(dim=dim) % P


class Limbs:
    """A canonical int64 matrix pre-split into its two 16-bit float64 limb matrices (reused across ``mm_mod`` calls)."""

    def __init__(self, B: torch.Tensor) -> None:
        B = B.to(torch.int64)
        self.shape = B.shape
        self.lo = (B & 0xFFFF).to(torch.float64)
        self.hi = (B >> 16).to(torch.float64)


def mm_mod(A: torch.Tensor | Limbs, B: torch.Tensor | Limbs) -> torch.Tensor:
    """``(A @ B) mod p`` for canonical int64 matrices via exact float64 limb matmuls (``k ≤ 2^21``)."""
    if isinstance(A, Limbs):
        a_lo, a_hi = A.lo, A.hi
    else:
        A = A.to(torch.int64)
        a_lo = (A & 0xFFFF).to(torch.float64)
        a_hi = (A >> 16).to(torch.float64)
    k = a_lo.shape[-1]
    if k > (1 << 21):
        raise ValueError("mm_mod: contraction too long for exact float64 limbs")
    if isinstance(B, Limbs):
        b_lo, b_hi = B.lo, B.hi
    else:
        B = B.to(torch.int64)
        b_lo = (B & 0xFFFF).to(torch.float64)
        b_hi = (B >> 16).to(torch.float64)
    ll = (a_lo @ b_lo).to(torch.int64) % P
    lh = ((a_lo @ b_hi).to(torch.int64) + (a_hi @ b_lo).to(torch.int64)) % P
    hh = (a_hi @ b_hi).to(torch.int64) % P
    return (ll + lh * _R16 + hh * _R32) % P


_POW8 = [pow(2, 8 * s, P) for s in range(8)]
_INT8_MAX_ELEMS = 1 << 28        # cuBLASLt faults above ~2^31 operand elements; chunk rows well below (packed lane's probe)
_INT8_K_CHUNK = 1 << 16          # 4 . 2^14 . 2^16 = 2^32 > int32 -- but each group of equal limb weight has <= 4 terms
#                                  summed in int64, and one _int_mm holds K . 2^14 <= 2^30


class Limbs8:
    """A canonical matrix pre-split into its four balanced INT8 limbs ``x = Σ_i 2^(8i) l_i``, ``l_i ∈ [-128, 128)``
    (each ``(rows, cols)`` int8, contiguous), for the INT8 tensor-core ``mm_mod_int8``."""

    @staticmethod
    def padded_shape(M: int, K: int, side: str) -> tuple[int, int]:
        """The stored (zero-padded) int32 layout for a logical ``(M, K)`` operand: ``(Mp, Kp)`` for side ``a``,
        ``(Np, Kp)`` = the transpose for side ``b`` (cuBLASLt: rows of A a multiple of 32, K and N multiples of 8)."""
        if side == "a":
            return max(32, -(-M // 32) * 32), max(32, -(-K // 8) * 8)
        return max(32, -(-K // 8) * 8), max(32, -(-M // 8) * 8)

    def __init__(self, B: torch.Tensor, side: str = "a", shape: tuple[int, int] | None = None) -> None:
        """``side="b"``: the right operand of the GEMM, stored column-major (cuBLASLt's IGEMM runs at 485 TMAC/s
        with a column-major ``B`` and at 50 with a row-major one, probed on the H100).  ``shape``: pass the logical
        ``(M, K)`` when ``B`` is already the padded int32 storage of :meth:`padded_shape` (no copy is made)."""
        self.side = side
        if shape is not None:
            self.shape = tuple(shape)
            v = B
            if v.dtype != torch.int32 or tuple(v.shape) != self.padded_shape(*shape, side) or not v.is_contiguous():
                raise ValueError("Limbs8: pre-padded operand has the wrong dtype / shape")
        else:
            self.shape = tuple(B.shape)
            M, K = B.shape
            v = torch.zeros(self.padded_shape(M, K, side), dtype=torch.int32, device=B.device)
            if side == "a":
                v[:M, :K] = B.to(torch.int32)
            else:
                v[:K, :M] = B.to(torch.int32).t()
        ls8 = _split_limbs(v)                                   # (4, *v.shape) int8, one pass
        self.limbs = [ls8[i].t() if side == "b" else ls8[i] for i in range(4)]


def _split_limbs(v: torch.Tensor) -> torch.Tensor:
    """``(4, *v.shape)`` int8: the balanced limbs of a canonical int32 tensor (fused kernel on CUDA)."""
    if _fast_split_limbs is not None and v.is_cuda:
        return _fast_split_limbs(v)
    out = []
    for _ in range(4):
        l = ((v + 128) & 255) - 128
        out.append(l.to(torch.int8))
        v = (v - l) >> 8
    return torch.stack(out, 0)


def _int_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``torch._int_mm`` on ``a (M, K) int8``, ``b (K, N) int8`` -> int32, with the shape rules of cuBLASLt (M a multiple
    of 32 and > 16, K and N multiples of 8) met by zero padding, and the operand-size cap by row chunking."""
    M, K = a.shape
    N = b.shape[1]
    Mp, Kp, Np = max(32, -(-M // 32) * 32), max(32, -(-K // 8) * 8), max(32, -(-N // 8) * 8)
    if (Mp, Kp) != (M, K):
        ap = torch.zeros((Mp, Kp), dtype=torch.int8, device=a.device)
        ap[:M, :K] = a
        a = ap
    if (Kp, Np) != (K, N) or b.stride(0) != 1:
        bp = torch.zeros((Np, Kp), dtype=torch.int8, device=b.device).t()      # column-major
        bp[:K, :N] = b
        b = bp
    if a.numel() > _INT8_MAX_ELEMS:
        rows = max(32, (_INT8_MAX_ELEMS // Kp) // 32 * 32)
        out = torch.cat([torch._int_mm(a[r:r + rows].contiguous(), b) for r in range(0, Mp, rows)], 0)
    else:
        out = torch._int_mm(a.contiguous(), b)
    return out[:M, :N]


def mm_mod_int8(A: torch.Tensor | Limbs8, B: torch.Tensor | Limbs8) -> torch.Tensor:
    """``(A @ B) mod p`` for canonical matrices on INT8 tensor cores: 16 limb GEMMs (INT32 accumulate, exact for
    ``K ≤ 2^16`` per chunk since ``|l_i l_j| < 2^14``), the limb products of equal weight ``2^(8(i+j))`` summed in
    int64 and recombined mod ``p``.  Bit-exact with ``mm_mod``."""
    a = A if isinstance(A, Limbs8) else Limbs8(A, "a")
    b = B if isinstance(B, Limbs8) else Limbs8(B, "b")
    K = a.shape[1]
    dev = a.limbs[0].device
    out = torch.zeros((a.shape[0], b.shape[1]), dtype=torch.int64, device=dev)
    for lo in range(0, K, _INT8_K_CHUNK):
        hi = min(K, lo + _INT8_K_CHUNK)
        if lo == 0 and hi == K:
            al, bl = a.limbs, b.limbs
        else:
            al = [l[:, lo:hi].contiguous() for l in a.limbs]
            bl = [l[lo:hi].t().contiguous().t() for l in b.limbs]
        for s in range(7):
            acc = None
            for i in range(4):
                j = s - i
                if 0 <= j < 4:
                    pr = _int_mm(al[i], bl[j])[: a.shape[0], : b.shape[1]].to(torch.int64)
                    acc = pr if acc is None else acc + pr
            out = (out + (acc % P) * _POW8[s]) % P
    return out


def ext_contract(X: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``Σ_g X[..., g, :] · w[g, :]`` in the extension: ``X`` ``(m, k, 6)``, ``w`` ``(k, 6)`` -> ``(m, 6)``."""
    k = w.shape[0]
    wmat = torch.zeros((k, DEG, DEG), dtype=torch.int64, device=w.device)   # [g, i, d] = coefficient of X_i w_j
    w = w.to(torch.int64)
    for i in range(DEG):
        for j in range(DEG):
            d = i + j
            if d < DEG:
                wmat[:, i, d] = w[:, j]
            else:
                wmat[:, i, d - DEG] = (W * w[:, j]) % P
    return mm_mod(X.reshape(X.shape[0], k * DEG), wmat.reshape(k * DEG, DEG))


def base_contract_ext(X: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``Σ_g X[m, g] · w[g, :]`` with base ``X`` ``(m, k)`` and extension ``w`` ``(k, 6)`` -> ``(m, 6)``."""
    return mm_mod(X, w)


def fold(v: torch.Tensor, r: torch.Tensor, chunk_elems: int = 1 << 22) -> torch.Tensor:
    """``v[i] + (v[i + half] - v[i]) r`` on the leading axis (extension ``v``, ``(2h, ..., 6)``); bounded temporaries."""
    if _fast_fold is not None and v.is_cuda:
        return _fast_fold(v, r)
    half = v.shape[0] // 2
    per_row = v[0].numel() // DEG
    step = max(1, chunk_elems // per_row)
    out = torch.empty((half,) + tuple(v.shape[1:]), dtype=torch.int64, device=v.device)
    for lo_i in range(0, half, step):
        hi_i = min(half, lo_i + step)
        lo = v[lo_i:hi_i].to(torch.int64)
        hi = v[half + lo_i:half + hi_i].to(torch.int64)
        out[lo_i:hi_i] = (lo + ext_mul((hi - lo) % P, r)) % P
    return out


def fold_base(v: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """First fold of a base-field table into the extension: ``lo + (hi - lo) r``."""
    if _fast_fold is not None and v.is_cuda:
        return _fast_fold(v, r, base=True)
    half = v.shape[0] // 2
    lo = v[:half].to(torch.int64)
    d = (v[half:].to(torch.int64) - lo) % P
    out = (d.unsqueeze(-1) * r) % P
    out[..., 0] = (out[..., 0] + lo) % P
    return out


def eq_table(point: list[Ext], device, out_dtype=torch.int64) -> torch.Tensor:
    """``eq(point, x)`` for every ``x`` on the hypercube ``(2^n, 6)``, top-bit-first (``field.rs::eq_table``).  Large
    tables (``n > 16``, CUDA) are the tensor product of the two halves' tables, written once in ``out_dtype`` by a fused
    kernel (the doubling loop would materialise ~4x the table in int64 temporaries: 26 GB at ``n = 28``)."""
    n = len(point)
    if _fast_eq_outer is not None and n > 16 and torch.device(device).type == "cuda":
        n_hi = n // 2
        return _fast_eq_outer(eq_table(point[:n_hi], device), eq_table(point[n_hi:], device), out_dtype)
    t = torch.zeros((1, DEG), dtype=torch.int64, device=device)
    t[0, 0] = 1
    for r in reversed(point):
        rt = t_ext(r, device)
        one_minus = (t_ext(ONE, device) - rt) % P
        t = torch.cat([ext_mul_chunked(t, one_minus), ext_mul_chunked(t, rt)], dim=0)
    return t if out_dtype == torch.int64 else t.to(out_dtype)


def eval_mle(values: torch.Tensor, point: list[Ext]) -> Ext:
    v = values.to(torch.int64)
    for r in point:
        v = fold(v, t_ext(r, v.device))
    return py_ext(v[0])


def geometric(rho: Ext, geo: Ext, n: int, device) -> torch.Tensor:
    """``rho · geo^k`` for ``k < n`` as an ``(n, 6)`` tensor (doubling steps)."""
    pw = t_ext(rho, device).reshape(1, DEG)
    g = geo
    while pw.shape[0] < n:
        pw = torch.cat([pw, ext_mul(pw, t_ext(g, device))], dim=0)
        g = e_mul(g, g)
    return pw[:n]


# ---------------------------------------------------------------------------------------------------------------------
# Base-field helpers (roots of unity for the Reed-Solomon code)
# ---------------------------------------------------------------------------------------------------------------------

GENERATOR = 31  # a primitive element of BabyBear (Plonky3's choice); order p - 1 = 2^27 * 15
TWO_ADICITY = 27


def root_of_unity(order: int) -> int:
    if order & (order - 1) or order > 2**TWO_ADICITY:
        raise ValueError("order must be a power of two <= 2^27")
    return pow(GENERATOR, (P - 1) // order, P)


def inv_mod(a: int) -> int:
    return pow(a % P, P - 2, P)


# the fused kernel (Triton, CUDA); ``VERITY_GPU_TORCH_ONLY=1`` disables it
_fast_ext_mul = None
_fast_fold = None
_fast_logup_round = None
_fast_phase1 = None
_fast_scatter = None
_fast_gate_eval = None
_fast_combine_level = None
_fast_split_limbs = None
_fast_eq_outer = None
_fast_rank1_add = None
_fast_eq_rows_dot = None
try:
    import os as _os

    if not _os.environ.get("VERITY_GPU_TORCH_ONLY"):
        from . import kernels as _kernels

        if _kernels.AVAILABLE:
            _fast_ext_mul = _kernels.ext_mul
            _fast_fold = _kernels.fold
            _fast_logup_round = _kernels.logup_round
            _fast_phase1 = _kernels.phase1_round
            _fast_scatter = _kernels.scatter_terms
            _fast_gate_eval = _kernels.gate_eval
            _fast_combine_level = _kernels.combine_level
            _fast_split_limbs = _kernels.split_limbs
            _fast_eq_outer = _kernels.eq_outer
            _fast_rank1_add = _kernels.rank1_add
            _fast_eq_rows_dot = _kernels.eq_rows_dot
except Exception:  # noqa: BLE001 -- no Triton / no CUDA: the torch path is used
    _fast_ext_mul = _fast_fold = _fast_logup_round = _fast_phase1 = _fast_scatter = _fast_gate_eval = _fast_combine_level = None
    _fast_split_limbs = _fast_eq_outer = _fast_rank1_add = _fast_eq_rows_dot = None
