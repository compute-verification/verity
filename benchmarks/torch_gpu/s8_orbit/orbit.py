"""One Strassen level of the int8 GEMM O = X·Wᵀ, exact in int32: the numpy reference of the fused orbit kernel.

X is A (M×K) and W is stored as W (N×K), so B = Wᵀ. On the halves m0|m1 of M, n0|n1 of N and k0|k1 of K:
A11 = X[m0, k0], A12 = X[m0, k1], A21 = X[m1, k0], A22 = X[m1, k1], and B11 = W[n0, k0], B12 = W[n1, k0],
B21 = W[n0, k1], B22 = W[n1, k1], each B block held as W holds it ([N/2, K/2]).

Strassen 1969 (`A_FORMS`, `B_FORMS`, `C_SUMS`): every form has at most two terms. So a form F is w + 256·c, where w is
F's int8 wrap and c ∈ {-1, 0, 1} its carry, and each product is exactly

    P = F_A·F_Bᵀ = w_A·w_Bᵀ + 256·(c_A·F_Bᵀ + w_A·c_Bᵀ).

The tensor cores compute w_A·w_Bᵀ, and the carries are sparse corrections. int32 arithmetic is Z/2^32, and |O| < 2^31
for int8 operands at K < 2^17, so wrapped int32 products, corrections and assembly give O bit for bit. Odd M, N or K are
padded with a zero row or column, as the kernel's loads zero-fill.

B's forms and their carries are made once per weight (`b_forms`). The carries are kept as metadata (`b_carries`): per
form and tile of `tile_n` output columns, entries k << 9 | n_local << 1 | (c < 0), sorted by k. A's forms are made per
call (`a_forms`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

A_FORMS = (
    (("11", 1), ("22", 1)),
    (("21", 1), ("22", 1)),
    (("11", 1),),
    (("22", 1),),
    (("11", 1), ("12", 1)),
    (("21", 1), ("11", -1)),
    (("12", 1), ("22", -1)),
)
B_FORMS = (
    (("11", 1), ("22", 1)),
    (("11", 1),),
    (("12", 1), ("22", -1)),
    (("21", 1), ("11", -1)),
    (("22", 1),),
    (("11", 1), ("12", 1)),
    (("21", 1), ("22", 1)),
)
C_SUMS = {
    "11": ((0, 1), (3, 1), (4, -1), (6, 1)),
    "12": ((2, 1), (4, 1)),
    "21": ((1, 1), (3, 1)),
    "22": ((0, 1), (1, -1), (2, 1), (5, 1)),
}
TILE_N = 96
K_SHIFT, N_SHIFT = 9, 1
N_LOCAL_MAX = 1 << (K_SHIFT - N_SHIFT)


def pad_even(a: np.ndarray) -> np.ndarray:
    pads = [(0, n % 2) for n in a.shape]
    return np.pad(a, pads) if any(p for _, p in pads) else a


def x_blocks(x: np.ndarray) -> dict[str, np.ndarray]:
    hm, hk = x.shape[0] // 2, x.shape[1] // 2
    return {"11": x[:hm, :hk], "12": x[:hm, hk:], "21": x[hm:, :hk], "22": x[hm:, hk:]}


def w_blocks(w: np.ndarray) -> dict[str, np.ndarray]:
    hn, hk = w.shape[0] // 2, w.shape[1] // 2
    return {"11": w[:hn, :hk], "12": w[hn:, :hk], "21": w[:hn, hk:], "22": w[hn:, hk:]}


def form(blocks: dict[str, np.ndarray], terms: tuple) -> np.ndarray:
    out = np.zeros(blocks["11"].shape, np.int16)
    for name, sign in terms:
        out += sign * blocks[name].astype(np.int16)
    return out


def wrap(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """F's int8 wrap w and carry c, F = w + 256·c."""
    w = ((f.astype(np.int16) + 128) & 0xFF) - 128
    c = (f - w) >> 8
    if np.abs(c).max(initial=0) > 1:
        raise ValueError("a form of more than two int8 terms")
    return w.astype(np.int8), c.astype(np.int8)


def _forms(blocks: dict[str, np.ndarray], forms: tuple) -> tuple[np.ndarray, np.ndarray]:
    pairs = [wrap(form(blocks, terms)) for terms in forms]
    return np.stack([w for w, _ in pairs]), np.stack([c for _, c in pairs])


def a_forms(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A's 7 forms of an int8 X [M, K], wrapped, and their carries: two int8 [7, M/2, K/2]."""
    return _forms(x_blocks(pad_even(x)), A_FORMS)


def b_forms(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """B's 7 forms of an int8 W [N, K], wrapped, and their carries: two int8 [7, N/2, K/2], as W is laid out."""
    return _forms(w_blocks(pad_even(w)), B_FORMS)


@dataclass(frozen=True)
class Carries:
    """Carry lists of [7, R, C] carries: `entries` [E] uint32, form j's tile t in entries[offsets[j, t]:offsets[j, t + 1]]."""
    offsets: np.ndarray
    entries: np.ndarray
    rows: int
    cols: int
    tile: int


def b_carries(c: np.ndarray, tile_n: int = TILE_N) -> Carries:
    """B's carries as the kernel reads them: per form and tile of `tile_n` rows of W (output columns), sorted by k."""
    if not 0 < tile_n <= N_LOCAL_MAX:
        raise ValueError(f"tile_n {tile_n} not in 1..{N_LOCAL_MAX}")
    forms, rows, cols = c.shape
    tiles = -(-rows // tile_n)
    if cols >= 1 << (32 - K_SHIFT):
        raise ValueError(f"K/2 = {cols} does not fit an entry")
    offsets = np.zeros((forms, tiles + 1), np.int64)
    chunks = []
    total = 0
    for j in range(forms):
        n, k = np.nonzero(c[j])
        tile = n // tile_n
        order = np.lexsort((n, k, tile))
        n, k, tile = n[order], k[order], tile[order]
        neg = (c[j][n, k] < 0).astype(np.uint32)
        chunks.append((k.astype(np.uint32) << K_SHIFT) | ((n % tile_n).astype(np.uint32) << N_SHIFT) | neg)
        offsets[j, 1:] = total + np.cumsum(np.bincount(tile, minlength=tiles))
        offsets[j, 0] = total
        total += len(n)
    return Carries(offsets, np.concatenate(chunks) if chunks else np.zeros(0, np.uint32), rows, cols, tile_n)


def carry_matrix(cl: Carries, j: int) -> np.ndarray:
    """Form j's carries [R, C] int8, from the lists alone."""
    out = np.zeros((cl.rows, cl.cols), np.int8)
    for t in range(cl.offsets.shape[1] - 1):
        e = cl.entries[cl.offsets[j, t]:cl.offsets[j, t + 1]]
        n = t * cl.tile + ((e >> N_SHIFT) & (N_LOCAL_MAX - 1)).astype(np.int64)
        k = (e >> K_SHIFT).astype(np.int64)
        out[n, k] = np.where(e & 1, -1, 1)
    return out


def _mm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a·bᵀ for small-integer operands, exact: float64 sums of integers below 2^53."""
    return (a.astype(np.float64) @ b.astype(np.float64).T).astype(np.int64)


def products(wa: np.ndarray, ca: np.ndarray, wb: np.ndarray, cl: Carries) -> np.ndarray:
    """The 7 products [7, M/2, N/2] int64: the wrapped forms' products plus 256 × the carry corrections, B's from its lists."""
    out = []
    for j in range(7):
        cb = carry_matrix(cl, j)
        fb = wb[j].astype(np.int16) + 256 * cb.astype(np.int16)
        p = _mm(wa[j], wb[j])
        if ca[j].any():
            p += 256 * _mm(ca[j], fb)
        if cb.any():
            p += 256 * _mm(wa[j], cb)
        out.append(p)
    return np.stack(out)


def assemble(p: np.ndarray) -> np.ndarray:
    """O's four quadrants from the 7 products, as one [M, N] int64."""
    q = {name: sum(s * p[j] for j, s in terms) for name, terms in C_SUMS.items()}
    return np.block([[q["11"], q["12"]], [q["21"], q["22"]]])


def to_int32(v: np.ndarray) -> np.ndarray:
    return (((v + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)).astype(np.int32)


def plain(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """The plain int8 GEMM X·Wᵀ in int32."""
    o = _mm(x, w)
    if np.abs(o).max(initial=0) >= 1 << 31:
        raise ValueError("X·Wᵀ leaves int32")
    return o.astype(np.int32)


def orbit(x: np.ndarray, w: np.ndarray, tile_n: int = TILE_N, b: tuple | None = None) -> np.ndarray:
    """X·Wᵀ in int32 through the orbit; `b` is (w_B, Carries) from a previous `b_forms` and `b_carries` of W."""
    m, n = x.shape[0], w.shape[0]
    if b is None:
        wb, cb = b_forms(w)
        b = (wb, b_carries(cb, tile_n))
    wa, ca = a_forms(x)
    return to_int32(assemble(products(wa, ca, *b)))[:m, :n]


def wrapped_only(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """The orbit with no corrections, in int32: equal to O wherever no form carries, and to O mod 2^8 everywhere."""
    wa, _ = a_forms(x)
    wb, _ = b_forms(w)
    p = np.stack([_mm(wa[j], wb[j]) for j in range(7)])
    return to_int32(assemble(p))[:x.shape[0], :w.shape[0]]
