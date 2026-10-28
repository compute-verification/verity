"""One Strassen level of the int8 GEMM, exact: O = A Bᵀ (A int8 [M, K], B int8 [N, K], O int32 [M, N]) from seven
half-size int8 products, the CPU reference of orbit.cu and its metadata.

Blocks: A(i, j) = A[i-th half of the rows, j-th half of K], B(i, j) = B[i-th half of the rows, j-th half of K], O(i, j)
likewise. With X = A and Y = Bᵀ, Strassen 1969's products are P = form_A · form_Bᵀ:

    M1 = (A00 + A11)(B00 + B11)ᵀ   M2 = (A10 + A11) B00ᵀ        M3 = A00 (B10 − B11)ᵀ     M4 = A11 (B01 − B00)ᵀ
    M5 = (A00 + A01) B11ᵀ          M6 = (A10 − A00)(B00 + B10)ᵀ  M7 = (A01 − A11)(B01 + B11)ᵀ
    O00 = M1 + M4 − M5 + M7   O01 = M3 + M5   O10 = M2 + M4   O11 = M1 − M2 + M3 + M6

Every form has at most two blocks, so a form F lies in [−256, 255] and is w + 256 c with w = wrap8(F) its int8 wrap and
c in {−1, 0, 1} its carry. The tensor cores multiply the wrapped forms; with F_B = wB + 256 cB,

    F_A F_Bᵀ = wA wBᵀ + 256 (cA F_Bᵀ + wA cBᵀ)

so a product is exact once each carry adds 256 c times a column of the other side's form: a B carry at (n, k) adds
256 c wA[:, k] to column n, an A carry at (m, k) adds 256 c F_B[:, k] to row m. int32 arithmetic is mod 2^32 and |O| < 2^31
for K < 2^17, so wrapped int32 accumulation and assembly give O exactly.

The kernel runs the products in ORDER through one pipeline with three accumulators (three is the fewest any order needs:
`min_live_accumulators`), assembling as `assemble` does. B's wrapped forms are laid out along K in ORDER (`prepare_b`:
B' [N/2, 7 K/2]), and so are A's (`forms_a`: A' [M/2, 7 K/2]), so to the producer the seven products are one GEMM of
shape (M/2, N/2, 7 K/2).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

# each form: ((half of rows, half of K, sign), ...); Strassen's numbering, 1-based
A_FORMS = {1: ((0, 0, 1), (1, 1, 1)), 2: ((1, 0, 1), (1, 1, 1)), 3: ((0, 0, 1),), 4: ((1, 1, 1),),
           5: ((0, 0, 1), (0, 1, 1)), 6: ((1, 0, 1), (0, 0, -1)), 7: ((0, 1, 1), (1, 1, -1))}
B_FORMS = {1: ((0, 0, 1), (1, 1, 1)), 2: ((0, 0, 1),), 3: ((1, 0, 1), (1, 1, -1)), 4: ((0, 1, 1), (0, 0, -1)),
           5: ((1, 1, 1),), 6: ((0, 0, 1), (1, 0, 1)), 7: ((0, 1, 1), (1, 1, 1))}
# O's quadrant (row half, column half) as signed products
QUADRANTS = {(0, 0): {1: 1, 4: 1, 5: -1, 7: 1}, (0, 1): {3: 1, 5: 1}, (1, 0): {2: 1, 4: 1}, (1, 1): {1: 1, 2: -1, 3: 1, 6: 1}}
# the kernel's product order: products 6 and 7 accumulate onto the live sums, the rest into a fresh accumulator
ORDER = (2, 4, 1, 6, 7, 3, 5)


def _block(x: np.ndarray, i: int, j: int) -> np.ndarray:
    r, c = x.shape[0] // 2, x.shape[1] // 2
    return x[i * r:(i + 1) * r, j * c:(j + 1) * c]


def form(x: np.ndarray, terms) -> np.ndarray:
    """A form of x's blocks, exact (int16)."""
    out = np.zeros((x.shape[0] // 2, x.shape[1] // 2), np.int16)
    for i, j, s in terms:
        out += s * _block(x, i, j).astype(np.int16)
    return out


def wrap(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(w, c): f = w + 256 c with w int8 and c in {−1, 0, 1} for f in [−384, 383]."""
    w = ((f.astype(np.int32) + 128) & 0xFF) - 128
    c = (f.astype(np.int32) - w) >> 8
    return w.astype(np.int8), c.astype(np.int8)


@dataclass
class PreparedB:
    """B's forms in ORDER, made once at load: `wide` [N/2, 7 K/2] int8 (the wrapped forms side by side along K), `carry`
    [7, N/2, K/2] int8 (each form's carries, mostly zero) and `full` [7, N/2, K/2] int16 (each form exactly)."""
    n: int
    k: int
    wide: np.ndarray
    carry: np.ndarray
    full: np.ndarray

    def carry_list(self, slot: int) -> np.ndarray:
        """[r, 3] int32 rows (n, k, c) of slot's nonzero carries, sorted by n then k."""
        n, k = np.nonzero(self.carry[slot])
        return np.stack([n, k, self.carry[slot][n, k]], axis=1).astype(np.int32)

    def carry_rate(self) -> dict[int, float]:
        """Each two-block form's carry rate, by Strassen's number."""
        return {p: float(np.count_nonzero(self.carry[s])) / self.carry[s].size
                for s, p in enumerate(ORDER) if len(B_FORMS[p]) == 2}


def _check_shape(m: int, n: int, k: int) -> None:
    if m % 2 or n % 2 or k % 2:
        raise ValueError(f"M, N and K must be even, got {m}, {n}, {k}")


def prepare_b(b: np.ndarray) -> PreparedB:
    """B int8 [N, K] -> its forms in ORDER (see PreparedB)."""
    n, k = b.shape
    _check_shape(2, n, k)
    full = np.stack([form(b, B_FORMS[p]) for p in ORDER])
    w, c = wrap(full)
    wide = np.ascontiguousarray(w.transpose(1, 0, 2).reshape(n // 2, 7 * (k // 2)))
    return PreparedB(n, k, wide, c, full)


def forms_a(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A int8 [M, K] -> (wide [M/2, 7 K/2] int8 wrapped forms in ORDER, carry [7, M/2, K/2] int8, full [7, M/2, K/2] int16)."""
    m, k = a.shape
    _check_shape(m, 2, k)
    full = np.stack([form(a, A_FORMS[p]) for p in ORDER])
    w, c = wrap(full)
    wide = np.ascontiguousarray(w.transpose(1, 0, 2).reshape(m // 2, 7 * (k // 2)))
    return wide, c, full


def _mm(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """x yᵀ as int64, exact: float64 BLAS while every partial sum stays below 2^53, int64 otherwise."""
    bound = float(np.abs(x).max(initial=0)) * float(np.abs(y).max(initial=0)) * x.shape[1]
    if bound < 2.0 ** 53:
        return (x.astype(np.float64) @ y.astype(np.float64).T).astype(np.int64)
    return x.astype(np.int64) @ y.astype(np.int64).T


def products(a_wide: np.ndarray, a_carry: np.ndarray, pb: PreparedB, corrected: bool = True) -> list[np.ndarray]:
    """The seven products in ORDER as int64: wrapped forms on the tensor cores, plus (corrected) each carry's 256 c column."""
    h = pb.k // 2
    out = []
    for s in range(7):
        wa = a_wide[:, s * h:(s + 1) * h]
        wb = pb.wide[:, s * h:(s + 1) * h]
        p = _mm(wa, wb)
        if corrected:
            p += 256 * (_mm(a_carry[s], pb.full[s]) + _mm(wa, pb.carry[s]))
        out.append(p)
    return out


def assemble(p: list[np.ndarray]) -> np.ndarray:
    """O int32 [M, N] from the products in ORDER, as the kernel does: three accumulators, int32 wrap-around arithmetic."""
    q = [x.astype(np.int64).astype(np.uint32).view(np.int32) for x in p]
    m2, m4, m1, m6, m7, m3, m5 = q
    with np.errstate(over="ignore"):
        r0, r1 = m2.copy(), m4.copy()
        o10 = r0 + r1
        r0 = -r0
        r2 = m1
        r0 += r2
        r1 += r2
        r0 += m6
        r1 += m7
        r2 = m3
        o11 = r0 + r2
        r0 = m5
        o01 = r2 + r0
        o00 = r1 - r0
    return np.block([[o00, o01], [o10, o11]])


def orbit_mm(a: np.ndarray, pb: PreparedB, corrected: bool = True) -> np.ndarray:
    """O = A Bᵀ int32 through the orbit (exact when corrected)."""
    if a.shape[1] != pb.k:
        raise ValueError(f"A has K {a.shape[1]}, B {pb.k}")
    a_wide, a_carry, _ = forms_a(a)
    return assemble(products(a_wide, a_carry, pb, corrected))


def plain_mm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """O = A Bᵀ int32, the plain GEMM's output (torch._int_mm's)."""
    return _mm(a, b).astype(np.uint32).view(np.int32)


def min_live_accumulators() -> int:
    """The fewest accumulators any product order needs: the largest rank, over the products' sequence, of the open
    quadrants' partial sums in the products done so far (a quadrant closes once its last product is in)."""
    quads = list(QUADRANTS)
    g = np.array([[QUADRANTS[q].get(p, 0) for p in range(1, 8)] for q in quads])
    need = [set(QUADRANTS[q]) for q in quads]
    best = 7
    for perm in itertools.permutations(range(1, 8)):
        done: set[int] = set()
        live = set(range(4))
        peak = 0
        for p in perm:
            cols = [c - 1 for c in sorted(done | {p})]
            peak = max(peak, int(np.linalg.matrix_rank(g[np.ix_(sorted(live), cols)])) if live else 0)
            done.add(p)
            live = {i for i in live if not need[i] <= done}
        best = min(best, peak)
    return best
