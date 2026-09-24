"""Zero knowledge for the Ligerito backend (lane ligerito-zk): what the prover adds so that the zero-check sumcheck and
the recursive Ligerito openings reveal nothing about the witness, against a malicious verifier, keeping PROTOCOL.md 8c.

Design (note ``~/.research/notes/lanes/ligerito-zk/20260923T1700Z-report-ligerito-zk.md`` sections 1-2); the four pieces
and the leak each one covers, with ``f`` the witness polynomial (N = 2^n base cells), round ``i`` of the recursion
committing the column-major matrix ``M~_i`` (``2^{k_i}`` rows x ``2^{k'_i}`` columns, column bits = the high variables):

1. **Mask rows inside the committed polynomial** (``mask_cells`` / ``sample_masks`` / ``apply_masks``): round-1
   columns ``[0, 1)`` and ``[C_1 - 8, C_1)`` are uniform.  Every entry of ``y_1 = M~_1 rbar_1`` -- and every fold of it,
   the expected symbols ``<gen_s, y_i>``, the final vector, the evaluation claims ``f^(z)`` (whose points must have no
   boolean coordinate on the mask rows' bits: the eval claim has to REACH the mask rows) -- is a functional of >= 6
   uniform base cells whose extension weights span ``F_{p^6}`` over ``F_p`` (8 tensor weights of 3 random coordinates),
   hence exactly uniform (masks are base cells, functionals are extension elements: one cell blinds one of 6 coordinates).
   The round-1 partial sumcheck's messages sum the columns of one half of a bit split: a mask column in both halves (0
   and ``C_1 - 1``) makes ``u[c] = <rowpart, M~_1[:, c]>`` uniform there and the 2 free extension values per round
   uniform.  The Libra coefficients (3) live in the last mask column.  The layout must leave these cells unconstrained
   AND unreferenced (``check_layout``); fp8-ada's layout has 319 zero rows -- 144 of them do it, so 0 extra cells.
   The zero-check's final sub-cube claims ``Az(r), Bz(r), Cz(r)`` are bare witness functionals (boolean row bits):
   the layout adds ONE trivial quadratic constraint ``w(i1, c) . w(i2, c) = w(i3, c)`` at a free constraint slot on
   three mask rows (``i1, i2`` uniform, ``i3 = i1 * i2`` derived by ``apply_masks``; ``ZkParams.product_rows``), so each
   of the three claims contains ``eq(r_k, k_m) . m~(r_c)`` -- jointly uniform given the rank event (``m3~(r_c)`` is a
   linear functional of ``m1`` with coefficients ``eq_c m2_c``, independent of ``m1~(r_c)``'s ``eq_c`` generically).
2. **Per-column RS padding** (``pad_columns`` / ``padding_combination`` / ``row_check_rhs``): every column's message is
   ``(M~_i[:, c] || mu_c)`` with ``mu_c`` uniform of length ``t_pad > |S_i|`` (Ligero 8b in coefficient form), so the
   opened rows are uniform; the padding's ``rbar``-combination ``ybar_i = mu^T rbar_i`` travels in the clear with the
   next root and the verifier's row check is ``<U_i[s,:], rbar_i> = <gen_s, y_i> + <gen'_s, ybar_i>`` (the claimed
   combination is the message ``(y_i || ybar_i)``; ``ybar_i`` is fixed before ``S_i`` is drawn, so the paper's
   consistency guarantee applies with rate ``(2^{k_i} + t_pad) / n_i`` -- ``queries_for`` sizes ``|S_i|`` for it).
3. **Libra-style sumcheck masks** (``SumcheckMask``; Xie et al. CRYPTO'19 ePrint 2019/317 section 4.1 Construction 1 /
   Theorem 3, the cheap instance of Chiesa-Forbes-Spooner ePrint 2017/305), ONE PER SUMCHECK over witness tables (the
   zero-check, the rows sumcheck, the shift sumcheck: their public factors vanish on the mask rows, so mask rows do not
   reach their messages): ``g(x) = a_0 + sum_i sum_{e=1..d} a_{i,e} x_i^e`` with ``dn + 1`` uniform extension
   coefficients committed in the sumcheck's G-block, ``G = sum_x g(x)`` sent, coin ``rho``, sumcheck on ``summand + rho
   g(x)``; the verifier gets ``g(r)`` as an inner-product claim on the committed polynomial (``SumcheckMask.claim``).
4. **Committed coins** (``coin_schedule`` / ``CommittedCoins`` / ``challenge_from``): every verifier message is a slot
   committed at step 0 (``c_k = Blake2b(tag || k || r_k || s_k)``), opened in order; the challenge of slot ``k`` is a
   function of ``(stmt, C = H(c_1..c_K), k, r_k)`` alone -- no prover message enters a seed (8c, red team F2); the
   prover aborts publicly on a mismatched opening (``CoinOpeningError``).

Field: BabyBear; extension ``F_{p^6} = F_p[x] / (x^6 - 31)`` as lane ligerito-proto's ``ext.py`` (an element is an int64
array whose last axis has length 6); the small extension helpers here are private and match that convention exactly so
they can be swapped for the shared module when it lands.  Everything here is numpy; torch is imported lazily and only
for the device mask sampler (``backends/direct/ligero/mask_sampler.py``, ChaCha20 + rejection, exactly uniform).
"""
from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field

import numpy as np

P = 2013265921                  # BabyBear 2^31 - 2^27 + 1
BETA = 31                       # F_{p^6} = F_p[x] / (x^6 - 31)   (ligerito-proto ext.py)
D = 6
T_PAD = 256                     # padding coefficients per committed column (> |S_i| = 192; Ligero 8b's value)
COIN_BYTES = 32
COIN_TAG = b"ligerito-zk/coin-commit|v1"
COIN_SEED_TAG = b"ligerito-zk/coin|v1"

__all__ = ["BETA", "COIN_BYTES", "D", "MASK_COLS_HIGH_LOG2", "MASK_COLS_LOW_LOG2", "P", "T_PAD", "AbortingCoins", "AdversarialCoins", "BlockClaim", "CoinOpeningError",
           "CommittedCoins", "EquivocatingCoins", "MaskCells", "Masks", "SumcheckMask", "ZkParams", "apply_masks",
           "challenge_from", "check_layout", "coin_commit", "coin_schedule", "mask_cells", "omitted", "pad_columns",
           "padding_combination", "queries_for", "rate", "row_check_rhs", "sample_masks", "set_g_coeffs", "uniform_symbols",
           "verify_opening", "zk_statement",
           "EqSumcheckMask", "RowMaskLayout", "RowMasks", "zk_mode_rows", "default_row_mask_layout", "f_index", "fill_mask_rows", "libra_block_claim",
           "permute_point", "z_to_f"]


# ---------------------------------------------------------------------------------------------------------------------
# F_{p^6} helpers (numpy, last axis = 6 coordinates), same convention as ligerito-proto's ext.py
# ---------------------------------------------------------------------------------------------------------------------

def ext_zero(shape=()) -> np.ndarray:
    return np.zeros(tuple(shape) + (D,), dtype=np.int64)


def ext_one(shape=()) -> np.ndarray:
    z = ext_zero(shape)
    z[..., 0] = 1
    return z


def ext_from_base(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.int64) % P
    z = np.zeros(a.shape + (D,), dtype=np.int64)
    z[..., 0] = a
    return z


def ext_mul(a, b) -> np.ndarray:
    """(..., 6) x (..., 6) -> (..., 6); every coordinate product reduced before summing (products < 2^62)."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    prod = (a[..., :, None] * b[..., None, :]) % P                     # (..., 6, 6), each < 2^31
    shape = prod.shape[:-2]
    return (prod.reshape(shape + (D * D,)) @ _EXT_MUL_TABLE) % P       # 36 -> 6 fold: sums < 36 * 31 * 2^31


def _ext_mul_table() -> np.ndarray:
    t = np.zeros((D * D, D), dtype=np.int64)
    for i in range(D):
        for j in range(D):
            k = i + j
            if k >= D:
                t[i * D + j, k - D] = BETA
            else:
                t[i * D + j, k] = 1
    return t


_EXT_MUL_TABLE = _ext_mul_table()


def ext_mul_base(a, s) -> np.ndarray:
    s = np.asarray(s, dtype=np.int64) % P
    return (np.asarray(a, dtype=np.int64) * s[..., None]) % P


def ext_pow(a, e: int) -> np.ndarray:
    result, base = ext_one(), np.asarray(a, dtype=np.int64) % P
    while e:
        if e & 1:
            result = ext_mul(result, base)
        base = ext_mul(base, base)
        e >>= 1
    return result


def ext_inv(a) -> np.ndarray:
    return ext_pow(a, P ** D - 2)


def ext_sum(a, axis=0) -> np.ndarray:
    return np.sum(np.asarray(a, dtype=np.int64), axis=axis) % P


def ext_is_zero(a) -> bool:
    return not np.any(np.asarray(a, dtype=np.int64) % P)


# ---------------------------------------------------------------------------------------------------------------------
# Parameters and the reserved cells
# ---------------------------------------------------------------------------------------------------------------------

MASK_COLS_HIGH_LOG2 = 3         # top mask block = the top 2^3 = 8 round-1 columns (>= D base cells per extension functional)
MASK_COLS_LOW_LOG2 = 0          # bottom mask block = round-1 column 0 (a mask column in BOTH halves of every column-bit split)


@dataclass(frozen=True)
class ZkParams:
    """``n``: log2 cells of the committed polynomial; ``k1``/``kp1``: round-1 rows / columns (log2); ``sumchecks``: the
    sumchecks over witness tables that get a Libra mask, as ``(name, n_vars, degree)`` -- fp8-ada / ligerito-sumcheck:
    ``(("zc", 30, 3), ("rows", 12, 2), ("shift", 18, 2))``; ``t_pad``: RS padding coefficients per committed column.

    Why blocks of 8 columns, at both ends: masks are BASE-field cells while every leaked functional (a fold ``y_i[row]``,
    an evaluation ``f^(z)``, a round-1 contraction ``u[c]``) is an EXTENSION element ``sum_x w_x f^[x]``, ``w_x in F_{p^6}``;
    one uniform base cell blinds only the F_p-line ``F_p w_x``.  ``m >= D = 6`` uniform base cells whose weights span
    ``F_{p^6}`` over ``F_p`` make the functional exactly uniform; the 8 cells of a 3-bit sub-cube have weights
    ``{prod_j pair_j(b_j)}_{b in {0,1}^3}`` (tensor factors of 3 random extension coordinates), which span with
    probability ``1 - O(1/p)`` -- the rank event of the simulator's distance.  The round-``j`` message of the round-1
    partial sumcheck sums the columns whose bit ``j`` is 0 (resp. 1): a mask column must sit in BOTH halves of every
    split and stay there under the fold ``u'[c] = (1-r) u[c] + r u[c + C/2]`` -- column 0 and column ``C_1 - 1`` do."""
    n: int
    k1: int
    kp1: int
    sumchecks: tuple = (("zc", 30, 3), ("rows", 12, 2), ("shift", 18, 2))
    t_pad: int = T_PAD
    enabled: bool = True
    mask_cols_high_log2: int = MASK_COLS_HIGH_LOG2
    mask_cols_low_log2: int = MASK_COLS_LOW_LOG2
    n_c: int | None = None             # layout column bits (a layout ROW = 2^{n_c} cells; rows are the high bits)
    product_rows: tuple = ()           # (i1, i2, i3) triples of layout rows: i1, i2 uniform masks, i3 := i1 * i2 pointwise
    row_layout: bool = False           # True: masks are relation ROWS (RowMaskLayout); no column blocks in f

    def __post_init__(self):
        if self.k1 + self.kp1 != self.n:
            raise ValueError(f"round-1 shape 2^{self.k1} x 2^{self.kp1} does not cover 2^{self.n} cells")
        if self.enabled and not self.row_layout and (1 << self.k1) < self.g_block_cells + 2:
            raise ValueError("the mask columns are too short for the G-blocks")
        if self.enabled and not self.row_layout and (1 << self.kp1) <= self.mask_cols_high + self.mask_cols_low:
            raise ValueError("round 1 needs more columns than the mask blocks")
        names = [nm for nm, _, _ in self.sumchecks]
        if len(set(names)) != len(names):
            raise ValueError("sumcheck names must be distinct")
        if self.product_rows and self.n_c is None:
            raise ValueError("product_rows need n_c (the layout's column bits)")
        if self.enabled:
            for tri in self.product_rows:
                for i in tri:
                    if not self.row_in_mask_block(i):
                        raise ValueError(f"product row {i} is not inside a mask block")

    def row_in_mask_block(self, i: int) -> bool:
        """Is layout row ``i`` (2^{n_c} cells) inside the bottom or top mask block?"""
        lo, hi = i << self.n_c, (i + 1) << self.n_c
        mc = mask_cells(self)
        return (mc.low.start <= lo and hi <= mc.low.stop) or (mc.high.start <= lo and hi <= mc.high.stop)

    # -- sumcheck masks
    def g_coeffs(self, idx: int) -> int:
        """Libra coefficients of sumcheck ``idx``: a_0 + d per variable."""
        _, n, d = self.sumchecks[idx]
        return d * n + 1

    @property
    def g_coeffs_total(self) -> int:
        return sum(self.g_coeffs(i) for i in range(len(self.sumchecks)))

    @property
    def g_block_cells(self) -> int:
        return D * self.g_coeffs_total

    def g_block_offset(self, idx: int) -> int:
        """Cell offset of sumcheck ``idx``'s block inside the last mask column."""
        return D * sum(self.g_coeffs(i) for i in range(idx))

    # -- shapes
    @property
    def N(self) -> int:
        return 1 << self.n

    @property
    def columns1(self) -> int:
        return 1 << self.kp1

    @property
    def rows1(self) -> int:
        return 1 << self.k1

    @property
    def mask_cols_high(self) -> int:
        return 0 if self.row_layout else 1 << self.mask_cols_high_log2

    @property
    def mask_cols_low(self) -> int:
        return 0 if self.row_layout else 1 << self.mask_cols_low_log2

    @property
    def mask_cell_count(self) -> int:
        return (self.mask_cols_high + self.mask_cols_low) * self.rows1 if self.enabled else 0


@dataclass(frozen=True)
class MaskCells:
    low: range                    # the bottom mask block: round-1 columns [0, 2^low_log2)
    high: range                   # the top mask block: round-1 columns [C_1 - 2^high_log2, C_1)
    g_blocks: tuple               # per sumcheck: the cells of its Libra coefficients (start of the LAST mask column)

    def all(self) -> np.ndarray:
        return np.concatenate([np.arange(self.low.start, self.low.stop, dtype=np.int64),
                               np.arange(self.high.start, self.high.stop, dtype=np.int64)])

    @property
    def count(self) -> int:
        return len(self.low) + len(self.high)


def mask_cells(zp: ZkParams) -> MaskCells:
    """The reserved cells: the bottom block ``[0, 2^low . 2^{k1})``, the top block ``[N - 2^high . 2^{k1}, N)``; the
    G-blocks are consecutive at the START of the last column (rows ``0 ..`` of column ``C_1 - 1``)."""
    if not zp.enabled or zp.row_layout:
        return MaskCells(range(0, 0), range(0, 0), ())
    low = range(0, zp.mask_cols_low * zp.rows1)
    high = range(zp.N - zp.mask_cols_high * zp.rows1, zp.N)
    g0 = zp.N - zp.rows1
    gb = tuple(np.arange(g0 + zp.g_block_offset(i), g0 + zp.g_block_offset(i) + D * zp.g_coeffs(i), dtype=np.int64)
               for i in range(len(zp.sumchecks)))
    return MaskCells(low, high, gb)


# ---------------------------------------------------------------------------------------------------------------------
# 1b. Mask ROWS in the relation's table (the real layout; ligerito-sumcheck's z (R, C), f = z with the bits interleaved)
# ---------------------------------------------------------------------------------------------------------------------
# ligerito-sumcheck commits ``f[i + R c] = z[i, c]`` (row bits LOW), every column ``c`` is a constrained unit column (pad
# sub-batches hold valid pad witnesses), and rows ``[m + 200, R)`` are free.  So a round-1 PCS "column" (the top k'_1 bits
# of x = a slice of 2^{n_c - k'_1} relation columns, ALL rows) cannot be uniform -- only its free rows can; the primitive
# is a MASK ROW: a free relation row filled with 2^{n_c} uniform base cells.  What mask rows reach (weights = tensor
# products of the challenges, generic on every cell):
#   * every round-i column contraction ``u_i[c]`` and both halves of every column-sumcheck split (a relation row meets
#     every PCS column of every round) -- covers the round messages of the PCS recursion;
#   * the eval claims at fully-extension points (``w~(r_i, r_c)`` of the rows sumcheck), the block claims;
#   * the FINAL VECTOR ``y_L[rho]``, indexed by the LOW k_L bits of x: with row bits low, ``rho`` is a function of the row
#     index ``i mod 2^{k_L}`` -- 2^{k_L} mask rows would be needed (k_L = 10..13: impossible in 319 free rows).  Fix (a
#     relabelling of the PCS variables, done by the relation lane when it builds f and permutes the claim points):
#     interleave ``a = k_L - b`` column bits BELOW the row bits, ``x = (c mod 2^a) + 2^a i + 2^{a + n_i} (c >> a)``; then
#     ``rho = (c mod 2^a, i mod 2^b)`` and 2^b mask rows with distinct residues mod 2^b cover every final entry with
#     ``2^{n_c - a}`` uniform cells each.  ``b = 3`` -> 8 mask rows.  PCS round-1 columns are still the top k'_1 bits of
#     ``c`` (unchanged), the sumcheck lane's row-bits-first points are permuted by ``permute_point``.
#   * the zero-check's final ``Az(r), Bz(r), Cz(r)``: bare witness functionals (A, B, C vanish on free rows) -> the
#     trivial constraint ``w(i1) w(i2) = w(i3)`` on three COMMITTED mask rows (``product_rows``; the relation adds the row).
#   * the shift sumcheck's claims ``w~(bits(c_x) || rho)`` have BOOLEAN row coordinates -> bare functionals of row c_x;
#     they must be reduced by a rows-type sumcheck to an extension row point first (``## For ligerito-relation``).
# Cost at fp8-ada (R = 4096, C = 2^18): (8 + 3 + 1) rows = 12 . 2^18 = 3.1 M cells = 0.29 % of N; the sampler does that
# in < 1 ms; the 8 uniform rows live in the free region (0 extra rows), the 3 product rows + 1 Libra row are committed
# rows (4 of the 296 spare committed rows).

@dataclass(frozen=True)
class RowMaskLayout:
    """Which relation rows are masks.  ``uniform_rows``: free rows, uniform cells (>= 2^b of them, all residues mod 2^b
    present); ``product_rows``: (i1, i2, i3) committed rows with the trivial quadratic constraint ``i1 . i2 = i3``;
    ``g_row``: the committed row whose first ``6 . sum(dn + 1)`` cells hold the Libra coefficients of ``sumchecks``
    (rest uniform); ``a``: column bits interleaved below the row bits in f (``k_L - b``)."""
    n_i: int
    n_c: int
    k_L: int
    uniform_rows: tuple
    product_rows: tuple
    g_row: int
    sumchecks: tuple = (("zc", 30, 2, True), ("rows", 12, 2), ("shift", 18, 2))
    b: int = 3

    def __post_init__(self):
        rows = list(self.uniform_rows) + [i for t in self.product_rows for i in t] + [self.g_row]
        if len(set(rows)) != len(rows):
            raise ValueError("mask rows must be distinct")
        if any(not (0 <= i < (1 << self.n_i)) for i in rows):
            raise ValueError("mask row out of range")
        if self.a < 0 or self.a > self.n_c:
            raise ValueError(f"k_L - b = {self.a} column bits cannot be interleaved (n_c = {self.n_c})")
        residues = {i % (1 << self.b) for i in self.uniform_rows}
        if len(residues) != (1 << self.b):
            raise ValueError(f"uniform_rows must cover every residue mod 2^{self.b} (the final vector's row bits); "
                             f"got {sorted(residues)}")
        if (1 << self.n_c) < self.g_cells:
            raise ValueError("a row is too short for the Libra coefficients")

    @property
    def a(self) -> int:
        return self.k_L - self.b

    @staticmethod
    def n_coeffs(sc) -> int:
        """``(name, n, d[, cross])`` -> 1 + dn [+ d^2] (the cross block of the bivariate opening round, F3)."""
        _, n, d = sc[:3]
        return 1 + d * n + (d * d if len(sc) > 3 and sc[3] else 0)

    @property
    def g_cells(self) -> int:
        return D * sum(self.n_coeffs(sc) for sc in self.sumchecks)

    def g_offset(self, idx: int) -> int:
        return D * sum(self.n_coeffs(sc) for sc in self.sumchecks[:idx])

    @property
    def all_rows(self) -> tuple:
        return tuple(self.uniform_rows) + tuple(i for t in self.product_rows for i in t) + (self.g_row,)

    @property
    def cells(self) -> int:
        return len(self.all_rows) << self.n_c

    def zk_params(self, n: int, k1: int, kp1: int, t_pad: int = T_PAD) -> ZkParams:
        """The ``ZkParams`` the padding / Libra / coin helpers take (no column blocks: those are the toy's layout)."""
        return ZkParams(n, k1, kp1, sumchecks=tuple(sc[:3] for sc in self.sumchecks), t_pad=t_pad, row_layout=True)


def default_row_mask_layout(n_i: int, n_c: int, k_L: int, m: int, free_from: int, b: int = 3,
                            sumchecks=(("zc", 30, 2, True), ("rows", 12, 2), ("shift", 18, 2))) -> RowMaskLayout:
    """fp8-ada: ``m`` committed rows, free rows ``[free_from, 2^n_i)``: the 2^b uniform rows are the top free rows, the
    product rows and the Libra row the last four committed rows ``[m - 4, m)`` (the relation must leave them
    unconstrained except for the trivial constraint)."""
    R = 1 << n_i
    if R - free_from < (1 << b):
        raise ValueError("not enough free rows")
    return RowMaskLayout(n_i, n_c, k_L, tuple(range(R - (1 << b), R)), ((m - 4, m - 3, m - 2),), m - 1, sumchecks, b)


def f_index(i, c, lay: RowMaskLayout):
    """The PCS cell of relation cell ``(i, c)``: ``x = (c mod 2^a) + 2^a i + 2^{a + n_i} (c >> a)`` (numpy-broadcastable)."""
    a = lay.a
    return (c & ((1 << a) - 1)) + (i << a) + ((c >> a) << (a + lay.n_i))


def z_to_f(z, lay: RowMaskLayout):
    """``f`` (R C,) from ``z`` (R, C) (numpy or torch, any dtype): one transpose of the (2^{n_c - a}, R, 2^a) view."""
    R, C = z.shape
    a = lay.a
    v = z.reshape(R, C >> a, 1 << a)
    v = v.transpose(1, 0, 2) if isinstance(v, np.ndarray) else v.permute(1, 0, 2)
    return v.reshape(-1)


def permute_point(point, lay: RowMaskLayout):
    """A claim point in ligerito-sumcheck's order (row bits ``0..n_i-1`` then column bits) -> PCS variable order for
    ``f = z_to_f(z)`` (column bits ``0..a-1``, row bits, column bits ``a..``).  ``point`` (n_i + n_c, 6); numpy or torch."""
    n_i, a = lay.n_i, lay.a
    parts = [point[n_i:n_i + a], point[:n_i], point[n_i + a:]]
    return np.concatenate(parts, axis=0) if isinstance(point, np.ndarray) else __import__("torch").cat(parts, dim=0)


@dataclass
class RowMasks:
    g_coeffs: list                # per sumcheck: (dn + 1, 6) Libra coefficients (numpy)
    cells: int


def fill_mask_rows(z, lay: RowMaskLayout, device=None) -> RowMasks:
    """Write the masks into ``z`` (R, C) in place (numpy int64 or torch int32/int64 on ``device``): uniform rows, the
    Libra row (coefficients first, uniform after), ``i3 := i1 . i2`` for every product triple.  Returns the Libra
    coefficients (the prover's ``SumcheckMask`` inputs).  ``2^{n_c}`` symbols per row from one sampler call."""
    C = 1 << lay.n_c
    rows = list(lay.uniform_rows) + [i for (i1, i2, _) in lay.product_rows for i in (i1, i2)] + [lay.g_row]
    if isinstance(z, np.ndarray):
        u = uniform_symbols(len(rows) * C, None).reshape(len(rows), C)
        z[rows, :] = u
        for i1, i2, i3 in lay.product_rows:
            z[i3, :] = (z[i1, :].astype(np.int64) * z[i2, :].astype(np.int64)) % P
        grow = z[lay.g_row, :lay.g_cells].astype(np.int64)
    else:
        import torch
        u = uniform_symbols_device(len(rows) * C, device).reshape(len(rows), C).to(z.dtype)
        z[torch.as_tensor(rows, device=z.device), :] = u
        for i1, i2, i3 in lay.product_rows:
            z[i3, :] = ((z[i1, :].to(torch.int64) * z[i2, :].to(torch.int64)) % P).to(z.dtype)
        grow = z[lay.g_row, :lay.g_cells].to(torch.int64).cpu().numpy()
    gs = [grow[lay.g_offset(k): lay.g_offset(k) + D * lay.n_coeffs(sc)].reshape(lay.n_coeffs(sc), D).copy()
          for k, sc in enumerate(lay.sumchecks)]
    return RowMasks(gs, len(lay.all_rows) * C)


def libra_block_claim(lay: RowMaskLayout, idx: int, gm, r) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The block claim ``<W, f> = g(r)`` on the real layout: ``(cells (6(dn+1),) PCS indices, weights (6(dn+1), 6), value)``
    -- a SPARSE round-1 term for the PCS (one relation row, 6(dn + 1) consecutive relation columns)."""
    if hasattr(gm, "claim_weights"):                                     # EqSumcheckMask
        w, v = gm.claim_weights(r).reshape(-1, D), gm.final_value(r)
    else:
        bc = gm.claim(r)
        w, v = bc.weights, bc.value
    cols = np.arange(lay.g_offset(idx), lay.g_offset(idx) + len(w), dtype=np.int64)
    return f_index(lay.g_row, cols, lay), w, v


def check_layout(zp: ZkParams, referenced) -> None:
    """``referenced``: a boolean array over the N cells (True = some constraint is located at or reads the cell) or a
    callable ``cells -> bool array``.  Raises if any mask cell is touched -- the assumption the whole design rests on."""
    mc = mask_cells(zp)
    cells = mc.all()
    if cells.size == 0:
        return
    hit = referenced(cells) if callable(referenced) else np.asarray(referenced, dtype=bool)[cells]
    hit = np.asarray(hit, dtype=bool)
    if hit.any():
        bad = cells[hit]
        raise ValueError(f"layout constrains or references {int(hit.sum())} mask cells (first: {bad[:8].tolist()}); "
                         f"the ZK masking needs round-1 columns [0, {zp.mask_cols_low}) and "
                         f"[{zp.columns1 - zp.mask_cols_high}, {zp.columns1}) unconstrained and unreferenced")


# ---------------------------------------------------------------------------------------------------------------------
# Uniform sampling (device ChaCha20 + rejection when available; host SHAKE-256 + the same rejection otherwise)
# ---------------------------------------------------------------------------------------------------------------------

def uniform_symbols(total: int, device=None, key: bytes | None = None) -> np.ndarray:
    """``total`` exactly uniform BabyBear symbols as int64 numpy (host) -- see ``uniform_symbols_device`` for the
    device path.  Host: SHAKE-256 stream under a fresh 32-byte key, ``u < 2p`` rejection then ``u mod p``."""
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    if device is not None and str(device).startswith("cuda"):
        return uniform_symbols_device(total, device).cpu().numpy()
    key = os.urandom(32) if key is None else key
    out, got, ctr = [], 0, 0
    while got < total:
        need = total - got
        raw = hashlib.shake_256(b"ligerito-zk/masks|v1|" + key + ctr.to_bytes(4, "little")).digest(4 * (need + need // 8 + 64))
        u = np.frombuffer(raw, dtype="<u4").astype(np.int64)
        u = u[u < 2 * P] % P
        out.append(u)
        got += u.shape[0]
        ctr += 1
    return np.concatenate(out)[:total]


def uniform_symbols_device(total: int, device):
    """Device path: ``backends/direct/ligero/mask_sampler.py`` (ChaCha20 counter mode under a fresh key, rejection)."""
    import torch
    from backends.direct.ligero.mask_sampler import sampler_for
    s = sampler_for(device)
    if s is None:
        return torch.from_numpy(uniform_symbols(total)).to(device)
    return s.uniform(total)[0]


# ---------------------------------------------------------------------------------------------------------------------
# 1. Mask cells in the committed polynomial
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class Masks:
    low: np.ndarray               # (2^low . 2^{k1},) the bottom block's cells
    high: np.ndarray              # (2^high . 2^{k1},) the top block's cells (the G-blocks = the first cells of its last column)
    g_coeffs: list                # per sumcheck: (d n + 1, 6) Libra coefficients (a_0 first, then a_{i,e} row-major in (i, e))


def sample_masks(zp: ZkParams, device=None) -> Masks:
    """Fresh uniform masks: both blocks; the Libra coefficients are read out of the last column's first cells."""
    if not zp.enabled:
        return Masks(np.zeros(0, np.int64), np.zeros(0, np.int64), [])
    nl, nh = zp.mask_cols_low * zp.rows1, zp.mask_cols_high * zp.rows1
    flat = uniform_symbols(nl + nh, device)
    low, high = flat[:nl].copy(), flat[nl:].copy()
    g0 = nh - zp.rows1
    gs = [high[g0 + zp.g_block_offset(i): g0 + zp.g_block_offset(i) + D * zp.g_coeffs(i)].reshape(zp.g_coeffs(i), D).copy()
          for i in range(len(zp.sumchecks))]
    return Masks(low, high, gs)


def set_g_coeffs(masks: Masks, zp: ZkParams, idx: int, coeffs: np.ndarray) -> None:
    """Overwrite sumcheck ``idx``'s Libra coefficients in ``masks`` (keeps the block's cells consistent)."""
    coeffs = np.asarray(coeffs, dtype=np.int64) % P
    g0 = masks.high.shape[0] - zp.rows1 + zp.g_block_offset(idx)
    masks.high[g0: g0 + D * zp.g_coeffs(idx)] = coeffs.reshape(-1)
    masks.g_coeffs[idx] = coeffs


def apply_masks(f_flat, masks: Masks, zp: ZkParams):
    """Write the masks into the flat witness vector (numpy or torch, length N), in place, then derive the product rows
    (``f^[i3, :] = f^[i1, :] * f^[i2, :]`` -- the witness of the trivial constraint ``m1 . m2 = m3`` that masks the
    zero-check's final sub-cube claims ``Az(r), Bz(r), Cz(r)``); returns it."""
    if not zp.enabled:
        return f_flat
    mc = mask_cells(zp)
    if hasattr(f_flat, "device") and not isinstance(f_flat, np.ndarray):        # torch
        import torch
        dev = f_flat.device
        f_flat[mc.low.start:mc.low.stop] = torch.as_tensor(masks.low, device=dev)
        f_flat[mc.high.start:mc.high.stop] = torch.as_tensor(masks.high, device=dev)
    else:
        f_flat[mc.low.start:mc.low.stop] = masks.low
        f_flat[mc.high.start:mc.high.stop] = masks.high
    for i1, i2, i3 in zp.product_rows:
        L = 1 << zp.n_c
        f_flat[i3 * L:(i3 + 1) * L] = (f_flat[i1 * L:(i1 + 1) * L] * f_flat[i2 * L:(i2 + 1) * L]) % P
    return f_flat


# ---------------------------------------------------------------------------------------------------------------------
# 2. Per-column RS padding (the hooks pcs.py calls around each committed round)
# ---------------------------------------------------------------------------------------------------------------------

def pad_columns(zp: ZkParams, M: np.ndarray, device=None) -> tuple[np.ndarray, np.ndarray]:
    """``M`` (R, C) base [or (6, R, C) extension planes]: returns ``(message (R + t_pad, C) [or (6, R + t_pad, C)],
    mu (t_pad, C) [or (6, t_pad, C)])`` -- the per-column message the RS encoder takes (coefficient basis: the padding
    is the coefficients ``x^R .. x^{R + t_pad - 1}``).  Without ZK: ``(M, empty)``."""
    planes = M.ndim == 3
    R, C = M.shape[-2], M.shape[-1]
    if not zp.enabled:
        return M, np.zeros(((D,) if planes else ()) + (0, C), dtype=np.int64)
    tp = zp.t_pad
    mu = uniform_symbols((D if planes else 1) * tp * C, device).reshape(((D,) if planes else ()) + (tp, C))
    msg = np.concatenate([np.asarray(M, dtype=np.int64) % P, mu], axis=-2)
    return msg, mu


def padding_combination(mu: np.ndarray, rbar: np.ndarray) -> np.ndarray:
    """``ybar = mu^T rbar``: (t_pad, 6) extension elements; ``mu`` (t_pad, C) base or (6, t_pad, C) planes, ``rbar``
    (C, 6).  Sent in the clear with the next commitment."""
    rbar = np.asarray(rbar, dtype=np.int64) % P
    if mu.ndim == 2:                                                   # base padding x extension weights
        return _limb_matmul(np.asarray(mu, dtype=np.int64) % P, rbar)
    mu = np.asarray(mu, dtype=np.int64) % P                            # (6, tp, C) planes
    out = np.zeros((mu.shape[1], D), dtype=np.int64)
    for i in range(D):                                                 # plane i of mu times rbar: shift by x^i, fold x^6 = beta
        pi = _limb_matmul(mu[i], rbar)                                 # (tp, 6): coefficient j of sum_c mu_i[:, c] rbar[c]
        out[:, i:] += pi[:, :D - i]
        if i:
            out[:, :i] += pi[:, D - i:] * BETA
    return out % P


def _limb_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``(a @ b) mod p`` exactly for base matrices ``a`` (m, C), ``b`` (C, k) in [0, p): 16-bit limbs of ``b`` keep the
    int64 partial sums below ``C 2^47`` (C <= 2^16)."""
    if a.shape[1] > (1 << 16):
        raise ValueError("limb matmul: C <= 2^16")
    lo, hi = b & 0xFFFF, b >> 16
    return ((a @ lo) % P + ((a @ hi) % P) * 65536) % P


def row_check_rhs(zp: ZkParams, k: int, eta_s: int, ybar: np.ndarray) -> np.ndarray:
    """``<gen'_s, ybar> = sum_j eta_s^{2^k + j} ybar[j]`` (6,): what the verifier adds to ``<gen_s, y_i>`` in the row
    check of an opened row ``s`` of round ``i`` (``2^k`` rows)."""
    if not zp.enabled or ybar.shape[0] == 0:
        return ext_zero()
    tp = ybar.shape[0]
    pw = np.empty(tp, dtype=np.int64)
    pw[0] = pow(int(eta_s), 1 << k, P)
    e = int(eta_s) % P
    for j in range(1, tp):
        pw[j] = pw[j - 1] * e % P
    return ext_sum((ybar * pw[:, None]) % P, axis=0)


def rate(zp: ZkParams, k: int, n_i: int) -> float:
    """RS rate of a committed round with ``2^k`` rows and codeword length ``n_i`` (padding included)."""
    msg = (1 << k) + (zp.t_pad if zp.enabled else 0)
    if msg > n_i:
        raise ValueError(f"padded message {msg} does not fit the codeword {n_i}")
    return msg / n_i


def queries_for(zp: ZkParams, k: int, n_i: int, target_log2: float | None = None, levels: int = 4,
                total_log2: float = -128.0) -> int:
    """``|S_i|`` for the per-level proximity term of the paper's (4) in the unique-decoding regime with the PADDED rate
    ``(2^k + t_pad) / n_i`` (red team F1): ``((1 + rate) / 2)^{|S|} <= 2^{target_log2}``.  The default target is the union
    rule ``total_log2 - log2(levels)`` (F7: -130 per level is not a union rule at L = 5), e.g. L = 4 -> -130, L = 5 -> -130.32."""
    if target_log2 is None:
        target_log2 = total_log2 - math.log2(levels)
    rho = rate(zp, k, n_i)
    return math.ceil(target_log2 / math.log2((1 + rho) / 2))


# ---------------------------------------------------------------------------------------------------------------------
# 3. Libra-style mask, one per sumcheck over witness tables (zero-check, rows, shift)
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class BlockClaim:
    """An inner-product claim ``<W, f^> = value`` with ``W`` supported on ``cells`` (flat indices, all inside one
    round-1 column) with extension ``weights`` (m, 6).  High part (the column bits): one-hot -> tensorizable; low
    part: the sparse vector -- the "block term" pcs.py batches like an eq / geom term."""
    cells: np.ndarray
    weights: np.ndarray
    value: np.ndarray

    def low_vector(self, rows: int, col_start: int) -> np.ndarray:
        """The dense (rows, 6) vector over the column's rows (``col_start`` = the column's first flat index)."""
        out = ext_zero((rows,))
        out[self.cells - col_start] = self.weights
        return out


class SumcheckMask:
    """``g(x) = a_0 + sum_{i < n} g_i(x_i)``, ``g_i(X) = sum_{e=1..d} a_{i,e} X^e``, coefficients in ``F_{p^6}``
    (``coeffs`` (dn + 1, 6): ``a_0`` then ``a_{i,e}`` row-major in (i, e)).  Round ``j`` eliminates the variable the
    prover calls ``x_j``; ``r_prefix`` are the challenges of rounds ``< j`` in that order."""

    def __init__(self, zp: ZkParams, idx: int, coeffs: np.ndarray):
        self.zp, self.idx = zp, idx
        self.name, self.n, self.d = zp.sumchecks[idx]
        coeffs = np.asarray(coeffs, dtype=np.int64) % P
        if coeffs.shape != (zp.g_coeffs(idx), D):
            raise ValueError(f"expected {zp.g_coeffs(idx)} extension coefficients, got {coeffs.shape}")
        self.a0 = coeffs[0]
        self.a = coeffs[1:].reshape(self.n, self.d, D)

    # -- the polynomial -------------------------------------------------------------------------------------------
    def g_i(self, i: int, x) -> np.ndarray:
        """``g_i(x)`` for an extension point ``x`` (6,) or a base scalar."""
        if np.ndim(x) == 0:
            return self.g_base(int(x) % P)[i]
        return self.g_all(np.asarray(x, dtype=np.int64)[None, :].repeat(self.n, 0))[i]

    def g_all(self, r) -> np.ndarray:
        """(n, 6): ``g_i(r_i)`` for extension points ``r`` (n, 6) -- ``d`` batched extension products."""
        r = np.asarray(r, dtype=np.int64).reshape(self.n, D) % P
        acc, rp = ext_zero((self.n,)), r.copy()
        for e in range(self.d):
            acc = (acc + ext_mul(self.a[:, e], rp)) % P
            if e + 1 < self.d:
                rp = ext_mul(rp, r)
        return acc

    def g_base(self, X: int) -> np.ndarray:
        """(n, 6): ``g_i(X)`` at a base point -- base-scalar products only."""
        acc, xp = ext_zero((self.n,)), X % P
        for e in range(self.d):
            acc = (acc + self.a[:, e] * xp) % P
            xp = (xp * X) % P
        return acc

    def g_i_ones(self) -> np.ndarray:
        """(n, 6): ``g_i(1) = sum_e a_{i,e}``."""
        return ext_sum(self.a, axis=1)

    def total_sum(self) -> np.ndarray:
        """``G = sum_{x in {0,1}^n} g(x) = 2^n a_0 + 2^{n-1} sum_i g_i(1)`` (6,)."""
        two_n = pow(2, self.n, P)
        two_n1 = pow(2, self.n - 1, P)
        return (ext_mul_base(self.a0, two_n) + ext_mul_base(ext_sum(self.g_i_ones(), axis=0), two_n1)) % P

    def round_values(self, j: int, r_prefix, xs=None) -> np.ndarray:
        """(len(xs), 6): ``sum_{x_{>j}} g(r_0..r_{j-1}, X, x_{>j})`` for ``X in xs`` (base points) -- what the prover
        adds (times ``rho``) to its round-``j`` message values."""
        xs = tuple(range(self.d + 1)) if xs is None else xs
        r_prefix = np.asarray(r_prefix, dtype=np.int64).reshape(-1, D) % P
        if r_prefix.shape[0] != j:
            raise ValueError("r_prefix must hold the challenges of the rounds before j")
        n = self.n
        const = self.a0.copy()
        if j:
            pre = np.concatenate([r_prefix, np.zeros((n - j, D), dtype=np.int64)], axis=0)
            const = (const + ext_sum(self.g_all(pre)[:j], axis=0)) % P
        ones = self.g_i_ones()
        tail = ext_sum(ones[j + 1:], axis=0) if j + 1 < n else ext_zero()
        f1 = pow(2, n - 1 - j, P)
        f2 = pow(2, n - 2 - j, P) if n - 2 - j >= 0 else 0
        out = ext_zero((len(xs),))
        for t, X in enumerate(xs):
            v = (const + self.g_base(int(X) % P)[j]) % P
            out[t] = (ext_mul_base(v, f1) + ext_mul_base(tail, f2)) % P
        return out

    def final_value(self, r) -> np.ndarray:
        """``g(r) = a_0 + sum_i g_i(r_i)`` (6,)."""
        return (self.a0 + ext_sum(self.g_all(r), axis=0)) % P

    # -- the commitment side ------------------------------------------------------------------------------------------
    def cells(self) -> np.ndarray:
        """The base cells (6 (dn + 1),) in G-block order: coordinate ``d`` of coefficient ``t`` at ``6 t + d``."""
        return np.concatenate([self.a0[None, :], self.a.reshape(-1, D)], axis=0).reshape(-1)

    def claim(self, r) -> BlockClaim:
        """``<W_g, f^> = g(r)``: weight ``x^d`` at ``a_0``'s cell ``d``, ``x^d r_i^e`` at ``a_{i,e}``'s cell ``d``
        (``x^d`` = the extension basis element with a 1 in coordinate ``d``)."""
        mc = mask_cells(self.zp)
        r = np.asarray(r, dtype=np.int64).reshape(self.n, D) % P
        w = ext_zero((self.zp.g_coeffs(self.idx), D))  # (coefficient t, coordinate d, 6)
        basis = np.eye(D, dtype=np.int64)               # x^d
        w[0] = basis
        t = 1
        for i in range(self.n):
            rp = r[i].copy()
            for _ in range(self.d):
                w[t] = ext_mul(basis, rp[None, :])
                rp = ext_mul(rp, r[i])
                t += 1
        cells = mc.g_blocks[self.idx].copy() if mc.g_blocks else np.arange(D * self.zp.g_coeffs(self.idx), dtype=np.int64)
        return BlockClaim(cells, w.reshape(-1, D), self.final_value(r))


class EqSumcheckMask:
    """Libra mask for ligerito-sumcheck's zero-check wire form (red team F3): the summand is ``eq(tau, x) (q(x) + rho g(x))``
    -- the mask is multiplied by the SAME public ``eq`` factor as the witness part, so every round keeps Gruen's
    ``(q(0), q(1), q(inf))`` form with per-variable degree ``d = 2`` and the verifier's checks are unchanged except that
    the zero-check's initial claim is ``rho . G`` with ``G = sum_x eq(tau, x) g(x)`` (sent by the prover).
    ``g(x) = a_0 + sum_i sum_{e<=d} a_{i,e} x_i^e + [cross] sum_{a,b<=d} c_{ab} x_0^a x_1^b`` -- the cross block blinds the
    ``X1^a X2^b`` coefficients of the bivariate OPENING round (variables 0 and 1 in round order), which ``g_0 + g_1`` alone
    leaves in the clear; with it the 9 coefficients of the 3 x 3 opening message are spanned (1 + 2 + 2 + 4 = 9).
    ``coeffs`` (1 + dn [+ d^2], 6): ``a_0``, ``a_{i,e}`` row-major, then ``c_{ab}`` row-major.  ``tau`` (n, 6) in round order
    (``tau[t]`` multiplies the variable eliminated at round ``t``)."""

    def __init__(self, n: int, d: int, coeffs: np.ndarray, tau, cross: bool = True):
        self.n, self.d, self.cross = n, d, cross
        coeffs = np.asarray(coeffs, dtype=np.int64) % P
        if coeffs.shape != (self.n_coeffs(n, d, cross), D):
            raise ValueError(f"expected {self.n_coeffs(n, d, cross)} coefficients, got {coeffs.shape}")
        self.a0 = coeffs[0]
        self.a = coeffs[1:1 + d * n].reshape(n, d, D)
        self.c = coeffs[1 + d * n:].reshape(d, d, D) if cross else None
        self.tau = np.asarray(tau, dtype=np.int64).reshape(n, D) % P
        self.g_ones = ext_sum(self.a, axis=1)                            # (n, 6): g_i(1)
        self.bar = ext_mul(self.tau, self.g_ones)                        # (n, 6): sum_{x_i} eq(tau_i, x_i) g_i(x_i) = tau_i g_i(1)

    @staticmethod
    def n_coeffs(n: int, d: int, cross: bool) -> int:
        return 1 + d * n + (d * d if cross else 0)

    # -- pieces
    def _g_at(self, i: int, x) -> np.ndarray:
        """g_i(x), x an extension (6,)."""
        x = np.asarray(x, dtype=np.int64) % P
        acc, xp = ext_zero(), x.copy()
        for e in range(self.d):
            acc = (acc + ext_mul(self.a[i, e], xp)) % P
            xp = ext_mul(xp, x)
        return acc

    def _pows(self, x) -> np.ndarray:
        """(d, 6): x^1 .. x^d."""
        x = np.asarray(x, dtype=np.int64) % P
        out, xp = ext_zero((self.d,)), x.copy()
        for e in range(self.d):
            out[e] = xp
            xp = ext_mul(xp, x)
        return out

    def _cross(self, x0, x1) -> np.ndarray:
        """sum_{a,b} c_ab x0^a x1^b (6,)."""
        if not self.cross:
            return ext_zero()
        p0, p1 = self._pows(x0), self._pows(x1)
        return ext_sum(ext_sum(ext_mul(ext_mul(self.c, p0[:, None, :]), p1[None, :, :]), axis=1), axis=0)

    # -- what the prover sends / adds
    def zk_sum(self) -> np.ndarray:
        """``G = sum_x eq(tau, x) g(x) = a_0 + sum_i tau_i g_i(1) + tau_0 tau_1 sum_ab c_ab`` (6,)."""
        G = (self.a0 + ext_sum(self.bar, axis=0)) % P
        if self.cross:
            G = (G + ext_mul(ext_mul(self.tau[0], self.tau[1]), ext_sum(ext_sum(self.c, axis=0), axis=0))) % P
        return G

    def _tail(self, t: int) -> np.ndarray:
        return ext_sum(self.bar[t + 1:], axis=0) if t + 1 < self.n else ext_zero()

    def message(self, t: int, r_prefix, xs=None) -> np.ndarray:
        """(len(xs), 6): the mask's part of round ``t``'s message ``q_t(X)`` (the prover adds ``rho`` times it):
        ``sum_{x>t} prod_{i>t} eq(tau_i, x_i) g(r_<t, X, x_>t)``.  Univariate rounds only (``t >= 2`` when the opening is
        bivariate; ``t >= 0`` otherwise)."""
        xs = tuple(range(self.d + 1)) if xs is None else xs
        r_prefix = np.asarray(r_prefix, dtype=np.int64).reshape(-1, D) % P
        if r_prefix.shape[0] != t:
            raise ValueError("r_prefix must hold the challenges of the rounds before t")
        const = self.a0.copy()
        for i in range(t):
            const = (const + self._g_at(i, r_prefix[i])) % P
        const = (const + self._tail(t)) % P
        if self.cross:
            if t >= 2:
                const = (const + self._cross(r_prefix[0], r_prefix[1])) % P
            elif t == 1:
                pass                                                     # handled per X below
            else:
                pass
        out = ext_zero((len(xs),))
        for k, X in enumerate(xs):
            Xe = ext_from_base(X) if np.ndim(X) == 0 else np.asarray(X, dtype=np.int64)
            v = (const + self._g_at(t, Xe)) % P
            if self.cross and t == 0:                                    # sum_{x1} eq(tau_1, x1) x1^b = tau_1
                v = (v + ext_mul(self._cross(Xe, ext_one()), self.tau[1])) % P
            elif self.cross and t == 1:
                v = (v + self._cross(r_prefix[0], Xe)) % P
            out[k] = v
        return out

    def message_pair(self, xs=None) -> np.ndarray:
        """(len(xs), len(xs), 6): the mask's part of the bivariate OPENING message on the grid ``(X1, X2) in xs x xs``
        (variables 0 and 1): ``a_0 + g_0(X1) + g_1(X2) + sum c_ab X1^a X2^b + sum_{i>1} tau_i g_i(1)``."""
        xs = tuple(range(self.d + 1)) if xs is None else xs
        const = (self.a0 + self._tail(1)) % P
        out = ext_zero((len(xs), len(xs)))
        for i, X1 in enumerate(xs):
            X1e = ext_from_base(X1) if np.ndim(X1) == 0 else np.asarray(X1, dtype=np.int64)
            g0 = self._g_at(0, X1e)
            for j, X2 in enumerate(xs):
                X2e = ext_from_base(X2) if np.ndim(X2) == 0 else np.asarray(X2, dtype=np.int64)
                out[i, j] = (const + g0 + self._g_at(1, X2e) + self._cross(X1e, X2e)) % P
        return out

    def final_value(self, r) -> np.ndarray:
        """``g(r)`` (6,) -- the block claim's value on the committed coefficients."""
        r = np.asarray(r, dtype=np.int64).reshape(self.n, D) % P
        acc = self.a0.copy()
        for i in range(self.n):
            acc = (acc + self._g_at(i, r[i])) % P
        return (acc + self._cross(r[0], r[1])) % P

    def claim_weights(self, r) -> np.ndarray:
        """(n_coeffs, 6, 6): weight of coefficient ``t``'s base cell ``d`` in ``<W, f> = g(r)`` (``x^d`` times the monomial)."""
        r = np.asarray(r, dtype=np.int64).reshape(self.n, D) % P
        mons = [ext_one()]
        for i in range(self.n):
            mons.extend(self._pows(r[i]))
        if self.cross:
            p0, p1 = self._pows(r[0]), self._pows(r[1])
            mons.extend(ext_mul(p0[a], p1[b]) for a in range(self.d) for b in range(self.d))
        mons = np.stack(mons)                                            # (n_coeffs, 6)
        basis = np.eye(D, dtype=np.int64)
        return ext_mul(basis[None, :, :], mons[:, None, :])             # (t, d, 6)

    def eval_brute(self, x_bits) -> np.ndarray:
        """g at a boolean point (test helper)."""
        x = [ext_from_base(b) for b in x_bits]
        acc = self.a0.copy()
        for i in range(self.n):
            acc = (acc + self._g_at(i, x[i])) % P
        return (acc + self._cross(x[0], x[1])) % P


# ---------------------------------------------------------------------------------------------------------------------
# 4. Committed coins (PROTOCOL.md 8c over K slots)
# ---------------------------------------------------------------------------------------------------------------------

class CoinOpeningError(RuntimeError):
    def __init__(self, slot: int, label: str, sent: list[str]):
        super().__init__(f"coin slot {slot} ({label}): opening does not match the step-0 commitment (verifier "
                         f"equivocation); prover aborted after sending {sent}")
        self.slot, self.label, self.sent = slot, label, list(sent)


def coin_commit(k: int, r: bytes, s: bytes) -> bytes:
    if len(r) != COIN_BYTES or len(s) != COIN_BYTES:
        raise ValueError("coin randomness and salt are 32 bytes each")
    return hashlib.blake2b(COIN_TAG + k.to_bytes(4, "little") + r + s, digest_size=32).digest()


def verify_opening(c_k: bytes, k: int, r: bytes, s: bytes) -> bool:
    return len(r) == COIN_BYTES and len(s) == COIN_BYTES and coin_commit(k, r, s) == c_k


def commitments_digest(cs: list[bytes]) -> bytes:
    return hashlib.blake2b(b"ligerito-zk/coin-set|v1" + b"".join(cs), digest_size=32).digest()


def coin_schedule(zp: ZkParams, kprimes: list[int]) -> list[str]:
    """The fixed slot order (a function of the parameters alone).  Per sumcheck ``s`` (in ``zp.sumchecks`` order):
    ``s/setup`` (its batching coins: tau / gamma, drawn after the commitment and the previous sumcheck's claims),
    ``s/rho`` (after ``G_s``; ZK only), the ``n_s`` challenges ``s/r{j}``; then per committed round ``i``: ``pcs/S{i-1}``
    (i >= 2), the batching ``pcs/alpha{i}``, the ``k'_i`` partial-sumcheck challenges; finally ``pcs/S{ell}``."""
    sched = []
    for name, n, _ in zp.sumchecks:
        sched.append(f"{name}/setup")
        if zp.enabled:
            sched.append(f"{name}/rho")
        sched += [f"{name}/r{j}" for j in range(n)]
    for i, kp in enumerate(kprimes, start=1):
        if i >= 2:
            sched.append(f"pcs/S{i - 1}")
        sched.append(f"pcs/alpha{i}")
        sched += [f"pcs/r{i}.{j}" for j in range(kp)]
    sched.append(f"pcs/S{len(kprimes)}")
    return sched


def challenge_seed(stmt: bytes, cset: bytes, k: int, r_k: bytes) -> bytes:
    """``Blake2b(tag || stmt || C || k || r_k)`` -- the verifier's coin alone; ``stmt`` and ``C`` are domain separation."""
    return hashlib.blake2b(COIN_SEED_TAG + stmt + cset + k.to_bytes(4, "little") + r_k, digest_size=32).digest()


def challenge_from(stmt: bytes, cset: bytes, k: int, r_k: bytes, count: int, *, indices_n: int | None = None):
    """``count`` extension elements (count, 6) from the slot's seed, or ``count`` distinct indices in ``[0, indices_n)``
    (repeats rejected inside the stream: deterministic, no prover-controlled restart)."""
    seed = challenge_seed(stmt, cset, k, r_k)
    if indices_n is None:
        raw = hashlib.shake_256(seed + b"|ext").digest(8 * count * D)
        u = np.frombuffer(raw, dtype="<u8").astype(np.int64, copy=False) & ((1 << 63) - 1)
        return (u % P).reshape(count, D)                              # bias 2^-32 per coordinate (proto's convention)
    out, seen, ctr = [], set(), 0
    while len(out) < count:
        raw = hashlib.shake_256(seed + b"|idx" + ctr.to_bytes(4, "little")).digest(8 * 2 * count)
        for u in np.frombuffer(raw, dtype="<u8"):
            i = int(u % indices_n)
            if i not in seen:
                seen.add(i)
                out.append(i)
                if len(out) == count:
                    break
        ctr += 1
    return out


class CommittedCoins:
    """The honest interactive verifier's coins: ``K`` slots sampled (from ``os.urandom`` or deterministically from
    ``seed`` for the diagnostic mode) and committed at step 0; ``open(k)`` returns what was committed, whatever the
    prover sent.  The prover side: ``ProverCoins`` (checks every opening, aborts on mismatch)."""

    def __init__(self, schedule: list[str], seed: bytes | None = None):
        self.schedule = list(schedule)
        K = len(schedule)
        if seed is None:
            self.r = [os.urandom(COIN_BYTES) for _ in range(K)]
            self.s = [os.urandom(COIN_BYTES) for _ in range(K)]
        else:
            self.r = [hashlib.blake2b(b"r|" + seed + k.to_bytes(4, "little"), digest_size=32).digest() for k in range(K)]
            self.s = [hashlib.blake2b(b"s|" + seed + k.to_bytes(4, "little"), digest_size=32).digest() for k in range(K)]
        self.cs = [coin_commit(k, self.r[k], self.s[k]) for k in range(K)]
        self.cset = commitments_digest(self.cs)

    @property
    def K(self) -> int:
        return len(self.schedule)

    def commitments(self) -> list[bytes]:
        return list(self.cs)

    def open(self, k: int, transcript: list[tuple[str, bytes]]) -> tuple[bytes, bytes] | None:
        """The opening of slot ``k`` after the prover's messages so far (``None`` = the verifier aborts)."""
        return self.r[k], self.s[k]

    def challenge(self, stmt: bytes, k: int, count: int, indices_n: int | None = None):
        return challenge_from(stmt, self.cset, k, self.r[k], count, indices_n=indices_n)


class ProverCoins:
    """What the prover holds: the ``K`` commitments; ``next(...)`` asks the verifier for the next slot, checks the
    opening against the commitment and derives the challenge; a mismatch is a public abort."""

    def __init__(self, stmt: bytes, schedule: list[str], commitments: list[bytes], verifier: CommittedCoins):
        if len(commitments) != len(schedule):
            raise ValueError("one commitment per slot")
        self.stmt, self.schedule, self.cs, self.v = stmt, list(schedule), list(commitments), verifier
        self.cset = commitments_digest(self.cs)
        self.k = 0
        self.sent: list[tuple[str, bytes]] = []
        self.opened: list[tuple[bytes, bytes]] = []

    def send(self, label: str, data: bytes) -> None:
        self.sent.append((label, data))

    def next(self, label: str, count: int, indices_n: int | None = None):
        k = self.k
        if k >= len(self.schedule) or self.schedule[k] != label:
            raise RuntimeError(f"coin schedule mismatch at slot {k}: expected {self.schedule[k] if k < len(self.schedule) else None}, got {label}")
        op = self.v.open(k, list(self.sent))
        if op is None or not verify_opening(self.cs[k], k, op[0], op[1]):
            raise CoinOpeningError(k, label, [m for m, _ in self.sent])
        self.opened.append(op)
        self.k += 1
        return challenge_from(self.stmt, self.cset, k, op[0], count, indices_n=indices_n)


class EquivocatingCoins(CommittedCoins):
    """Commits honestly, then opens slot ``slot`` to a value that depends on the prover's messages (the adaptive coin
    choice the commitment forbids): the prover must abort there, having sent only blinded messages."""

    def __init__(self, schedule, slot: int, seed=None):
        super().__init__(schedule, seed)
        self.slot = slot

    def open(self, k, transcript):
        if k != self.slot:
            return self.r[k], self.s[k]
        h = hashlib.blake2b(b"adaptive|" + b"|".join(d for _, d in transcript), digest_size=32).digest()
        return h, self.s[k]


class AdversarialCoins(CommittedCoins):
    """Commits to coins the verifier CHOSE (no entropy, or ground in advance against the statement): allowed -- the
    challenges are then a fixed function of these coins and the masking is coin-oblivious for any in-domain coin."""

    def __init__(self, schedule, fixed: dict[int, bytes] | None = None, zero: bool = False):
        super().__init__(schedule, seed=b"adversarial")
        if zero:
            self.r = [b"\0" * COIN_BYTES for _ in self.r]
        for k, rk in (fixed or {}).items():
            self.r[k] = rk
        self.cs = [coin_commit(k, self.r[k], self.s[k]) for k in range(self.K)]
        self.cset = commitments_digest(self.cs)


class AbortingCoins(CommittedCoins):
    """Aborts (refuses to open) at slot ``k`` when ``predicate(k, transcript)`` -- the selectively-aborting V* of the
    GK argument; a ``PublicCheckVerifier`` is ``predicate`` = "a public consistency check failed" and must abort in
    neither world."""

    def __init__(self, schedule, predicate, seed=None):
        super().__init__(schedule, seed)
        self.predicate = predicate
        self.aborted_at: int | None = None

    def open(self, k, transcript):
        if self.predicate(k, transcript):
            self.aborted_at = k
            return None
        return self.r[k], self.s[k]


# ---------------------------------------------------------------------------------------------------------------------
# Fingerprint text (relchain.zk_mode / omitted / zk_statement)
# ---------------------------------------------------------------------------------------------------------------------

ZK_MODE = ("Ligerito ZK: mask rows in the committed polynomial (round-1 columns [0, {lo}) and the top {hi} uniform -> every "
           "fold entry, expected symbol and final-vector entry is a functional of >= 6 uniform base cells with F_p-spanning "
           "extension weights, and both halves of every round-1 column split hold a mask column), t_pad = {t_pad} uniform RS "
           "padding coefficients per committed column with the padding combination ybar_i in the clear (opened rows uniform "
           "for |S_i| <= t_pad), one Libra mask polynomial g = a_0 + sum_i g_i(x_i) per sumcheck over witness tables ({scs}; "
           "{g} F_p^6 coefficients committed in the G-blocks) each with a post-G coin; malicious-verifier: every verifier "
           "message is a step-0 committed slot (K = {K}), challenges are functions of the coins alone, prover aborts publicly "
           "on a mismatched opening; simulator by rewinding V* after step 0 (CRH only)")

ZK_MODE_ROWS = ("Ligerito ZK (relation layout, f = z with {a} column bits interleaved below the {n_i} row bits): {u} uniform "
                "mask rows in the free region covering every residue mod 2^{b} (so every fold entry, round contraction, final-"
                "vector entry and fully-extension eval claim is a functional of >= 6 uniform base cells with F_p-spanning "
                "tensor weights), 3 committed mask rows under the trivial constraint m1 . m2 = m3 (masks the zero-check's "
                "Az, Bz, Cz claims), one committed Libra row ({g} F_p^6 coefficients: eq-weighted masks g with a 2 x 2 cross "
                "block for the bivariate opening round, one per sumcheck over witness tables: {scs}), t_pad = {t_pad} uniform RS "
                "padding coefficients per committed column with ybar_i in the clear (opened rows uniform for |S_i| <= t_pad); "
                "malicious-verifier: every verifier message is a step-0 committed slot (K = {K}), challenges are functions of "
                "the coins alone, prover aborts publicly on a mismatched opening; simulator by rewinding V* after step 0 (CRH only)")


def zk_mode_rows(lay: "RowMaskLayout", t_pad: int = T_PAD, K: int | None = None) -> str:
    """The ``relchain.zk_mode`` text for the real layout."""
    return ZK_MODE_ROWS.format(a=lay.a, n_i=lay.n_i, u=len(lay.uniform_rows), b=lay.b, g=lay.g_cells // D,
                               scs=", ".join(f"{sc[0]}({sc[1]} vars, d={sc[2]}{', cross' if len(sc) > 3 and sc[3] else ''})"
                                             for sc in lay.sumchecks),
                               t_pad=t_pad, K=K if K is not None else "per coin_schedule")


OMITTED_ZK = ("unopened Merkle leaves hidden by Blake3 of a (marginally) uniform row -- random-oracle-flavoured, "
              "salted leaves not implemented; GK rewinding capped in the demonstrated simulator")


def zk_statement(zp: ZkParams, mode: str = "interactive", private_operands: bool = True) -> str:
    if not zp.enabled:
        return "none (no masking)"
    who = ("malicious-verifier ZK (simulator by rewinding V* after step 0, CRH only)" if mode == "interactive"
           else "HVZK; NIZK only in the programmable random-oracle model")
    priv = ("" if private_operands else "; operands and output words are PUBLIC in this statement, so privacy is vacuous today")
    return who + "; the verifier learns no functional of the witness: opened rows, sumcheck messages, expected symbols, " \
                 "final vector and evaluation claims are uniform on their public consistency sets (up to the O(1/p) rank " \
                 "event of the mask weights)" + priv


def omitted(zp: ZkParams) -> str:
    return OMITTED_ZK if zp.enabled else "ZK masking"


def zk_mode(zp: ZkParams, K: int) -> str:
    if not zp.enabled:
        return "none"
    scs = ", ".join(f"{nm}: {n} vars deg {d}" for nm, n, d in zp.sumchecks)
    return ZK_MODE.format(lo=zp.mask_cols_low, hi=zp.mask_cols_high, t_pad=zp.t_pad, scs=scs, g=zp.g_coeffs_total, K=K)
