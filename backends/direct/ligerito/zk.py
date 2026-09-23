"""Zero knowledge for the Ligerito backend (lane ligerito-zk): what the prover adds so that the zero-check sumcheck and
the recursive Ligerito openings reveal nothing about the witness, against a malicious verifier, keeping PROTOCOL.md 8c.

Design (note ``~/.research/notes/lanes/ligerito-zk/20260923T1700Z-report-ligerito-zk.md`` sections 1-2); the four pieces
and the leak each one covers, with ``f`` the witness polynomial (N = 2^n base cells), round ``i`` of the recursion
committing the column-major matrix ``M~_i`` (``2^{k_i}`` rows x ``2^{k'_i}`` columns, column bits = the high variables):

1. **Mask cells inside the committed polynomial** (``mask_cells`` / ``sample_masks`` / ``apply_masks``): the top 8
   round-1 columns ``[N - 8 . 2^{k_1}, N)`` (the mask block) are uniform, so every entry of ``y_1 = M~_1 rbar_1`` -- and
   every fold of it, the expected symbols ``<gen_s, y_i>``, the final vector -- is a functional of >= 6 uniform base
   cells whose extension weights span ``F_{p^6}`` (8 tensor weights of 3 random coordinates), hence exactly uniform;
   8 uniform cells (rows ``q..q+7``) in every other round-1 column (the A-cells) make round 1's column contraction
   ``u = M~_1^T rowpart`` uniform (the round-1 partial-sumcheck messages) and the evaluation claims ``f^(z)`` uniform
   PROVIDED the claim points reach them: the zero-check must sum over the padding rows with weight 0 (``tau`` with the
   padding-selector coordinate fixed to 0) so ``z`` has an extension coordinate there -- a claim on the constrained
   sub-cube alone is a bare functional of the witness.  The Libra coefficients (3) live in the last mask column.  The
   layout must leave these cells unconstrained AND unreferenced (``check_layout``).
2. **Per-column RS padding** (``pad_columns`` / ``padding_combination`` / ``row_check_rhs``): every column's message is
   ``(M~_i[:, c] || mu_c)`` with ``mu_c`` uniform of length ``t_pad > |S_i|`` (Ligero 8b in coefficient form), so the
   opened rows are uniform; the padding's ``rbar``-combination ``ybar_i = mu^T rbar_i`` travels in the clear with the
   next root and the verifier's row check is ``<U_i[s,:], rbar_i> = <gen_s, y_i> + <gen'_s, ybar_i>`` (the claimed
   combination is the message ``(y_i || ybar_i)``; ``ybar_i`` is fixed before ``S_i`` is drawn, so the paper's
   consistency guarantee applies with rate ``(2^{k_i} + t_pad) / n_i`` -- ``queries_for`` sizes ``|S_i|`` for it).
3. **Libra-style zero-check mask** (``SumcheckMask``; Xie et al. CRYPTO'19 ePrint 2019/317 section 4.1 Construction 1 /
   Theorem 3, the cheap instance of Chiesa-Forbes-Spooner ePrint 2017/305): ``g(x) = a_0 + sum_i sum_{e=1..d} a_{i,e}
   x_i^e`` with ``dn + 1`` uniform extension coefficients committed in the G-block, ``G = sum_x g(x)`` sent, coin
   ``rho``, sumcheck on ``eq(tau, x) C(x) + rho g(x)``; the verifier gets ``g(r)`` as an inner-product claim on the
   committed polynomial (``SumcheckMask.claim``).
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

__all__ = ["A_CELLS_LOG2", "BETA", "COIN_BYTES", "D", "MASK_COLS_LOG2", "P", "T_PAD", "AbortingCoins", "AdversarialCoins", "BlockClaim", "CoinOpeningError",
           "CommittedCoins", "EquivocatingCoins", "MaskCells", "Masks", "SumcheckMask", "ZkParams", "apply_masks",
           "challenge_from", "check_layout", "coin_commit", "coin_schedule", "mask_cells", "omitted", "pad_columns",
           "padding_combination", "queries_for", "rate", "row_check_rhs", "sample_masks", "uniform_symbols",
           "verify_opening", "zk_statement"]


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
    shape = np.broadcast_shapes(a.shape[:-1], b.shape[:-1])
    out = np.zeros(shape + (D,), dtype=np.int64)
    for i in range(D):
        ai = a[..., i]
        for j in range(D):
            prod = (ai * b[..., j]) % P
            k = i + j
            if k >= D:
                out[..., k - D] += prod * BETA
            else:
                out[..., k] += prod
    return out % P


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

MASK_COLS_LOG2 = 3              # the mask block = the top 2^3 = 8 round-1 columns (>= D base cells per extension functional)
A_CELLS_LOG2 = 3                # 8 A-cells per other round-1 column (rows q .. q+7, q 8-aligned)


@dataclass(frozen=True)
class ZkParams:
    """``n``: log2 cells of the committed polynomial; ``k1``/``kp1``: round-1 rows / columns (log2); ``degree``: the
    zero-check's per-variable degree (eq x quadratic = 3); ``n_sumcheck``: variables of the zero-check (defaults to
    ``n``); ``q``: the (8-aligned) padding row offset the layout reserves inside every round-1 column for the A-cells.

    Why 8 and not 1: masks are BASE-field cells while every leaked functional (a fold ``y_i[row]``, an evaluation
    ``f^(z)``, a round-1 contraction ``u[c]``) is an EXTENSION element ``sum_x w_x f^[x]`` with ``w_x in F_{p^6}``;
    one uniform base cell blinds only the F_p-line ``F_p w_x`` of it.  ``m >= D = 6`` uniform base cells whose weights
    ``w_{x_1} .. w_{x_m}`` span ``F_{p^6}`` over ``F_p`` make the functional exactly uniform; the 8 cells of a 3-bit
    sub-cube have weights ``{prod_j pair_j(b_j)}_{b in {0,1}^3}`` (tensor factors of 3 random extension coordinates),
    which span with probability ``1 - O(1/p)`` -- the rank event of the simulator's distance."""
    n: int
    k1: int
    kp1: int
    t_pad: int = T_PAD
    degree: int = 3
    n_sumcheck: int | None = None
    q: int = 0
    enabled: bool = True
    mask_cols_log2: int = MASK_COLS_LOG2
    a_cells_log2: int = A_CELLS_LOG2

    def __post_init__(self):
        if self.k1 + self.kp1 != self.n:
            raise ValueError(f"round-1 shape 2^{self.k1} x 2^{self.kp1} does not cover 2^{self.n} cells")
        if not 0 <= self.q < (1 << self.k1) or self.q % self.a_cells_per_col:
            raise ValueError(f"q must be a {self.a_cells_per_col}-aligned row offset of the round-1 column")
        if self.enabled and (1 << self.k1) < self.g_block_cells + 2:
            raise ValueError("the mask columns are too short for the G-block")
        if self.enabled and self.kp1 <= self.mask_cols_log2:
            raise ValueError("round 1 needs more than 2^mask_cols_log2 columns")
        if self.enabled and self.q + self.a_cells_per_col > (1 << self.k1):
            raise ValueError("the A-cells overflow the column")

    @property
    def n_vars(self) -> int:
        return self.n if self.n_sumcheck is None else self.n_sumcheck

    @property
    def g_coeffs(self) -> int:
        """Libra coefficients: a_0 + d per variable."""
        return self.degree * self.n_vars + 1

    @property
    def g_block_cells(self) -> int:
        return D * self.g_coeffs

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
    def mask_cols(self) -> int:
        return 1 << self.mask_cols_log2

    @property
    def a_cells_per_col(self) -> int:
        return 1 << self.a_cells_log2

    @property
    def other_cols(self) -> int:
        return self.columns1 - self.mask_cols

    @property
    def mask_cell_count(self) -> int:
        return (self.mask_cols * self.rows1 + self.a_cells_per_col * self.other_cols) if self.enabled else 0


@dataclass(frozen=True)
class MaskCells:
    block: range                  # the mask block: the top 2^mask_cols_log2 round-1 columns (contiguous cells)
    a_cells: np.ndarray           # (other_cols * a_cells_per_col,): rows q..q+7 of every other round-1 column
    g_block: np.ndarray           # the Libra coefficients' cells (the first cells of the LAST mask column)

    def all(self) -> np.ndarray:
        return np.concatenate([np.arange(self.block.start, self.block.stop, dtype=np.int64), self.a_cells])

    @property
    def count(self) -> int:
        return len(self.block) + len(self.a_cells)


def mask_cells(zp: ZkParams) -> MaskCells:
    """The reserved cells.  The mask block is the top ``2^3`` round-1 columns ``[N - 8 . 2^{k1}, N)``; the G-block sits at
    the START of its last column (rows ``0 .. 6(dn+1)`` of column ``C_1 - 1``); the A-cells are rows ``q .. q + 7`` of
    every other round-1 column."""
    if not zp.enabled:
        return MaskCells(range(0, 0), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    top = zp.N - zp.mask_cols * zp.rows1
    blk = range(top, zp.N)
    cols = np.arange(zp.other_cols, dtype=np.int64) * zp.rows1 + zp.q
    a = (cols[:, None] + np.arange(zp.a_cells_per_col, dtype=np.int64)[None, :]).reshape(-1)
    g0 = zp.N - zp.rows1
    g = np.arange(g0, g0 + zp.g_block_cells, dtype=np.int64)
    return MaskCells(blk, a, g)


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
                         "the ZK masking needs the top 8 round-1 columns and 8 padding cells per round-1 column free")


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
    block: np.ndarray             # (8 . 2^{k1},) the mask block's cells (the G-block = the first cells of the last column)
    a_cells: np.ndarray           # (other_cols * 8,) in ``MaskCells.a_cells`` order
    g_coeffs: np.ndarray          # (dn + 1, 6) the Libra coefficients (a_0 first, then a_{i,e} for i, e = 1..d)


def sample_masks(zp: ZkParams, device=None) -> Masks:
    """Fresh uniform masks: the mask block, the A-cells and the Libra coefficients (the latter are the first
    ``6 (dn + 1)`` cells of the block's last column, read as extension elements)."""
    if not zp.enabled:
        return Masks(np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, D), np.int64))
    nb = zp.mask_cols * zp.rows1
    flat = uniform_symbols(nb + zp.a_cells_per_col * zp.other_cols, device)
    blk = flat[:nb].copy()
    a = flat[nb:].copy()
    g0 = nb - zp.rows1
    g = blk[g0:g0 + zp.g_block_cells].reshape(zp.g_coeffs, D).copy()
    return Masks(blk, a, g)


def apply_masks(f_flat, masks: Masks, zp: ZkParams):
    """Write the masks into the flat witness vector (numpy or torch, length N), in place; returns it."""
    if not zp.enabled:
        return f_flat
    mc = mask_cells(zp)
    if hasattr(f_flat, "device") and not isinstance(f_flat, np.ndarray):        # torch
        import torch
        dev = f_flat.device
        f_flat[mc.block.start:mc.block.stop] = torch.as_tensor(masks.block, device=dev)
        f_flat[torch.as_tensor(mc.a_cells, device=dev)] = torch.as_tensor(masks.a_cells, device=dev)
    else:
        f_flat[mc.block.start:mc.block.stop] = masks.block
        f_flat[mc.a_cells] = masks.a_cells
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
        tp, C = mu.shape
        out = np.zeros((tp, D), dtype=np.int64)
        for c in range(C):
            out = (out + (mu[:, c, None] * rbar[c][None, :]) % P) % P
        return out
    _, tp, C = mu.shape
    mu_ext = np.transpose(mu, (1, 2, 0))                               # (tp, C, 6)
    acc = ext_zero((tp,))
    for c in range(C):
        acc = (acc + ext_mul(mu_ext[:, c], rbar[c])) % P
    return acc


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


def queries_for(zp: ZkParams, k: int, n_i: int, target_log2: float = -130.0) -> int:
    """``|S_i|`` for the per-round proximity term of the paper's (4) in the unique-decoding regime,
    ``((1 + rate) / 2)^{|S|} <= 2^{target_log2}`` (proto: rate 1/4 -> (5/8)^192 = 2^-130.2)."""
    rho = rate(zp, k, n_i)
    return math.ceil(target_log2 / math.log2((1 + rho) / 2))


# ---------------------------------------------------------------------------------------------------------------------
# 3. Libra-style mask for the zero-check sumcheck
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

    def __init__(self, zp: ZkParams, coeffs: np.ndarray):
        self.zp = zp
        self.n, self.d = zp.n_vars, zp.degree
        coeffs = np.asarray(coeffs, dtype=np.int64) % P
        if coeffs.shape != (zp.g_coeffs, D):
            raise ValueError(f"expected {zp.g_coeffs} extension coefficients, got {coeffs.shape}")
        self.a0 = coeffs[0]
        self.a = coeffs[1:].reshape(self.n, self.d, D)

    # -- the polynomial -------------------------------------------------------------------------------------------
    def g_i(self, i: int, x) -> np.ndarray:
        """``g_i(x)`` for an extension point ``x`` (6,) or a base scalar."""
        x = ext_from_base(x) if np.ndim(x) == 0 else np.asarray(x, dtype=np.int64) % P
        acc, xp = ext_zero(), x.copy()
        for e in range(self.d):
            acc = (acc + ext_mul(self.a[i, e], xp)) % P
            xp = ext_mul(xp, x)
        return acc

    def g_i_ones(self) -> np.ndarray:
        """(n, 6): ``g_i(1) = sum_e a_{i,e}``."""
        return ext_sum(self.a, axis=1)

    def total_sum(self) -> np.ndarray:
        """``G = sum_{x in {0,1}^n} g(x) = 2^n a_0 + 2^{n-1} sum_i g_i(1)`` (6,)."""
        two_n = pow(2, self.n, P)
        two_n1 = pow(2, self.n - 1, P)
        return (ext_mul_base(self.a0, two_n) + ext_mul_base(ext_sum(self.g_i_ones(), axis=0), two_n1)) % P

    def round_values(self, j: int, r_prefix, xs=(0, 1, 2, 3)) -> np.ndarray:
        """(len(xs), 6): ``sum_{x_{>j}} g(r_0..r_{j-1}, X, x_{>j})`` for ``X in xs`` (base points) -- what the prover
        adds (times ``rho``) to its round-``j`` message values."""
        r_prefix = np.asarray(r_prefix, dtype=np.int64).reshape(-1, D) % P
        if r_prefix.shape[0] != j:
            raise ValueError("r_prefix must hold the challenges of the rounds before j")
        n = self.n
        const = self.a0.copy()
        for i in range(j):
            const = (const + self.g_i(i, r_prefix[i])) % P
        ones = self.g_i_ones()
        tail = ext_sum(ones[j + 1:], axis=0) if j + 1 < n else ext_zero()
        f1 = pow(2, n - 1 - j, P)
        f2 = pow(2, n - 2 - j, P) if n - 2 - j >= 0 else 0
        out = ext_zero((len(xs),))
        for t, X in enumerate(xs):
            v = (const + self.g_i(j, int(X) % P)) % P
            out[t] = (ext_mul_base(v, f1) + ext_mul_base(tail, f2)) % P
        return out

    def final_value(self, r) -> np.ndarray:
        """``g(r) = a_0 + sum_i g_i(r_i)`` (6,)."""
        r = np.asarray(r, dtype=np.int64).reshape(self.n, D) % P
        acc = self.a0.copy()
        for i in range(self.n):
            acc = (acc + self.g_i(i, r[i])) % P
        return acc

    # -- the commitment side ------------------------------------------------------------------------------------------
    def cells(self) -> np.ndarray:
        """The base cells (6 (dn + 1),) in G-block order: coordinate ``d`` of coefficient ``t`` at ``6 t + d``."""
        return np.concatenate([self.a0[None, :], self.a.reshape(-1, D)], axis=0).reshape(-1)

    def claim(self, r) -> BlockClaim:
        """``<W_g, f^> = g(r)``: weight ``x^d`` at ``a_0``'s cell ``d``, ``x^d r_i^e`` at ``a_{i,e}``'s cell ``d``
        (``x^d`` = the extension basis element with a 1 in coordinate ``d``)."""
        mc = mask_cells(self.zp)
        r = np.asarray(r, dtype=np.int64).reshape(self.n, D) % P
        w = ext_zero((self.zp.g_coeffs, D))            # (coefficient t, coordinate d, 6)
        basis = np.eye(D, dtype=np.int64)               # x^d
        w[0] = basis
        t = 1
        for i in range(self.n):
            rp = r[i].copy()
            for _ in range(self.d):
                w[t] = ext_mul(basis, rp[None, :])
                rp = ext_mul(rp, r[i])
                t += 1
        return BlockClaim(mc.g_block.copy(), w.reshape(-1, D), self.final_value(r))


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
    """The fixed slot order (a function of the parameters alone): batching coins after the commitment, ``rho_zk``
    after ``G``, the ``n`` zero-check challenges, then per committed round ``i``: ``S_{i-1}`` (i >= 2), the batching
    ``alpha_i``, the ``k'_i`` partial-sumcheck challenges; finally ``S_{ell-1}`` after the final vector."""
    sched = ["zc/batch"]
    if zp.enabled:
        sched.append("zc/rho")
    sched += [f"zc/r{j}" for j in range(zp.n_vars)]
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

ZK_MODE = ("Ligerito ZK: mask cells in the committed polynomial (top 8 round-1 columns uniform -> every fold, expected "
           "symbol and final-vector entry is a functional of >= 6 uniform base cells with F_p-spanning extension weights; "
           "8 uniform padding cells per other round-1 column for the round-1 partial sumcheck and the evaluation claims, "
           "which must reach them: zero-check summed over padding rows with weight 0), t_pad = {t_pad} uniform RS padding "
           "coefficients per committed column with the padding combination ybar_i in the clear (opened rows uniform for "
           "|S_i| <= t_pad), Libra mask polynomial g = a_0 + sum_i g_i(x_i) (deg {degree}, {g} F_p^6 coefficients committed "
           "in the G-block) on the zero-check with a post-G coin; malicious-verifier: every verifier message is a step-0 "
           "committed slot (K = {K}), challenges are functions of the coins alone, prover aborts publicly on a mismatched "
           "opening; simulator by rewinding V* after step 0 (CRH only)")

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
    return ZK_MODE.format(t_pad=zp.t_pad, degree=zp.degree, g=zp.g_coeffs, K=K)
