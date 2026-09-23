"""Test scaffolding for ``zk.py`` (lane ligerito-zk): a toy Ligerito PCS + degree-3 zero-check in numpy, interactive with
the step-0 committed coins, wired through every hook of ``zk.py`` exactly as ``pcs.py`` / ``sumcheck.py`` are asked to
(the ``## Interface`` section of the lane note).  It exists so the masking can be exercised end to end -- honest proofs,
negatives, the simulator and the mask-free control -- before the real modules land; it is NOT a prover for real sizes
(pure numpy, Python Merkle trees) and is meant to be deleted once ``pcs.py``/``sumcheck.py`` carry the hooks.

Toy relation (mirrors the real layout's structure: padding rows inside every round-1 column, an unconstrained region
at the top): cells ``x`` of ``f`` (N = 2^n) are indexed ``[plane (2 bits) | col_rest | flag (1 bit) | row_low (k1-1 bits)]``;
for every ``z = (col_rest, flag = 0, row_low)`` the constraint ``f(0,0,z) f(0,1,z) - f(1,0,z) = 0`` holds (a . b = c);
``flag = 1`` rows are padding (the A-cells sit there), plane 3 is unreferenced (the mask column is its top round-1
column).  Zero-check: ``sum_z eq(tau', z) (a(z) b(z) - c(z)) = 0`` over ALL ``n - 2`` variables (padding rows
included) with ``tau'_flag = 0`` so the padding rows have weight ``(1 - z_flag) = 0``: degree stays 3, the honest sum is
0 whatever the padding rows hold, and the final eq-claims ``a(r), b(r), c(r)`` are at a point whose flag coordinate is
an extension element -- so each plane's A-cells enter the claim with nonzero weight (this is the requirement the design
puts on ``sumcheck.py``: the eval claims must reach the mask cells; a claim restricted to the constrained sub-cube would
be a bare functional of the witness).

Ligerito (paper section 6, proto's conventions): round ``i`` commits ``M~_i`` (``2^{k_i}`` x ``2^{k'_i}``, column-major),
coefficient-basis RS along the tall axis on ``<w_{n_i}>``, ``n_i = 4 . 2^{k_i}``; claims are tensor terms; the partial
sumcheck folds the column bits highest first; the opened rows of round ``i`` give geom claims on ``y_i = M~_i rbar_i``
(committed as ``M~_{i+1}``); the last ``y_ell`` is sent in the clear.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np

from . import zk
from .zk import D, P, ZkParams, ext_from_base, ext_inv, ext_mul, ext_mul_base, ext_one, ext_sum, ext_zero

GEN = 31                     # multiplicative generator of BabyBear
INV = {i: pow(i, P - 2, P) for i in range(1, 8)}


# ---------------------------------------------------------------------------------------------------------------------
# NTT (coefficient basis, subgroup <w_n>) and the toy layout
# ---------------------------------------------------------------------------------------------------------------------

def root_of_unity(n: int) -> int:
    return pow(GEN, (P - 1) // n, P)


def ntt_cols(msg: np.ndarray, n: int) -> np.ndarray:
    """``msg`` (m, C) coefficient columns (m <= n) -> (n, C) evaluations at ``w^s``, iterative radix-2 (numpy)."""
    m, C = msg.shape
    x = np.zeros((n, C), dtype=np.int64)
    x[:m] = msg % P
    bits = n.bit_length() - 1
    rev = np.array([int(f"{i:0{bits}b}"[::-1], 2) for i in range(n)]) if bits else np.zeros(1, dtype=np.int64)
    x = x[rev]
    L = 2
    w_n = root_of_unity(n)
    while L <= n:
        w = pow(w_n, n // L, P)
        tw = np.array([pow(w, j, P) for j in range(L // 2)], dtype=np.int64)
        x = x.reshape(n // L, L, C)
        a, b = x[:, :L // 2], (x[:, L // 2:] * tw[None, :, None]) % P
        x = np.concatenate([(a + b) % P, (a - b) % P], axis=1)
        L *= 2
    return x.reshape(n, C)


@dataclass(frozen=True)
class Toy:
    n: int                       # log2 cells
    kprimes: tuple               # column bits per committed round
    n_rows_open: int             # |S| per round
    t_pad: int

    @property
    def k1(self):
        return self.n - self.kprimes[0]

    @property
    def ks(self):
        ks, cur = [], self.n
        for kp in self.kprimes:
            cur -= kp
            ks.append(cur)
        return ks

    @property
    def k_final(self):
        return self.ks[-1]

    @property
    def n_sc(self):
        return self.n - 2

    @property
    def flag_var(self) -> int:
        """Index (highest first) of the flag bit among the n_sc zero-check variables."""
        return self.n_sc - self.k1

    def zk_params(self, enabled=True) -> ZkParams:
        return ZkParams(self.n, self.k1, self.kprimes[0], t_pad=self.t_pad, degree=3, n_sumcheck=self.n_sc,
                        q=(1 << self.k1) - 8, enabled=enabled)

    # cell index of (plane, z, flag): bits [plane | col_rest | flag | row_low]; z over the n_sc = n - 2 low bits
    def cell(self, plane: int, z: np.ndarray, flag: int | None = None) -> np.ndarray:
        k1 = self.k1
        if flag is None:
            return (plane << (self.n - 2)) | z
        row_low = z & ((1 << (k1 - 1)) - 1)
        col_rest = z >> k1
        return (plane << (self.n - 2)) | (col_rest << k1) | (flag << (k1 - 1)) | row_low

    def referenced(self) -> np.ndarray:
        ref = np.zeros(1 << self.n, dtype=bool)
        z = np.arange(1 << (self.n_sc - 1))
        for plane in (0, 1, 2):
            ref[self.cell(plane, z, 0)] = True
        return ref

    def witness(self, rng: np.random.Generator, trivial: bool = False) -> np.ndarray:
        """A structured witness (constant rows, small values -- what leaks in the clear) or the all-zero one."""
        f = np.zeros(1 << self.n, dtype=np.int64)
        if trivial:
            return f
        z = np.arange(1 << (self.n_sc - 1))
        a = rng.integers(0, 16, size=z.size)                    # small operands (like FP8 codes)
        a[: z.size // 4] = 3                                    # a constant stretch (like a constant row)
        b = rng.integers(0, 16, size=z.size)
        f[self.cell(0, z, 0)] = a
        f[self.cell(1, z, 0)] = b
        f[self.cell(2, z, 0)] = a * b % P
        return f

    def claim_point(self, plane: int, r_sc: np.ndarray) -> np.ndarray:
        """The f-point (n, 6) of a(r) / b(r) / c(r): bits high -> low: plane (boolean), then r_sc (all n - 2 low bits,
        the flag coordinate included).  ``r_sc`` (n_sc, 6) is ordered highest bit first."""
        pt = ext_zero((self.n,))
        pt[0, 0], pt[1, 0] = plane >> 1, plane & 1
        pt[2:] = r_sc
        return pt

    def weights(self, tau: np.ndarray) -> np.ndarray:
        """tau' = tau with the flag coordinate set to 0: eq(tau', z) vanishes on the padding rows."""
        t = tau.copy()
        t[self.flag_var] = 0
        return t


# ---------------------------------------------------------------------------------------------------------------------
# Tensor terms (eq / geom / block) over a vector of 2^m cells, bits highest first
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class Term:
    coef: np.ndarray             # (6,)
    pairs: np.ndarray            # (m_top, 2, 6): tensor factors of the top bits (highest first)
    tail: np.ndarray             # (2^{m_low}, 6): dense vector over the lowest bits
    label: str = ""

    @property
    def nbits(self) -> int:
        return self.pairs.shape[0] + (self.tail.shape[0].bit_length() - 1)

    def colpart(self, kp: int) -> np.ndarray:
        """(2^kp, 6): tensor of the top kp pairs; index bit j of the column <-> pair (kp - 1 - j)."""
        vec = ext_one((1,))
        for j in reversed(range(kp)):                      # pair 0 is the highest bit: appended last
            pr = self.pairs[j]
            vec = np.concatenate([ext_mul(vec, pr[0]), ext_mul(vec, pr[1])], axis=0)   # new bit becomes the HIGH bit
        return vec

    def rowvec(self, k: int) -> np.ndarray:
        """(2^k, 6): the vector over the low k bits (pairs below the top ones, then the tail)."""
        kp = self.pairs.shape[0] - (k - (self.tail.shape[0].bit_length() - 1))
        vec = ext_one((1,))
        for j in reversed(range(kp, self.pairs.shape[0])):
            pr = self.pairs[j]
            vec = np.concatenate([ext_mul(vec, pr[0]), ext_mul(vec, pr[1])], axis=0)
        # vec indexes the pair bits (highest first -> low bit last); tail indexes the lowest bits
        out = ext_mul(vec[:, None, :], self.tail[None, :, :]).reshape(-1, D)
        return out

    def restrict(self, kp: int, r: np.ndarray, extra=None) -> "Term":
        """Fix the top kp bits at the challenges r (kp, 6), highest first: coef *= prod_j pair_j(r_j)."""
        coef = self.coef.copy()
        for j in range(kp):
            pr = self.pairs[j]
            one_minus = (ext_one() - r[j]) % P
            coef = ext_mul(coef, (ext_mul(pr[0], one_minus) + ext_mul(pr[1], r[j])) % P)
        if extra is not None:
            coef = ext_mul(coef, extra)
        return Term(coef, self.pairs[kp:], self.tail, self.label)


def eq_term(point: np.ndarray, coef=None, label="eq") -> Term:
    pt = np.asarray(point, dtype=np.int64) % P
    pairs = np.zeros((pt.shape[0], 2, D), dtype=np.int64)
    pairs[:, 1] = pt
    pairs[:, 0] = (ext_one() - pt) % P
    return Term(ext_one() if coef is None else coef, pairs, ext_one((1,)), label)


def geom_term(eta: int, nbits: int, coef=None, label="geom") -> Term:
    pairs = np.zeros((nbits, 2, D), dtype=np.int64)
    pairs[:, 0, 0] = 1
    for j in range(nbits):
        pairs[j, 1, 0] = pow(eta, 1 << (nbits - 1 - j), P)
    return Term(ext_one() if coef is None else coef, pairs, ext_one((1,)), label)


def block_term(claim: zk.BlockClaim, nbits: int, coef=None, label="block") -> Term:
    """One-hot pairs on the high bits, dense tail on the lowest bits spanning the block."""
    lo, hi = int(claim.cells.min()), int(claim.cells.max())
    span_bits = max(1, (hi ^ lo).bit_length())
    base = lo >> span_bits << span_bits
    assert base <= lo and hi < base + (1 << span_bits)
    pairs = np.zeros((nbits - span_bits, 2, D), dtype=np.int64)
    for j in range(nbits - span_bits):
        bit = (base >> (nbits - 1 - j)) & 1
        pairs[j, bit, 0] = 1
    tail = ext_zero((1 << span_bits,))
    tail[claim.cells - base] = claim.weights
    return Term(ext_one() if coef is None else coef, pairs, tail, label)


def tensor_eval(term: Term, vec: np.ndarray) -> np.ndarray:
    """<term, vec> for a dense vec (2^nbits, 6) -- test helper."""
    return ext_mul(term.coef, ext_sum(ext_mul(term.rowvec(term.nbits), vec), axis=0))


def eq_eval(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    acc = ext_one()
    for j in range(a.shape[0]):
        one_a, one_b = (ext_one() - a[j]) % P, (ext_one() - b[j]) % P
        acc = ext_mul(acc, (ext_mul(a[j], b[j]) + ext_mul(one_a, one_b)) % P)
    return acc


def lagrange_at(vals: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Polynomial through (e, vals[e]) for e = 0..d, evaluated at the extension point r."""
    d = vals.shape[0] - 1
    acc = ext_zero()
    for e in range(d + 1):
        num, den = ext_one(), 1
        for m in range(d + 1):
            if m == e:
                continue
            num = ext_mul(num, (r - ext_from_base(m)) % P)
            den = den * ((e - m) % P) % P
        acc = (acc + ext_mul_base(ext_mul(vals[e], num), pow(den, P - 2, P))) % P
    return acc


# ---------------------------------------------------------------------------------------------------------------------
# Merkle (toy)
# ---------------------------------------------------------------------------------------------------------------------

def _h(*parts: bytes) -> bytes:
    return hashlib.blake2b(b"|".join(parts), digest_size=32).digest()


class Merkle:
    def __init__(self, rows: np.ndarray):
        leaves = [_h(b"leaf", np.ascontiguousarray(r).tobytes()) for r in rows]
        levels = [leaves]
        while len(levels[-1]) > 1:
            cur = levels[-1]
            levels.append([_h(b"node", cur[i], cur[i + 1]) for i in range(0, len(cur), 2)])
        self.levels = levels

    @property
    def root(self) -> bytes:
        return self.levels[-1][0]

    def path(self, i: int) -> list[bytes]:
        out = []
        for lvl in self.levels[:-1]:
            out.append(lvl[i ^ 1])
            i //= 2
        return out

    @staticmethod
    def verify(root: bytes, i: int, row: np.ndarray, path: list[bytes]) -> bool:
        h = _h(b"leaf", np.ascontiguousarray(row).tobytes())
        for sib in path:
            h = _h(b"node", h, sib) if i % 2 == 0 else _h(b"node", sib, h)
            i //= 2
        return h == root


# ---------------------------------------------------------------------------------------------------------------------
# Transcript container
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class Proof:
    roots: list = field(default_factory=list)
    G: np.ndarray | None = None
    zc_msgs: list = field(default_factory=list)          # (4, 6) per round
    evals: dict = field(default_factory=dict)            # a, b, c, g
    pcs_msgs: list = field(default_factory=list)         # per round: list of (3, 6)
    ybars: list = field(default_factory=list)            # per committed round: (t_pad, 6)
    opened: list = field(default_factory=list)           # per committed round: (rows (|S|, C[,6]), paths)
    final: np.ndarray | None = None                      # (2^{k_final}, 6)
    aborted: zk.CoinOpeningError | None = None

    def components(self) -> dict:
        """Flattened numeric components (for the distinguisher statistics)."""
        out = {"zc_msgs": np.concatenate([m.reshape(-1) for m in self.zc_msgs]) if self.zc_msgs else np.zeros(0),
               "evals": np.concatenate([np.asarray(v).reshape(-1) for v in self.evals.values()]) if self.evals else np.zeros(0),
               "pcs_msgs": np.concatenate([np.asarray(m).reshape(-1) for rd in self.pcs_msgs for m in rd]) if self.pcs_msgs else np.zeros(0),
               "final": self.final.reshape(-1) if self.final is not None else np.zeros(0)}
        for i, (rows, _) in enumerate(self.opened):
            out[f"opened{i + 1}"] = np.asarray(rows).reshape(-1)
        for i, yb in enumerate(self.ybars):
            out[f"ybar{i + 1}"] = np.asarray(yb).reshape(-1)
        return out


# ---------------------------------------------------------------------------------------------------------------------
# Prover
# ---------------------------------------------------------------------------------------------------------------------

def _to_matrix(vec: np.ndarray, k: int) -> np.ndarray:
    """(2^{k+kp}[, 6]) column-major -> (2^k, 2^kp[, 6]); the column bits are the HIGH bits."""
    R = 1 << k
    if vec.ndim == 1:
        return vec.reshape(-1, R).T.copy()
    return np.transpose(vec.reshape(-1, R, D), (1, 0, 2)).copy()


def _encode(msg: np.ndarray, n_i: int) -> np.ndarray:
    """msg (m, C) base or (m, C, 6) ext -> codeword rows (n_i, C) / (n_i, C, 6)."""
    if msg.ndim == 2:
        return ntt_cols(msg, n_i)
    m, C, _ = msg.shape
    return ntt_cols(msg.reshape(m, C * D), n_i).reshape(n_i, C, D)


def _contract_cols(M: np.ndarray, rowvec: np.ndarray) -> np.ndarray:
    """u[c] = sum_row M[row, c] rowvec[row] -> (C, 6); M (R, C, 6) extension."""
    return ext_sum(ext_mul(M, rowvec[:, None, :]), axis=0)


def _fold_rows(M: np.ndarray, rbar: np.ndarray) -> np.ndarray:
    """y = M rbar -> (R, 6); M (R, C, 6) extension."""
    return ext_sum(ext_mul(M, rbar[None, :, :]), axis=1)


def _matmul_safe(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """(A @ B) mod p with int64 accumulation kept < 2^63 by chunking the contraction (entries < 2^31)."""
    K = A.shape[1]
    step = 2 ** 62 // (P * P)
    out = np.zeros((A.shape[0], B.shape[1]), dtype=np.int64)
    for k0 in range(0, K, step):
        out = (out + A[:, k0:k0 + step] @ B[k0:k0 + step]) % P
    return out


def _contract_cols_base(M: np.ndarray, rowvec: np.ndarray) -> np.ndarray:
    out = ext_zero((M.shape[1],))
    for d in range(D):
        out[:, d] = _matmul_safe(M.T, rowvec[:, d:d + 1])[:, 0]
    return out


def _fold_rows_base(M: np.ndarray, rbar: np.ndarray) -> np.ndarray:
    return _matmul_safe(M, rbar) % P


class ColumnSumcheck:
    """Partial sumcheck over the kp column bits (highest first) of sum_t coef_t <u_t, colpart_t>."""

    def __init__(self, terms: list[Term], us: list[np.ndarray], kp: int):
        self.A = np.stack(us) % P                                   # (T, C, 6)
        self.B = np.stack([t.colpart(kp) for t in terms])           # (T, C, 6)
        self.coef = np.stack([t.coef for t in terms])               # (T, 6)
        self.rs: list[np.ndarray] = []

    def claim(self) -> np.ndarray:
        return ext_sum(ext_mul(self.coef, ext_sum(ext_mul(self.A, self.B), axis=1)), axis=0)

    def message(self) -> np.ndarray:
        C = self.A.shape[1]
        h = C // 2
        Alo, Ahi, Blo, Bhi = self.A[:, :h], self.A[:, h:], self.B[:, :h], self.B[:, h:]
        A2, B2 = (2 * Ahi - Alo) % P, (2 * Bhi - Blo) % P
        out = []
        for a, b in ((Alo, Blo), (Ahi, Bhi), (A2, B2)):
            out.append(ext_sum(ext_mul(self.coef, ext_sum(ext_mul(a, b), axis=1)), axis=0))
        return np.stack(out)

    def fold(self, r: np.ndarray) -> None:
        C = self.A.shape[1]
        h = C // 2
        om = (ext_one() - r) % P
        self.A = (ext_mul(self.A[:, :h], om) + ext_mul(self.A[:, h:], r)) % P
        self.B = (ext_mul(self.B[:, :h], om) + ext_mul(self.B[:, h:], r)) % P
        self.rs.append(r)


def rbar_of(rs: list[np.ndarray]) -> np.ndarray:
    """(2^kp, 6) with rs highest bit first: rbar[c] = prod_j ((1 - r_j) if bit_j(c) = 0 else r_j)."""
    vec = ext_one((1,))
    for r in reversed(rs):                                 # rs[0] is the highest bit: appended last
        vec = np.concatenate([ext_mul(vec, (ext_one() - r) % P), ext_mul(vec, r)], axis=0)
    return vec


def interp2(g: np.ndarray, r: np.ndarray) -> np.ndarray:
    return lagrange_at(g, r)


def prove(toy: Toy, f: np.ndarray, zp: ZkParams, verifier: zk.CommittedCoins, stmt: bytes, *,
          zc_override=None, masks: zk.Masks | None = None, evals_override=None) -> Proof:
    """The toy prover.  ``zc_override``: (messages, evals) the simulator substitutes for the zero-check; ``masks``: the
    masks to use (the simulator's solved tableau) instead of fresh ones."""
    n, n_sc = toy.n, toy.n_sc
    pr = Proof()
    sched = zk.coin_schedule(zp, list(toy.kprimes))
    coins = zk.ProverCoins(stmt, sched, verifier.commitments(), verifier)
    try:
        # ---- witness + masks -> f^, round-1 commitment
        fh = f.copy() % P
        masks = zk.sample_masks(zp) if masks is None else masks
        zk.apply_masks(fh, masks, zp)
        gmask = zk.SumcheckMask(zp, masks.g_coeffs) if zp.enabled else None
        ks = toy.ks
        M1 = _to_matrix(fh, ks[0])                                  # (2^k1, C1) base
        msg1, mu1 = zk.pad_columns(zp, M1)
        n1 = 4 << ks[0]
        U1 = _encode(msg1, n1)
        T1 = Merkle(U1)
        pr.roots.append(T1.root)
        coins.send("root1", T1.root)
        # ---- zero-check
        tau = coins.next("zc/batch", n_sc)
        if zp.enabled:
            pr.G = gmask.total_sum()
            coins.send("G", pr.G.tobytes())
            rho = coins.next("zc/rho", 1)[0]
        z = np.arange(1 << n_sc)
        a = ext_from_base(fh[toy.cell(0, z)])
        b = ext_from_base(fh[toy.cell(1, z)])
        c = ext_from_base(fh[toy.cell(2, z)])
        eqt = rbar_of(list(toy.weights(tau)))                      # eq(tau', z), z bits highest first <-> tau order
        rs_sc = []
        for j in range(n_sc):
            if zc_override is not None:
                msg = zc_override[0][j]
            else:
                h = a.shape[0] // 2
                vals = []
                for X in (0, 1, 2, 3):
                    def at(T):
                        return (T[:h] + ext_mul_base(T[h:] - T[:h], X)) % P
                    e_, a_, b_, c_ = at(eqt), at(a), at(b), at(c)
                    vals.append(ext_sum(ext_mul(e_, (ext_mul(a_, b_) - c_) % P), axis=0))
                msg = np.stack(vals)
                if zp.enabled:
                    msg = (msg + ext_mul(rho, gmask.round_values(j, np.array(rs_sc).reshape(-1, D)))) % P
            pr.zc_msgs.append(msg)
            coins.send(f"zc_msg{j}", msg.tobytes())
            r = coins.next(f"zc/r{j}", 1)[0]
            rs_sc.append(r)
            h = a.shape[0] // 2
            om = (ext_one() - r) % P
            eqt, a, b, c = [(ext_mul(T[:h], om) + ext_mul(T[h:], r)) % P for T in (eqt, a, b, c)]
        r_sc = np.array(rs_sc)
        pts = {name: toy.claim_point(plane, r_sc) for name, plane in (("a", 0), ("b", 1), ("c", 2))}
        if evals_override is not None:
            pr.evals = dict(evals_override)
        else:
            pr.evals = {"a": a[0], "b": b[0], "c": c[0]}
            if zp.enabled:
                pr.evals["g"] = gmask.final_value(r_sc)
        coins.send("evals", b"".join(v.tobytes() for v in pr.evals.values()))
        # ---- PCS rounds.  Claims are GROUPS (list of tensor terms, value); the batching coin weights group g by
        # alpha^g; after a round the reduced claim is one group (all the terms, value = the last message at r).
        groups = [([eq_term(pts[nm], label=nm)], pr.evals[nm]) for nm in ("a", "b", "c")]
        if zp.enabled:
            groups.append(([block_term(gmask.claim(r_sc), n)], pr.evals["g"]))
        M, mu, U, T = M1, mu1, U1, T1
        prev_rbar = None
        for i, kp in enumerate(toy.kprimes, start=1):
            k = ks[i - 1]
            if i >= 2:
                S = coins.next(f"pcs/S{i - 1}", toy.n_rows_open, indices_n=U_prev.shape[0])
                rows = U_prev[S]
                paths = [T_prev.path(s) for s in S]
                pr.opened.append((rows, paths))
                coins.send(f"opened{i - 1}", np.ascontiguousarray(rows).tobytes())
                # expected symbols (the verifier recomputes them from the opened rows): geom claims on vec(M~_i)
                w_prev = root_of_unity(U_prev.shape[0])
                for idx, s in enumerate(S):
                    eta = pow(w_prev, s, P)
                    row = rows[idx] if rows.ndim == 3 else ext_from_base(rows[idx])
                    val = (ext_sum(ext_mul(row, prev_rbar), axis=0) - zk.row_check_rhs(zp, ks[i - 2], eta, pr.ybars[-1])) % P
                    groups.append(([geom_term(eta, k + kp, label=f"geom{s}")], val))
            alpha = coins.next(f"pcs/alpha{i}", 1)[0]
            terms = batch_groups(groups, alpha)
            us = [(_contract_cols_base(M, t.rowvec(k)) if M.ndim == 2 else _contract_cols(M, t.rowvec(k))) for t in terms]
            sc = ColumnSumcheck(terms, us, kp)
            msgs = []
            for j in range(kp):
                m = sc.message()
                msgs.append(m)
                coins.send(f"pcs_msg{i}.{j}", m.tobytes())
                r = coins.next(f"pcs/r{i}.{j}", 1)[0]
                sc.fold(r)
            pr.pcs_msgs.append(msgs)
            rbar = rbar_of(sc.rs)
            groups = [([t.restrict(kp, np.array(sc.rs)) for t in terms], sc.claim())]
            y = _fold_rows_base(M, rbar) if M.ndim == 2 else _fold_rows(M, rbar)          # (2^k, 6)
            ybar = zk.padding_combination(mu, rbar)
            pr.ybars.append(ybar)
            U_prev, T_prev, prev_rbar = U, T, rbar
            if i < len(toy.kprimes):
                Mn = _to_matrix(y, ks[i])                                                 # (2^{k_{i+1}}, C_{i+1}, 6)
                planes = np.transpose(Mn, (2, 0, 1))
                msgp, mup = zk.pad_columns(zp, planes)
                msg_ext = np.transpose(msgp, (1, 2, 0))
                U = _encode(msg_ext, 4 << ks[i])
                T = Merkle(U.reshape(U.shape[0], -1))
                M, mu = Mn, mup
                pr.roots.append(T.root)
                coins.send(f"root{i + 1}", T.root + ybar.tobytes())
            else:
                pr.final = y
                coins.send("final", y.tobytes() + ybar.tobytes())
        i = len(toy.kprimes)
        S = coins.next(f"pcs/S{i}", toy.n_rows_open, indices_n=U_prev.shape[0])
        rows = U_prev[S]
        pr.opened.append((rows, [T_prev.path(s) for s in S]))
    except zk.CoinOpeningError as e:
        pr.aborted = e
    return pr


# ---------------------------------------------------------------------------------------------------------------------
# Verifier (its own coins; every check of the design)
# ---------------------------------------------------------------------------------------------------------------------

def verify(toy: Toy, pr: Proof, zp: ZkParams, coins: zk.CommittedCoins, stmt: bytes) -> tuple[bool, str]:
    if pr.aborted is not None:
        return False, "prover aborted"
    n, n_sc, ks = toy.n, toy.n_sc, toy.ks
    sched = zk.coin_schedule(zp, list(toy.kprimes))
    slot = {lab: k for k, lab in enumerate(sched)}
    ch = lambda lab, cnt, ni=None: coins.challenge(stmt, slot[lab], cnt, ni)   # noqa: E731
    tau = ch("zc/batch", n_sc)
    if zp.enabled:
        rho = ch("zc/rho", 1)[0]
        cur = ext_mul(rho, pr.G)
    else:
        cur = ext_zero()
    rs_sc = []
    for j in range(n_sc):
        m = pr.zc_msgs[j]
        if not np.array_equal((m[0] + m[1]) % P, cur):
            return False, f"zero-check round {j}: h(0)+h(1) != claim"
        r = ch(f"zc/r{j}", 1)[0]
        rs_sc.append(r)
        cur = lagrange_at(m, r)
    r_sc = np.array(rs_sc)
    ev = pr.evals
    lhs = ext_mul(eq_eval(toy.weights(tau), r_sc), (ext_mul(ev["a"], ev["b"]) - ev["c"]) % P)
    if zp.enabled:
        lhs = (lhs + ext_mul(rho, ev["g"])) % P
    if not np.array_equal(lhs % P, cur):
        return False, "zero-check final check failed"
    pts = {name: toy.claim_point(plane, r_sc) for name, plane in (("a", 0), ("b", 1), ("c", 2))}
    groups = [([eq_term(pts[nm])], ev[nm]) for nm in ("a", "b", "c")]
    if zp.enabled:
        # the verifier builds the block claim's weights itself from r (public), value = the prover's g(r)
        dummy = zk.SumcheckMask(zp, np.zeros((zp.g_coeffs, D), dtype=np.int64))
        bc = dummy.claim(r_sc)
        groups.append(([block_term(zk.BlockClaim(bc.cells, bc.weights, ev["g"]), n)], ev["g"]))
    prev_rbar = None
    for i, kp in enumerate(toy.kprimes, start=1):
        k = ks[i - 1]
        if i >= 2:
            n_prev = 4 << ks[i - 2]
            S = ch(f"pcs/S{i - 1}", toy.n_rows_open, n_prev)
            rows, paths = pr.opened[i - 2]
            w_prev = root_of_unity(n_prev)
            for idx, s in enumerate(S):
                if not Merkle.verify(pr.roots[i - 2], s, rows[idx].reshape(-1) if rows.ndim == 3 else rows[idx], paths[idx]):
                    return False, f"round {i - 1}: Merkle path {s}"
                eta = pow(w_prev, s, P)
                row = rows[idx] if rows.ndim == 3 else ext_from_base(rows[idx])
                val = (ext_sum(ext_mul(row, prev_rbar), axis=0) - zk.row_check_rhs(zp, ks[i - 2], eta, pr.ybars[i - 2])) % P
                groups.append(([geom_term(eta, k + kp)], val))
        alpha = ch(f"pcs/alpha{i}", 1)[0]
        terms = batch_groups(groups, alpha)
        claim = batch_values(groups, alpha)
        rs = []
        for j in range(kp):
            m = pr.pcs_msgs[i - 1][j]
            if not np.array_equal((m[0] + m[1]) % P, claim):
                return False, f"pcs round {i}.{j}: g(0)+g(1) != claim"
            r = ch(f"pcs/r{i}.{j}", 1)[0]
            rs.append(r)
            claim = interp2(m, r)
        prev_rbar = rbar_of(rs)
        terms = [t.restrict(kp, np.array(rs)) for t in terms]
        if i < len(toy.kprimes):
            # the reduced claim on y_i = vec(M~_{i+1}) is sum_t coef_t <rowvec_t, y_i> = claim: ONE group
            groups = [(terms, claim)]
        else:
            y = pr.final
            n_prev = 4 << k
            S = ch(f"pcs/S{i}", toy.n_rows_open, n_prev)
            rows, paths = pr.opened[i - 1]
            w_prev = root_of_unity(n_prev)
            for idx, s in enumerate(S):
                if not Merkle.verify(pr.roots[i - 1], s, rows[idx].reshape(-1) if rows.ndim == 3 else rows[idx], paths[idx]):
                    return False, f"round {i}: Merkle path {s}"
                eta = pow(w_prev, s, P)
                row = rows[idx] if rows.ndim == 3 else ext_from_base(rows[idx])
                lhs = ext_sum(ext_mul(row, prev_rbar), axis=0)
                gen = geom_term(eta, k).rowvec(k)
                rhs = (ext_sum(ext_mul(gen, y), axis=0) + zk.row_check_rhs(zp, k, eta, pr.ybars[i - 1])) % P
                if not np.array_equal(lhs, rhs):
                    return False, f"final row check failed at row {s}"
            tot = ext_zero()
            for t in terms:
                tot = (tot + ext_mul(t.coef, ext_sum(ext_mul(t.rowvec(k), y), axis=0))) % P
            if not np.array_equal(tot, claim):
                return False, "final sumcheck check failed"
    return True, "accepted"


def batch_groups(groups, alpha) -> list[Term]:
    """Group g's terms get their coefficients multiplied by alpha^g."""
    out, apow = [], ext_one()
    for terms, _ in groups:
        out += [Term(ext_mul(t.coef, apow), t.pairs, t.tail, t.label) for t in terms]
        apow = ext_mul(apow, alpha)
    return out


def batch_values(groups, alpha) -> np.ndarray:
    acc, apow = ext_zero(), ext_one()
    for _, v in groups:
        acc = (acc + ext_mul(apow, v)) % P
        apow = ext_mul(apow, alpha)
    return acc


# ---------------------------------------------------------------------------------------------------------------------
# Simulator (malicious verifier, interactive): dummy run -> rewind -> fake tableau + the honest PCS prover
# ---------------------------------------------------------------------------------------------------------------------

class PublicCheckVerifier(zk.AbortingCoins):
    """Aborts iff a public consistency check on the transcript so far fails (red-team-zk-2 G1): must abort in
    neither world.  Checks: every zero-check message satisfies h(0) + h(1) = previous claim."""

    def __init__(self, schedule, toy: Toy, zp: ZkParams, stmt: bytes, seed=None):
        super().__init__(schedule, self._pred, seed)
        self.toy, self.zp, self.stmt = toy, zp, stmt

    def _pred(self, k, transcript):
        msgs = [np.frombuffer(d, dtype=np.int64).reshape(4, D) for lab, d in transcript if lab.startswith("zc_msg")]
        if not msgs:
            return False
        slot = {lab: kk for kk, lab in enumerate(self.schedule)}
        G = [np.frombuffer(d, dtype=np.int64) for lab, d in transcript if lab == "G"]
        if not G:
            return False
        rho = self.challenge(self.stmt, slot["zc/rho"], 1)[0]
        cur = ext_mul(rho, G[0])
        for j, m in enumerate(msgs):
            if not np.array_equal((m[0] + m[1]) % P, cur):
                return True
            if j < len(msgs) - 1 or k > slot[f"zc/r{j}"]:
                cur = lagrange_at(m, self.challenge(self.stmt, slot[f"zc/r{j}"], 1)[0])
        return False


def _uniform_ext(count: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, P, size=(count, D), dtype=np.int64)


def simulate(toy: Toy, zp: ZkParams, verifier: zk.CommittedCoins, stmt: bytes, rng: np.random.Generator,
             max_rewinds: int = 8) -> Proof:
    """The simulator of the note's section 2, against the coin object ``verifier`` (honest or malicious).  Returns the
    transcript it outputs (an aborted one if V* aborts)."""
    n, n_sc = toy.n, toy.n_sc
    sched = zk.coin_schedule(zp, list(toy.kprimes))
    # 1. dummy run: a random tableau with fresh masks, honest prover algorithm on it, honest-observable zero-check
    #    messages (uniform on the consistency lines).  Extracts the openings (or an abort).
    fake = rng.integers(0, P, size=1 << n, dtype=np.int64)
    dummy_masks = zk.sample_masks(zp)
    gm = zk.SumcheckMask(zp, dummy_masks.g_coeffs)
    probe = _ProbeCoins(verifier)
    zc_msgs, evals = _sample_zero_check(toy, zp, gm.total_sum(), probe, stmt, rng)
    pr0 = prove(toy, fake, zp, probe, stmt, zc_override=(zc_msgs, evals), masks=dummy_masks)
    if pr0.aborted is not None:
        return pr0
    # 2. rewind: all coins known (probe.opened); build the fake tableau consistent with a fresh honest-observable
    #    zero-check transcript and run the honest prover on it.
    slot = {lab: k for k, lab in enumerate(sched)}
    known = _KnownCoins(verifier, probe.opened)
    for _ in range(max_rewinds):
        masks = zk.sample_masks(zp)
        gm = zk.SumcheckMask(zp, masks.g_coeffs)
        G = gm.total_sum()
        zc_msgs, evals, r_sc, rho = _sample_zero_check(toy, zp, G, known, stmt, rng, return_r=True)
        # g*(r) must equal evals["g"]: shift a_{0,1} by +delta1 and a_{0,2} by -delta1 (keeps G)
        delta = (evals["g"] - gm.final_value(r_sc)) % P
        r0 = r_sc[0]
        denom = (r0 - ext_mul(r0, r0)) % P
        d1 = ext_mul(delta, ext_inv(denom))
        coeffs = np.concatenate([gm.a0[None], gm.a.reshape(-1, D)])
        coeffs[1] = (coeffs[1] + d1) % P
        coeffs[2] = (coeffs[2] - d1) % P
        masks.g_coeffs = coeffs
        g0 = masks.block.shape[0] - zp.rows1
        masks.block[g0:g0 + zp.g_block_cells] = coeffs.reshape(-1)
        gm = zk.SumcheckMask(zp, coeffs)
        assert np.array_equal(gm.total_sum(), G) and np.array_equal(gm.final_value(r_sc), evals["g"])
        # f^*(z_a) = v_a etc.: solve three mask-column cells (outside the G-block)
        fh = fake.copy()
        zk.apply_masks(fh, masks, zp)
        pts = {nm: toy.claim_point(pl, r_sc) for nm, pl in (("a", 0), ("b", 1), ("c", 2))}
        mc = zk.mask_cells(zp)
        # the claim of plane p only sees plane-p cells: solve, per plane, its A-cells (base field, 8 cells, extension
        # weights eq(z_p, cell) -- a 6 x 8 F_p system; the rank event is the simulator's O(1/p) distance term)
        ok = True
        for pl, nm in ((0, "a"), (1, "b"), (2, "c")):
            pt = pts[nm]
            lo, hi = pl << (n - 2), (pl + 1) << (n - 2)
            cells = mc.a_cells[(mc.a_cells >= lo) & (mc.a_cells < hi)][:8]
            bits = np.array([[(int(x) >> (n - 1 - b)) & 1 for b in range(n)] for x in cells])
            W = np.stack([eq_eval(pt, ext_from_base(bits[j])) for j in range(len(cells))])      # (8, 6)
            fh_zero = fh.copy()
            fh_zero[cells] = 0
            target = (evals[nm] - _mle_eval(fh_zero, pt)) % P
            sol = _solve_base(W.T, target)                                                      # W^T m = target
            if sol is None:
                ok = False
                break
            fh[cells] = sol
            masks.a_cells[np.searchsorted(mc.a_cells, cells)] = sol
        if not ok:
            continue
        return prove(toy, fake, zp, known, stmt, zc_override=(zc_msgs, evals), masks=masks)
    raise RuntimeError("simulator: no base-field solution for the eval cells after max_rewinds resamples")


def _mle_eval(vec: np.ndarray, pt: np.ndarray) -> np.ndarray:
    T = ext_from_base(vec)
    for j in range(pt.shape[0]):
        h = T.shape[0] // 2
        T = (ext_mul(T[:h], (ext_one() - pt[j]) % P) + ext_mul(T[h:], pt[j])) % P
    return T[0]


def _solve_base(A: np.ndarray, b: np.ndarray):
    """One solution of ``A m = b`` over F_p (A (r, c), b (r,)), or None if inconsistent (rank-deficient case)."""
    r, c = A.shape
    M = np.concatenate([A % P, (b % P)[:, None]], axis=1).astype(np.int64)
    piv_cols, row = [], 0
    for col in range(c):
        if row >= r:
            break
        nz = np.nonzero(M[row:, col])[0]
        if nz.size == 0:
            continue
        pr_ = row + nz[0]
        M[[row, pr_]] = M[[pr_, row]]
        M[row] = M[row] * pow(int(M[row, col]), P - 2, P) % P
        for rr in range(r):
            if rr != row and M[rr, col]:
                M[rr] = (M[rr] - M[rr, col] * M[row]) % P
        piv_cols.append(col)
        row += 1
    if np.any(M[row:, -1]):
        return None
    m = np.zeros(c, dtype=np.int64)
    for i, col in enumerate(piv_cols):
        m[col] = M[i, -1]
    return m


def _solve_ext(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Gaussian elimination over F_{p^6}: A (m, m, 6), b (m, 6) -> x (m, 6)."""
    m = A.shape[0]
    A, b = A.copy(), b.copy()
    for col in range(m):
        piv = next(r for r in range(col, m) if not zk.ext_is_zero(A[r, col]))
        A[[col, piv]], b[[col, piv]] = A[[piv, col]], b[[piv, col]]
        inv = ext_inv(A[col, col])
        A[col] = ext_mul(A[col], inv)
        b[col] = ext_mul(b[col], inv)
        for r in range(m):
            if r != col and not zk.ext_is_zero(A[r, col]):
                fct = A[r, col].copy()
                A[r] = (A[r] - ext_mul(fct, A[col])) % P
                b[r] = (b[r] - ext_mul(fct, b[col])) % P
    return b


def _sample_zero_check(toy, zp, G, coins, stmt, rng, return_r=False):
    """Honest-observable zero-check transcript without a witness: messages uniform on h(0)+h(1) = claim, evaluation
    claims uniform, g(r) solved from the final check.  Needs the coins (they are queried through ``coins``)."""
    sched = zk.coin_schedule(zp, list(toy.kprimes))
    slot = {lab: k for k, lab in enumerate(sched)}
    tau = coins.peek(stmt, slot["zc/batch"], toy.n_sc)
    rho = coins.peek(stmt, slot["zc/rho"], 1)[0]
    cur = ext_mul(rho, G)
    msgs, rs = [], []
    for j in range(toy.n_sc):
        m = _uniform_ext(4, rng)
        m[1] = (cur - m[0]) % P
        msgs.append(m)
        r = coins.peek(stmt, slot[f"zc/r{j}"], 1)[0]
        rs.append(r)
        cur = lagrange_at(m, r)
    r_sc = np.array(rs)
    ev = {nm: _uniform_ext(1, rng)[0] for nm in ("a", "b", "c")}
    rest = ext_mul(eq_eval(toy.weights(tau), r_sc), (ext_mul(ev["a"], ev["b"]) - ev["c"]) % P)
    ev["g"] = ext_mul((cur - rest) % P, ext_inv(rho))
    if return_r:
        return msgs, ev, r_sc, rho
    return msgs, ev


class _ProbeCoins(zk.CommittedCoins):
    """The simulator's view of V* during the dummy run: forwards openings, records them; ``peek`` derives the
    challenge of a slot from V*'s opening (the simulator asks V* in order, as the prover would)."""

    def __init__(self, v: zk.CommittedCoins):
        self.schedule, self.cs, self.cset, self.v = v.schedule, v.cs, v.cset, v
        self.opened: dict[int, tuple[bytes, bytes]] = {}
        self._transcript: list = []

    def commitments(self):
        return list(self.cs)

    def open(self, k, transcript):
        op = self.v.open(k, transcript)
        if op is not None:
            self.opened[k] = op
        return op

    def peek(self, stmt, k, count, indices_n=None):
        if k not in self.opened:
            # the dummy run has not reached this slot yet: sampled messages so far do not depend on it, so we
            # use the value V* WILL open (bound by the commitment) -- for an honest/committed V* that is its coin.
            op = self.v.open(k, [])
            if op is None:
                raise _EarlyAbort(k)
            self.opened[k] = op
        return zk.challenge_from(stmt, self.cset, k, self.opened[k][0], count, indices_n=indices_n)


class _EarlyAbort(Exception):
    pass


class _KnownCoins(_ProbeCoins):
    """After the rewind: replays V* (binding forces the same openings or an abort); ``peek`` uses the extracted ones."""

    def __init__(self, v, opened):
        super().__init__(v)
        self.opened = dict(opened)


# ---------------------------------------------------------------------------------------------------------------------
# Statistics for the distinguisher tests (as redteam_zk.py)
# ---------------------------------------------------------------------------------------------------------------------

BINS = 64


def chi2_sf(stat: float, dof: int) -> float:
    z = ((stat / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
    return 0.5 * math.erfc(z / math.sqrt(2))


def chi2_two_sample(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    cx = np.bincount((np.asarray(x, dtype=np.int64).ravel() % P) * BINS // P, minlength=BINS).astype(float)
    cy = np.bincount((np.asarray(y, dtype=np.int64).ravel() % P) * BINS // P, minlength=BINS).astype(float)
    kx, ky = math.sqrt(cy.sum() / cx.sum()), math.sqrt(cx.sum() / cy.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = (kx * cx - ky * cy) ** 2 / (cx + cy)
    stat = float(np.nansum(terms))
    return stat, chi2_sf(stat, BINS - 1)


def chi2_uniform(x: np.ndarray) -> tuple[float, float]:
    counts = np.bincount((np.asarray(x, dtype=np.int64).ravel() % P) * BINS // P, minlength=BINS)
    e = x.size / BINS
    stat = float(((counts - e) ** 2 / e).sum())
    return stat, chi2_sf(stat, BINS - 1)


def compare(proofs_x: list[Proof], proofs_y: list[Proof]) -> dict:
    """Two-sample chi-square per component, pooled over the proofs."""
    out = {}
    cx = {k: np.concatenate([p.components()[k] for p in proofs_x]) for k in proofs_x[0].components()}
    cy = {k: np.concatenate([p.components()[k] for p in proofs_y]) for k in proofs_y[0].components()}
    for k in cx:
        if cx[k].size == 0:
            continue
        stat, pv = chi2_two_sample(cx[k], cy[k])
        out[k] = {"chi2": stat, "p": pv, "n": int(cx[k].size)}
    return out


DEFAULT_TOY = Toy(n=16, kprimes=(5, 1, 1), n_rows_open=16, t_pad=24)     # 2^11 x 32, then 2^10 x 2, 2^9 x 2 (ext)
