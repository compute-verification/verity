"""J10's shared-noise forms against H1 + fx1's after-coin work, bit-exact on `h1tile` and `segment`.

A call is M rows of A (committed before the coin) and N rows of B (registered), at contraction K = 32 KS. Its noise is
per call row (K dither bits b and the digit coins z) and per call column (K coin bits c and U's 32 values), and a 32 × 32
unit (I, J) reads rows 32 I .. 32 I + 31 and columns 32 J .. 32 J + 31 of it. Each form below is a rule for that noise.
For each form the check builds every unit's credited words with `segment.call_words`, exactly as `h1tile.reference`
does, and receipts with fx1, and then the cheapest after-coin route a cheater has given the AB running partials
P_s = A^(s) B^(s)T, which it computes before the coin:

    T_s = sum_(s' <= s) P_s' + E_s',  E_s = A N^T + b (B + N)^T - 256 (wX (B + N)^T + X wW^T)   (segment s, mod 2^32)

with N = c + U (U repeated every segment), X = A + b - 256 wX and W' = B + N - 256 wW the mod 2^8 wraps (wX at A = 127,
b = 1; wW in {-1, 0, 1}), O' = T_(KS-1) - U A_0 and D_d = U A_d from X's residue digits. The route factors b's and N's
segment slices as the form allows (b^(s) = F G_s, N^(s) = H L_s, U = H_U L_U), its words must equal call_words' bit for
bit, and its operations are counted: MACs with both operands reading coins (the products the record's floor counts:
`Pouw.Assumptions.ChargeVF3j16`, Theorem K's `Slp` slots), MACs of committed data by a coin (bit-affine words, free in
the record's model), sparse wrap adds, and one write a credited word.

`collect()` returns (passed, failed) and `report()` the per-form table, at the check's shape and at model shapes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from typing import Callable

import numpy as np

from verity.ml.protocols.pouw.circuit import h1tile as H, seeds as SD, segment as SG

M32 = (1 << 32) - 1
P61 = (1 << 61) - 1
TILE = 32


def _sx(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.int64) & 0xFF
    return np.where(v & 0x80, v - 256, v)


def _bytes(rng: random.Random, n: int = 64) -> bytes:
    return bytes(rng.getrandbits(8) for _ in range(n))


def _synthetic_weights(rng: random.Random, n: int, k: int) -> list[list[int]]:
    """Synthetic int8 weights, Gaussian with sigma 32 clipped at ±127: about 4.7% with |B| > 63, near served_h1's
    measured 4.66% on served weights. They are not served weights, so the wrap densities below are modeled."""
    return [[max(-127, min(127, round(rng.gauss(0.0, 32.0)))) for _ in range(k)] for _ in range(n)]


def _rank_mod_p(mat) -> int:
    rows = [[int(x) % P61 for x in r] for r in mat]
    rank, ncol = 0, len(rows[0]) if rows else 0
    for col in range(ncol):
        piv = next((i for i in range(rank, len(rows)) if rows[i][col]), None)
        if piv is None:
            continue
        rows[rank], rows[piv] = rows[piv], rows[rank]
        inv = pow(rows[rank][col], P61 - 2, P61)
        rows[rank] = [x * inv % P61 for x in rows[rank]]
        for i in range(len(rows)):
            if i != rank and rows[i][col]:
                f = rows[i][col]
                rows[i] = [(x - f * y) % P61 for x, y in zip(rows[i], rows[rank])]
        rank += 1
    return rank


# -- a form: the call's noise and the factorizations its rule gives the cheater -----------------------------------------

@dataclass
class Fac:
    """``X = F @ G(s)``: F (n × rho) is `sel` (one-hot rows: a gather, no MACs) or coin-read (MACs), G(s) is rho × 32."""

    F: np.ndarray
    G: Callable[[int], np.ndarray]
    sel: bool

    @property
    def rho(self) -> int:
        return self.F.shape[1]


@dataclass
class Form:
    name: str
    what: str
    rows: list  # per call row: (K dither bits, [m - 1 lists of 32 digit coins])
    cols: list  # per call column: (K coin bits, 32 int8 encodings of U)
    row_fac: Fac | None = None  # b's segment slices
    col_fac: Fac | None = None  # N's
    u_fac: Fac | None = None  # U's (the same every segment)
    precoin_cols: bool = False  # the column noise is known before A is committed


def _arrays(form: Form, m: int):
    b = np.array([r[0] for r in form.rows], dtype=np.int64)
    z = np.array([r[1] for r in form.rows], dtype=np.int64).reshape(len(form.rows), m - 1, TILE)
    c = np.array([q[0] for q in form.cols], dtype=np.int64)
    u = _sx(np.array([q[1] for q in form.cols], dtype=np.int64))
    return b, z, c, u


def _onehot(classes: list[int], rho: int) -> np.ndarray:
    F = np.zeros((len(classes), rho), dtype=np.int64)
    F[np.arange(len(classes)), classes] = 1
    return F


def _selection(base_rows, base_cols, rcls: list[int], ccls: list[int], m: int, name: str, what: str) -> Form:
    """Rows take noise ``base_rows[rcls[i]]`` and columns ``base_cols[ccls[j]]``: rho_b, rho_N distinct streams."""
    rows = [base_rows[k] for k in rcls]
    cols = [base_cols[k] for k in ccls]
    rb, cb = sorted(set(rcls)), sorted(set(ccls))
    ri, ci = {k: n for n, k in enumerate(rb)}, {k: n for n, k in enumerate(cb)}
    pb = np.array([base_rows[k][0] for k in rb], dtype=np.int64)
    pc = np.array([base_cols[k][0] for k in cb], dtype=np.int64)
    pu = _sx(np.array([base_cols[k][1] for k in cb], dtype=np.int64))
    Fb, Fc = _onehot([ri[k] for k in rcls], len(rb)), _onehot([ci[k] for k in ccls], len(cb))
    return Form(name, what, rows, cols,
                row_fac=Fac(Fb, lambda s: pb[:, 32 * s:32 * s + 32], True) if len(rb) < len(rcls) else None,
                col_fac=Fac(Fc, lambda s: pc[:, 32 * s:32 * s + 32] + pu, True) if len(cb) < len(ccls) else None,
                u_fac=Fac(Fc, lambda s: pu, True) if len(cb) < len(ccls) else None)


def _u129(x: int) -> int:
    return H.u129(x)


def forms(M: int, N: int, K: int, rng: random.Random) -> tuple[list[Form], dict]:
    """The candidate forms on one call (M × N at K), with the step's and the window's seeds."""
    lay = H.Layout(M, N, K)
    m = lay.m
    salt, rootA, rootB = _bytes(rng), _bytes(rng), _bytes(rng)
    seed_step = SD.seed(salt, 7, rootA, rootB)
    rows0, cols0 = H.noise(seed_step, lay)
    out = [Form("J10-R", "J10 as JOINT-POUW and kernels state it: one SHA-512 stream per call row and per call column, "
                "seeded once a step from the step's roots (h1tile's layout at call scale)", rows0, cols0)]
    out.append(_selection(rows0, cols0, [0] * M, list(range(N)), m, "row-shared",
                          "one row stream for every row of the call (b, z shared by all rows)"))
    out.append(_selection(rows0, cols0, list(range(M)), [0] * N, m, "col-shared",
                          "one column stream for every column (c, U shared by all columns)"))
    out.append(_selection(rows0, cols0, [0] * M, [0] * N, m, "both-shared", "one row and one column stream a call"))
    out.append(_selection(rows0, cols0, [i % 4 for i in range(M)], [j % 4 for j in range(N)], m, "classes-4",
                          "rows and columns in 4 classes each, a stream a class (rank-4 selection)"))
    lab_r = [rng.randrange(8) for _ in range(M)]
    lab_c = [rng.randrange(8) for _ in range(N)]
    out.append(_selection(rows0, cols0, lab_r, lab_c, m, "label-prf-3",
                          "a cheap PRF over (row, column) keyed by a 3-bit row label and a 3-bit column label: any "
                          "F(label, k) has at most 2^3 distinct rows (inner product <x_i, y_k> mod 2 included)"))
    out.append(_selection(rows0, cols0, [i % TILE for i in range(M)], [j % TILE for j in range(N)], m, "unit-periodic",
                          "one unit's noise reused by every unit (period 32 in rows and columns): rank 32 a segment, "
                          "but 32 distinct rows"))

    # outer products: b = f g^T, c = f' g'^T, U = alpha beta^T
    f = np.array([rng.getrandbits(1) for _ in range(M)], dtype=np.int64)
    g = np.array([rng.getrandbits(1) for _ in range(K)], dtype=np.int64)
    f2 = np.array([rng.getrandbits(1) for _ in range(N)], dtype=np.int64)
    g2 = np.array([rng.getrandbits(1) for _ in range(K)], dtype=np.int64)
    al = np.array([rng.choice((-1, 1)) for _ in range(N)], dtype=np.int64)
    be = np.array([rng.randint(-64, 64) for _ in range(TILE)], dtype=np.int64)
    rows = [([int(f[i] & g[k]) for k in range(K)], rows0[i][1]) for i in range(M)]
    cols = [([int(f2[j] & g2[k]) for k in range(K)], [int(al[j] * be[t]) & 0xFF for t in range(TILE)]) for j in range(N)]
    out.append(Form("outer", "outer products: b_ik = f_i g_k, c_jk = f'_j g'_k, U_jt = alpha_j beta_t", rows, cols,
                    row_fac=Fac(f[:, None], lambda s: g[None, 32 * s:32 * s + 32], False),
                    col_fac=Fac(np.stack([f2, al], 1), lambda s: np.stack([g2[32 * s:32 * s + 32], be]), False),
                    u_fac=Fac(al[:, None], lambda s: be[None, :], False)))

    # XOR-separable: b_ik = h_i ^ g_k = h + g - 2 h g (rank 3); U from bytes x_j ^ y_t through u129 (no factor)
    h = np.array([rng.getrandbits(1) for _ in range(M)], dtype=np.int64)
    h2 = np.array([rng.getrandbits(1) for _ in range(N)], dtype=np.int64)
    xb = [rng.getrandbits(8) for _ in range(N)]
    yb = [rng.getrandbits(8) for _ in range(TILE)]
    rows = [([int(h[i] ^ g[k]) for k in range(K)], rows0[i][1]) for i in range(M)]
    cols = [([int(h2[j] ^ g2[k]) for k in range(K)], [_u129(xb[j] ^ yb[t]) for t in range(TILE)]) for j in range(N)]
    one = np.ones(M, dtype=np.int64)
    out.append(Form("xor-sep", "XOR-separable PRF: b_ik = h_i xor g_k, c_jk = h'_j xor g'_k, U_jt = u129(x_j xor y_t); "
                    "the route factors b only (U's rank is about 16 through u129, which a rank route would also use)",
                    rows, cols,
                    row_fac=Fac(np.stack([h, one, h], 1),
                                lambda s: np.stack([np.ones(32, dtype=np.int64), g[32 * s:32 * s + 32],
                                                    -2 * g[32 * s:32 * s + 32]]), False)))

    # a cheap, well-mixed PRF (segment.mix32 in counter mode): no structure a route can use
    k1, k2, k3 = (rng.getrandbits(32) for _ in range(3))
    rows = [([SG.mix32((k1 + i * K + k) & M32) >> 31 for k in range(K)], rows0[i][1]) for i in range(M)]
    cols = [([SG.mix32((k2 + j * K + k) & M32) >> 31 for k in range(K)],
             [_u129(SG.mix32((k3 + j * TILE + t) & M32)) for t in range(TILE)]) for j in range(N)]
    out.append(Form("mix32-prf", "a cheap well-mixed PRF (mix32 counter mode) per (row, k) and (column, k): not SHA-512",
                    rows, cols))

    # the column streams seeded once a window, without the step's A root (the joint-PoUW model's unsound variant)
    seed_win = SD.seed(salt, 7, bytes(64), rootB)
    _r, cols_w = H.noise(seed_win, lay)
    out.append(Form("window-cols", "column streams seeded once a window without root(A): known before A is committed",
                    rows0, cols_w, precoin_cols=True))
    return out, {"salt": salt, "rootA": rootA, "rootB": rootB, "seed_step": seed_step}


# -- the reference words: segment.call_words lane by lane, as h1tile.reference ------------------------------------------

def unit_words(a, b, rows, cols, K: int) -> list[list[list[int]]]:
    """Each lane's absorbed sets (T_0 .. T_(KS-1), O', D_1 .. D_(m-1)) of one 32 × 32 unit, from injected noise."""
    lay = H.Layout(TILE, TILE, K)
    ks, m = lay.ks, lay.m
    out = []
    for ln in range(lay.lanes):
        R, Cs = lay.rows(ln), lay.cols(ln)
        seg = [range(32 * s, 32 * s + 32) for s in range(ks)]
        xr = [[[a[i][t] for t in ts] for i in R] for ts in seg]
        bb = [[[rows[i][0][t] for t in ts] for i in R] for ts in seg]
        wr = [[[b[j][t] for t in ts] for j in Cs] for ts in seg]
        cc = [[[cols[j][0][t] for t in ts] for j in Cs] for ts in seg]
        out.append(SG.call_words(xr, bb, wr, cc, [cols[j][1] for j in Cs],
                                 [[rows[i][1][d] for i in R] for d in range(m - 1)]))
    return out


def receipt(words: list[list[list[int]]], keys: list[list[int]]) -> list[int]:
    """CP-W: each lane folds its sets with fx1 from its keys; word 2 s, 2 s + 1 the XOR over lanes of y and z."""
    cp = [[0, 0] for _ in range(len(words[0]))]
    for ln, sets in enumerate(words):
        st = list(keys[ln])
        for s, p in enumerate(sets):
            st = SG.absorb(H.FOLD, st, p)
            for hh in range(2):
                for v in st[8 * hh:8 * hh + 8]:
                    cp[s][hh] ^= v
    return [w & M32 for s in cp for w in s]


def _unit_slices(form: Form, A, B, I: int, J: int):
    rs, cs = slice(TILE * I, TILE * I + TILE), slice(TILE * J, TILE * J + TILE)
    return A[rs], B[cs], form.rows[rs], form.cols[cs]


# -- the cheater's route --------------------------------------------------------------------------------------------

def _times(fac: Fac | None, s: int, left: np.ndarray | None = None, right: np.ndarray | None = None) -> np.ndarray:
    """``left @ X^(s)T`` (left given) or ``X^(s) @ right`` for X factored as ``fac`` (n × 32 a segment)."""
    if left is not None:
        return (left @ fac.G(s).T) @ fac.F.T
    return fac.F @ (fac.G(s) @ right)


def route(form: Form, A: np.ndarray, B: np.ndarray, K: int, m: int):
    """The cheater's words (T: KS × M × N, O': M × N, D: m - 1 × M × N, mod 2^32) from P_s = A^(s) B^(s)T, which it
    computes before the coin, and the form's factors after it: E_s = A N^T + b B^T + b N^T - 256 (wX (B + N)^T + X wW^T),
    b N^T through both factors where the form has them (`form_costs` counts this route)."""
    M, N = A.shape[0], B.shape[0]
    ks = K // 32
    b, z, c, u = _arrays(form, m)
    Nn = c + np.tile(u, (1, ks))
    X = _sx(A + b)
    wX = (A + b > 127).astype(np.int64)
    BN = B + Nn
    wW = (BN > 127).astype(np.int64) - (BN < -128).astype(np.int64)
    T = np.zeros((ks, M, N), dtype=np.int64)
    acc = np.zeros((M, N), dtype=np.int64)
    rf, cf = form.row_fac, form.col_fac
    for s in range(ks):
        sl = slice(32 * s, 32 * s + 32)
        acc += A[:, sl] @ B[:, sl].T
        if form.precoin_cols:
            Wp = BN[:, sl] - 256 * wW[:, sl]
            acc += A[:, sl] @ (Wp - B[:, sl]).T
            acc += b[:, sl] @ Wp.T - 256 * (wX[:, sl] @ Wp.T)
        else:
            acc += A[:, sl] @ Nn[:, sl].T if cf is None else _times(cf, s, left=A[:, sl])
            acc += b[:, sl] @ B[:, sl].T if rf is None else _times(rf, s, right=B[:, sl].T)
            if rf is not None and cf is not None:
                acc += rf.F @ (rf.G(s) @ cf.G(s).T) @ cf.F.T
            elif rf is not None:
                acc += _times(rf, s, right=Nn[:, sl].T)
            elif cf is not None:
                acc += _times(cf, s, left=b[:, sl])
            else:
                acc += b[:, sl] @ Nn[:, sl].T
            acc -= 256 * (wX[:, sl] @ BN[:, sl].T + X[:, sl] @ wW[:, sl].T)
        T[s] = acc & M32
    Y = sum(X[:, 32 * s:32 * s + 32] for s in range(ks))
    Ad = np.zeros((m, M, 32), dtype=np.int64)
    R = Y.copy()
    for d in range(m - 1):
        q = R >> 7
        zd = np.where((d == m - 2) & (q == 127), 0, z[:, d, :] & 1)
        Ad[d] = (R & 127) | (zd << 7)
        R = q + zd
    Ad[m - 1] = R & 0xFF
    Ads = _sx(Ad)

    def uprod(L):
        return L @ u.T if form.u_fac is None else _times(form.u_fac, 0, left=L)

    Op = (T[-1] - uprod(Ads[0])) & M32
    D = np.stack([uprod(Ads[d]) & M32 for d in range(1, m)]) if m > 1 else np.zeros((0, M, N), dtype=np.int64)
    return T, Op, D


def honest_macs(M: int, N: int, K: int, m: int) -> int:
    return 32 * M * N * (K // 32 + m)


def compare(form: Form, A_l, B_l, T, Op, D, K: int, units: list[tuple[int, int]]):
    """Whether the route's words equal `segment.call_words`' at every lane, entry and set of the listed units."""
    lay = H.Layout(TILE, TILE, K)
    ks = lay.ks
    for I, J in units:
        a, b, rows, cols = _unit_slices(form, A_l, B_l, I, J)
        words = unit_words(a, b, rows, cols, K)
        for ln, sets in enumerate(words):
            Rr, Cs = lay.rows(ln), lay.cols(ln)
            for q in range(32):
                i, j = TILE * I + Rr[SG.row_slot(q)], TILE * J + Cs[SG.col_slot(q)]
                want = [T[s][i, j] for s in range(ks)] + [Op[i, j]] + [D[d][i, j] for d in range(len(D))]
                got = [sets[s][q] & M32 for s in range(len(sets))]
                if [int(w) for w in want] != got:
                    return False, f"unit ({I},{J}) lane {ln} entry {q}: route {want[:3]}.. call_words {got[:3]}.."
    return True, ""


# -- checks ------------------------------------------------------------------------------------------------------------

def _operands(rng, M, N, K):
    A = [[rng.randint(-127, 127) for _ in range(K)] for _ in range(M)]
    B = _synthetic_weights(rng, N, K)
    return A, B


def check_injected_reference_is_h1tile(rng):
    """Words built from injected noise and folded with fx1 give `h1tile.reference`'s receipt, bit for bit."""
    K = 128
    A, B = _operands(rng, TILE, TILE, K)
    salt, pos, roots = _bytes(rng), rng.randint(0, 1 << 20), (_bytes(rng), _bytes(rng))
    lay = H.Layout(TILE, TILE, K)
    rows, cols = H.noise(SD.seed(salt, pos, *roots), lay)
    keys = H.fold_keys(SD.receipt_seed(salt, pos, *roots), lay)
    got = receipt(unit_words(A, B, rows, cols, K), keys)
    want = H.reference(A, B, salt, pos, roots)[1]
    return got == want, f"injected-noise receipt == h1tile.reference's ({len(want)} words, 32x32, K={K})"


def check_routes_bit_exact(rng, M=64, N=64, K=128):
    """Every form's after-coin route gives call_words' words at every unit, lane, entry and set, and so every receipt."""
    fs, _ = forms(M, N, K, rng)
    A_l, B_l = _operands(rng, M, N, K)
    A, B = np.array(A_l, dtype=np.int64), np.array(B_l, dtype=np.int64)
    m = H.Layout(M, N, K).m
    units = [(I, J) for I in range(M // TILE) for J in range(N // TILE)]
    bad = []
    lay = H.Layout(TILE, TILE, K)
    keys = [[rng.getrandbits(32) for _ in range(16)] for _ in range(lay.lanes)]
    for f in fs:
        T, Op, D = route(f, A, B, K, m)
        ok, why = compare(f, A_l, B_l, T, Op, D, K, units)
        if not ok:
            bad.append(f"{f.name}: {why}")
            continue
        sets = list(T) + [Op] + list(D)
        I, J = units[-1]
        routed = [[[int(sets[s][TILE * I + lay.rows(ln)[SG.row_slot(q)], TILE * J + lay.cols(ln)[SG.col_slot(q)]])
                    for q in range(32)] for s in range(lay.sets)] for ln in range(lay.lanes)]
        if receipt(routed, keys) != receipt(unit_words(*_unit_slices(f, A_l, B_l, I, J), K), keys):
            bad.append(f"{f.name}: the route's receipt differs")
    return not bad, "; ".join(bad) or f"{len(fs)} forms' routes bit-exact at {len(units)} units ({M}x{N}, K={K}), receipts too"


def _structure(form: Form, K: int, m: int):
    b, _z, c, u = _arrays(form, m)
    ks = K // 32
    Nn = c + np.tile(u, (1, ks))
    rb = min(_rank_mod_p(b[:, 32 * s:32 * s + 32]) for s in range(ks))
    rn = min(_rank_mod_p(Nn[:, 32 * s:32 * s + 32]) for s in range(ks))
    return {"rank_b": rb, "rank_N": rn, "rank_U": _rank_mod_p(u),
            "distinct_rows": len({tuple(r[0]) for r in form.rows}), "distinct_cols": len({tuple(q[0]) + tuple(q[1]) for q in form.cols})}


def _prod(M: int, N: int, rf, cf) -> int:
    """MACs of ``X^(s) Y^(s)T`` (M × 32 by 32 × N, both coins) by its cheapest route given the factors (rho, sel)."""
    best = 32 * M * N
    if rf:
        best = min(best, 32 * rf[0] * N + (0 if rf[1] else M * rf[0] * N))
    if cf:
        best = min(best, 32 * M * cf[0] + (0 if cf[1] else M * cf[0] * N))
    if rf and cf:
        best = min(best, rf[0] * 32 * cf[0] + (0 if rf[1] else M * rf[0] * cf[0]) + (0 if cf[1] else M * cf[0] * N))
    return best


def _data(n_out: int, other: int, fac) -> int:
    """MACs of committed data (other × 32) by a coin (32 × n_out, factored as ``fac``): a bit-affine word."""
    if not fac:
        return 32 * other * n_out
    return 32 * other * fac[0] + (0 if fac[1] else other * fac[0] * n_out)


def form_costs(shape: dict, M: int, N: int, K: int, m: int, dX: float, dW: float) -> dict:
    """The after-coin operations at (M, N, K) of a form whose factors are ``shape`` (rho and sel of b's, N's and U's
    segment factors, or None), against the honest unit's 32 MN (KS + m) MACs, all of them coin by coin.

    * ``rec``: the record's products (`ChargeVF3j16`, Theorem K's slots), the cheapest route's MACs whose two operands
      both read coins: b N^T a segment and U A_d a digit set; A N^T and b B^T are bit-affine words, free.
    * ``ops``: every after-coin operation of the cheaper of the honest route and the decomposition (A N^T + b (B + N)^T,
      the wrap's sparse adds, a write a credited word a segment, the elementwise X, B + N, masks and digits)."""
    ks = K // 32
    rf, cf, uf = shape["row"], shape["col"], shape["u"]
    honest = 32 * M * N * (ks + m)
    elem = 3 * (M + N) * K + M * K + m * M * 32
    writes = (ks + m) * M * N
    sparse = round(32 * M * N * ks * (dX + dW))
    if shape["precoin"]:
        rec = 0
        dense = (ks + m) * 32 * M * N
    else:
        rec = ks * _prod(M, N, rf, cf) + m * _prod(M, N, None, uf)
        dense = ks * (_data(N, M, cf) + _prod(M, N, rf, None)) + m * _prod(M, N, None, uf)
    ops = min(honest, dense + sparse + writes + elem)
    return {"honest": honest, "rec": rec, "ops": ops, "dense": dense, "sparse": sparse, "writes": writes, "elem": elem}


def _shape(form: Form) -> dict:
    f = lambda fac: (fac.rho, fac.sel) if fac else None  # noqa: E731
    return {"row": f(form.row_fac), "col": f(form.col_fac), "u": f(form.u_fac), "precoin": form.precoin_cols}


def _shares(c: dict) -> dict:
    h = c["honest"]
    alu = (c["sparse"] + c["writes"] + c["elem"]) / h
    return {"skip_ops": 1 - c["ops"] / h, "skip_record": 1 - c["rec"] / h, "dense_share": c["dense"] / h,
            "wrap_share": c["sparse"] / h, "alu_share": alu,
            "r_break": (1 - c["dense"] / h) / alu if c["ops"] < h else 0.0,
            "r_wrap": (1 - (c["dense"] + c["writes"] + c["elem"]) / h) / (c["sparse"] / h) if c["sparse"] and c["ops"] < h
            else 0.0}


def check_forms_verdicts(rng, M=64, N=64, K=128):
    """At a model shape (256 × 4096, K = 4096): J10-R and mix32 keep both the record's count and the op count at the
    honest unit's; every shared or structured form loses the record's count; only forms structured on both sides lose
    the op count; window-cols loses the record's whole count."""
    fs, _ = forms(M, N, K, rng)
    m = H.Layout(M, N, K).m
    sv = {}
    for f in fs:
        sh = _shares(form_costs(_shape(f), 256, 4096, 4096, H.Layout(256, 4096, 4096).m, 1 / 510, 0.00456))
        sv[f.name] = (sh, _structure(f, K, m))
    bad = []
    for name in ("J10-R", "mix32-prf"):
        sh, st = sv[name]
        if sh["skip_record"] > 1e-9 or sh["skip_ops"] > 1e-9 or st["rank_b"] < 32 or st["rank_N"] < 32:
            bad.append(f"{name} {sh} {st}")
    for name in ("row-shared", "col-shared", "both-shared", "classes-4", "label-prf-3", "unit-periodic", "outer", "xor-sep"):
        if sv[name][0]["skip_record"] < 0.5:
            bad.append(f"{name} keeps the record's count: {sv[name][0]}")
    for name in ("both-shared", "classes-4", "label-prf-3", "unit-periodic", "outer"):
        if sv[name][0]["skip_ops"] < 0.5:
            bad.append(f"{name} keeps the op count: {sv[name][0]}")
    for name in ("row-shared", "col-shared"):
        if sv[name][0]["skip_ops"] > 1e-9:
            bad.append(f"{name} skips ops with the other side's noise J10's: {sv[name][0]}")
    if sv["window-cols"][0]["skip_record"] < 0.99:
        bad.append(f"window-cols keeps the record's count: {sv['window-cols'][0]}")
    if sv["unit-periodic"][1]["rank_b"] != 32:
        bad.append("unit-periodic's segment rank isn't 32")
    return not bad, "; ".join(bad) or "J10-R and mix32 keep both counts; shared forms lose the record's, two-sided ones the op count"


def wrap_density(rng, n=1024, k=4096):
    """W' = B + c + U wraps at |B + c + U| > 127 on the synthetic weights; X at A = 127, b = 1 on uniform A."""
    B = np.array(_synthetic_weights(rng, n, k), dtype=np.int64)
    c = np.array([[rng.getrandbits(1) for _ in range(k)] for _ in range(n)], dtype=np.int64)
    U = np.array([[rng.randint(-64, 64) for _ in range(32)] for _ in range(n)], dtype=np.int64)
    BN = B + c + np.tile(U, (1, k // 32))
    big = float(np.mean(np.abs(B) > 63))
    dW = float(np.mean((BN > 127) | (BN < -128)))
    return big, dW, 1 / 510


def check_wrap_does_not_rescue(rng):
    """The wrap correction is (dX + dW) of a contraction in sparse adds: on the synthetic weights about 0.456% + 0.2%, so it
    rescues no structured form in op count; on hardware it rescues only where a sparse add costs ~ 1/(dX + dW) MACs."""
    big, dW, dX = wrap_density(rng, n=256, k=2048)
    ok = 0.035 < big < 0.06 and 0.003 < dW < 0.007
    return ok, f"|B| > 63 on {big:.2%} of the synthetic weights, W' wraps on {dW:.3%}, X on {dX:.3%}"


def check_luck_correlation(rng, S=96, K=128):
    """Units (0, 0) and (0, 1) of a 32 × 64 call: a cheater that leaves out b at one entry of row 0 passes a unit iff that
    bit is 0. Under J10 the two units read the same bit, so their pass events are one event (P[both] = P[one] = ½);
    under per-unit seeds they are independent (P[both] about ¼). Each unit's luck is the same ½ either way."""
    M, N = TILE, 2 * TILE
    lay = H.Layout(M, N, K)
    i0, k0 = 0, 5
    res = {"J10-R": [], "per-unit": []}
    for _ in range(S):
        A = np.array([[rng.randint(-120, 120) for _ in range(K)] for _ in range(M)], dtype=np.int64)
        B = np.array(_synthetic_weights(rng, N, K), dtype=np.int64)
        salt, rootA, rootB = _bytes(rng), _bytes(rng), _bytes(rng)
        rows, cols = H.noise(SD.seed(salt, 3, rootA, rootB), lay)
        per_unit = [H.noise(SD.seed(salt, 3 * 64 + J, rootA, hashlib.sha512(rootB + bytes([J])).digest()),
                            H.Layout(TILE, TILE, K)) for J in range(2)]
        for name in res:
            ev = []
            for J in range(2):
                if name == "J10-R":
                    b = np.array([r[0] for r in rows], dtype=np.int64)
                    cc = np.array([q[0] for q in cols[TILE * J:TILE * J + TILE]], dtype=np.int64)
                    uu = _sx(np.array([q[1] for q in cols[TILE * J:TILE * J + TILE]], dtype=np.int64))
                else:
                    pr, pc = per_unit[J]
                    b = np.array([r[0] for r in pr], dtype=np.int64)
                    cc = np.array([q[0] for q in pc], dtype=np.int64)
                    uu = _sx(np.array([q[1] for q in pc], dtype=np.int64))
                Wp = _sx(B[TILE * J:TILE * J + TILE] + cc + np.tile(uu, (1, K // 32)))
                X = _sx(A + b)
                Xc = X.copy()
                Xc[i0, k0] = A[i0, k0]
                honest = np.cumsum(np.stack([X[:, 32 * s:32 * s + 32] @ Wp[:, 32 * s:32 * s + 32].T
                                             for s in range(K // 32)]), 0)
                cheat = np.cumsum(np.stack([Xc[:, 32 * s:32 * s + 32] @ Wp[:, 32 * s:32 * s + 32].T
                                            for s in range(K // 32)]), 0)
                ev.append(bool(np.array_equal(honest & M32, cheat & M32)))
            res[name].append(ev)
    stats = {}
    for name, evs in res.items():
        p0 = sum(e[0] for e in evs) / S
        p1 = sum(e[1] for e in evs) / S
        both = sum(e[0] and e[1] for e in evs) / S
        same = all(e[0] == e[1] for e in evs)
        stats[name] = (p0, p1, both, same)
    j, u = stats["J10-R"], stats["per-unit"]
    ok = j[3] and not u[3] and 0.3 < j[0] < 0.7 and 0.3 < u[0] < 0.7 and u[2] < 0.4
    return ok, (f"J10-R: P[pass] {j[0]:.2f}/{j[1]:.2f}, P[both] {j[2]:.2f}, events identical {j[3]}; per-unit: "
                f"{u[0]:.2f}/{u[1]:.2f}, P[both] {u[2]:.2f}, identical {u[3]} (S={S})")


# -- generation cost ---------------------------------------------------------------------------------------------------

def gen_cost(M: int, N: int, K: int) -> dict:
    """SHA-512 compressions a 32 × 32 unit: per-unit (h1tile today) and J10-R (streams a call row and column, seeds and
    σ_R a call, 32 key blocks a unit kept), and the joint-PoUW noise-only model of both."""
    lay = H.Layout(TILE, TILE, K)
    rb, cb = lay.rb, lay.cb
    units = M * N / (TILE * TILE)
    per_unit = TILE * rb + TILE * cb + 32 + 4
    j10_noise = (M * rb + N * cb) / units
    j10 = j10_noise + 32 + 4 / units
    return {"M": M, "N": N, "K": K, "rb": rb, "cb": cb, "per_unit": per_unit, "per_unit_noise": TILE * (rb + cb),
            "j10_noise": round(j10_noise, 2), "j10": round(j10, 2), "ratio": round(j10 / per_unit, 4),
            "joint_pouw_ratio": round(16 * (1 / M + 1 / N), 4)}


#: int8 Llama-3.1-8B's linears a layer as (K, N), at 32 layers and 20k tokens/s: the joint-PoUW window model
#: (benchmarks/sampling_plan/joint_pouw.py at 8e436d9a89, `LINEARS`, `CALLS`, `tokens_per_s`)
LLAMA_CALLS = ((4096, 6144), (4096, 4096), (4096, 28672), (14336, 4096))
LLAMA_LAYERS, TOKENS_PER_S = 32, 20_000
#: served_h1's per-unit kernel: a K = 4096 unit's 612 compressions in 89.6 ns
COMPRESSION_S = 89.6e-9 / 612


def serve_b(T: int) -> dict:
    """Hashing time a token over plain time at T tokens a step: per unit, J10-R, and J10-R without σ_R's 32 key blocks a
    unit, the terms the joint-PoUW shared model counts (it reports 2.92 per unit, 0.188 shared at T = 256, 0.736 at
    64)."""
    tot = {"per_unit": 0.0, "j10": 0.0, "j10_noise": 0.0}
    for K, N in LLAMA_CALLS:
        g = gen_cost(T, N, K)
        for key in tot:
            tot[key] += N / 1024 * g[key]
    scale = LLAMA_LAYERS * COMPRESSION_S * TOKENS_PER_S
    return {"T": T, **{k: round(v * scale, 4) for k, v in tot.items()}}


def check_generation_cost():
    """J10-R a unit at K = 4096: 9216 (1/M + 1/N) noise blocks + 32 keys, against 612; at decode (M = 32) half (kernels'
    'halves it'). A token's hashing at T = 256: the joint-PoUW per-unit and shared figures reproduce within 8% from the
    stream counts alone, and the 32 key blocks a unit it leaves out add about 0.126."""
    g256 = gen_cost(256, 4096, 4096)
    g32 = gen_cost(32, 4096, 4096)
    b256, b64 = serve_b(256), serve_b(64)
    near = lambda x, y: abs(x / y - 1) < 0.08  # noqa: E731
    ok = (g256["per_unit"] == 612 and abs(g256["j10_noise"] - 38.25) < 1e-6 and 0.45 < g32["ratio"] < 0.56
          and g256["ratio"] > 1.7 * g256["joint_pouw_ratio"] and near(b256["per_unit"], 2.92)
          and near(b256["j10_noise"], 0.188) and near(b64["j10_noise"], 0.736) and 0.28 < b256["j10"] < 0.33)
    return ok, (f"T=256: J10-R {g256['j10']} a unit ({g256['ratio']:.3f} of 612; noise-only model {g256['joint_pouw_ratio']}); "
                f"T=32: {g32['j10']} ({g32['ratio']:.3f}); a token's hashing over plain: per unit {b256['per_unit']}, "
                f"J10-R {b256['j10']} at T=256 ({b256['j10_noise']} without keys), {b64['j10']} at T=64 "
                f"({b64['j10_noise']})")


CHECKS = [check_injected_reference_is_h1tile, check_routes_bit_exact, check_forms_verdicts, check_wrap_does_not_rescue,
          check_luck_correlation, check_generation_cost]


def collect(seed: int = 20261019, quick: bool = False):
    passed, failed = [], []
    for fn in CHECKS:
        rng = random.Random(f"{seed}:{fn.__name__}")
        args = () if fn is check_generation_cost else (rng,)
        kw = {"S": 32} if quick and fn is check_luck_correlation else {}
        ok, detail = fn(*args, **kw)
        (passed if ok else failed).append((fn.__name__, detail))
    return passed, failed


def report(seed: int = 20261019) -> dict:
    rng = random.Random(f"{seed}:report")
    M, N, K = 64, 64, 128
    fs, _ = forms(M, N, K, rng)
    m = H.Layout(M, N, K).m
    big, dW, dX = wrap_density(rng, n=512, k=4096)
    shapes = [(32, 4096, 4096), (256, 4096, 4096), (256, 14336, 4096), (1024, 4096, 14336), (8192, 4096, 4096)]
    rows = []
    for f in fs:
        st = _structure(f, K, m)
        per = {}
        for (Ms, Ns, Ks) in shapes:
            ms = H.Layout(Ms, Ns, Ks).m
            per[f"{Ms}x{Ns}x{Ks}"] = {k: round(v, 4) for k, v in
                                       _shares(form_costs(_shape(f), Ms, Ns, Ks, ms, dX, dW)).items()}
        rows.append({"form": f.name, "what": f.what, "structure": st, "at_shape": per})
    gens = [gen_cost(Ms, Ns, Ks) for (Ms, Ns, Ks) in shapes + [(64, 4096, 4096), (32, 14336, 4096)]]
    return {"wrap": {"B_over_63": big, "dW": dW, "dX": dX}, "forms": rows, "generation": gens,
            "serve_b": [serve_b(T) for T in (32, 64, 256)]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--out", help="write the report JSON here")
    a = ap.parse_args()
    passed, failed = collect(a.seed)
    for name, d in passed:
        print(f"PASS {name}: {d}")
    for name, d in failed:
        print(f"FAIL {name}: {d}")
    rep = report(a.seed)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"passed": passed, "failed": failed, "report": rep}, fh, indent=1)
    for r in rep["forms"]:
        s = r["at_shape"]["256x4096x4096"]
        print(f"{r['form']:>14}: ranks b/N/U {r['structure']['rank_b']}/{r['structure']['rank_N']}/{r['structure']['rank_U']}, "
              f"distinct rows/cols {r['structure']['distinct_rows']}/{r['structure']['distinct_cols']}; at 256x4096x4096 "
              f"skip ops {s['skip_ops']:.3f}, skip record {s['skip_record']:.3f}, wrap {s['wrap_share']:.4f}, "
              f"R_break {s['r_break']:.1f}, R_wrap {s['r_wrap']:.0f}")
    for g in rep["generation"]:
        print(f"gen {g['M']}x{g['N']}x{g['K']}: per-unit {g['per_unit']}, J10-R {g['j10']} ({g['ratio']:.3f}; "
              f"noise-only {g['joint_pouw_ratio']:.4f})")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
