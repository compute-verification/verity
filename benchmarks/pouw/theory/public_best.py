"""D5 (CMP-125, sprint CMP-135): the best public matmul method's priced cost on protocol v1's operands, c_mm.

Protocol v1 (PROTO-DESIGN, `accel_score/nofold.py` at 1bfece4240; BET-HYBRID `accel_hybrid/largel.py`):

~~~text
coins    live, after A and B are committed: b (m x K), c (n x K) bits; V (m x L) uniform on ±64; U (n x L) uniform on
         ±80; pi_s a fresh permutation of U's columns for each segment s
operands X = wrap8(A + b + V_rep),  W' = wrap8(B + c + U[:, pi_s])        (A, B any int8)
words    T_s = sum_{s' <= s} X_s' W'_s'^T mod 2^32 per cell and segment of L inner indices, every word a SHA-512 leaf
~~~

c_mm is the cheapest public method's priced time for every word over the honest kernel's (dense INT8 MMA, wgmma k32 on
the H200, mma.sync k32 on sm_120). A method's time is the max over θ's resources of its load over the resource's rate
(CALIB's product form: `theta.toml`'s `joint` census shows MMA and the other pipes overlapping). The resources are the
tensor pipe (dense s8, 2:4 sparse s8, f16), issue, the ALU pipe, the FMA pipe (FFMA one slot, IMAD and DP4A two),
shared memory and, on sm_120, register-file reads. A linear program picks, for every class of Strassen leaf, how its
product and its int8 overflow corrections are computed:

- main product: s8 on wrapped forms, f16 (forms of up to 11 bits are exact, FP32 accumulation of a segment is exact
  with overwhelming probability at these magnitudes), two s8 bit planes, DP4A or FFMA beside the tensor pipe;
- each overflow correction 256 e·D (e the sparse excess of a form): 2:4 sparse s8 (leftover groups of 3 or 4 nonzeros
  scattered), two 2:4 layers, or a scatter through shared memory with int16- or int8-packed rows, adds on the ALU or
  the FMA pipe.

Strassen and Winograd come with their isotropy orbit (P, Q, R in GL2(Z) with entries in {-1, 0, 1}, and transposition),
so block pairing and the choice of scheme are the adversary's. Depth d needs the leaf tiles to be MMA shapes and the
output super-tile to fit the register file; leaf tiles set the operand traffic. Data: A = B = 0 is the minimum overflow
over all data (BET-HYBRID-2's `width.py`), so it is the adversary's case; real weights are Gaussian at σ 30.7.

`python public_best.py` prints the table; `--check` runs the exact CPU checks of the words at small sizes.
"""

from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog

# --------------------------------------------------------------------------------------------------------------------
# θ (CALIB, `machine_model/data/devices/sm_90/theta.toml` and `machine_model/data/theta.toml` at 7fb72a68ca), per SM
# per clock. Rates CALIB doesn't list are marked in `assumed`.


@dataclass(frozen=True)
class Card:
    name: str
    sms: int
    s8: float
    sp: float
    f16: float
    issue: float
    alu: float
    fma: float
    smem: float
    rf: float
    mma_rf: float          # RF words a MAC that the MMA reads (mma.sync's A, B and C fragments; 0 for wgmma)
    mma_issue: float       # lane-ops a MAC to issue the MMA and load its register operands
    mac_rate: float        # dense s8 MAC/s (nofold's CARDS)
    sha: float             # our measured SHA-512 compressions/s
    c_op: float            # MAC-times an ALU lane-op (nofold's CARDS)
    acc_words: int         # register words the output super-tile and one leaf accumulator may take
    min_m: int             # the MMA's smallest m
    honest_tile: tuple     # the honest kernel's (m, n) tile
    facets: tuple          # θ's joint census: (a, b) with a·(tensor time) + b·(other warp instructions) ≤ 4 SMSP-cycles
    facet_other: str       # what θ's facet counts as other: every instruction (issue) or vector-pipe ones (vec)
    assumed: tuple = field(default_factory=tuple)

    def leaf_rate(self, nl: int) -> float:
        """MMA rate at leaf width nl over the full rate (wgmma measured at n256 and n32; mma.sync is m16n8)."""
        if self.name != "H200" or nl >= 128:
            return 1.0
        return 2729.9 / 4095.6 if nl <= 32 else (2729.9 / 4095.6 + 1) / 2


H200 = Card(
    name="H200", sms=132, s8=4095.6, sp=8192.0, f16=2047.8, issue=3.998 * 32, alu=63.01, fma=127.9, smem=127.96,
    rf=math.inf, mma_rf=0.0, mma_issue=1 / 2048, mac_rate=1978.9e12 / 2, sha=4.79e9, c_op=64,
    acc_words=48 * 1024, min_m=64, honest_tile=(64, 256),
    # JI90_0: IGMMA.64x256x32 62.741 + every other warp instruction 0.981 SMSP-cycles; an IGMMA is 131072 MACs a SMSP
    facets=((62.741 * 4095.6 / 131072, 0.981),), facet_other="issue",
    assumed=("2:4 sparse s8 at the data sheet's 8192 (3958 TOPS), unmeasured",
             "wgmma at n128 at the measured n256 rate, n64 halfway to n32's measured 2729.9",
             "DP4A at IMAD's 64 lanes, unmeasured on sm_90",
             "48K of the 64K register words a SM hold accumulators",
             "every MMA instruction weighs in JI90_0 in proportion to its tensor time"),
)
SM120 = Card(
    name="RTX PRO 6000", sms=188, s8=1023.8, sp=2048.0, f16=511.9, issue=4.0 * 32, alu=64.0, fma=124.3, smem=127.6,
    rf=253.8, mma_rf=(4 + 2 + 4) * 32 / 4096, mma_issue=1 / 512, mac_rate=1.0e15 / 2, sha=7.13e9, c_op=8,
    acc_words=160 * 32, min_m=16, honest_tile=(64, 64),
    # JI3e_0..2: MMA a + every other vector-pipe warp instruction b SMSP-cycles; an mma.sync k32 is 4096 MACs a SMSP
    facets=tuple((a * 1023.8 / 4096, b) for a, b in ((8.115, 1.0), (15.03, 0.317), (15.539, 0.172))),
    facet_other="vec",
    assumed=("160 accumulator registers a lane (a warp's super-tile)",
             "every MMA instruction weighs in JI3e in proportion to its tensor time"),
)
CARDS = (H200, SM120)

# Protocol v1 (nofold.py's v1_card): HYB at U ±80, V ±64, e 0.75%, L a multiple of 128 (64 for PV).
U_X, U_W = 64, 80
A_COMMIT, ETA, HYB_RHO, HYB_P, E_V1 = 0.17, 0.85, 5, 0.0119 + 0.00071, 0.0075
C_H200 = 2 * ETA * H200.mac_rate / (32 * H200.sha)
STATIC = (("qkv", 16384, 16512), ("o", 16384, 16384), ("up", 16384, 65536), ("down", 65536, 16384))
TOKENS = 4096                      # m: tokens in a batch (the forms on W' amortize over them)


def v1_L(card: Card, K: int, step: int = 128) -> int:
    c = 2 * ETA * card.mac_rate / (32 * card.sha)
    a = 1 + A_COMMIT * c / C_H200 + card.c_op * HYB_P / 2
    q = 1 - HYB_RHO * E_V1
    g = (a + math.sqrt(a * a + 4 * q * c / (E_V1 * K))) / (2 * q)
    return max(step, step * math.floor(E_V1 * g * K / step))


# --------------------------------------------------------------------------------------------------------------------
# Operand entries and Strassen forms, exactly.

INT8 = np.arange(-128, 128)


def wrap8(v):
    return ((v + 128) & 0xFF) - 128


def data_pmf(kind: str, sigma: float = 0.0) -> np.ndarray:
    """pmf over int8 of a data entry: zero, a clipped discrete Gaussian, or uniform."""
    if kind == "zero":
        p = (INT8 == 0).astype(float)
    elif kind == "gauss":
        p = np.exp(-0.5 * (INT8 / sigma) ** 2)
    else:
        p = np.ones(256)
    return p / p.sum()


def entry_pmf(data: np.ndarray, u: int) -> np.ndarray:
    """pmf over int8 of wrap8(d + bit + uniform{-u..u}), d drawn from `data`."""
    out = np.zeros(256)
    for d, pd in zip(INT8, data):
        if pd == 0:
            continue
        for bit in (0, 1):
            vals = wrap8(d + bit + np.arange(-u, u + 1))
            np.add.at(out, vals + 128, pd * 0.5 / (2 * u + 1))
    return out


def form_pmf(entry: np.ndarray, coefs) -> tuple[np.ndarray, int]:
    """pmf of sum_t coef_t Z_t over independent entries; returns (pmf, offset) with value v at index v + offset."""
    pmf, off = np.array([1.0]), 0
    for c in coefs:
        sc = np.zeros(255 * abs(c) + 1)
        sc[(INT8 * c - (-128 * c if c > 0 else 127 * c))] = entry
        pmf = np.convolve(pmf, sc)
        off += -128 * c if c > 0 else 127 * c
    return pmf, -off


@dataclass(frozen=True)
class FormStats:
    terms: int
    q: float           # P(form leaves int8): the density of its excess e
    left24: float      # density of e's nonzeros beyond two in a group of four (2:4's leftover)
    max_abs: int


def form_stats(entry: np.ndarray, coefs) -> FormStats:
    coefs = [c for c in coefs if c]
    pmf, off = form_pmf(entry, coefs)
    v = np.arange(len(pmf)) - off
    q = float(pmf[(v < -128) | (v > 127)].sum())
    left = sum(math.comb(4, j) * q ** j * (1 - q) ** (4 - j) * (j - 2) for j in (3, 4)) / 4
    return FormStats(len(coefs), q, left, int(np.abs(v[pmf > 0]).max()))


# --------------------------------------------------------------------------------------------------------------------
# Bilinear schemes for <2,2,2>: (a_r, b_r, c_r), M_r = <a_r, A blocks> <b_r, B blocks>, C = sum_r c_r M_r.

def _m(rows):
    return np.array(rows, dtype=np.int64)


STRASSEN = [
    (_m([[1, 0], [0, 1]]), _m([[1, 0], [0, 1]]), _m([[1, 0], [0, 1]])),
    (_m([[0, 0], [1, 1]]), _m([[1, 0], [0, 0]]), _m([[0, 0], [1, -1]])),
    (_m([[1, 0], [0, 0]]), _m([[0, 1], [0, -1]]), _m([[0, 1], [0, 1]])),
    (_m([[0, 0], [0, 1]]), _m([[-1, 0], [1, 0]]), _m([[1, 0], [1, 0]])),
    (_m([[1, 1], [0, 0]]), _m([[0, 0], [0, 1]]), _m([[-1, 1], [0, 0]])),
    (_m([[-1, 0], [1, 0]]), _m([[1, 1], [0, 0]]), _m([[0, 0], [0, 1]])),
    (_m([[0, 1], [0, -1]]), _m([[0, 0], [1, 1]]), _m([[1, 0], [0, 0]])),
]
WINOGRAD = [
    (_m([[1, 0], [0, 0]]), _m([[1, 0], [0, 0]]), _m([[1, 1], [1, 1]])),
    (_m([[0, 1], [0, 0]]), _m([[0, 0], [1, 0]]), _m([[1, 0], [0, 0]])),
    (_m([[1, 1], [-1, -1]]), _m([[0, 0], [0, 1]]), _m([[0, 1], [0, 0]])),
    (_m([[0, 0], [0, 1]]), _m([[1, -1], [-1, 1]]), _m([[0, 0], [-1, 0]])),
    (_m([[0, 0], [1, 1]]), _m([[-1, 1], [0, 0]]), _m([[0, 1], [0, 1]])),
    (_m([[-1, 0], [1, 1]]), _m([[1, -1], [0, 1]]), _m([[0, 1], [1, 1]])),
    (_m([[1, 0], [-1, 0]]), _m([[0, -1], [0, 1]]), _m([[0, 0], [1, 1]])),
]


def is_valid(scheme) -> bool:
    """The scheme's tensor equals <2,2,2>'s: sum_r a_r[i,k] b_r[k',j] c_r[i',j'] = [i=i'][k=k'][j=j']."""
    t = np.zeros((2, 2, 2, 2, 2, 2), dtype=np.int64)
    for a, b, c in scheme:
        t += np.einsum("ik,lj,pq->ikljpq", a, b, c)
    want = np.zeros_like(t)
    for i, k, j in itertools.product(range(2), repeat=3):
        want[i, k, k, j, i, j] = 1
    return bool(np.array_equal(t, want))


def _gl2():
    out = []
    for e in itertools.product((-1, 0, 1), repeat=4):
        g = _m([e[:2], e[2:]])
        if abs(round(np.linalg.det(g))) == 1:
            out.append(g)
    return out


def _inv(g):
    return _m([[g[1, 1], -g[0, 1]], [-g[1, 0], g[0, 0]]]) * int(round(np.linalg.det(g)))


def transform(scheme, P, Q, R):
    """The scheme for C = A B obtained from (P^-1 A Q)(Q^-1 B R) = P^-1 C R."""
    Pi, Qi, Ri = _inv(P), _inv(Q), _inv(R)
    return [(Pi.T @ a @ Q.T, Qi.T @ b @ R.T, P @ c @ Ri) for a, b, c in scheme]


def transpose(scheme):
    """C^T = B^T A^T: the sides swap, so W' forms become A-side forms."""
    return [(b.T, a.T, c.T) for a, b, c in scheme]


def orbit(limit_monomial: bool = True):
    """Distinct valid schemes reachable from Strassen and Winograd (by their per-side coefficient profiles)."""
    gl = _gl2()
    if limit_monomial:
        mono = [g for g in gl if (g != 0).sum() == 2]
        reps, seen = [], set()
        for g in gl:
            key = min(tuple((m @ g).ravel()) for m in mono)
            if key not in seen:
                seen.add(key)
                reps.append(g)
        gl = reps
    out = {}
    for base in (STRASSEN, WINOGRAD):
        for P, Q, R in itertools.product(gl, repeat=3):
            for s in (transform(base, P, Q, R), transpose(transform(base, P, Q, R))):
                key = (tuple(sorted(tuple(int(x) for x in a.ravel()) for a, _, _ in s)),
                       tuple(sorted(tuple(int(x) for x in b.ravel()) for _, b, _ in s)))
                if key not in out and max(int(np.abs(np.concatenate([a.ravel(), b.ravel()])).max()) for a, b, _ in s) <= 1:
                    out[key] = s
    return [s for s in out.values() if is_valid(s)]


def kron(outer, inner):
    """Depth-2 scheme on a 4x4 block grid: leaf (r1, r2) has forms a_r1 (x) a_r2, flattened per side."""
    return [(np.kron(a1, a2), np.kron(b1, b2), np.kron(c1, c2)) for a1, b1, c1 in outer for a2, b2, c2 in inner]


def power(scheme, d: int):
    s = scheme
    for _ in range(d - 1):
        s = kron(s, scheme)
    return s


# --------------------------------------------------------------------------------------------------------------------
# Pricing: per naive MAC, loads on each resource; the time is the max of load / rate.

RES = ("tensor", "issue", "alu", "fma", "smem", "rf", "vec")


def cap(card: Card) -> dict:
    return dict(tensor=1.0, issue=card.issue, alu=card.alu, fma=card.fma, smem=card.smem, rf=card.rf, vec=math.inf)


def scatter_loads(variant: str) -> dict:
    """Loads a contribution (one nonzero of e times one row of the dense side, added into its output).

    Rows are kept packed in shared memory: int16 pairs (one LDS.32 brings two rows) or int8 quads (one LDS.32 brings
    four rows, one PRMT spreads two of them into an int16 pair). Adds: IADD3 on the ALU (two packed words an op) or
    IMAD on the FMA pipe (one packed word an op, two slots). `vec` counts vector-pipe lane-ops (θ's sm_120 facets)."""
    return {
        "s16-alu": dict(smem=2.0, issue=0.5 + 0.25, alu=0.25, rf=0.5 + 0.75, vec=0.25),
        "s16-fma": dict(smem=2.0, issue=0.5 + 0.5, fma=1.0, rf=0.5 + 1.5, vec=0.5),
        "s8-alu": dict(smem=1.0, issue=0.25 + 0.5 + 0.25, alu=0.5 + 0.25, rf=0.25 + 1.0 + 0.75, vec=0.75),
        "s8-fma": dict(smem=1.0, issue=0.25 + 0.5 + 0.5, alu=0.5, fma=1.0, rf=0.25 + 1.0 + 1.5, vec=1.0),
    }[variant]


SCATTER = ("s16-alu", "s16-fma", "s8-alu", "s8-fma")
BESIDE_TILE = 8                     # DP4A/FFMA register tile (t x t outputs a lane)


@dataclass
class Leaf:
    weight: float      # leaf MACs over naive MACs
    A: FormStats
    B: FormStats
    nnz_c: int


def leaf_tile(card: Card, d: int):
    """The leaf tile (m_l, n_l, rate) for depth d: the output super-tile (2^d m_l x 2^d n_l) plus one leaf accumulator
    must fit the register budget, and m_l the MMA's smallest m. Highest rate first, then least operand traffic."""
    if d == 0:
        return card.honest_tile + (1.0,)
    best = None
    for ml in (16, 32, 64, 128, 256):
        for nl in (8, 16, 32, 64, 128, 256):
            if ml < card.min_m or (4 ** d + 1) * ml * nl > card.acc_words:
                continue
            key = (card.leaf_rate(nl), -(1 / ml + 1 / nl))
            if best is None or key > best[0]:
                best = (key, ml, nl, card.leaf_rate(nl))
    return None if best is None else best[1:]


@dataclass
class Priced:
    c: float
    binding: list
    loads: dict
    routes: dict
    feasible: bool = True


def price(card: Card, leaves: list, L: int, n: int, m: int, d: int, beside: bool = True,
          wgmma_smem: bool = True, facets: bool = True, items: tuple = (), extra_fixed: dict | None = None) -> Priced:
    """LP: minimize T (SM-clocks a naive MAC) with every resource's load at most T times its rate, and θ's joint
    facets a·tensor + b·other ≤ 4T. c = T over the honest kernel's 1/s8.

    wgmma_smem: wgmma's operand reads (B, and A when it isn't in registers) are charged to the 128 B/clk shared-memory
    port; False charges only the LDS that brings A to registers (the H200 only; θ doesn't measure it).
    items: more work a naive MAC, each (name, weight, {route: loads a unit}), one convex mix of routes an item.
    extra_fixed: loads a naive MAC with no choice of route (the hash)."""
    tile = leaf_tile(card, d)
    if tile is None:
        return Priced(math.inf, ["no tile"], {}, {}, False)
    ml, nl, rate = tile
    smem_mma = (1 / ml + 1 / nl) if (wgmma_smem or card is not H200) else 1 / nl
    tens = 1 / (card.s8 * rate)
    names, cols, eqs = [], [], []

    def var(name, loads):
        names.append(name)
        cols.append([loads.get(r, 0.0) for r in RES])

    mma = dict(smem=smem_mma, rf=card.mma_rf, issue=card.mma_issue)
    t = BESIDE_TILE
    route_loads = {
        "s8": dict(mma, tensor=tens),
        "f16": {**{k: 2 * v for k, v in mma.items()}, "tensor": 1 / (card.f16 * rate)},
        "split": {**{k: 2 * v for k, v in mma.items()}, "tensor": 2 * tens},
        "dp4a": dict(fma=0.5, issue=0.25 + 1 / (8 * t), smem=2 / t, rf=0.75, vec=0.25),
        "ffma": dict(fma=1.0, issue=1 + 1 / (2 * t), smem=8 / t, rf=3.0, vec=1.0),
    }
    main = ["s8", "f16", "split"] + (["dp4a", "ffma"] if beside else [])
    sc0 = scatter_loads("s16-alu")
    sp = dict(mma, tensor=1 / (card.sp * rate))
    for k, lf in enumerate(leaves):
        mrow = {}
        for r in main:
            if r == "f16" and max(lf.A.max_abs, lf.B.max_abs) > 2048:
                continue
            ld = dict(route_loads[r])
            if r in ("s8", "dp4a") and lf.A.q > 0 and lf.B.q > 0:
                for res_, x in sc0.items():         # the double term 65536 eA eB^T, scattered
                    ld[res_] = ld.get(res_, 0.0) + lf.A.q * lf.B.q * x
            var(f"{k}:{r}", ld)
            mrow[f"{k}:{r}"] = 1.0
        eqs.append((mrow, lf.weight))
        needs = [f"{k}:s8"] + ([f"{k}:dp4a"] if beside else [])
        for side, fs in (("A", lf.A), ("B", lf.B)):
            if fs.q <= 0:
                continue
            row = {nm: -1.0 for nm in needs}
            var(f"{k}:{side}:2:4", {r: sp.get(r, 0) + fs.left24 * sc0.get(r, 0) for r in RES})
            var(f"{k}:{side}:2:4x2", {r: 2 * sp.get(r, 0) for r in RES})
            for v in SCATTER:
                var(f"{k}:{side}:{v}", {r: fs.q * x for r, x in scatter_loads(v).items()})
            for nm in ("2:4", "2:4x2") + SCATTER:
                row[f"{k}:{side}:{nm}"] = 1.0
            eqs.append((row, 0.0))
    for name, w, routes_ in items:
        row = {}
        for r, ld in routes_.items():
            var(f"{name}:{r}", ld)
            row[f"{name}:{r}"] = 1.0
        eqs.append((row, w))
    # Fixed loads a naive MAC: every word stored for the hash (STS, both kernels), Strassen's output assembly in
    # registers (one IADD a leaf entry it enters) and the forms (amortized over the other operand's rows).
    fixed = dict(issue=1 / L, smem=4 / L)
    for r, v in (extra_fixed or {}).items():
        fixed[r] = fixed.get(r, 0.0) + v
    if d > 0:
        adds = sum(lf.nnz_c for lf in leaves) / 4 ** d / L
        forms = sum(lf.weight * 8 ** d / 4 ** d * ((lf.A.terms - 1 + 2 * (lf.A.terms > 1)) / n
                                                   + (lf.B.terms - 1 + 2 * (lf.B.terms > 1)) / m) for lf in leaves)
        for r, v in (("alu", adds + forms), ("rf", 3 * (adds + forms)), ("vec", adds + forms), ("issue", adds + forms)):
            fixed[r] = fixed.get(r, 0.0) + v
    caps = cap(card)
    nv = len(names)
    ix = {r: i for i, r in enumerate(RES)}
    A_ub, b_ub = [], []
    for r in RES:
        if not math.isinf(caps[r]):
            A_ub.append([-caps[r]] + [cols[j][ix[r]] for j in range(nv)])
            b_ub.append(-fixed.get(r, 0.0))
    other = card.facet_other
    if facets:
        for a, b in card.facets:
            A_ub.append([-4.0] + [a * cols[j][ix["tensor"]] + b * cols[j][ix[other]] / 32 for j in range(nv)])
            b_ub.append(-(b * fixed.get(other, 0.0) / 32))
    A_eq = [[0.0] + [row.get(nm, 0.0) for nm in names] for row, _ in eqs]
    b_eq = [rhs for _, rhs in eqs]
    res = linprog([1.0] + [0.0] * nv, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * (nv + 1),
                  method="highs")
    assert res.status == 0, res.message
    T, x = res.x[0], res.x[1:]
    tot = {r: sum(cols[j][ix[r]] * x[j] for j in range(nv)) + fixed.get(r, 0.0) for r in RES}
    loads = {r: tot[r] / caps[r] * card.s8 for r in RES if not math.isinf(caps[r])}
    if facets:
        for i, (a, b) in enumerate(card.facets):
            loads[f"joint{i}"] = (a * tot["tensor"] + b * tot[other] / 32) / 4 * card.s8
    binding = [r for r, v in loads.items() if v >= T * card.s8 * (1 - 1e-6)]
    routes = {}
    for nm, v in zip(names, x):
        if v > 1e-9:
            key = nm.split(":", 1)[1]
            routes[key] = routes.get(key, 0.0) + v
    return Priced(T * card.s8, binding, loads, routes)


_STATS: dict = {}


def _stats(side: str, entry: np.ndarray, coefs: tuple) -> FormStats:
    key = (side, entry.tobytes(), coefs)
    if key not in _STATS:
        _STATS[key] = form_stats(entry, coefs)
    return _STATS[key]


def leaves_of(scheme, d: int, eX: np.ndarray, eW: np.ndarray) -> list:
    s = power(scheme, d) if d > 0 else [(np.ones((1, 1), np.int64),) * 3]
    out = []
    for a, b, c in s:
        ka = tuple(int(x) for x in sorted(a.ravel()[a.ravel() != 0]))
        kb = tuple(int(x) for x in sorted(b.ravel()[b.ravel() != 0]))
        out.append(Leaf(1 / 8 ** d, _stats("A", eX, ka), _stats("B", eW, kb), int((c != 0).sum())))
    return out


def profile(scheme) -> str:
    """Term counts of the X-side and W'-side forms, e.g. 'X 1112224 / W' 1112234'."""
    ta = "".join(str(int((a != 0).sum())) for a in sorted((a for a, _, _ in scheme), key=lambda a: (a != 0).sum()))
    tb = "".join(str(int((b != 0).sum())) for b in sorted((b for _, b, _ in scheme), key=lambda b: (b != 0).sum()))
    return f"X {ta} / W' {tb}"


def best_scheme(card: Card, d: int, eX, eW, L: int, n: int, m: int, schemes, **kw):
    """The cheapest scheme of the orbit at depth d (the same scheme at every level): (Priced, scheme)."""
    best, seen = None, set()
    for s in schemes:
        lv = leaves_of(s, d, eX, eW)
        key = tuple(sorted((lf.A.terms, lf.A.q, lf.B.terms, lf.B.q, lf.nnz_c) for lf in lv))
        if key in seen:
            continue
        seen.add(key)
        p = price(card, lv, L, n, m, d, **kw)
        if best is None or p.c < best[0].c:
            best = (p, s)
    return best


# --------------------------------------------------------------------------------------------------------------------
# Exact CPU checks of the words (small sizes).

def v1_operands(A, B, L: int, rng, uX: int = U_X, uW: int = U_W, permute: bool = True):
    m, K = A.shape
    n, S = B.shape[0], K // L
    b, c = rng.integers(0, 2, (m, K)), rng.integers(0, 2, (n, K))
    V, U = rng.integers(-uX, uX + 1, (m, L)), rng.integers(-uW, uW + 1, (n, L))
    perms = [rng.permutation(L) if permute else np.arange(L) for _ in range(S)]
    Up = np.concatenate([U[:, p] for p in perms], axis=1)
    return wrap8(A + b + np.tile(V, (1, S))), wrap8(B + c + Up)


def honest_words(X, W, L: int) -> np.ndarray:
    S = X.shape[1] // L
    seg = np.stack([X[:, s * L:(s + 1) * L] @ W[:, s * L:(s + 1) * L].T for s in range(S)])
    return np.cumsum(seg, axis=0) & 0xFFFFFFFF


@dataclass
class Count:
    s8: int = 0
    sp: int = 0
    left: int = 0
    scatter_q: list = field(default_factory=list)
    f16_exact: bool = True


def _split24(e: np.ndarray):
    """e = e24 + rest with at most two nonzeros of e24 in each group of four along the inner axis (axis 1)."""
    r, k = e.shape
    g = e.reshape(r, k // 4, 4)
    nz = np.cumsum(g != 0, axis=2)
    keep = (g != 0) & (nz <= 2)
    e24 = np.where(keep, g, 0).reshape(r, k)
    return e24, e - e24


def leaf_product(Xf, Wf, cnt: Count, fp: bool = False):
    """Xf (m x k) forms, Wf (n x k) forms, by s8 on the wrapped forms plus 2:4 corrections (and leftovers)."""
    if fp:
        out = (Xf.astype(np.float32) @ Wf.astype(np.float32).T).astype(np.float64)
        exact = Xf @ Wf.T
        cnt.f16_exact &= bool(np.array_equal(out.astype(np.int64), exact)) and max(np.abs(Xf).max(), np.abs(Wf).max()) <= 2048
        return exact
    Xw, Ww = wrap8(Xf), wrap8(Wf)
    eX, eW = (Xf - Xw) // 256, (Wf - Ww) // 256
    m, k = Xf.shape
    n = Wf.shape[0]
    cnt.s8 += m * k * n
    out = Xw @ Ww.T
    for e, other, side in ((eX, Ww, "A"), (eW, Xw, "B")):
        if not e.any():
            continue
        e24, rest = _split24(e)
        cnt.sp += m * k * n
        cnt.left += int((rest != 0).sum()) * (n if side == "A" else m)
        cnt.scatter_q.append((side, float((e != 0).mean())))
        corr = e24 @ other.T + rest @ other.T
        out = out + 256 * (corr if side == "A" else corr.T)
    out = out + 65536 * (eX @ eW.T)
    return out


def scheme_product(Xs, Ws, scheme, d: int, cnt: Count, fp: bool = False):
    """Xs (m x k) times Ws^T (k x n) through the depth-d scheme, every leaf by `leaf_product`."""
    if d == 0:
        return leaf_product(Xs, Ws, cnt, fp)
    m, k = Xs.shape
    n = Ws.shape[0]
    Ab = [[Xs[i * m // 2:(i + 1) * m // 2, j * k // 2:(j + 1) * k // 2] for j in range(2)] for i in range(2)]
    # B = Ws^T (k x n): block B[p][q] is rows p of k, cols q of n, stored transposed as Ws[q-block, p-block]
    Bb = [[Ws[q * n // 2:(q + 1) * n // 2, p * k // 2:(p + 1) * k // 2] for q in range(2)] for p in range(2)]
    C = [[None, None], [None, None]]
    for a, b, c in scheme:
        Af = sum(a[i, j] * Ab[i][j] for i in range(2) for j in range(2) if a[i, j])
        Bf = sum(b[p, q] * Bb[p][q] for p in range(2) for q in range(2) if b[p, q])
        M = scheme_product(Af, Bf, scheme, d - 1, cnt, fp)
        for i, j in itertools.product(range(2), repeat=2):
            if c[i, j]:
                C[i][j] = c[i, j] * M if C[i][j] is None else C[i][j] + c[i, j] * M
    return np.block(C)


def check_words(scheme, d: int, data: str, m: int = 32, n: int = 32, L: int = 128, S: int = 3, seed: int = 1,
                fp: bool = False):
    """Every word by the depth-d scheme (int8 wrapped forms, 2:4 corrections and leftovers) equals the honest word."""
    rng = np.random.default_rng(seed)
    if data == "zero":
        A, B = np.zeros((m, S * L), np.int64), np.zeros((n, S * L), np.int64)
    else:
        A = np.clip(np.rint(rng.normal(0, 20, (m, S * L))), -128, 127).astype(np.int64)
        B = np.clip(np.rint(rng.normal(0, 30.7, (n, S * L))), -128, 127).astype(np.int64)
    X, W = v1_operands(A, B, L, rng)
    want = honest_words(X, W, L)
    cnt = Count()
    acc = np.zeros((m, n), np.int64)
    for s in range(S):
        acc = acc + scheme_product(X[:, s * L:(s + 1) * L], W[:, s * L:(s + 1) * L], scheme, d, cnt, fp)
        if not np.array_equal(acc & 0xFFFFFFFF, want[s]):
            return False, cnt
    return True, cnt


def check_packing_invariance(L: int = 256, S: int = 4, n: int = 64, seed: int = 2):
    """HY1's packing needs the W'-side wrap pattern of a Strassen form to repeat across segments when B is periodic.
    Returns the share of form overflows that repeat in the next segment, without and with v1's permutation of U."""
    rng = np.random.default_rng(seed)
    B0 = rng.integers(-64, 64, (n, L))
    out = []
    for permute in (False, True):
        B = np.tile(B0, (1, S))
        _, W = v1_operands(np.zeros((1, S * L), np.int64), B, L, rng, permute=permute)
        h = L // 2
        e = [((W[:, s * L:s * L + h] + W[:, s * L + h:(s + 1) * L]) > 127) for s in range(S)]
        rep = sum(int((e[s] & e[s + 1]).sum()) for s in range(S - 1))
        tot = sum(int(e[s].sum()) for s in range(S - 1))
        out.append(rep / max(tot, 1))
    return tuple(out)


# --------------------------------------------------------------------------------------------------------------------
# Report.

def operand_cases():
    return {
        "A=B=0": (entry_pmf(data_pmf("zero"), U_X), entry_pmf(data_pmf("zero"), U_W)),
        "real (A σ20, B σ30.7)": (entry_pmf(data_pmf("gauss", 20.0), U_X), entry_pmf(data_pmf("gauss", 30.7), U_W)),
    }


def classes(card: Card):
    rows = [(nm, K, N, v1_L(card, K)) for nm, K, N in STATIC]
    for T in (2048, 8192, 32768):
        rows.append((f"PV T{T}", T, 128, v1_L(card, T, 64)))
    return rows


# SHA-512 compressions/s: the fastest public implementation we can cite per card, with its source.
SHA_PUBLIC = {
    "H200": (4.79e9, "ours (CMP-107), placeholder"),
    "RTX PRO 6000": (7.13e9, "ours (r20261016-110413-c5d1), placeholder"),
}


# --------------------------------------------------------------------------------------------------------------------
# The whole job a naive MAC: words, the peels (HYB_RHO products of inner size L, L/K a MAC each), the wrap corrections
# (nofold's c_op·p/2), A's commitments and every word's SHA-512 leaf (two compressions per 32 words).

def hash_loads(card: Card, L: int, r_sha: float = 1.0) -> dict:
    """SHA-512 as the ALU-bound kernel we measured: card.sha compressions/s saturate the ALU at the clock that
    nofold's MAC rate implies."""
    per_clk = card.sha * card.s8 / card.mac_rate * r_sha          # compressions a SM-clock
    comps = 1 / (16 * L) + A_COMMIT * H200.sha / (ETA * H200.mac_rate)
    ops = comps * card.alu / per_clk
    return dict(alu=ops, issue=ops, vec=ops)


def job_items(card: Card, L: int, K: int, honest: bool, beside: bool) -> tuple:
    tens = 1 / card.s8
    mma = dict(smem=1 / card.honest_tile[0] + 1 / card.honest_tile[1], rf=card.mma_rf, issue=card.mma_issue)
    wraps = {"alu": dict(alu=HYB_P / 2, issue=HYB_P / 2, vec=HYB_P / 2)}
    peel = {"s8": dict(mma, tensor=tens)}
    if not honest:
        sp = dict(mma, tensor=1 / card.sp)
        wraps["2:4 W, alu X"] = {**sp, "alu": 0.00071 / 2, "issue": sp["issue"] + 0.00071 / 2, "vec": 0.00071 / 2}
        wraps["2:4 both"] = {k: 2 * v for k, v in sp.items()}
        for v in SCATTER:
            wraps[v] = {r: HYB_P * x for r, x in scatter_loads(v).items()}
        if beside:
            peel["dp4a"] = dict(fma=0.5, issue=0.25 + 1 / 64, smem=0.25, rf=0.75, vec=0.25)
    return (("wraps", 1.0, wraps), ("peels", HYB_RHO * L / K, peel))


def job(card: Card, L: int, K: int, n: int, eX, eW, schemes, r_sha: float = 1.0, wgmma_smem: bool = True) -> dict:
    """Honest additive γ (nofold's sum, at θ's rates), honest overlapped (θ's max with facets), and the best public
    overlapped schedule (depth 0 to 2, beside routes, wraps on any route, SHA-512 r_sha times faster)."""
    h = hash_loads(card, L)
    add = 1 + HYB_RHO * L / K + (h["alu"] + HYB_P / 2) / card.alu * card.s8
    kw = dict(wgmma_smem=wgmma_smem)
    hon = price(card, leaves_of(None, 0, eX, eW), L, n, TOKENS, 0, beside=False, items=job_items(card, L, K, True, False),
                extra_fixed=h, **kw)
    hr = hash_loads(card, L, r_sha)
    best = ("d0", price(card, leaves_of(None, 0, eX, eW), L, n, TOKENS, 0, beside=True,
                        items=job_items(card, L, K, False, True), extra_fixed=hr, **kw))
    for d in (1, 2):
        r = best_scheme(card, d, eX, eW, L, n, TOKENS, schemes, beside=True, items=job_items(card, L, K, False, True),
                        extra_fixed=hr, **kw)
        if r and r[0].c < best[1].c:
            best = (f"d{d}", r[0])
    return dict(additive=add, honest=hon, best=best, hash_share=h["alu"] / card.alu * card.s8 / add)


def job_report(schemes, r_pub: dict) -> dict:
    out = {}
    eX, eW = operand_cases()["A=B=0"]
    print("\n== the whole job a naive MAC (words, peels, wraps, A's and the words' SHA-512), MAC-times at θ's s8 rate")
    for card in CARDS:
        for r_name, r in (("ours", 1.0), ("public best", r_pub[card.name])):
            print(f"  {card.name}, SHA-512 {r_name} ({r:.2f}x ours)")
            print(f"    {'class':10} {'L':>5} | {'additive':>8} {'hash':>5} | {'honest overlapped':>24} | "
                  f"{'best public overlapped':>28} | eps if credit additive, overlapped")
            for nm, K, N, L in classes(card):
                j = job(card, L, K, N, eX, eW, schemes, r)
                hon, (lab, best) = j["honest"], j["best"]
                out[(card.name, r_name, nm)] = j
                print(f"    {nm:10} {L:5} | {j['additive']:8.3f} {j['hash_share']:5.2f} | {hon.c:7.3f} "
                      f"{','.join(hon.binding)[:16]:>16} | {best.c:7.3f} {lab} {','.join(best.binding)[:17]:>17} | "
                      f"{1 - best.c / j['additive']:6.1%} {1 - best.c / hon.c:6.1%}")
    return out


def variants(card: Card):
    out = [("tensor", dict(beside=False)), ("beside", dict(beside=True))]
    if card is H200:
        out.append(("beside, wgmma reads free", dict(beside=True, wgmma_smem=False)))
    return out


def report(schemes, depths=(1, 2)) -> dict:
    """c_mm per data case, card, matmul class and pricing variant: the minimum over depth 0 (naive, with beside routes
    if allowed) and Strassen depths. The scheme is chosen once per (case, card, variant, depth) at qkv's L and reused
    for the other classes, which differ only in L and n (1/L and 1/n terms)."""
    out = {}
    for case, (eX, eW) in operand_cases().items():
        print(f"\n== operands {case}: int8 overflow of two-term forms X {form_stats(eX, (1, 1)).q:.4f}, "
              f"W' {form_stats(eW, (1, 1)).q:.4f}; three-term {form_stats(eX, (1, 1, 1)).q:.4f}, "
              f"{form_stats(eW, (1, 1, 1)).q:.4f}; four-term {form_stats(eX, (1, 1, 1, 1)).q:.4f}, "
              f"{form_stats(eW, (1, 1, 1, 1)).q:.4f}")
        for card in CARDS:
            rows = classes(card)
            _, _, N0, L0 = rows[0]
            vs = variants(card)
            chosen = {}
            print(f"  {card.name}")
            for vname, kw in vs:
                for d in depths:
                    chosen[vname, d] = best_scheme(card, d, eX, eW, L0, N0, TOKENS, schemes, **kw)
                    r = chosen[vname, d]
                    if r:
                        p = r[0]
                        top = ", ".join(f"{k} {v:.3f}" for k, v in sorted(p.routes.items(), key=lambda kv: -kv[1])[:6])
                        print(f"    [{vname}] depth {d}: c {p.c:.3f} bound by {','.join(p.binding)}; {profile(r[1])}; "
                              f"leaf {leaf_tile(card, d)[:2]}; at qkv {top}")
            print(f"    {'class':10} {'K':>6} {'L':>5} | " + " | ".join(f"{v:>26}" for v, _ in vs))
            for nm, K, N, L in rows:
                cells = []
                for vname, kw in vs:
                    cand = [("d0", price(card, leaves_of(None, 0, eX, eW), L, N, TOKENS, 0, **kw))]
                    for d in depths:
                        if chosen[vname, d]:
                            cand.append((f"d{d}", price(card, leaves_of(chosen[vname, d][1], d, eX, eW), L, N, TOKENS,
                                                         d, **kw)))
                    lab, p = min(cand, key=lambda t: t[1].c)
                    out[(case, card.name, vname, nm)] = (lab, p)
                    cells.append(f"{p.c:6.3f} {lab} {','.join(p.binding)[:15]:>15}")
                print(f"    {nm:10} {K:6} {L:5} | " + " | ".join(f"{c:>26}" for c in cells))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--job-only", action="store_true")
    args = ap.parse_args()
    if args.check:
        for nm, sch in (("strassen", STRASSEN), ("winograd", WINOGRAD)):
            for d in (1, 2):
                for data in ("zero", "real"):
                    ok, cnt = check_words(sch, d, data)
                    print(f"{nm} depth {d} {data}: words exact {ok}; s8 MACs {cnt.s8}, 2:4 MACs {cnt.sp}, "
                          f"leftover contributions {cnt.left}")
        print("packing invariance (no permutation, v1's permutation):", check_packing_invariance())
        return
    schemes = orbit()
    print(f"orbit: {len(schemes)} valid schemes (Strassen, Winograd, GL2(Z) images mod monomials, transposes)")
    if not args.job_only:
        report(schemes)
    job_report(schemes, {c.name: SHA_PUBLIC[c.name][0] / c.sha for c in CARDS})


if __name__ == "__main__":
    main()
