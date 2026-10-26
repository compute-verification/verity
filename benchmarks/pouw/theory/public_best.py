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
import re
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path

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
    family: str = "sm_120"
    assumed: tuple = field(default_factory=tuple)

    def leaf_rate(self, nl: int) -> float:
        """MMA rate at leaf width nl over the full rate (wgmma measured at n256 and n32; mma.sync is m16n8)."""
        if self.family != "sm_90" or nl >= 128:
            return 1.0
        return 2729.9 / 4095.6 if nl <= 32 else (2729.9 / 4095.6 + 1) / 2


H200 = Card(
    name="H200", sms=132, s8=4095.6, sp=8192.0, f16=2047.8, issue=3.998 * 32, alu=63.01, fma=127.9, smem=127.96,
    rf=math.inf, mma_rf=0.0, mma_issue=1 / 2048, mac_rate=1978.9e12 / 2, sha=4.79e9, c_op=64,
    acc_words=48 * 1024, min_m=64, honest_tile=(64, 256),
    # JI90_0: IGMMA.64x256x32 62.741 + every other warp instruction 0.981 SMSP-cycles; an IGMMA is 131072 MACs a SMSP
    facets=((62.741 * 4095.6 / 131072, 0.981),), facet_other="issue", family="sm_90",
    assumed=("2:4 sparse s8 at the data sheet's 8192 (3958 TOPS), unmeasured",
             "wgmma at n128 at the measured n256 rate, n64 halfway to n32's measured 2729.9",
             "DP4A at IMAD's 64 lanes, unmeasured on sm_90",
             "48K of the 64K register words a SM hold accumulators",
             "every MMA instruction weighs in JI90_0 in proportion to its tensor time"),
)
# RT-MODEL M1/M2 (note 20261026T0310Z-draft-accel-rt-model): the H200 is power-limited, so over a long window its
# tensor pipe holds 499,712 MACs a tick (the sheet's 989 TOPS at 1980 MHz), 8.2% under θ's short-window 4095.6; one ALU
# op is then 3785.7 / 63.01 = 60.1 MAC-times (M2 reads 59.2 at 64 lanes). Every tensor format is scaled alike.
M1_S8 = 499712 / 132
H200_M1 = replace(
    H200, name="H200, M1 cap", s8=M1_S8, sp=H200.sp * M1_S8 / H200.s8, f16=H200.f16 * M1_S8 / H200.s8,
    facets=((62.741 * M1_S8 / 131072, 0.981),),
    assumed=H200.assumed + ("RT-MODEL M1: the long-window cap scales s8, 2:4 and f16 alike",))
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
CARDS = (H200, H200_M1, SM120)

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


def grid(scheme) -> tuple:
    """(bm, bk, bn): the block grid of A (bm x bk), B (bk x bn) and C (bm x bn)."""
    a, b, _ = scheme[0]
    return a.shape[0], a.shape[1], b.shape[1]


def is_valid(scheme) -> bool:
    """The scheme's tensor equals <bm,bk,bn>'s: sum_r a_r[i,k] b_r[k',j] c_r[i',j'] = [i=i'][k=k'][j=j']."""
    bm, bk, bn = grid(scheme)
    t = np.zeros((bm, bk, bk, bn, bm, bn), dtype=np.int64)
    for a, b, c in scheme:
        if a.shape != (bm, bk) or b.shape != (bk, bn) or c.shape != (bm, bn):
            return False
        t += np.einsum("ik,lj,pq->ikljpq", a, b, c)
    want = np.zeros_like(t)
    for i, k, j in itertools.product(range(bm), range(bk), range(bn)):
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


# Sedoglavic's catalogue (fmm.univ-lille.fr, the `<shape>_LRP.mpl` files, kept in `fmm/`): Hopcroft–Kerr's <2,k,n>
# family (RT-ALGO A2; the catalogue's <2,3,5;25> is a parametric family, left out) and Laderman's <3,3,3;23> (the
# catalogue's family, read at l = 1).
FMM = Path(__file__).resolve().parent / "fmm"
CATALOGUE = ("2x3x3", "2x3x4", "2x4x4", "2x4x5", "2x5x5", "3x3x3", "4x4x4")


def read_lrp(path: Path) -> tuple:
    """The file's (L, R, P): L is R x (m k), R is R x (k n), P is (m n) x R, as Fractions."""
    txt = path.read_text().replace("1/l", "1").replace("l", "1").replace("LRP:=", "")
    mats = []
    for mt in re.finditer(r"Matrix\((\d+), (\d+), \[(.*?)\]\)", txt, re.S):
        r, c, body = int(mt.group(1)), int(mt.group(2)), mt.group(3)
        rows = re.findall(r"\[([^\[\]]*)\]", body)
        mats.append([[Fraction(x.strip()) for x in row.split(",")] for row in rows])
        assert len(mats[-1]) == r and all(len(x) == c for x in mats[-1])
    return tuple(mats)


def lrp_scheme(name: str):
    """The catalogue scheme as (a, b, c) triples in this file's convention, or None when a coefficient isn't an integer
    (a scheme over Z/2^32 needs 2-adic units, and int8 forms need integers)."""
    m, k, n = (int(x) for x in name.split("x"))
    Lm, Rm, Pm = read_lrp(FMM / f"{name}_LRP.mpl")
    if any(x.denominator != 1 for M in (Lm, Rm, Pm) for row in M for x in row):
        return None
    Li, Ri, Pi = (np.array([[int(x) for x in row] for row in M], dtype=np.int64) for M in (Lm, Rm, Pm))
    shapes = [(m, k, n)] + [p for p in set(itertools.permutations((m, k, n))) if p != (m, k, n)]
    for (mm, kk, nn), ta, tb, tc in itertools.product(shapes, (0, 1), (0, 1), (0, 1)):
        try:
            s = [((Li[r].reshape(kk, mm).T if ta else Li[r].reshape(mm, kk)),
                  (Ri[r].reshape(nn, kk).T if tb else Ri[r].reshape(kk, nn)),
                  (Pi[:, r].reshape(nn, mm).T if tc else Pi[:, r].reshape(mm, nn))) for r in range(len(Li))]
        except ValueError:
            continue
        if is_valid(s):
            return s
    raise ValueError(f"{name}: no flattening convention reproduces the matmul tensor")


def cyclic(scheme):
    """<m,k,n> -> <k,n,m>: Tr(A B C^T) is invariant under cycling, so (a, b, c) -> (b, c^T, a^T)."""
    return [(b, c.T, a.T) for a, b, c in scheme]


def orientations(scheme):
    """The scheme's six orientations (three cycles, each with its transpose), the valid ones."""
    out, s = [], scheme
    for _ in range(3):
        for t in (s, transpose(s)):
            if is_valid(t):
                out.append(t)
        s = cyclic(s)
    return out


def catalogue():
    """{name: [valid orientations]} of the catalogue's integral schemes."""
    out = {}
    for name in CATALOGUE:
        s = lrp_scheme(name)
        if s is not None:
            out[name] = orientations(s)
    return out


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


def lookup_routes(L: int, n: int, m: int, kappa: float = 1.0) -> dict:
    """O-IO's lookup routes (art:092f373e…, `io_lp.py` and `merkle_lp.py`), loads a naive MAC, at A = B = 0 only.

    There X = b + V_rep and W' = c + U[:, pi_s] don't wrap, and X W'^T = X c^T + (X pi_s^-1) U^T: permuting X's
    segment columns (m L moves a segment, 1/n a MAC) leaves U static, so U-specialized code and tables over U apply
    despite v1's permutation. Each route is that static-U product plus the binary product X c^T by Four Russians
    (g-bit chunks of c index per-row tables of X's chunk subset sums, two rows packed an entry, built once per row
    pair and chunk for the n columns). O-IO's routes need the noise's period: today's binary R and HYB's V U^T
    computed once a period don't exist under v1, whose V (U pi_s)^T is a fresh product every segment.
    - lut1: T_jt[r] = r U_jt (256 int16 entries); one lookup a MAC, tables rebuilt for each use by the m rows;
    - swar5: U-specialized IADD3 sums of 5-bit chunk subset sums in registers (`io_lp.fullwidth_routes` d_swar5).
    lut2 (two products a lookup, 2^16 entries a (j, inner pair)) is left out: at v1's L a column's tables are
    L/2 x 128 KB, far past the 228 KB a SM holds, and a rebuild costs 2^16/m adds a lookup.
    `vec` counts the ALU lane-ops, `rf` two words an ALU op and one a lookup."""
    def lanes(alu, other=0.0, smem=0.0):
        return dict(alu=alu, issue=alu + other, vec=alu, rf=2 * alu + other, smem=smem)

    def cpart(gb: int):
        lk = math.ceil(32 / gb) / 2                        # lookups a word of 32 MACs (two rows an entry)
        j_each = min(9, (228 * 1024) // (4 * 2 ** gb))
        build = lk * 2 ** gb / n
        d = lanes(lk / 2 + 2 * lk / j_each + 2 * lk / 30 + build, lk + build, 4 * (lk + build) * kappa)
        return {k: v / 32 for k, v in d.items()}

    c = min((cpart(gb) for gb in range(6, 14)), key=lambda d: d["issue"] + d["smem"] / 4)
    lut1 = lanes(0.5 + 2 / 9 + 256 / m, 1 + 256 / m, 4 * (1 + 256 / m) * kappa)
    nch = math.ceil(32 / 5)
    swar5 = lanes((8 * math.ceil((nch - 1) / 2) / 2 + 2.5) / 32 + (3 + 2 + 2 + 0.3) / L)
    return {name: {k: r.get(k, 0.0) + c.get(k, 0.0) for k in set(r) | set(c)}
            for name, r in (("lut1", lut1), ("swar5", swar5))}


@dataclass
class Leaf:
    weight: float      # leaf MACs over naive MACs
    A: FormStats
    B: FormStats
    nnz_c: int


NAIVE = (1, 1, 1)


def leaf_tile(card: Card, g: tuple):
    """The leaf tile (m_l, n_l, rate) for a scheme on block grid g = (bm, bk, bn): the output super-tile
    (bm m_l x bn n_l) plus one leaf accumulator must fit the register budget, and m_l the MMA's smallest m. Highest
    rate first, then least operand traffic."""
    if g == NAIVE:
        return card.honest_tile + (1.0,)
    best = None
    for ml in (16, 32, 64, 128, 256):
        for nl in (8, 16, 32, 64, 128, 256):
            if ml < card.min_m or (g[0] * g[2] + 1) * ml * nl > card.acc_words:
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


def price(card: Card, leaves: list, L: int, n: int, m: int, g: tuple = NAIVE, beside: bool = True,
          wgmma_smem: bool = True, facets: bool = True, items: tuple = (), extra_fixed: dict | None = None,
          lookups: bool = False) -> Priced:
    """LP: minimize T (SM-clocks a naive MAC) with every resource's load at most T times its rate, and θ's joint
    facets a·tensor + b·other ≤ 4T. c = T over the honest kernel's 1/s8.

    g: the scheme's block grid (NAIVE for the plain product). The scheme covers whole MMA blocks, floor(L / 32 bk)
    32 bk of a segment's L inner indices and floor(n / bn n_l) bn n_l of the n columns; the rest is a naive leaf.
    wgmma_smem: wgmma's operand reads (B, and A when it isn't in registers) are charged to the 128 B/clk shared-memory
    port; False charges only the LDS that brings A to registers (sm_90 only; θ doesn't measure it).
    items: more work a naive MAC, each (name, weight, {route: loads a unit}), one convex mix of routes an item.
    extra_fixed: loads a naive MAC with no choice of route.
    lookups: O-IO's lookup routes for naive leaves (`lookup_routes`; valid at B = 0 only)."""
    tile = leaf_tile(card, g)
    if tile is None:
        return Priced(math.inf, ["no tile"], {}, {}, False)
    ml, nl, rate = tile
    BM, BK, BN = g
    if g != NAIVE:
        f = (L // (32 * BK)) * 32 * BK / L * ((n // (BN * nl)) * BN * nl / n)
        if f <= 0:
            return Priced(math.inf, ["no block"], {}, {}, False)
        one = FormStats(1, 0.0, 0.0, 128)
        leaves = [replace(lf, weight=lf.weight * f) for lf in leaves] + ([Leaf(1 - f, one, one, 0)] if f < 1 else [])
    smem_mma = (1 / ml + 1 / nl) if (wgmma_smem or card.family != "sm_90") else 1 / nl
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
    if lookups and beside:
        lk = lookup_routes(L, n, m)
        route_loads.update(lk)
    sc0 = scatter_loads("s16-alu")
    sp = dict(mma, tensor=1 / (card.sp * rate))
    for k, lf in enumerate(leaves):
        mrow = {}
        naive_leaf = lf.A.terms == 1 and lf.B.terms == 1
        for r in main + (list(lk) if lookups and beside and naive_leaf else []):
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
    if g != NAIVE:
        adds = sum(lf.weight * lf.nnz_c * BK / L for lf in leaves)
        forms = sum(lf.weight * (BN * (lf.A.terms - 1 + 2 * (lf.A.terms > 1)) / n
                                 + BM * (lf.B.terms - 1 + 2 * (lf.B.terms > 1)) / m) for lf in leaves)
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


def powered(scheme, d: int):
    return power(scheme, d) if d > 0 else [(np.ones((1, 1), np.int64),) * 3]


def leaves_of(scheme, d: int, eX: np.ndarray, eW: np.ndarray) -> list:
    s = powered(scheme, d)
    bm, bk, bn = grid(s)
    out = []
    for a, b, c in s:
        ka = tuple(int(x) for x in sorted(a.ravel()[a.ravel() != 0]))
        kb = tuple(int(x) for x in sorted(b.ravel()[b.ravel() != 0]))
        out.append(Leaf(1 / (bm * bk * bn), _stats("A", eX, ka), _stats("B", eW, kb), int((c != 0).sum())))
    return out


def profile(scheme) -> str:
    """Term counts of the X-side and W'-side forms, e.g. 'X 1112224 / W' 1112234'."""
    ta = "".join(str(int((a != 0).sum())) for a in sorted((a for a, _, _ in scheme), key=lambda a: (a != 0).sum()))
    tb = "".join(str(int((b != 0).sum())) for b in sorted((b for _, b, _ in scheme), key=lambda b: (b != 0).sum()))
    return f"X {ta} / W' {tb}"


def price_scheme(card: Card, scheme, d: int, eX, eW, L: int, n: int, m: int, **kw) -> Priced:
    return price(card, leaves_of(scheme, d, eX, eW), L, n, m, grid(powered(scheme, d)), **kw)


def best_scheme(card: Card, d: int, eX, eW, L: int, n: int, m: int, schemes, **kw):
    """The cheapest scheme of the list at depth d (the same scheme at every level): (Priced, scheme)."""
    best, seen = None, set()
    for s in schemes:
        lv = leaves_of(s, d, eX, eW)
        key = (grid(powered(s, d)), tuple(sorted((lf.A.terms, lf.A.q, lf.B.terms, lf.B.q, lf.nnz_c) for lf in lv)))
        if key in seen:
            continue
        seen.add(key)
        p = price(card, lv, L, n, m, grid(powered(s, d)), **kw)
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
    bm, bk, bn = grid(scheme)
    m, k = Xs.shape
    n = Ws.shape[0]
    Ab = [[Xs[i * m // bm:(i + 1) * m // bm, j * k // bk:(j + 1) * k // bk] for j in range(bk)] for i in range(bm)]
    # B = Ws^T (k x n): block B[p][q] is rows p of k, cols q of n, stored transposed as Ws[q-block, p-block]
    Bb = [[Ws[q * n // bn:(q + 1) * n // bn, p * k // bk:(p + 1) * k // bk] for q in range(bn)] for p in range(bk)]
    C = [[None] * bn for _ in range(bm)]
    for a, b, c in scheme:
        Af = sum(a[i, j] * Ab[i][j] for i in range(bm) for j in range(bk) if a[i, j])
        Bf = sum(b[p, q] * Bb[p][q] for p in range(bk) for q in range(bn) if b[p, q])
        M = scheme_product(Af, Bf, scheme, d - 1, cnt, fp)
        for i, j in itertools.product(range(bm), range(bn)):
            if c[i, j]:
                C[i][j] = c[i, j] * M if C[i][j] is None else C[i][j] + c[i, j] * M
    return np.block(C)


def check_words(scheme, d: int, data: str, S: int = 3, seed: int = 1, fp: bool = False, uX: int = U_X):
    """Every word by the depth-d scheme (int8 wrapped forms, 2:4 corrections and leftovers) equals the honest word."""
    bm, bk, bn = grid(scheme)
    m, n, L = 16 * bm ** d, 16 * bn ** d, 32 * bk ** d
    rng = np.random.default_rng(seed)
    if data == "zero":
        A, B = np.zeros((m, S * L), np.int64), np.zeros((n, S * L), np.int64)
    else:
        A = np.clip(np.rint(rng.normal(0, 20, (m, S * L))), -128, 127).astype(np.int64)
        B = np.clip(np.rint(rng.normal(0, 30.7, (n, S * L))), -128, 127).astype(np.int64)
    X, W = v1_operands(A, B, L, rng, uX=uX)
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
    """name -> (X entry pmf, W' entry pmf, lookups allowed). A = B = 0 is the adversary's case (the minimum overflow
    over all data); O-IO's lookup routes need B = 0."""
    zero = data_pmf("zero")
    return {
        "A=B=0, v1 (V ±64, U ±80)": (entry_pmf(zero, U_X), entry_pmf(zero, U_W), True),
        "A=B=0, ±80 both (v2's likely)": (entry_pmf(zero, 80), entry_pmf(zero, 80), True),
        "real, v1 (A σ20, B σ30.7)": (entry_pmf(data_pmf("gauss", 20.0), U_X),
                                      entry_pmf(data_pmf("gauss", 30.7), U_W), False),
    }


def classes(card: Card):
    rows = [(nm, K, N, v1_L(card, K)) for nm, K, N in STATIC]
    for T in (2048, 8192, 32768):
        rows.append((f"PV T{T}", T, 128, v1_L(card, T, 64)))
    return rows


def methods(orbit_schemes, cat) -> dict:
    """name -> (schemes, depth): the plain product, the 2x2 orbit at depth 1 and 2, and each catalogue scheme at depth
    1 in its best orientation."""
    out = {"naive": ([None], 0), "2x2 orbit d1": (orbit_schemes, 1), "2x2 orbit d2": (orbit_schemes, 2)}
    for name, ors in cat.items():
        out[f"<{name.replace('x', ',')};{len(ors[0])}>"] = (ors, 1)
    return out


def variants(card: Card, lookups: bool):
    out = [("tensor", dict(beside=False)), ("beside", dict(beside=True, lookups=lookups))]
    if card.family == "sm_90":
        out.append(("beside, wgmma reads free", dict(beside=True, lookups=lookups, wgmma_smem=False)))
    return out


def report(menu) -> dict:
    """c_mm per operand case, card, pricing variant and matmul class: the cheapest method of the menu. Each method's
    scheme (orientation, orbit member) is chosen at qkv's L and reused for the other classes, which differ in L and n."""
    out = {}
    for case, (eX, eW, lk) in operand_cases().items():
        print(f"\n== operands {case}: int8 overflow of 2-, 3- and 4-term forms, X "
              + " ".join(f"{form_stats(eX, (1,) * t).q:.4f}" for t in (2, 3, 4)) + "; W' "
              + " ".join(f"{form_stats(eW, (1,) * t).q:.4f}" for t in (2, 3, 4)))
        for card in CARDS:
            rows = classes(card)
            _, _, N0, L0 = rows[0]
            print(f"  {card.name}: every method at qkv (L {L0}, n {N0})")
            vs = variants(card, lk)
            chosen = {}
            for vname, kw in vs:
                for mname, (schemes, d) in menu.items():
                    chosen[vname, mname] = best_scheme(card, d, eX, eW, L0, N0, TOKENS, schemes, **kw)
                print(f"    [{vname}] " + "; ".join(f"{mn} {chosen[vname, mn][0].c:.3f}" for mn in menu))
                best_m = min(menu, key=lambda mn: chosen[vname, mn][0].c)
                p, s = chosen[vname, best_m]
                top = ", ".join(f"{k} {v:.3f}" for k, v in sorted(p.routes.items(), key=lambda kv: -kv[1])[:6])
                print(f"      cheapest {best_m}: c {p.c:.3f} bound by {','.join(p.binding)}"
                      + (f"; {profile(s)}" if s is not None else "") + f"; routes {top}")
            print(f"    {'class':10} {'K':>6} {'L':>5} | " + " | ".join(f"{v:>34}" for v, _ in vs))
            for nm, K, N, L in rows:
                cells = []
                for vname, kw in vs:
                    cand = [(mn, price_scheme(card, chosen[vname, mn][1], d, eX, eW, L, N, TOKENS, **kw))
                            for mn, (_, d) in menu.items()]
                    mname, p = min(cand, key=lambda t: t[1].c)
                    out[(case, card.name, vname, nm)] = (mname, p)
                    cells.append(f"{p.c:6.3f} {mname[:14]:>14} {','.join(p.binding)[:11]:>11}")
                print(f"    {nm:10} {K:6} {L:5} | " + " | ".join(f"{c:>34}" for c in cells))
    return out


# --------------------------------------------------------------------------------------------------------------------
# SHA-512. Compression schedules (RT-UNITS, note 20261026T0320Z-draft-accel-rt-units, "SHA-512 schedules", counted
# against the default build's SASS): lane instructions a compression as ALU, IMAD-pipe slots (IMAD 1, IMAD.WIDE 2,
# IMAD.HI 2.09), issue, and on sm_120 the register words they read (SHF 2, LOP3 3, IADD.64 4, IMAD 1, IMAD.WIDE 3,
# IMAD.HI 1, no operand reuse), against θ's measured register-file port there.

SHA_SCHEDULES = {
    "sm_90": {"default build": (3400, 152, 3608, 0.0),
              "A-wide + R-hi": (2460, 2462, 4002, 0.0),
              "A-wide + R-shf": (2550, 2550, 4306, 0.0)},
    "sm_120": {"default build": (3415, 0, 3415, 1728 * 2 + 927 * 3 + 760 * 4),
               "R-shf": (2563, 2564, 4267, 876 * 2 + 927 * 3 + 760 * 4 + 852 * (1 + 3)),
               "R-hi, IMAD.HI at 32 (unmeasured)": (2441, 2440, 3902, 754 * 2 + 927 * 3 + 760 * 4 + 487 * (1 + 3 + 1))},
}


def sha_loads(card: Card, schedule: str) -> dict:
    alu, slots, issue, rf = SHA_SCHEDULES[card.family][schedule]
    return dict(alu=alu, fma=slots * card.fma / 64, issue=issue, vec=issue, rf=rf)


def sha_clocks(card: Card, schedule: str) -> tuple:
    """SM-clocks a compression alone (the max over θ's resources) and the binding resource."""
    ld, cp = sha_loads(card, schedule), cap(card)
    t = {r: v / cp[r] for r, v in ld.items() if not math.isinf(cp[r])}
    r = max(t, key=t.get)
    return t[r], r


def sha_menu(card: Card) -> list:
    """The schedules a kernel may use: those on measured rates (honest hashes at the best of them, c_sha = 1)."""
    return [s for s in SHA_SCHEDULES[card.family] if "unmeasured" not in s]


# Measured public SHA-512 rates (hashes a second, whole card), each with its SMs, the clock it ran at (the rated
# boost where the source gives none) and its message-structure factor: hashcat's -m 1700 optimized kernel hashes a
# short password (message words 7-14 zero, no feed-forward, an early exit on two digest words, so its last three
# rounds are dead), about 1.135x less work a hash than a general compression (CALIB 03:35Z, via RT-UNITS).
SHA_MEASURED = (
    ("sm_90", "ours, commit_gpu default build (CMP-107)", 4.79e9, 132, 1980, 1.0, "r20261018-062010-1c6b"),
    ("sm_90", "hashcat -m 1700 on H100 SXM (GH100, 132 SMs), 5.40-5.48 GH/s", 5.44e9, 132, 1980, 1.135,
     "onlinehashcrack.com/tools-benchmark-hashcat-nvidia-H100-SXM.php; gist.github.com/laluka/2ee80e40ad485adc26934109f4ab75ef"),
    ("sm_120", "ours, commit_gpu default build (2092 MHz lock)", 7.13e9, 188, 2092, 1.0, "r20261016-110413-c5d1"),
    ("sm_120", "hashcat -m 1700 on RTX 5090 (GB202, 170 SMs), 9.99-10.07 GH/s, at its rated boost", 10.03e9, 170,
     2407, 1.135, "gist.github.com/Chick3nman/09bac0775e6393468c2925c1e1363d5c; hashcat.net/forum/thread-12395-page-2.html"),
)


def sha_report() -> dict:
    print("\n== SHA-512: SM-clocks a general compression (lower is faster)")
    out = {}
    for fam, card in (("sm_90", H200), ("sm_120", SM120)):
        ours = None
        for f, what, rate, sms, mhz, factor, src in SHA_MEASURED:
            if f != fam:
                continue
            clk = sms * mhz * 1e6 / rate * factor
            ours = ours or clk
            out[fam, what] = clk
            print(f"  {card.name}: {what}: {rate / 1e9:.2f} G/s at {mhz} MHz, {clk:.1f} SM-clocks, {ours / clk:.2f}x "
                  f"ours a SM-clock [{src}]")
        base = sha_clocks(card, "default build")[0]
        for sch in SHA_SCHEDULES[fam]:
            t, r = sha_clocks(card, sch)
            out[fam, sch] = t
            print(f"  {card.name}: schedule {sch} (RT-UNITS, on paper): {t:.2f} SM-clocks at θ, bound by {r}; "
                  f"{base / t:.2f}x the default build")
    return out


# --------------------------------------------------------------------------------------------------------------------
# The whole job a naive MAC: words, the peels (HYB_RHO products of inner size L, L/K a MAC each), the wrap corrections
# (nofold's c_op·p/2), A's commitments and every word's SHA-512 leaf (two compressions per 32 words).

def comps_per_mac(L: int) -> float:
    return 1 / (16 * L) + A_COMMIT * H200.sha / (ETA * H200.mac_rate)


def job_items(card: Card, L: int, K: int, honest: bool, beside: bool, schedules: list) -> tuple:
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
    sha = {s: sha_loads(card, s) for s in schedules}
    return (("wraps", 1.0, wraps), ("peels", HYB_RHO * L / K, peel), ("sha", comps_per_mac(L), sha))


def job(card: Card, L: int, K: int, n: int, eX, eW, menu: dict, lookups: bool) -> dict:
    """nofold's additive γ at θ (the default build's hash) and four overlapped prices (θ's max with facets): the
    deployed kernel (dense s8, wraps on the ALU, the default build's hash), the same with the best SHA-512 schedule
    (c_sha = 1), the dense kernel with every non-matmul choice at its best (wraps on any route), and the best public
    method of the menu (OPT*)."""
    hs = sha_menu(card)
    add = (1 + HYB_RHO * L / K + comps_per_mac(L) * sha_clocks(card, "default build")[0] * card.s8
           + HYB_P / 2 / card.alu * card.s8)
    naive = leaves_of(None, 0, eX, eW)
    deployed = price(card, naive, L, n, TOKENS, beside=False, items=job_items(card, L, K, True, False, hs[:1]))
    hon = price(card, naive, L, n, TOKENS, beside=False, items=job_items(card, L, K, True, False, hs))
    dense = price(card, naive, L, n, TOKENS, beside=False, items=job_items(card, L, K, False, False, hs))
    best = None
    for mname, (schemes, d) in menu.items():
        r = best_scheme(card, d, eX, eW, L, n, TOKENS, schemes, beside=True, lookups=lookups,
                        items=job_items(card, L, K, False, True, hs))
        if best is None or r[0].c < best[1].c:
            best = (mname, r[0])
    return dict(additive=add, deployed=deployed, honest=hon, dense=dense, best=best,
                hash_share=comps_per_mac(L) * sha_clocks(card, "default build")[0] * card.s8 / add)


def job_report(menu: dict, case: str = "A=B=0, v1 (V ±64, U ±80)") -> dict:
    out = {}
    eX, eW, lk = operand_cases()[case]
    print(f"\n== the whole job a naive MAC, operands {case}, MAC-times at θ's s8 rate: nofold's additive sum (default "
          "hash; its hash share); overlapped: the deployed kernel (default hash, wraps on the ALU), with the best "
          "SHA-512 schedule (c_sha = 1), the dense kernel at its best (wraps on 2:4 or scattered), and OPT* (the best "
          "public method); gaps 1 - OPT*/each")
    for card in CARDS:
        print(f"  {card.name}")
        print(f"    {'class':10} {'L':>5} | {'additive':>8} {'hash':>5} | {'deployed':>18} | {'best SHA':>18} | "
              f"{'dense at best':>18} | {'OPT*':>34} | gaps: deployed, best SHA, dense")
        for nm, K, N, L in classes(card):
            j = job(card, L, K, N, eX, eW, menu, lk)
            (lab, b) = j["best"]
            out[card.name, nm] = j
            cells = " | ".join(f"{j[k].c:7.3f} {','.join(j[k].binding)[:10]:>10}" for k in ("deployed", "honest", "dense"))
            print(f"    {nm:10} {L:5} | {j['additive']:8.3f} {j['hash_share']:5.2f} | {cells} | {b.c:7.3f} "
                  f"{lab[:12]:>12} {','.join(b.binding)[:13]:>13} | "
                  + " ".join(f"{1 - b.c / j[k].c:6.1%}" for k in ("deployed", "honest", "dense")))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exact CPU checks of the words at small sizes")
    ap.add_argument("--job-only", action="store_true")
    ap.add_argument("--sha-only", action="store_true")
    args = ap.parse_args()
    cat = catalogue()
    if args.check:
        checks = [("strassen", STRASSEN, (1, 2)), ("winograd", WINOGRAD, (1, 2))]
        checks += [(name, ors[0], (1,)) for name, ors in cat.items() if math.prod(grid(ors[0])) <= 50]
        for nm, sch, depths in checks:
            for d in depths:
                for data in ("zero", "real"):
                    for uX in (U_X, 80):
                        ok, cnt = check_words(sch, d, data, uX=uX)
                        print(f"{nm} {grid(sch)} depth {d} {data} uX {uX}: words exact {ok}; s8 MACs {cnt.s8}, "
                              f"2:4 MACs {cnt.sp}, leftover contributions {cnt.left}")
        print("packing invariance (no permutation, v1's permutation):", check_packing_invariance())
        return
    print("catalogue: " + ", ".join(f"{k} rank {len(v[0])}, {len(v)} orientations" for k, v in cat.items())
          + "; not integral (left out): " + ", ".join(n for n in CATALOGUE if n not in cat))
    sha_report()
    if args.sha_only:
        return
    orb = orbit()
    print(f"orbit: {len(orb)} valid <2,2,2> schemes (Strassen, Winograd, GL2(Z) images mod monomials, transposes)")
    menu = methods(orb, cat)
    if not args.job_only:
        report(menu)
    job_report(menu)


if __name__ == "__main__":
    main()
