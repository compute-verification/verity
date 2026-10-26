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


@dataclass(frozen=True, order=True)
class FormStats:
    terms: int
    q: float           # P(form leaves int8): the density of its excess e
    left24: float      # density of e's nonzeros beyond two in a group of four (2:4's leftover)
    max_abs: int
    binary: float = 0.0  # P(entry = 1) when every entry is 0 or 1 (A = 0 without V: X = b), else 0


def form_stats(entry: np.ndarray, coefs) -> FormStats:
    coefs = [c for c in coefs if c]
    pmf, off = form_pmf(entry, coefs)
    v = np.arange(len(pmf)) - off
    q = float(pmf[(v < -128) | (v > 127)].sum())
    left = sum(math.comb(4, j) * q ** j * (1 - q) ** (4 - j) * (j - 2) for j in (3, 4)) / 4
    support = v[pmf > 0]
    binary = float(pmf[v == 1].sum()) if set(support.tolist()) <= {0, 1} else 0.0
    return FormStats(len(coefs), q, left, int(np.abs(support).max()), binary)


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


FORK_FLOOR_222 = 4      # ALGO-COST A2 (fork_deliveries.py 24d9d2e6e4): IADD3s a segment for one 2x2 level, forked chains


def _iadd3(scheme) -> int:
    """IADD3s a segment to add every product into the output words, flat: an output block with t contributions takes
    ceil(t / 2) (two terms and the running word an IADD3; signs are free)."""
    t = sum((c != 0).astype(int) for _, _, c in scheme)
    return int(np.ceil(t / 2).sum())


def assembly(scheme, d: int) -> float:
    """IADD3s a segment a output entry for the depth-d scheme's output assembly, the cheaper of flat (every leaf into
    the words) and level by level (each level's products assembled first, the forked-chain floor on one 2x2 level)."""
    if d == 0:
        return 0.0
    bm, _, bn = grid(scheme)
    one = _iadd3(scheme)
    if (bm, bn) == (2, 2) and len(scheme) == 7:
        one = min(one, FORK_FLOOR_222)
    per_level = one / (bm * bn)
    hier = per_level * sum((len(scheme) / (bm * bn)) ** j for j in range(d))
    pw = power(scheme, d)
    gm, _, gn = grid(pw)
    return min(hier, _iadd3(pw) / (gm * gn))


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


PACK_MAX = 4                          # the narrow side's |form entry| bound for BET-HYBRID-2 §I's packing
PACK_CHUNKS = (64, 128, 256, 512)     # inner positions between decodes, the LP's choice


def _left24(q: float) -> float:
    return sum(math.comb(4, j) * q ** j * (1 - q) ** (4 - j) * (j - 2) for j in (3, 4)) / 4


def lanes_alu(x: float) -> dict:
    return dict(alu=x, issue=x, vec=x, rf=2 * x)


def packed_loads(sp: dict, sc0: dict, q: float, narrow: int, amort: int) -> dict:
    """BET-HYBRID-2 §I's packing, loads a leaf MAC for each chunk length C: with the other side's form narrow (|entry|
    <= `narrow`; A = 0 without V), two columns' excesses share one int8 operand E2 = e1 + 64 e2 on the 2:4 MMA (half
    the correction MACs) and E2's leftover beyond two a group of four is scattered. Each chunk of C inner positions
    accumulates afresh and each output pair is decoded (5 lane-ops: zero T; S1 = ((T + 32) & 63) - 32 into its
    output, T - S1 into the other's running sum, shifted by 6 once at the end). The decode needs |S1| <= 31, which
    holds when e1 has at most 31 // narrow nonzeros in the chunk; e is formed first, so the prover knows each chunk's
    counts, puts the sparser column first, and does a chunk where both columns exceed it unpacked. E2 is formed
    (one IMAD) and compressed to 2:4 (two lane-ops an entry) once for the `amort` rows of the other side."""
    cap_nz = 31 // narrow
    q2 = 1 - (1 - q) ** 2
    out = {}
    for C in PACK_CHUNKS:
        over = 1 - sum(math.comb(C, j) * q ** j * (1 - q) ** (C - j) for j in range(cap_nz + 1))
        bad = over ** 2
        packed = {r: 0.5 * sp.get(r, 0.0) + _left24(q2) * sc0.get(r, 0.0) for r in RES}
        unpacked = {r: sp.get(r, 0.0) + _left24(q) * sc0.get(r, 0.0) for r in RES}
        extra = lanes_alu((1 - bad) * 5 / (2 * C) + 3 / (2 * amort))
        out[C] = {r: (1 - bad) * packed[r] + bad * unpacked[r] + extra.get(r, 0.0) for r in RES}
    return out


def binary_loads(sp: dict, sc0: dict, p: float, amort_bin: int, amort_dense: int) -> dict:
    """RT-MODEL's M7, loads a leaf MAC, for a form whose entries are 0 or 1 with P(1) = p (A = 0 without V): in each
    group of four inner positions with three or four ones the bits are complemented, x = 1 - x', so X = Y + Z with Y
    2:4-conformant (at most two nonzeros, entries in {-1, 0, 1}) and Z the indicator of complemented groups. Y W^T is
    one 2:4 MMA pass; Z W^T = Zg G^T with G the dense side's group sums (|G| <= 512: two int8 digit planes) on 2:4 over
    a quarter of the inner positions, Zg's leftover beyond two a group of four scattered. Y and Zg are formed for the
    `amort_dense` columns (two lane-ops an entry), G for the `amort_bin` rows (five an entry of G)."""
    pz = math.comb(4, 3) * p ** 3 * (1 - p) + p ** 4
    out = {r: 1.5 * sp.get(r, 0.0) + _left24(pz) / 4 * sc0.get(r, 0.0) for r in RES}
    for r, x in lanes_alu(2 / amort_dense + 5 / 4 / amort_bin).items():
        out[r] += x
    return out


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
    mix: dict = field(default_factory=dict)


@dataclass
class Method:
    """One way to compute the product, for `price_methods`: a scheme's leaves on block grid g (rest and asm as in
    `price`), the loads a naive MAC it adds with no choice of route when it runs the whole product (fixed), and work
    that scales with its share (items, as in `price`)."""
    label: str
    leaves: list
    g: tuple = NAIVE
    rest: Leaf | None = None
    asm: float | None = None
    fixed: dict = field(default_factory=dict)
    items: tuple = ()


def price(card: Card, leaves: list, L: int, n: int, m: int, g: tuple = NAIVE, beside: bool = True,
          wgmma_smem: bool = True, facets: bool = True, items: tuple = (), extra_fixed: dict | None = None,
          lookups: bool = False, rest: Leaf | None = None, asm: float | None = None) -> Priced:
    """LP: minimize T (SM-clocks a naive MAC) with every resource's load at most T times its rate, and θ's joint
    facets a·tensor + b·other ≤ 4T. c = T over the honest kernel's 1/s8.

    g: the scheme's block grid (NAIVE for the plain product). The scheme covers whole MMA blocks, floor(L / 32 bk)
    32 bk of a segment's L inner indices and floor(n / bn n_l) bn n_l of the n columns; the rest is a naive leaf.
    wgmma_smem: wgmma's operand reads (B, and A when it isn't in registers) are charged to the 128 B/clk shared-memory
    port; False charges only the LDS that brings A to registers (sm_90 only; θ doesn't measure it).
    items: more work a naive MAC, each (name, weight, {route: loads a unit}), one convex mix of routes an item.
    extra_fixed: loads a naive MAC with no choice of route.
    lookups: O-IO's lookup routes for naive leaves (`lookup_routes`; valid at B = 0 only).
    rest: the operands' naive leaf, for the part the scheme's blocks don't cover (default: generic int8 entries).
    asm: the scheme's output assembly, IADD3s a segment a output entry (`assembly`)."""
    return price_methods(card, [Method("main", leaves, g, rest, asm)], L, n, m, beside=beside, wgmma_smem=wgmma_smem,
                         facets=facets, items=items, extra_fixed=extra_fixed, lookups=lookups)


def price_methods(card: Card, methods: list, L: int, n: int, m: int, beside: bool = True, wgmma_smem: bool = True,
                  facets: bool = True, items: tuple = (), extra_fixed: dict | None = None,
                  lookups: bool = False) -> Priced:
    """`price` over several methods at once. The prover picks a method a tile, so the LP picks every method's share
    of the product (the shares sum to 1) together with every route: the convex hull of the methods, which can beat
    each of them when they bind different resources (ALGO-COST A19). A method's leaves, fixed loads and items scale
    with its share; `items` and `extra_fixed` are the job's, whichever method runs. Priced.mix: the shares."""
    names, cols, eqs = [], [], []

    def var(name, loads):
        names.append(name)
        cols.append([loads.get(r, 0.0) for r in RES])

    sc0 = scatter_loads("s16-alu")
    rows_of = {"A": m, "B": n}
    lk = lookup_routes(L, n, m) if lookups and beside else {}
    shares = []
    for i, mt in enumerate(methods):
        tile = leaf_tile(card, mt.g)
        if tile is None:
            continue
        ml, nl, rate = tile
        BM, BK, BN = mt.g
        leaves, f = list(mt.leaves), 1.0
        if mt.g != NAIVE:
            f = (L // (32 * BK)) * 32 * BK / L * ((n // (BN * nl)) * BN * nl / n)
            if f <= 0:
                continue
            one = FormStats(1, 0.0, 0.0, 128)
            rest = replace(mt.rest, weight=1 - f) if mt.rest is not None else Leaf(1 - f, one, one, 0)
            leaves = [replace(lf, weight=lf.weight * f) for lf in leaves] + ([rest] if f < 1 else [])
        # The method's own loads a naive MAC: its output assembly in registers (`assembly`'s IADD3s; without it, one
        # IADD a leaf entry it enters) and the forms (amortized over the other operand's rows).
        own = dict(mt.fixed)
        if mt.g != NAIVE:
            if mt.asm is None:
                adds = sum(lf.weight * lf.nnz_c * BK / L for lf in leaves)
            else:
                adds = f * mt.asm / L
            forms = sum(lf.weight * (BN * (lf.A.terms - 1 + 2 * (lf.A.terms > 1)) / n
                                     + BM * (lf.B.terms - 1 + 2 * (lf.B.terms > 1)) / m) for lf in leaves)
            for r, v in (("alu", adds + forms), ("rf", 3 * (adds + forms)), ("vec", adds + forms),
                         ("issue", adds + forms)):
                own[r] = own.get(r, 0.0) + v
        share = f"share:{mt.label}"
        shares.append(share)
        var(share, own)
        pre = f"{i}." if len(methods) > 1 else ""
        smem_mma = (1 / ml + 1 / nl) if (wgmma_smem or card.family != "sm_90") else 1 / nl
        tens = 1 / (card.s8 * rate)
        mma = dict(smem=smem_mma, rf=card.mma_rf, issue=card.mma_issue)
        t = BESIDE_TILE
        route_loads = {
            "s8": dict(mma, tensor=tens),
            "f16": {**{k: 2 * v for k, v in mma.items()}, "tensor": 1 / (card.f16 * rate)},
            "split": {**{k: 2 * v for k, v in mma.items()}, "tensor": 2 * tens},
            "dp4a": dict(fma=0.5, issue=0.25 + 1 / (8 * t), smem=2 / t, rf=0.75, vec=0.25),
            "ffma": dict(fma=1.0, issue=1 + 1 / (2 * t), smem=8 / t, rf=3.0, vec=1.0),
            **lk,
        }
        main = ["s8", "f16", "split"] + (["dp4a", "ffma"] if beside else [])
        sp = dict(mma, tensor=1 / (card.sp * rate))
        for k, lf in enumerate(leaves):
            k = f"{pre}{k}"
            mrow = {share: -lf.weight}
            naive_leaf = lf.A.terms == 1 and lf.B.terms == 1
            for r in main + (list(lk) if naive_leaf else []):
                if r == "f16" and max(lf.A.max_abs, lf.B.max_abs) > 2048:
                    continue
                ld = dict(route_loads[r])
                if r in ("s8", "dp4a") and lf.A.q > 0 and lf.B.q > 0:
                    for res_, x in sc0.items():         # the double term 65536 eA eB^T, scattered
                        ld[res_] = ld.get(res_, 0.0) + lf.A.q * lf.B.q * x
                var(f"{k}:{r}", ld)
                mrow[f"{k}:{r}"] = 1.0
            bins = [(s, o) for s, fs, o in (("A", lf.A, "B"), ("B", lf.B, "A")) if fs.binary > 0]
            for s, o in bins:
                var(f"{k}:2:4 bin {s}", binary_loads(sp, sc0, (lf.A if s == "A" else lf.B).binary, rows_of[s],
                                                      rows_of[o]))
                mrow[f"{k}:2:4 bin {s}"] = 1.0
            eqs.append((mrow, 0.0))
            needs = [f"{k}:s8"] + ([f"{k}:dp4a"] if beside else []) + [f"{k}:2:4 bin {s}" for s, _ in bins]
            for side, fs, other_side in (("A", lf.A, lf.B), ("B", lf.B, lf.A)):
                if fs.q <= 0:
                    continue
                row = {nm: -1.0 for nm in needs}
                var(f"{k}:{side}:2:4", {r: sp.get(r, 0) + fs.left24 * sc0.get(r, 0) for r in RES})
                var(f"{k}:{side}:2:4x2", {r: 2 * sp.get(r, 0) for r in RES})
                for v in SCATTER:
                    var(f"{k}:{side}:{v}", {r: fs.q * x for r, x in scatter_loads(v).items()})
                names_c = ["2:4", "2:4x2"] + list(SCATTER)
                if other_side.max_abs <= PACK_MAX:
                    other_rows = rows_of["B" if side == "A" else "A"]
                    for C, ld in packed_loads(sp, sc0, fs.q, other_side.max_abs, other_rows).items():
                        var(f"{k}:{side}:2:4 packed C{C}", ld)
                        names_c.append(f"2:4 packed C{C}")
                for nm in names_c:
                    row[f"{k}:{side}:{nm}"] = 1.0
                eqs.append((row, 0.0))
        for name, w, routes_ in mt.items:
            row = {share: -w}
            for r, ld in routes_.items():
                var(f"{pre}{name}:{r}", ld)
                row[f"{pre}{name}:{r}"] = 1.0
            eqs.append((row, 0.0))
    if not shares:
        return Priced(math.inf, ["no tile"], {}, {}, False)
    eqs.append(({s: 1.0 for s in shares}, 1.0))
    for name, w, routes_ in items:
        row = {}
        for r, ld in routes_.items():
            var(f"{name}:{r}", ld)
            row[f"{name}:{r}"] = 1.0
        eqs.append((row, w))
    # Every word stored for the hash (STS, both kernels).
    fixed = dict(issue=1 / L, smem=4 / L)
    for r, v in (extra_fixed or {}).items():
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
            b_ub.append(-(a * fixed.get("tensor", 0.0) + b * fixed.get(other, 0.0) / 32))
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
    routes, mix = {}, {}
    for nm, v in zip(names, x):
        if v > 1e-9:
            kind, key = nm.split(":", 1)
            if kind == "share":
                mix[key] = v
            else:
                routes[key] = routes.get(key, 0.0) + v
    return Priced(T * card.s8, binding, loads, routes, mix=mix)


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
    return price(card, leaves_of(scheme, d, eX, eW), L, n, m, grid(powered(scheme, d)),
                 rest=leaves_of(None, 0, eX, eW)[0], asm=assembly(scheme, d), **kw)


def best_scheme(card: Card, d: int, eX, eW, L: int, n: int, m: int, schemes, **kw):
    """The cheapest scheme of the list at depth d (the same scheme at every level): (Priced, scheme)."""
    best, seen = None, set()
    for s in schemes:
        lv = leaves_of(s, d, eX, eW)
        key = (grid(powered(s, d)), tuple(sorted((lf.A, lf.B, lf.nnz_c) for lf in lv)), assembly(s, d))
        if key in seen:
            continue
        seen.add(key)
        p = price(card, lv, L, n, m, grid(powered(s, d)), rest=leaves_of(None, 0, eX, eW)[0], asm=assembly(s, d),
                  **kw)
        if best is None or p.c < best[0].c:
            best = (p, s)
    return best


def scheme_method(label: str, scheme, d: int, eX, eW) -> Method:
    return Method(label, leaves_of(scheme, d, eX, eW), grid(powered(scheme, d)), leaves_of(None, 0, eX, eW)[0],
                  assembly(scheme, d))


def price_hull(card: Card, chosen: dict, eX, eW, L: int, n: int, m: int, **kw) -> Priced:
    """The convex hull of the methods (label -> (scheme, depth)), priced as one LP (`price_methods`)."""
    return price_methods(card, [scheme_method(lb, s, d, eX, eW) for lb, (s, d) in chosen.items()], L, n, m, **kw)


def distinct_schemes(schemes, d: int, eX, eW) -> list:
    """The schemes of the list that price differently at depth d (`best_scheme`'s key)."""
    seen, out = set(), []
    for s in schemes:
        lv = leaves_of(s, d, eX, eW)
        key = (grid(powered(s, d)), tuple(sorted((lf.A, lf.B, lf.nnz_c) for lf in lv)), assembly(s, d))
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def rich_menu(orb, cat, eX, eW) -> dict:
    """Every distinct scheme as its own method: the plain product, the 2x2 orbit at depth 1, Strassen at depth 2 and
    every orientation of each catalogue scheme at depth 1."""
    out = {"naive": (None, 0), "Strassen d2": (STRASSEN, 2)}
    out.update({f"orbit d1 #{i}": (s, 1) for i, s in enumerate(distinct_schemes(orb, 1, eX, eW))})
    for name, ors in cat.items():
        out.update({f"<{name.replace('x', ',')};{len(ors[0])}> o{j}": (s, 1)
                    for j, s in enumerate(distinct_schemes(ors, 1, eX, eW))})
    return out


def shares(p: Priced) -> str:
    return "+".join(f"{k} {v:.2f}" for k, v in sorted(p.mix.items(), key=lambda kv: -kv[1]) if v > 0.005)


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


def check_no_v(L: int = 512, m: int = 32, n: int = 32, C: int = 128, seed: int = 3) -> dict:
    """Without V at A = B = 0 (X = b, W' = c + U[:, pi], U ±80), the two tricks `price` gives the adversary, exactly:
    M7's complement split of the binary X (`binary_loads`) and §I's packed correction (`packed_loads`) on a depth-1
    product of two-term forms on both sides, (X_a + X_b)(W'_a + W'_b)^T, chunks of C positions. Returns whether each
    product is exact, the operand
    constraints (Y 2:4 with entries in {-1, 0, 1}, Zg binary, G's digit planes and E2 in int8), Zg's 2:4 leftover
    density, and the share of chunks done unpacked."""
    rng = np.random.default_rng(seed)
    X, W = v1_operands(np.zeros((m, L), np.int64), np.zeros((n, L), np.int64), L, rng, uX=0)
    assert set(np.unique(X).tolist()) <= {0, 1}
    g = X.reshape(m, L // 4, 4)
    comp = g.sum(axis=2) >= 3
    Y = np.where(comp[:, :, None], g - 1, g).reshape(m, L)
    G = W.reshape(n, L // 4, 4).sum(axis=2)
    G_lo = wrap8(G)
    G_hi = (G - G_lo) // 256
    Zg = comp.astype(np.int64)
    m7 = Y @ W.T + Zg @ (256 * G_hi + G_lo).T
    y_ok = bool(((Y.reshape(m, L // 4, 4) != 0).sum(axis=2) <= 2).all() and np.abs(Y).max() <= 1)
    planes_ok = bool(np.abs(G_lo).max() <= 128 and np.abs(G_hi).max() <= 127)
    _, zg_rest = _split24(Zg[:, : (L // 16) * 4])
    h = L // 2
    Xc = X[:, :h] + X[:, h:]
    Wc = W[:, :h] + W[:, h:]
    Ww = wrap8(Wc)
    e = (Wc - Ww) // 256
    cap_nz = 31 // int(np.abs(Xc).max())
    corr = np.zeros((m, n), np.int64)
    bad = chunks = 0
    e2_ok = True
    for j in range(0, n, 2):
        for c0 in range(0, h, C):
            e1, e2 = e[j, c0:c0 + C], e[j + 1, c0:c0 + C]
            x = Xc[:, c0:c0 + C]
            chunks += 1
            if (e1 != 0).sum() > (e2 != 0).sum():
                e1, e2, cols = e2, e1, (j + 1, j)
            else:
                cols = (j, j + 1)
            if (e1 != 0).sum() > cap_nz:
                bad += 1
                corr[:, cols[0]] += x @ e1
                corr[:, cols[1]] += x @ e2
                continue
            E2 = e1 + 64 * e2
            e2_ok &= bool(np.abs(E2).max() <= 127)
            T = x @ E2
            S1 = ((T + 32) & 63) - 32
            corr[:, cols[0]] += S1
            corr[:, cols[1]] += (T - S1) >> 6
    packed = Xc @ Ww.T + 256 * corr
    return dict(m7_exact=bool(np.array_equal(m7, X @ W.T)), y_2to4=y_ok, g_planes_int8=planes_ok,
                zg_left=float((zg_rest != 0).mean()), packed_exact=bool(np.array_equal(packed, Xc @ Wc.T)),
                e2_int8=e2_ok, e_density=float((e != 0).mean()), unpacked_chunks=bad / chunks)


# --------------------------------------------------------------------------------------------------------------------
# Report.

def operand_cases():
    """name -> (X entry pmf, W' entry pmf, lookups allowed). A = B = 0 is the adversary's case (the minimum overflow
    over all data); O-IO's lookup routes need B = 0."""
    zero = data_pmf("zero")
    return {
        "A=B=0, v1 (V ±64, U ±80)": (entry_pmf(zero, U_X), entry_pmf(zero, U_W), True),
        "A=B=0, ±80 both (v2's likely)": (entry_pmf(zero, 80), entry_pmf(zero, 80), True),
        "A=B=0, no V (U ±80; is V needed?)": (entry_pmf(zero, 0), entry_pmf(zero, 80), True),
        "A=B=0, v2 draft (V ±96, U ±80)": (entry_pmf(zero, 96), entry_pmf(zero, 80), True),
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
                h = price_hull(card, {mn: (chosen[vname, mn][1], d) for mn, (_, d) in menu.items()}, eX, eW, L0, N0,
                               TOKENS, **kw)
                top = ", ".join(f"{k} {v:.3f}" for k, v in sorted(h.routes.items(), key=lambda kv: -kv[1])[:6])
                print(f"      hull of the methods: c {h.c:.3f} bound by {','.join(h.binding)}; shares {shares(h)}; "
                      f"routes {top}")
            print(f"    {'class':10} {'K':>6} {'L':>5} | " + " | ".join(f"{v + ' (hull)':>41}" for v, _ in vs))
            for nm, K, N, L in rows:
                cells = []
                for vname, kw in vs:
                    cand = [(mn, price_scheme(card, chosen[vname, mn][1], d, eX, eW, L, N, TOKENS, **kw))
                            for mn, (_, d) in menu.items()]
                    mname, p = min(cand, key=lambda t: t[1].c)
                    h = price_hull(card, {mn: (chosen[vname, mn][1], d) for mn, (_, d) in menu.items()}, eX, eW, L,
                                   N, TOKENS, **kw)
                    out[(case, card.name, vname, nm)] = (mname, p, h)
                    cells.append(f"{p.c:6.3f} ({h.c:.3f}) {mname[:14]:>14} {','.join(h.binding)[:11]:>11}")
                print(f"    {nm:10} {K:6} {L:5} | " + " | ".join(f"{c:>41}" for c in cells))
    return out


def hull_report(orb, cat) -> dict:
    """`report`'s hull takes each method in its best scheme alone; this one takes every distinct scheme (`rich_menu`)
    at the frozen qkv shape, at v1's and v2's L, on v1's operands and v2's (±80 both)."""
    out = {}
    zero = data_pmf("zero")
    print("\n== the hull over every distinct scheme (qkv, A = B = 0): c, shares, binding")
    for case, (eX, eW) in (("v1 (V ±64, U ±80)", (entry_pmf(zero, U_X), entry_pmf(zero, U_W))),
                           ("v2 (±80 both)", (entry_pmf(zero, V2_V), entry_pmf(zero, V2_U)))):
        menu = rich_menu(orb, cat, eX, eW)
        for card in CARDS:
            for L in (v1_L(card, 16384), V2_T[card.family] * V2_PICKS[card.family][0][4]):
                for vname, kw in variants(card, True):
                    h = price_hull(card, menu, eX, eW, L, 16512, TOKENS, **kw)
                    out[case, card.name, L, vname] = h
                    print(f"  {case:18} {card.name:12} L {L:5} [{vname}] ({len(menu)} schemes): c {h.c:.3f}; "
                          f"shares {shares(h)}; bound by {','.join(h.binding)}")
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


def job_items(card: Card, L: int, K: int, honest: bool, beside: bool, schedules: list,
              hash_scale: float = 1.0) -> tuple:
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
    return (("wraps", 1.0, wraps), ("peels", HYB_RHO * L / K, peel), ("sha", hash_scale * comps_per_mac(L), sha))


LANES = ("issue", "alu", "fma", "smem", "rf")


def lane_load(p: Priced) -> float:
    """The honest kernel's lane load: its busiest non-tensor resource over its tensor time."""
    return max(p.loads.get(r, 0.0) for r in LANES) / p.loads["tensor"]


def job(card: Card, L: int, K: int, n: int, eX, eW, menu: dict, lookups: bool, hash_scale: float = 1.0) -> dict:
    """nofold's additive γ at θ (the default build's hash) and four overlapped prices (θ's max with facets): the
    deployed kernel (dense s8, wraps on the ALU, the default build's hash), the same with the best SHA-512 schedule
    (c_sha = 1), the dense kernel with every non-matmul choice at its best (wraps on any route), and the best public
    method of the menu (OPT*). hash_scale scales the hashing a MAC (a design that hashes less)."""
    hs = sha_menu(card)
    comps = hash_scale * comps_per_mac(L)
    add = 1 + HYB_RHO * L / K + comps * sha_clocks(card, "default build")[0] * card.s8 + HYB_P / 2 / card.alu * card.s8
    naive = leaves_of(None, 0, eX, eW)

    def items(honest, beside, sch):
        return job_items(card, L, K, honest, beside, sch, hash_scale)

    deployed = price(card, naive, L, n, TOKENS, beside=False, items=items(True, False, hs[:1]))
    hon = price(card, naive, L, n, TOKENS, beside=False, items=items(True, False, hs))
    dense = price(card, naive, L, n, TOKENS, beside=False, items=items(False, False, hs))
    best, chosen = None, {}
    for mname, (schemes, d) in menu.items():
        r = best_scheme(card, d, eX, eW, L, n, TOKENS, schemes, beside=True, lookups=lookups,
                        items=items(False, True, hs))
        chosen[mname] = (r[1], d)
        if best is None or r[0].c < best[1].c:
            best = (mname, r[0])
    h = price_hull(card, chosen, eX, eW, L, n, TOKENS, beside=True, lookups=lookups, items=items(False, True, hs))
    if h.c < best[1].c * (1 - 1e-6):
        best = (f"hull {shares(h)}", h)
    return dict(additive=add, deployed=deployed, honest=hon, dense=dense, best=best,
                hash_share=comps * sha_clocks(card, "default build")[0] * card.s8 / add)


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


def sweep_report(menu: dict, Ls=(1280, 2688, 4096, 8192, 16384)) -> dict:
    """The job's c_mm (OPT* over the dense kernel at its best) as L grows towards K: the hash's share falls, and
    the matmul's public best shows once the tensor pipe binds. qkv (K 16384) and down (K 65536), both operand cases
    at A = B = 0."""
    out = {}
    print("\n== c_mm of the whole job (OPT* / the dense kernel at its best, both hashing at the best schedule) as L "
          "grows")
    for case in ("A=B=0, v1 (V ±64, U ±80)", "A=B=0, ±80 both (v2's likely)"):
        eX, eW, lk = operand_cases()[case]
        print(f"  operands {case}")
        for card in CARDS:
            for nm, K, N in (STATIC[0], STATIC[3]):
                cells = []
                for L in Ls:
                    if L > K:
                        continue
                    j = job(card, L, K, N, eX, eW, menu, lk)
                    lab, b = j["best"]
                    out[case, card.name, nm, L] = j
                    cells.append(f"L {L}: {b.c / j['dense'].c:.3f} {lab[:12]} ({','.join(b.binding)[:12]})")
                print(f"    {card.name:12} {nm:5} " + "; ".join(cells))
    return out


def lane_report(menu: dict, scales=(0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0)) -> dict:
    """ALGO-COST A8/A10: c_mm of the job against the honest kernel's lane load. The job's hashing is scaled by s (a
    design that hashes less a MAC: larger L, narrower words, fewer committed words); for each s, the dense kernel at
    its best gives the lane load (busiest lane resource over tensor time) and OPT* over it gives c_mm. The plain
    product and the 2x2 orbit at depth 1 and 2 (the catalogue never wins)."""
    sub = {k: v for k, v in menu.items() if k in ("naive", "2x2 orbit d1", "2x2 orbit d2")}
    out = {}
    print("\n== c_mm of the job against the honest kernel's lane load (lane time / tensor time), hashing scaled by s")
    for case in ("A=B=0, v1 (V ±64, U ±80)", "A=B=0, ±80 both (v2's likely)"):
        eX, eW, lk = operand_cases()[case]
        print(f"  operands {case}")
        for card in CARDS:
            for nm, K, N, L in (classes(card)[0], classes(card)[3]):
                cells = []
                for s in scales:
                    j = job(card, L, K, N, eX, eW, sub, lk, hash_scale=s)
                    lab, b = j["best"]
                    out[case, card.name, nm, s] = (lane_load(j["dense"]), b.c / j["dense"].c, lab)
                    cells.append(f"s {s:g}: lane {lane_load(j['dense']):.2f} c {b.c / j['dense'].c:.3f} "
                                 f"{lab.replace('2x2 orbit ', '')}")
                print(f"    {card.name:12} {nm:5} L {L:5} | " + "; ".join(cells))
    return out


# --------------------------------------------------------------------------------------------------------------------
# v2's noise tiles (PROTO-DESIGN-2, nofold.py eb880f9be8): a segment of L = t L' holds t tiles, V (m x L') repeats every
# tile and U's columns are permuted afresh in each, X = b + V, W' = c + U pi_tau. At A = B = 0 a prover can skip the t
# products a segment (ALGO-COST A12, PROTO-DESIGN-2's "new route"): T_s = b c^T + V (sum_tau c_tau)^T + Z U^T, with
# Z = sum_tau (b + V)_tau pi_tau^-1, the last two of inner L'.

V2_V, V2_U = 80, 80                 # v2 as frozen (PROTO-DESIGN-3, nofold.py 541efba7a3): ±80 on both cards
V2_V_MARGIN = 96                    # the RTX PRO 6000's stated margin
V2_T = {"sm_90": 2, "sm_120": 4}    # v2 as frozen: tiles a segment per card
V2_PEEL_DIGITS = 4                  # Z_X U^T and V Y_W^T, two base-128 digits each, inner L' once a call
# b c^T's cost in tensor time a naive MAC, and the Strassen depth it runs at (its output assembly then costs lanes);
# depth None: part of it runs on lanes the price leaves out, so it enters only the standalone table.
BIT_COST = {"sm_90": (("b1 wgmma, θ 32764.6 b1 MACs a SM-clock", 4095.6 / 32764.6, 0),),
            "sm_120": (("FP4 Strassen depth 2 (ALGO-COST A7, exact)", 0.383, 2),
                       ("A7 with b1 emulated on the ALU beside it", 0.26, None))}


def _tile_operands(t: int, v: int):
    zero = data_pmf("zero")
    eX, eW = entry_pmf(zero, v), entry_pmf(zero, V2_U)
    return eX, eW, Leaf(1.0, _stats("A", eX, (1,) * t), _stats("B", eW, (1,)), 1)


def tile_route(card: Card, t: int, Lp: int, K: int, n: int, bit: float, v: int = V2_V, **kw) -> dict:
    """The tile route at A = B = 0 against the honest kernel, both standalone at θ, a naive MAC: honest 1 (plus its peels,
    V2_PEEL_DIGITS products of inner L' a call); the route b c^T at `bit` over the whole segment, V (sum c)^T one int8
    product of inner L', and Z U^T priced by `price` (Z a t-term form of b + V entries, its excess on any route). The
    route pays no peels: at A = B = 0 the served output is 0 and the O unit's identity holds of the true words
    (nofold.v2_round's reading)."""
    _, _, zleaf = _tile_operands(t, v)
    honest = 1 + V2_PEEL_DIGITS * Lp / K
    z = price(card, [zleaf], Lp, n, TOKENS, **kw)
    vc = 1.0
    route = bit + (vc + z.c) / t
    return dict(honest=honest, z=z.c, z_routes=z.routes, route=route, c=route / honest)


def tile_job(card: Card, t: int, Lp: int, K: int, n: int, bit: float, depth: int, v: int = V2_V,
             hash_scale: float = 1.0) -> dict:
    """The tile route inside the job, θ's max with facets, a naive MAC of a segment of L = t L'. Honest: the dense
    product of X = b + V and W' = c + U pi (both within int8 at A = B = 0, so no wraps), v2's peels and every word's
    SHA-512 leaf at the best schedule. Route: the same words and hashing; b c^T (`bit` of tensor time, its MMAs' lane
    loads in proportion, and at Strassen `depth` its output assembly and one conversion a word on the ALU) and
    V (sum c)^T as fixed loads; Z U^T a leaf of weight 1/t priced on any route beside the hash; no peels. The
    prover may run the route on some groups of t tiles and the dense product (with its peels) on the rest, so c is
    the hull of the two over honest (`price_methods`); route_only is the route on every group. best is the honest
    words' best program (the dense product with every route beside the tensor pipe, OPT_honest = OPT*): c_opt, the
    route's leak against it, is the protocol's; eps_alg = 1 - best / honest is the deployed kernel's gap."""
    L, zleaf, sha, peels, dense, honest = _tile_honest(card, t, Lp, K, n, v, hash_scale)
    dense_m = Method("dense", dense, items=(peels,))
    best = price_methods(card, [dense_m], L, n, TOKENS, beside=True, items=(sha,))
    mma = dict(smem=1 / card.honest_tile[0] + 1 / card.honest_tile[1], rf=card.mma_rf, issue=card.mma_issue)
    fixed = {r: x * (bit + 1 / t) for r, x in mma.items()}
    fixed["tensor"] = (bit + 1 / t) / card.s8
    if depth:
        ops = (assembly(STRASSEN, depth) + 1) / L
        for r, x in (("alu", ops), ("issue", ops), ("rf", 3 * ops), ("vec", ops)):
            fixed[r] = fixed.get(r, 0.0) + x
    tiles = Method("tiles", [replace(zleaf, weight=1 / t)], fixed=fixed)
    route = price_methods(card, [tiles], L, n, TOKENS, beside=True, items=(sha,))
    hull = price_methods(card, [tiles, dense_m], L, n, TOKENS, beside=True, items=(sha,))
    return dict(lane=lane_load(honest), alu=_alu_load(honest), honest=honest, best=best, route=hull, route_only=route,
                c=hull.c / honest.c, c_route=route.c / honest.c, c_opt=min(1.0, hull.c / best.c),
                eps_alg=1 - best.c / honest.c)


def _alu_load(p: Priced) -> float:
    return p.loads["alu"] / p.loads["tensor"]


def _tile_honest(card: Card, t: int, Lp: int, K: int, n: int, v: int, hash_scale: float) -> tuple:
    eX, eW, zleaf = _tile_operands(t, v)
    L = t * Lp
    items = job_items(card, L, K, False, False, sha_menu(card), hash_scale)
    peels = ("peels", V2_PEEL_DIGITS * Lp / K, items[1][2])
    dense = leaves_of(None, 0, eX, eW)
    honest = price(card, dense, L, n, TOKENS, beside=False, items=(peels, items[2]))
    return L, zleaf, items[2], peels, dense, honest


def tile_job_at(card: Card, t: int, Lp: int, K: int, n: int, bit: float, depth: int, v: int, alu: float,
                s_max: float = 4.0, steps: int = 30) -> dict | None:
    """`tile_job` with the hashing scaled so that the honest kernel's ALU time over its tensor time is `alu` (the
    lane load nofold states), by bisection on the scale; None if no scale up to s_max reaches it."""
    def load(s):
        return _alu_load(_tile_honest(card, t, Lp, K, n, v, s)[-1])

    lo, hi = 0.0, s_max
    if load(hi) < alu:
        return None
    for _ in range(steps):
        mid = (lo + hi) / 2
        if load(mid) < alu:
            lo = mid
        else:
            hi = mid
    return tile_job(card, t, Lp, K, n, bit, depth, v, hi)


def tile_report(Lps=(256, 512, 1024), ts=(2, 3, 4, 5, 6)) -> dict:
    out = {}
    print(f"\n== v2's noise tiles at A = B = 0 (U ±{V2_U}, U permuted every tile): the tile route b c^T + V(Σc)^T + "
          f"Z U^T over the honest t products (plus {V2_PEEL_DIGITS} peel digits of inner L' a call), standalone, a "
          "naive MAC; each cell c (Z U^T's own price); the route wins from the first t with c < 1")
    for v in (V2_V, V2_V_MARGIN):
        for card in (H200, H200_M1, SM120):
            vs = [("", dict(beside=True))] + ([("reads free", dict(beside=True, wgmma_smem=False))]
                                              if card.family == "sm_90" else [])
            for bname, bit, _ in BIT_COST[card.family]:
                for vname, kw in vs:
                    for K, n in ((16384, 16512), (65536, 4096)):
                        for Lp in Lps:
                            cells, wins = [], None
                            for t in ts:
                                r = tile_route(card, t, Lp, K, n, bit, v, **kw)
                                out[v, card.name, bname, vname, K, Lp, t] = r
                                cells.append(f"t {t}: {r['c']:.3f} ({r['z']:.3f})")
                                wins = wins or (t if r["c"] < 1 else None)
                            print(f"  V ±{v} {card.name:12} {vname:10} b c^T by {bname[:20]:20} K {K:5} L' {Lp:4} | "
                                  f"wins from t {wins} | " + "; ".join(cells))
    return out


V2_LP = {"sm_90": (768, 1024, 1792), "sm_120": (192, 256)}    # the L' that V2_PICKS uses

# v2 as frozen, nofold.v2_pick at 070e970980 (ε_alg budget 0, overlapped, 16-bit words): per static shape (model, K, N,
# layers), the tile period L' (L = t L') and the honest ALU time without wraps over tensor time (A_nw / T).
V2_PICKS = {
    "sm_90": (
        ("frozen", 16384, 16512, 1, 1024, 1.100), ("frozen", 16384, 16384, 1, 1024, 1.100),
        ("frozen", 16384, 65536, 1, 1024, 1.087), ("frozen", 65536, 16384, 1, 1088, 1.060),
        ("Llama-3.1-8B", 4096, 6144, 32, 1024, 1.150), ("Llama-3.1-8B", 4096, 4096, 32, 1024, 1.180),
        ("Llama-3.1-8B", 4096, 28672, 32, 1024, 1.104), ("Llama-3.1-8B", 14336, 4096, 32, 1216, 1.005),
        ("Llama-3.1-8B", 4096, 128256, 1, 1024, 1.094),
        ("Llama-3.1-70B", 8192, 10240, 80, 1024, 1.118), ("Llama-3.1-70B", 8192, 8192, 80, 1024, 1.126),
        ("Llama-3.1-70B", 8192, 57344, 80, 1024, 1.092), ("Llama-3.1-70B", 28672, 8192, 80, 1152, 1.034),
        ("Llama-3.1-70B", 8192, 128256, 1, 1024, 1.089),
        ("DeepSeek-V4-Pro", 7168, 43008, 61, 896, 1.244), ("DeepSeek-V4-Pro", 3072, 50176, 61, 768, 1.448),
        ("DeepSeek-V4-Pro", 7168, 1536, 61, 1216, 1.154), ("DeepSeek-V4-Pro", 1536, 65536, 61, 768, 1.452),
        ("DeepSeek-V4-Pro", 7168, 576, 61, 1792, 1.272), ("DeepSeek-V4-Pro", 4096, 16384, 61, 1024, 1.114),
        ("DeepSeek-V4-Pro", 16384, 7168, 61, 1024, 1.123), ("DeepSeek-V4-Pro", 7168, 129280, 1, 896, 1.239),
    ),
    "sm_120": (
        ("frozen", 16384, 16512, 1, 192, 1.054), ("frozen", 16384, 16384, 1, 192, 1.055),
        ("frozen", 16384, 65536, 1, 192, 1.051), ("frozen", 65536, 16384, 1, 192, 1.006),
        ("Llama-3.1-8B", 4096, 6144, 32, 192, 1.262), ("Llama-3.1-8B", 4096, 4096, 32, 192, 1.270),
        ("Llama-3.1-8B", 4096, 28672, 32, 192, 1.250), ("Llama-3.1-8B", 14336, 4096, 32, 192, 1.076),
        ("Llama-3.1-8B", 4096, 128256, 1, 192, 1.247),
        ("Llama-3.1-70B", 8192, 10240, 80, 192, 1.109), ("Llama-3.1-70B", 8192, 8192, 80, 192, 1.111),
        ("Llama-3.1-70B", 8192, 57344, 80, 192, 1.102), ("Llama-3.1-70B", 28672, 8192, 80, 192, 1.027),
        ("Llama-3.1-70B", 8192, 128256, 1, 192, 1.101),
        ("DeepSeek-V4-Pro", 7168, 43008, 61, 192, 1.148), ("DeepSeek-V4-Pro", 3072, 50176, 61, 192, 1.187),
        ("DeepSeek-V4-Pro", 7168, 1536, 61, 192, 1.207), ("DeepSeek-V4-Pro", 1536, 65536, 61, 192, 1.318),
        ("DeepSeek-V4-Pro", 7168, 576, 61, 256, 1.001), ("DeepSeek-V4-Pro", 4096, 16384, 61, 192, 1.252),
        ("DeepSeek-V4-Pro", 16384, 7168, 61, 192, 1.061), ("DeepSeek-V4-Pro", 7168, 129280, 1, 192, 1.147),
    ),
}


SMS_MHZ = {"sm_90": (132, 1980), "sm_120": (188, 2092)}
PUB_SHA = {"sm_90": ("A-wide + R-hi", ("default build",)),
           "sm_120": ("R-hi, IMAD.HI at 32 (unmeasured)", ("R-shf", "default build"))}


def units(card: Card, p: Priced) -> dict:
    """A priced program's units a naive MAC by resource: tensor in s8 MAC-times, lanes in lane-ops."""
    caps = cap(card)
    return {r: p.loads[r] if r == "tensor" else p.loads[r] * caps[r] / card.s8 for r in RES if r in p.loads}


def pub_programs(menu: dict | None = None) -> dict:
    """D5's named programs for WORKSOUND-INST's PubBest, per card at v2 as frozen (A = B = 0, V and U ±80, the frozen
    qkv shape at v2's L): pub_mm standalone (lanes idle: the hull of the menu's methods, each in its best scheme, with
    every route beside the tensor pipe, O-IO's lookups included) and with the lanes busy (the honest dense kernel,
    c = 1, wherever V2_PICKS' tile route doesn't win); pub_sha, the best paper schedule at θ, with the measured rates
    beside it. SHA-512's fma units are θ's fma lanes (an IMAD takes two)."""
    menu = menu or methods(orbit(), catalogue())
    out = {}
    print("\n== D5's named public programs (units a naive MAC: tensor in s8 MAC-times, lanes in lane-ops; SHA-512 a "
          "compression)")
    zero = data_pmf("zero")
    eX, eW = entry_pmf(zero, V2_V), entry_pmf(zero, V2_U)
    for card in (H200, SM120):
        fam = card.family
        L = V2_T[fam] * V2_PICKS[fam][0][4]
        K, n = V2_PICKS[fam][0][1:3]
        chosen = {mn: (best_scheme(card, d, eX, eW, L, n, TOKENS, schemes, beside=True, lookups=True)[1], d)
                  for mn, (schemes, d) in menu.items()}
        alone = price_hull(card, chosen, eX, eW, L, n, TOKENS, beside=True, lookups=True)
        dense = price(card, leaves_of(None, 0, eX, eW), L, n, TOKENS, beside=False)
        name = f"pub_mm.{card.name.replace(' ', '')}.v2"
        out[name] = dict(standalone=alone, job=dense, schemes={mn: chosen[mn] for mn in alone.mix})
        for label, p in (("standalone", alone), ("lanes busy (v2's periods)", dense)):
            u = ", ".join(f"{r} {x:.4g}" for r, x in units(card, p).items())
            routes = ", ".join(f"{r} {x:.3f}" for r, x in p.routes.items())
            print(f"  {name} {label}: c {p.c:.3f} bound by {','.join(p.binding)} (L {L}, K {K}, n {n}); "
                  + (f"shares {shares(p)} ({'; '.join(profile(chosen[mn][0]) for mn in p.mix if chosen[mn][0])}); "
                     if label == "standalone" else "") + f"routes {routes}; units {u}")
        sms, mhz = SMS_MHZ[fam]
        best, others = PUB_SHA[fam]
        for sch in (best,) + others:
            t, r = sha_clocks(card, sch)
            ld = sha_loads(card, sch)
            tag = f"pub_sha.{card.name.replace(' ', '')}" if sch == best else "  beside it"
            out[tag, sch] = dict(clocks=t, rate=sms * mhz * 1e6 / t, loads=ld)
            print(f"  {tag}: {sch}: {t:.2f} SM-clocks a compression at θ ({r}-bound), {sms * mhz * 1e6 / t / 1e9:.2f} G/s "
                  f"at {mhz} MHz on {sms} SMs; units " + ", ".join(f"{k} {v:g}" for k, v in ld.items()))
        for f, what, rate, s_, m_, factor, src in SHA_MEASURED:
            if f == fam:
                clk = s_ * m_ * 1e6 / rate * factor
                print(f"    measured: {what}: {clk:.1f} SM-clocks a compression [{src}]")
    return out


def v2_report(ts: dict | None = None) -> dict:
    """c_mm of v2 as frozen per static shape: the tile route mixed with dense tiles inside the job at the shape's L'
    and honest ALU/tensor (V2_PICKS), c = c_opt (over the honest words' best program, OPT*); per model the leak
    sum w h (1 - c) / sum w h, w the shape's MACs and h OPT*'s cost a MAC, and the deployed kernel's gap eps_alg
    weighted the same way. ts overrides the card's tile count (the same L')."""
    out = {}
    print("\n== v2 as frozen (V ±80, nofold.v2_pick at 070e970980): the tile route mixed with dense tiles inside the "
          "job per static shape at its L' and honest ALU/tensor, over OPT*; cells: K x N, L', ALU/T: c (eps_alg); per "
          "model the MAC-and-cost-weighted leak and eps_alg")
    for card in (H200, H200_M1, SM120):
        t = (ts or V2_T)[card.family]
        _, bit, depth = BIT_COST[card.family][0]
        for model in ("frozen", "Llama-3.1-8B", "Llama-3.1-70B", "DeepSeek-V4-Pro"):
            num = den = alg = dep = 0.0
            cells = []
            for mdl, K, N, reps, Lp, alu in V2_PICKS[card.family]:
                if mdl != model:
                    continue
                r = tile_job_at(card, t, Lp, K, N, bit, depth, V2_V, alu)
                c = r["c_opt"]
                w = K * N * reps
                num, den = num + w * r["best"].c * (1 - c), den + w * r["best"].c
                alg, dep = alg + w * (r["honest"].c - r["best"].c), dep + w * r["honest"].c
                out[card.name, t, model, K, N] = dict(Lp=Lp, alu=alu, c=c, c_route=r["c_route"], mix=r["route"].mix,
                                                      eps_alg=r["eps_alg"])
                cells.append(f"{K}x{N} {Lp} {alu:.2f}: {c:.3f} ({r['eps_alg']:.1%})" + (
                    f" [route only {r['c_route']:.3f}; {shares(r['route'])}]" if c < 1 - 1e-6 else ""))
            out[card.name, t, model] = num / den
            out[card.name, t, model, "eps_alg"] = alg / dep
            print(f"  {card.name:12} t {t} {model:16} leak {num / den:6.2%} eps_alg {alg / dep:5.2%} | "
                  + "; ".join(cells))
    return out


def tile_lane_report(alus=(0.6, 0.82, 1.0, 1.05, 1.1, 1.5, 2.0)) -> dict:
    """The tile route inside the job against the honest kernel's lane load, at v2's frozen t per card (and t - 1,
    t + 1) and v2's L': the job's hashing scaled so that the honest ALU time over tensor time (nofold's lane load) is
    each of `alus`; c_opt, the hull of route and dense tiles over the honest words' best program (OPT*), and the
    deployed kernel's gap eps_alg against that program."""
    out = {}
    print("\n== v2's tile route inside the job (θ's max with facets) at the honest kernel's ALU/tensor (nofold's lane "
          "load; v2 as frozen: ≥ 1.00, 0.82 if the network certifier expanded the coins); cells: ALU/tensor: c_opt "
          "(the route mixed with dense tiles over OPT*) / eps_alg (the deployed kernel's gap to OPT*)")
    for v in (V2_V, V2_V_MARGIN):
        for card in (H200, H200_M1, SM120):
            t0 = V2_T[card.family]
            for bname, bit, depth in BIT_COST[card.family]:
                if depth is None:
                    continue
                for t in sorted({max(2, t0 - 1), t0, t0 + 1}):
                    for K, n in ((16384, 16512), (65536, 4096)):
                        for Lp in V2_LP[card.family]:
                            cells = []
                            for a in alus:
                                r = tile_job_at(card, t, Lp, K, n, bit, depth, v, a)
                                out[v, card.name, bname, t, K, Lp, a] = r
                                cells.append(f"{a:.2f}: " + (f"{r['c_opt']:.3f}/{r['eps_alg']:.1%}" if r else "-"))
                            print(f"  V ±{v} {card.name:12} {bname[:20]:20} t {t} K {K:5} L' {Lp:4} | "
                                  + "; ".join(cells))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exact CPU checks of the words at small sizes")
    ap.add_argument("--job-only", action="store_true")
    ap.add_argument("--sha-only", action="store_true")
    ap.add_argument("--sweep", action="store_true", help="only the job's c_mm as L grows")
    ap.add_argument("--lanes", action="store_true", help="only the job's c_mm against the honest lane load")
    ap.add_argument("--tiles", action="store_true", help="only v2's tile route (ALGO-COST A12)")
    ap.add_argument("--hull", action="store_true", help="only the hull over every distinct scheme (ALGO-COST A19)")
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
        print("without V (M7's complement split, §I's packed correction):", check_no_v())
        return
    if args.tiles:
        tile_report()
        tile_lane_report()
        v2_report()
        v2_report({"sm_90": 2, "sm_120": 3})
        pub_programs()
        return
    print("catalogue: " + ", ".join(f"{k} rank {len(v[0])}, {len(v)} orientations" for k, v in cat.items())
          + "; not integral (left out): " + ", ".join(n for n in CATALOGUE if n not in cat))
    sha_report()
    if args.sha_only:
        return
    orb = orbit()
    print(f"orbit: {len(orb)} valid <2,2,2> schemes (Strassen, Winograd, GL2(Z) images mod monomials, transposes)")
    if args.hull:
        hull_report(orb, cat)
        return
    menu = methods(orb, cat)
    if args.sweep:
        sweep_report(menu)
        return
    if args.lanes:
        lane_report(menu)
        return
    if not args.job_only:
        report(menu)
        hull_report(orb, cat)
    job_report(menu)
    sweep_report(menu)
    lane_report(menu)
    tile_report()
    tile_lane_report()
    v2_report()
    v2_report({"sm_90": 2, "sm_120": 3})
    pub_programs()


if __name__ == "__main__":
    main()
