#!/usr/bin/env python3
"""TT_OUT's falsification census for Pearl-C's ticket chain on the H100 (CPU, bit-exact).

Pearl-C binds output words only (the pre-peel accumulator C~ and the useful output U), so the work an adversary can skip in
the ticket chain is, per wgmma tile: an atom that leaves every accumulator of the tile unchanged (a tile identity, e.g. after
freezing), a run of atoms exact across the whole tile long enough for int8 plus Strassen to pay (14 or more atoms: one
Winograd level on the H100 breaks even at 450-700 of inner dimension), and a promotion add that leaves the total unchanged.
The census counts them over input families built to trigger each, under Pearl-shaped noise, for a pure wgmma chain and for
promotion from +0 every 4 atoms (G4). A cell is in Pi's domain when G4 is used, or when the chain is pure and k/delta^2 <= 2^15.
The census passes when every in-domain cell's skippable share is at most 1/400.

The atom is `verity.ml.tc`'s FP8 `mma`/`wgmma` step with a nonzero accumulator, parametric in the model (`--atom`,
`set_atom`): H100's one group of 32 on a 14-bit adder (`HOPPER_E4M3_K32`, the default), Ada's two groups of 16
(`ADA_E4M3_M16N8K32`), the RTX PRO 6000's one group of 32 on a 26-bit adder (`BLACKWELL_SM120_E4M3_M16N8K32`), or any
`groups:width:zero`.  Only the atom changes across cards (the forming atom and the chain both run on it); the families, the
debit and the domain rule are the same.  It is vectorized here and checked against `verity.ml.tc`'s `GroupSum.step` for
every atom by --selftest.  --exact-run-min is the card's exact-rewrite break-even in atoms. The noise is Pearl's shape, simplified: per row,
y = x + delta*rms(x)*Z with Z a rank-32 row of unit RMS (lines of entries +-(1..128)); alpha = 448/max|y|;
codes = RNE_E4M3(alpha*y). Seeds are the tuple (run, unit index, side), so duplicated units get independent noise.

    python benchmarks/pouw/pearlc_census.py --selftest
    python benchmarks/pouw/pearlc_census.py --ks 8192,16384,65536 --deltas 1 --tile 64x16 --out census.json
    python benchmarks/pouw/pearlc_census.py --atom sm120-e4m3-k32 --forming s5 --line-norm 16 --ks 8192,65536 --debit \
        --out census_sm120.json

--forming s5 is docs/pouw/cheap-binding.md §5's forming, bit-exact: s = max|x|, rho = RMS over every 8th position,
sigma = delta*rho, alpha = 448/(s + 4*sigma); F's lines are E4M3 codes of (N_F/sqrt(32))*w and E'_row = e4m3((sigma*alpha/N_F)*v)
(w, v unit-RMS lines of entries +-(1..128)); D = atom.step(C = f32(alpha*x), E'_row, F_l) per element; A' =
RNE_E4M3(D) with saturation. Pi's activation rules are checked per row: the noise floor sigma*alpha >= 1 and liveness (at
most k/64 entries with |x| >= 8*sigma); a cell whose rows break them is out of the domain ("rejected"). The collision rate
is how often two independent noise draws give an element the same code: the element-level noise entropy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_result  # noqa: E402

# ---- E4M3 codec -------------------------------------------------------------------------------------------------------
MAG = np.zeros(128, dtype=np.float64)
for _c in range(128):
    _e, _m = _c >> 3, _c & 7
    MAG[_c] = _m * 2.0 ** -9 if _e == 0 else (8 + _m) * 2.0 ** (_e - 10)
MAG[0x7F] = np.nan
FIN = MAG[:0x7F]
Q_MAX = 448.0
SIG = np.array([(c & 7) | (8 if c >> 3 else 0) for c in range(128)], dtype=np.int64)
EXP1 = np.array([max(c >> 3, 1) for c in range(128)], dtype=np.int64)
OFF, PSHIFT = -14, 17        # verity.ml.tc.term.E4M3: product_exponent_offset, product_shift
FP32_SIG = 24                # the FP32 significand width the adder rescales from

# The atom (`verity.ml.tc.models.GroupSum`): products summed in groups on a `width`-bit truncating adder with exponent
# floor `zero`, the incoming accumulator cut to `width` and joined to the first group, each group's result to the next.
ATOMS = {
    "hopper-e4m3-k32": ((32,), 14, -139),      # HOPPER_E4M3_K32 (Hawkeye H100)
    "ada-e4m3-k32": ((16, 16), 14, -139),      # ADA_E4M3_M16N8K32 (RTX 4090)
    "sm120-e4m3-k32": ((32,), 26, -133),       # BLACKWELL_SM120_E4M3_M16N8K32 (RTX PRO 6000, r20260930-031244-ecbc)
}
ATOM_GROUPS, ATOM_WIDTH, ATOM_ZERO = ATOMS["hopper-e4m3-k32"]
ATOM_NAME = "hopper-e4m3-k32"


def set_atom(name: str) -> None:
    """Select the atom model the chain and debit replay; `--atom`.  A custom `groups:width:zero` (e.g. `16,16:14:-139`)
    lets the captured sm_120 step run before it is pinned in `verity.ml.tc`."""
    global ATOM_GROUPS, ATOM_WIDTH, ATOM_ZERO, ATOM_NAME
    if name in ATOMS:
        ATOM_GROUPS, ATOM_WIDTH, ATOM_ZERO = ATOMS[name]
    else:
        g, w, z = name.split(":")
        ATOM_GROUPS, ATOM_WIDTH, ATOM_ZERO = tuple(int(x) for x in g.split(",")), int(w), int(z)
    ATOM_NAME = name


def enc(x: np.ndarray) -> np.ndarray:
    a = np.minimum(np.abs(x), Q_MAX)
    hi = np.searchsorted(FIN, a, side="left").clip(0, 126)
    lo = (hi - 1).clip(0, 126)
    dlo, dhi = a - FIN[lo], FIN[hi] - a
    pick = np.where(dlo < dhi, lo, np.where(dhi < dlo, hi, np.where(lo % 2 == 0, lo, hi)))
    pick = np.where(a == FIN[hi], hi, pick)
    return (pick | np.where((x < 0) & (pick > 0), 0x80, 0)).astype(np.uint8)


def bf16(x: np.ndarray) -> np.ndarray:
    """Round to BF16 (RNE) and back."""
    u = x.astype(np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32).astype(np.float64)


# ---- the atom with a nonzero accumulator ---------------------------------------------------------------------------------

def _fp32_parts(C: np.ndarray):
    a = np.abs(C)
    _, e = np.frexp(a)
    ex = (e - 1).astype(np.int64)
    sub = a < 2.0 ** -126
    ex = np.where(sub, -126, ex)
    mag = np.where(sub, np.ldexp(a, 149), np.ldexp(a, (23 - ex).astype(np.int64)))
    return C < 0, ex, np.where(a > 0, np.rint(mag), 0).astype(np.int64)


def _group_step(C: np.ndarray, a: np.ndarray, b: np.ndarray, width: int, zero: int):
    """One grouped-sum on a `width`-bit truncating adder (`verity.ml.tc.models.group_sum`), for a slice of up to 32 product
    pairs.  C: (P,) exact FP32 values; a, b: (P, g) codes.  Returns (D, exact) with exact = no bit lost in this group."""
    rs = FP32_SIG - width                        # rescale from the 24-bit FP32 grid to the adder's width (10 for width 14)
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    ma, mb = a & 0x7F, b & 0x7F
    ss = SIG[ma] * SIG[mb]
    pexp = EXP1[ma] + EXP1[mb] + OFF
    pneg = ((a ^ b) & 0x80) != 0
    cneg, cexp, cmag = _fp32_parts(C)
    mx = np.maximum(np.maximum(np.where(ss > 0, pexp, -10 ** 6).max(-1), np.where(cmag > 0, cexp, -10 ** 6)), zero)
    psh = np.clip(mx[:, None] - pexp, 0, 62)
    csh = np.clip(mx - cexp, 0, 62)
    p_w = (ss << PSHIFT) >> rs if rs >= 0 else (ss << PSHIFT) << -rs
    ps = np.where(ss > 0, p_w >> psh, 0)
    c_w = cmag >> rs if rs >= 0 else cmag << -rs
    cs = np.where(cmag > 0, c_w >> csh, 0)
    lost = (((cmag & ((np.int64(1) << rs) - 1)) != 0) if rs > 0 else np.zeros(C.shape, bool))
    lost |= (c_w & ((np.int64(1) << csh) - 1)) != 0
    lost |= ((ss > 0) & ((p_w & ((np.int64(1) << psh) - 1)) != 0)).any(-1)
    tot = np.where(pneg, -ps, ps).sum(-1) + np.where(cneg, -cs, cs)
    neg, m = tot < 0, np.abs(tot)
    bl = np.frexp(m.astype(np.float64))[1].astype(np.int64)
    ex = mx + bl - width
    over = np.clip(bl - width, 0, 62)
    lost |= (bl > width) & ((m & ((np.int64(1) << over) - 1)) != 0)
    m = np.where(bl > width, m >> over, m << np.clip(width - bl, 0, 62))
    sub = ex < -126
    subsh = np.clip(-126 - ex, 0, 62)
    lost |= sub & ((m & ((np.int64(1) << subsh) - 1)) != 0)
    m = np.where(sub, m >> subsh, m)
    ex = np.where(sub, -126, ex)
    v = np.ldexp((m << rs).astype(np.float64) if rs >= 0 else (m >> -rs).astype(np.float64), (ex - 23).astype(np.int64))
    return np.where(m == 0, 0.0, np.where(neg, -v, v)), ~lost


def step(C: np.ndarray, a: np.ndarray, b: np.ndarray):
    """The atom's k32 step under the selected model (`set_atom`): C: (P,) exact FP32; a, b: (P, k) codes with k = Σ groups.
    Returns (D, exact); exact = no bit lost in any group.  Each group's output is the next group's incoming accumulator."""
    exact = np.ones(C.shape, bool)
    start = 0
    for g in ATOM_GROUPS:
        D, ex = _group_step(C, a[:, start:start + g], b[:, start:start + g], ATOM_WIDTH, ATOM_ZERO)
        C, exact, start = D, exact & ex, start + g
    return C, exact


def selftest(cases: int = 3000) -> int:
    """Check the vectorized parametric step against `verity.ml.tc`'s `GroupSum.step` on nonzero-accumulator cases, for
    every atom in ATOMS and the current selection (restored at the end).  Returns the total mismatch count."""
    from verity.ml.tc.models import GroupSum
    saved = ATOM_NAME
    rng = np.random.default_rng(7)
    a = (rng.integers(0, 0x7F, (cases, 32)) | rng.integers(0, 2, (cases, 32)) * 0x80).astype(np.uint8)
    b = (rng.integers(0, 0x7F, (cases, 32)) | rng.integers(0, 2, (cases, 32)) * 0x80).astype(np.uint8)
    a[: cases // 3] &= 0x87
    C = np.where(rng.random(cases) < 0.2, 0.0, rng.standard_normal(cases) * np.exp2(rng.integers(-20, 30, cases)))
    C = C.astype(np.float32).astype(np.float64)
    bad = 0
    for name in dict.fromkeys([*ATOMS, saved]):
        set_atom(name)
        H = GroupSum(name=name, arch="census", dtype="e4m3", operand_bits=8, groups=ATOM_GROUPS, width=ATOM_WIDTH,
                     zero_exponent=ATOM_ZERO, validated="census selftest")
        D, _ = step(C, a, b)
        for i in range(cases):
            want = H.step(int(np.float32(C[i]).view(np.uint32)), [int(x) for x in a[i]], [int(x) for x in b[i]])
            got = int(np.float32(D[i]).view(np.uint32))
            bad += want != got and not (want & 0x7FFFFFFF == 0 and got & 0x7FFFFFFF == 0)
    set_atom(saved)
    return bad


# ---- inputs -------------------------------------------------------------------------------------------------------------

def _rng(*parts: object) -> np.random.Generator:
    h = hashlib.sha256("|".join(["verity/pouw/pearl-c/census/v1", *map(str, parts)]).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def lines(n: int, rng: np.random.Generator) -> np.ndarray:
    return (rng.integers(1, 129, (n, 32)) * rng.choice([-1, 1], (n, 32))).astype(np.float64)


def form(X: np.ndarray, F: np.ndarray, delta: float, rng: np.random.Generator):
    """Pearl-shaped forming of rows X against noise lines F (k x 32). Returns codes, the noise-free mask, and the rate at which
    rounding through BF16 first gives a different code than one rounding."""
    Z = lines(X.shape[0], rng) @ F.T
    Z /= np.sqrt((Z ** 2).mean(1, keepdims=True))
    sig = delta * np.sqrt((X ** 2).mean(1, keepdims=True))
    Y = X + bf16(sig * Z)
    alpha = Q_MAX / np.abs(Y).max(1, keepdims=True)
    codes = enc(alpha * Y)
    double = enc(bf16(alpha * Y))
    nf = enc(alpha * (X - 4 * sig)) == enc(alpha * (X + 4 * sig))
    return codes, nf, float((codes != double).mean())


def unit_lines(n: int, rng: np.random.Generator) -> np.ndarray:
    v = lines(n, rng)
    return v / np.sqrt((v ** 2).mean(1, keepdims=True))


def form_s5(X: np.ndarray, Fc: np.ndarray, delta: float, line_norm: float, rng: np.random.Generator, chunk: int = 1 << 17):
    """§5's forming of rows X against F's codes Fc (k x 32). Returns codes, a second draw's codes (for the collision rate),
    the share of E' codes that are zero or subnormal, and the rows' domain verdicts (noise floor, liveness)."""
    rows, k = X.shape
    s = np.abs(X).max(1)
    rho = np.sqrt((X[:, ::8] ** 2).mean(1))
    sig = delta * rho
    alpha = np.where(s + 4 * sig > 0, Q_MAX / np.maximum(s + 4 * sig, 1e-300), 0.0)
    floor_ok = sig * alpha >= 1.0
    live_ok = (np.abs(X) >= 8 * sig[:, None]).sum(1) * 64 <= k
    C = (alpha[:, None] * X).astype(np.float32).astype(np.float64)
    out, sub = [], []
    for _draw in range(2):
        Ec = enc(unit_lines(rows, rng) * (sig * alpha / line_norm)[:, None])
        sub.append(float(((Ec & 0x7F) < 8).mean()))
        codes = np.empty((rows, k), dtype=np.uint8)
        idx = np.arange(rows * k)
        for lo in range(0, rows * k, chunk):
            ii = idx[lo:lo + chunk]
            r_, l_ = ii // k, ii % k
            D, _ = step(C[r_, l_], Ec[r_], Fc[l_])
            codes[r_, l_] = enc(D)
        out.append(codes)
    return out[0], float((out[0] == out[1]).mean()), sub[0], bool(floor_ok.all()), bool(live_ok.all())


FAMILIES = ("gaussian", "spike1-first", "spikes-first", "spikes-permuted", "spikes-groupstart", "constant",
            "coherent-gaussian", "orthogonal-FA", "duplicated")
ACTIVATION_FAMILIES = ("zero-row", "zero-slices", "sparse10", "rank1", "saturated", "grid-aligned", "in-span-FA")
OUTLIER_FAMILIES = ("outliers-first", "outliers-groupstart")
FREEZE_FAMILIES = ("aligned-spikes-r64", "aligned-spikes-r160", "aligned-spikes-r220", "aligned-spikes-r300",
                   "aligned-spikes-pm1-r64", "aligned-spikes-pm1-r220", "aligned-spikes-pm1-r300", "aligned-spikes-pm1-r444")


def family(name: str, rows: int, cols: int, k: int, FA: np.ndarray, rng: np.random.Generator):
    L = k // 64 - 1
    if name == "gaussian":
        return rng.standard_normal((rows, k)), rng.standard_t(4, (cols, k))
    if name == "orthogonal-FA":
        A = rng.standard_normal((rows, k))
        Q, _ = np.linalg.qr(FA)
        return A - (A @ Q) @ Q.T, rng.standard_t(4, (cols, k))
    if name == "zero-row":
        return np.zeros((rows, k)), rng.standard_t(4, (cols, k))
    if name == "zero-slices":
        A = rng.standard_normal((rows, k))
        A.reshape(rows, k // 32, 32)[:, ::2, :] = 0.0
        return A, rng.standard_t(4, (cols, k))
    if name == "sparse10":
        return rng.standard_normal((rows, k)) * (rng.random((rows, k)) < 0.1), rng.standard_t(4, (cols, k))
    if name == "rank1":
        v = rng.standard_normal(k)
        return np.outer(rng.standard_normal(rows), v), np.outer(rng.standard_normal(cols), rng.standard_t(4, k))
    if name == "saturated":
        A = rng.standard_normal((rows, k)) * 0.1
        A[:, : k // 2] = np.sign(rng.standard_normal((rows, k // 2)))
        return A, rng.standard_t(4, (cols, k))
    if name == "grid-aligned":
        return val(rng.integers(0x30, 0x60, (rows, k)) | (rng.integers(0, 2, (rows, k)) * 0x80)), rng.standard_t(4, (cols, k))
    if name == "in-span-FA":
        return rng.standard_normal((rows, 32)) @ FA.T, rng.standard_t(4, (cols, k))
    if name == "duplicated":
        a, b = rng.standard_normal((1, k)), rng.standard_t(4, (1, k))
        return np.repeat(a, rows, 0), np.repeat(b, cols, 0)
    if name in ("constant", "coherent-gaussian"):
        v = rng.choice([-1.0, 1.0], k) if name == "constant" else rng.standard_normal(k)
        return np.tile(v, (rows, 1)), np.tile(v, (cols, 1))
    if name in ("outliers-first", "outliers-groupstart"):
        # P1's outlier families (pearlc_freeze_census.py): a +-1e-3 background and k/64 - 1 outliers of +-1, first or 64 per
        # 128-deep group; one sign vector for every row, so every word sees P1's sign-matched pair.
        sgn = rng.choice([-1.0, 1.0], k)
        A = 1e-3 * rng.choice([-1.0, 1.0], (rows, k))
        B = 1e-3 * rng.choice([-1.0, 1.0], (cols, k))
        pos = np.arange(L) if name == "outliers-first" else \
            np.concatenate([np.arange(g * 128, g * 128 + 64) for g in range(max(1, L // 64))])[:L]
        A[:, pos] = sgn[pos]
        B[:, pos] = sgn[pos]
        return A, B
    if name.startswith("aligned-spikes-"):
        # Two spikes per 128-deep group, at the same positions in every row of A and B, off the every-8th grid so that rho is
        # the background's: s/rho = R, which the noise floor allows up to 444 and liveness allows at k/64 entries. The
        # background is Gaussian, or +-1 (`aligned-spikes-pm1-rR`), which freezes whole atoms more (the reviewer's probe).
        spec = name.removeprefix("aligned-spikes-")
        pm1 = spec.startswith("pm1-")
        R = float(spec.removeprefix("pm1-").removeprefix("r"))
        pos = np.concatenate([np.arange(0, k, 128) + 1, np.arange(0, k, 128) + 9])
        if pm1:
            A, B = rng.choice([-1.0, 1.0], (rows, k)), rng.choice([-1.0, 1.0], (cols, k))
        else:
            A, B = rng.standard_normal((rows, k)), rng.standard_normal((cols, k))
        A[:, pos] = R
        B[:, pos] = R
        return A, B
    pos = {"spike1-first": np.arange(1), "spikes-first": np.arange(L),
           "spikes-permuted": rng.permutation(k)[:L],
           "spikes-groupstart": np.arange(0, k, 128)[:L]}[name]
    A, B = np.zeros((rows, k)), np.zeros((cols, k))
    A[:, pos] = 1.0
    B[:, pos] = 1.0
    return A, B


def chain(a: np.ndarray, b: np.ndarray, G: int):
    """a, b: (P, k) codes of the P outputs of one tile. Returns per-atom tile identity, per-atom tile exactness, word identity
    share, promotion identities (G > 0), each word's final value (the ticket word under G, the chained word when pure) and
    the share of (word, atom) steps that are exact."""
    P, k = a.shape
    T = k // 32
    C = np.zeros(P)
    total = np.zeros(P, dtype=np.float32)
    tid, tex = np.zeros(T, bool), np.zeros(T, bool)
    words, promo, wex = 0, 0, 0
    for t in range(T):
        if G and t % G == 0:
            C = np.zeros(P)
        D, ex = step(C, a[:, 32 * t:32 * t + 32], b[:, 32 * t:32 * t + 32])
        same = D == C
        tid[t], tex[t], words, wex = same.all(), ex.all(), words + int(same.sum()), wex + int(ex.sum())
        C = D
        if G and (t % G == G - 1 or t == T - 1):
            new = (total + C.astype(np.float32)).astype(np.float32)
            promo += int(((new == total) & (C != 0)).all())
            total = new
    return tid, tex, words / (P * T), promo, (total if G else C.astype(np.float32)), wex / (P * T)


def group_run(a: np.ndarray, b: np.ndarray, ts: list[int], skip: frozenset = frozenset()) -> tuple[np.ndarray, list[np.ndarray]]:
    """One promotion group's accumulator from +0 over atoms `ts`, the atoms in `skip` passed through; and each atom's
    identity flag (D == C) on the unskipped run."""
    C, ids = np.zeros(a.shape[0]), []
    for t in ts:
        if t in skip:
            continue
        D, _ = step(C, a[:, 32 * t:32 * t + 32], b[:, 32 * t:32 * t + 32])
        ids.append(D == C)
        C = D
    return C, ids


def debit(a: np.ndarray, b: np.ndarray, G: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """The replayed debit per word, P1's option (A) (`jointFlags`): on the noisy codes and the ticket's schedule (groups of G
    atoms from +0, each promoted into an FP32 total), flag every atom of every nonempty subset of one group whose joint
    skip leaves the word's final total unchanged. It contains the identities, the groups whose promotion is an identity,
    and every single or joint skip inside one group, however it is absorbed (by a later atom's grid or by a later
    promotion). Returns the (words, atoms) flags and P1's e_F per word (identities plus promotion-identity groups)."""
    return debit_parts(a, b, G)[:2]


def debit_parts(a: np.ndarray, b: np.ndarray, G: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`debit`, and which flags the group absorbs itself (a skip that leaves the group's accumulator unchanged, the atom's
    adder); the rest are absorbed only by the FP32 total's rounding in a later promotion."""
    P, k = a.shape
    T = k // 32
    groups = [list(range(g0, min(g0 + G, T))) for g0 in range(0, T, G)]
    Cs, before, eF = [], [], np.zeros(P, dtype=np.int64)
    total = np.zeros(P, dtype=np.float32)
    for ts in groups:
        C, ids = group_run(a, b, ts)
        new = (total + C.astype(np.float32)).astype(np.float32)
        eF += (np.stack(ids, 1) | (new == total)[:, None]).sum(1)
        Cs.append(C)
        before.append(total)
        total = new
    final = total
    flags = np.zeros((P, T), dtype=bool)
    inside = np.zeros((P, T), dtype=bool)
    for g, ts in enumerate(groups):
        for mask in range(1, 1 << len(ts)):
            skip = frozenset(t for i, t in enumerate(ts) if mask >> i & 1)
            Cm, _ = group_run(a, b, ts, skip)
            in_group = Cm == Cs[g]
            same = in_group
            if not same.all():
                t32 = (before[g] + Cm.astype(np.float32)).astype(np.float32)
                for h in range(g + 1, len(groups)):
                    t32 = (t32 + Cs[h].astype(np.float32)).astype(np.float32)
                same = same | (t32 == final)
            for t in skip:
                flags[:, t] |= same
                inside[:, t] |= in_group
    return flags, eF, inside


def group_sums(a: np.ndarray, b: np.ndarray, G: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Each word's group accumulators (words, groups) and its final promoted total (the ticket word)."""
    T = a.shape[1] // 32
    total, out = np.zeros(a.shape[0], dtype=np.float32), []
    for g0 in range(0, T, G):
        C, _ = group_run(a, b, list(range(g0, min(g0 + G, T))))
        out.append(C)
        total = (total + C.astype(np.float32)).astype(np.float32)
    return np.stack(out, 1), total


def _f32(x) -> np.float32:
    return np.float32(x)


def greedy_set(ap: np.ndarray, bp: np.ndarray, G: int = 4) -> set:
    """P1's greedy joint set for one word (`ap`, `bp`: (1, k) codes; pearlc_freeze_census.py, §6.2): atom pairs from two
    different groups whose joint skip leaves the ticket word unchanged while neither atom alone does, packed greedily, then
    every single atom that keeps it unchanged. The set is replayed from scratch before it is returned."""
    T = ap.shape[1] // 32
    n = -(-T // G)

    def gword(g: int, S: set) -> np.float32:
        C, _ = group_run(ap, bp, list(range(g * G, min((g + 1) * G, T))), frozenset(S))
        return _f32(C[0])

    W = np.array([gword(g, set()) for g in range(n)], dtype=np.float32)
    pre = np.zeros(n + 1, dtype=np.float32)
    for g in range(n):
        pre[g + 1] = _f32(pre[g] + W[g])
    out = pre[n]

    def output(W2: np.ndarray) -> np.float32:
        t = np.float32(0)
        for h in range(n):
            t = _f32(t + W2[h])
        return t

    Wskip = np.array([gword(t // G, {t}) for t in range(T)], dtype=np.float32)
    alone = np.array([output(np.where(np.arange(n) == t // G, Wskip[t], W)) == out for t in range(T)])
    pairs = []
    WsG = np.full(n * G, np.nan, dtype=np.float32)
    WsG[:T] = Wskip
    WsG = WsG.reshape(n, G)
    for t1 in range(T):
        g1 = t1 // G
        if alone[t1] or g1 + 1 >= n:
            continue
        run = np.empty(n + 1, dtype=np.float32)
        run[g1 + 1] = _f32(pre[g1] + Wskip[t1])
        for h in range(g1 + 1, n):
            run[h + 1] = _f32(run[h] + W[h])
        # X[i, j]: the final total with atom j of group g1 + 1 + i skipped as well
        X = (run[g1 + 1:n, None] + WsG[g1 + 1:n]).astype(np.float32)
        for h in range(g1 + 2, n):
            X[:h - g1 - 1] = (X[:h - g1 - 1] + W[h]).astype(np.float32)
        for i, j in np.argwhere(X == out):
            t2 = (g1 + 1 + int(i)) * G + int(j)
            if t2 < T and not alone[t2]:
                pairs.append((t1, t2))
    S: set = set()
    Wcur = W.copy()

    def try_add(new: set):
        S2 = S | new
        W2 = Wcur.copy()
        for g in {t // G for t in new}:
            W2[g] = gword(g, S2)
        return (S2, W2) if output(W2) == out else None

    for t1, t2 in pairs:
        if t1 not in S and t2 not in S and (r := try_add({t1, t2})):
            S, Wcur = r
    for t in range(T):
        if t not in S and (r := try_add({t})):
            S, Wcur = r
    assert word_with_skip(ap, bp, S, G) == out
    return S


def word_with_skip(ap: np.ndarray, bp: np.ndarray, S: set, G: int = 4) -> np.float32:
    """One word's ticket (the final promoted total) with the atoms in `S` skipped, replayed from scratch."""
    T = ap.shape[1] // 32
    total = np.float32(0)
    for g0 in range(0, T, G):
        C, _ = group_run(ap, bp, list(range(g0, min(g0 + G, T))), frozenset(S))
        total = _f32(total + _f32(C[0]))
    return total


def cross_group(a: np.ndarray, b: np.ndarray, flags: np.ndarray, words: np.ndarray, G: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """P1's greedy joint set for the sampled words: a lower bound on the largest jointly skippable set, whose cross-group part
    no replay-local debit covers (`joint_cross_group`). Returns the set size and its atoms outside the debit's flags."""
    sizes, beyond = np.zeros(len(words), dtype=np.int64), np.zeros(len(words), dtype=np.int64)
    for w_i, p in enumerate(words):
        S = greedy_set(a[p:p + 1], b[p:p + 1], G)
        sizes[w_i] = len(S)
        beyond[w_i] = sum(1 for t in S if not flags[p, t])
    return sizes, beyond


def transfer(a_sim: np.ndarray, b_sim: np.ndarray, a: np.ndarray, b: np.ndarray, flags: np.ndarray, words: np.ndarray,
             G: int = 4) -> dict:
    """The statement reviewer's transfer test (cross-group-transfer-probe.py): an adversary that knows the activations, the
    weights and F before the salt simulates a noise draw, finds the greedy set on it, and skips that set on the real draw.
    Per sampled word: does the real ticket word survive the skip, and how many of the set's atoms are outside the real
    draw's debit flags (the credit it would gain)? Beside it, the real draw's own set and its overlap with the simulated one."""
    rows = []
    for p in words:
        S_sim = greedy_set(a_sim[p:p + 1], b_sim[p:p + 1], G)
        ok = bool(word_with_skip(a[p:p + 1], b[p:p + 1], S_sim, G) == word_with_skip(a[p:p + 1], b[p:p + 1], set(), G))
        S_real = greedy_set(a[p:p + 1], b[p:p + 1], G)
        rows.append({"sim": len(S_sim), "transfers": ok and len(S_sim) > 0,
                     "undebited_if_transfers": sum(1 for t in S_sim if not flags[p, t]) if ok else 0,
                     "real": len(S_real), "overlap": len(S_sim & S_real)})
    nonempty = [r for r in rows if r["sim"] > 0]
    return {"transfer_words": len(rows), "transfer_nonempty_sim_sets": len(nonempty),
            "transfer_hits": sum(r["transfers"] for r in rows),
            "transfer_undebited_saving_max": max((r["undebited_if_transfers"] for r in rows), default=0),
            "transfer_overlap_share_of_real": round(float(np.mean([r["overlap"] / max(r["real"], 1) for r in rows])), 6)}


def not_needed_share(flags: np.ndarray, rows: int, cols: int) -> float:
    """P3's `needed`, on the tile: the share of (A row, atom) pairs that no word of the row needs (every word (i, j) has the
    atom flagged). A whole unit's rows meet many more columns, so the unit's share is at most the tile's."""
    return float(flags.reshape(rows, cols, -1).all(1).mean())


EXACT_RUN_MIN = 14      # atoms: one Winograd level's break-even on the H100 (450-700 of inner dimension); --exact-run-min


def exact_run_mask(tex: np.ndarray, min_run: int | None = None) -> np.ndarray:
    min_run = EXACT_RUN_MIN if min_run is None else min_run
    mask, start = np.zeros(len(tex), bool), None
    for i, e in enumerate(list(tex) + [False]):
        if e and start is None:
            start = i
        elif not e and start is not None:
            if i - start >= min_run:
                mask[start:i] = True
            start = None
    return mask


def exact_run_atoms(tex: np.ndarray, min_run: int | None = None) -> int:
    return int(exact_run_mask(tex, min_run).sum())


def in_domain(G: int, k: int, delta: float) -> bool:
    return G == 4 or k / delta ** 2 <= 2 ** 15


def val(codes: np.ndarray) -> np.ndarray:
    codes = codes.astype(np.int64)
    v = MAG[codes & 0x7F]
    return np.where(codes & 0x80, -v, v)


def family_rows(fam: str, k: int, delta: float, rows: int, cols: int, run: int):
    """A family's activation rows XA and weight rows XB, and the s5 F codes of each side, from the census's seeds."""
    FA = lines(k, _rng(run, "FA", k))
    XA, XB = family(fam, rows, cols, k, FA, _rng(run, fam, k, delta))
    return XA, XB, FA, lines(k, _rng(run, "FB", k))


def s5_lines(k: int, run: int, line_norm: float):
    return tuple(enc(unit_lines(k, _rng(run, f"{s}-lines", k)) * (line_norm / np.sqrt(32))) for s in ("FA", "FB"))


def cell(fam: str, k: int, delta: float, rows: int, cols: int, run: int, forming: str = "census",
         line_norm: float = 256.0, with_debit: bool = False, cross_words: int = 0, transfer_words: int = 0) -> list[dict]:
    XA, XB, FA, FB = family_rows(fam, k, delta, rows, cols, run)
    extra = {"forming": forming}
    if forming == "s5":
        FcA, FcB = s5_lines(k, run, line_norm)
        a, collA, subA, floorA, liveA = form_s5(XA, FcA, delta, line_norm, _rng(run, fam, k, delta, "unit-A"))
        b, collB, subB, floorB, liveB = form_s5(XB, FcB, delta, line_norm, _rng(run, fam, k, delta, "unit-B"))
        nfA, drA, drB = np.zeros(1), 0.0, 0.0
        ok = floorA and liveA and floorB and liveB
        extra.update({"line_norm": line_norm, "collision_A": round(collA, 6), "collision_B": round(collB, 6),
                      "e_prime_subnormal_A": round(subA, 6), "rules_ok": ok,
                      "rejected_by": [n for n, v in (("noise-floor-A", floorA), ("liveness-A", liveA),
                                                     ("noise-floor-B", floorB), ("liveness-B", liveB)) if not v]})
    else:
        a, nfA, drA = form(XA, FA, delta, _rng(run, fam, k, delta, "unit-A"))
        b, nfB, drB = form(XB, FB, delta, _rng(run, fam, k, delta, "unit-B"))
        ok = True
    ii, jj = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    A, B = a[ii.ravel()], b[jj.ravel()]
    out = []
    finals = {}
    for G in (0, 4):
        tid, tex, word_id, promo, finals[G], word_exact = chain(A, B, G)
        T = k // 32
        skip = int((tid | exact_run_mask(tex)).sum()) + promo
        row = {**extra, "family": fam, "k": k, "delta": delta, "layout": "G4" if G else "pure",
               "in_domain": in_domain(G, k, delta) and ok,
               "atoms": T, "tile_identity_atoms": int(tid.sum()), "exact_run_atoms": exact_run_atoms(tex),
               "tile_exact_atoms": int(tex.sum()), "word_exact_steps": round(word_exact, 6),
               "promotion_identities": promo, "word_identity": round(word_id, 6),
               "noise_free_entries": float(nfA.mean()), "double_rounding_differs": round((drA + drB) / 2, 6),
               "skippable_share": skip / T}
        if G:
            # The unpromoted attack: the pure chain's word in place of the ticket word, per word and whole tiles.
            same = finals[0] == finals[G]
            full = T // G * G
            row.update({"unpromoted_equal_words": round(float(same.mean()), 6), "unpromoted_equal_tile": bool(same.all()),
                        "tile_exact_groups": round(float(tex[:full].reshape(-1, G).all(1).mean()), 6) if full else 0.0})
        if G and with_debit:
            flags, eF, inside = debit_parts(A, B, G)
            per_word = flags.sum(1)
            # Every tile identity is flagged in every word, so under the debited credit only exact runs remain skippable.
            row.update({"debit_share": round(float(per_word.mean()) / T, 6), "debit_max_word_share": round(float(per_word.max()) / T, 6),
                        "debit_via_promotion_share": round(float((flags & ~inside).mean()), 6),
                        "debit_eF_share": round(float(eF.mean()) / T, 6), "not_needed_share": round(not_needed_share(flags, rows, cols), 6),
                        "tile_all_flagged_share": round(float(flags.all(0).mean()), 6), "residual_share": exact_run_atoms(tex) / T})
            if transfer_words and forming == "s5":
                sample = np.unique(np.linspace(0, len(A) - 1, transfer_words).astype(int))
                a_sim = form_s5(XA, FcA, delta, line_norm, _rng(run, fam, k, delta, "unit-A", "simulated"))[0]
                b_sim = form_s5(XB, FcB, delta, line_norm, _rng(run, fam, k, delta, "unit-B", "simulated"))[0]
                row.update(transfer(a_sim[ii.ravel()], b_sim[jj.ravel()], A, B, flags, sample, G))
            if cross_words:
                sample = np.unique(np.linspace(0, len(A) - 1, cross_words).astype(int))
                size, beyond = cross_group(A, B, flags, sample, G)
                row.update({"cross_group_words": len(sample), "cross_group_share_mean": round(float(size.mean()) / T, 6),
                            "cross_group_share_max": round(float(size.max()) / T, 6),
                            "cross_group_beyond_debit_mean": round(float(beyond.mean()) / T, 6),
                            "cross_group_beyond_debit_max": round(float(beyond.max()) / T, 6)})
            if forming == "s5":
                a2 = form_s5(XA, FcA, delta, line_norm, _rng(run, fam, k, delta, "unit-A", "salt-2"))[0]
                b2 = form_s5(XB, FcB, delta, line_norm, _rng(run, fam, k, delta, "unit-B", "salt-2"))[0]
                g1, t1 = group_sums(A, B, G)
                g2, t2 = group_sums(a2[ii.ravel()], b2[jj.ravel()], G)
                row.update({"group_sums_equal_across_salts": round(float((g1 == g2).mean()), 6),
                            "ticket_words_equal_across_salts": round(float((t1 == t2).mean()), 6)})
        out.append(row)
    return out


def main() -> None:
    global EXACT_RUN_MIN
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ks", default="8192,16384,65536")
    ap.add_argument("--deltas", default="1")
    ap.add_argument("--tile", default="64x16")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--forming", choices=("census", "s5"), default="census")
    ap.add_argument("--line-norm", type=float, default=256.0)
    ap.add_argument("--atom", default="hopper-e4m3-k32",
                    help=f"the FP8 atom model: one of {', '.join(ATOMS)}, or groups:width:zero")
    ap.add_argument("--exact-run-min", type=int, default=EXACT_RUN_MIN,
                    help="atoms in a tile-exact run before an exact rewrite pays (the card's break-even; 14 on the H100)")
    ap.add_argument("--debit", action="store_true",
                    help="replay each word's debit (eF and skip-one) under G4; the pass criterion becomes the undebited residual")
    ap.add_argument("--cross-words", type=int, default=0,
                    help="with --debit: run P1's greedy cross-group search on this many words per cell")
    ap.add_argument("--transfer-words", type=int, default=0,
                    help="with --debit and --forming s5: the reviewer's transfer test (simulated-draw greedy sets skipped on the real draw)")
    ap.add_argument("--out", default="pearlc_census.json")
    args = ap.parse_args()
    set_atom(args.atom)
    EXACT_RUN_MIN = args.exact_run_min
    bad = selftest()
    print(json.dumps({"selftest_cases": 3000, "atom": ATOM_NAME, "mismatches": bad}), flush=True)
    if args.selftest or bad:
        sys.exit(1 if bad else 0)
    rows, cols = map(int, args.tile.split("x"))
    t0, rows_out = time.time(), []
    for delta in map(float, args.deltas.split(",")):
        for k in map(int, args.ks.split(",")):
            for fam in args.families.split(","):
                for r in cell(fam, k, delta, rows, cols, args.run, args.forming, args.line_norm, args.debit, args.cross_words,
                              args.transfer_words):
                    rows_out.append(r)
                    print(json.dumps(r), flush=True)
    key = "residual_share" if args.debit else "skippable_share"
    scored = [r for r in rows_out if r["in_domain"] and key in r]
    worst = max((r[key] for r in scored), default=0.0)
    worst_row = max(scored, key=lambda r: r[key], default=None)
    doc = {"tool": "pouw_pearlc_census", "atom": ATOM_NAME, "atom_groups": list(ATOM_GROUPS), "atom_width": ATOM_WIDTH,
           "atom_zero": ATOM_ZERO, "exact_run_min": EXACT_RUN_MIN, "tile": args.tile, "forming": args.forming,
           "line_norm": args.line_norm,
           "selftest_mismatches": bad, "cells": rows_out,
           "criterion": key, "worst_in_domain_skippable_share": worst, "worst_in_domain_cell": worst_row, "threshold": 1 / 400,
           "seconds": round(time.time() - t0, 1)}
    out = Path(args.out)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    ok = bad == 0 and worst <= 1 / 400
    run_result.write(out, ok, "pearlc-census", f"worst in-domain {key} {worst:.5f} (threshold 0.0025)",
                     [("worst_in_domain_skippable_share", worst, "fraction"),
                      ("max_double_rounding_differs", max(r["double_rounding_differs"] for r in rows_out), "fraction")])
    print(json.dumps({k: v for k, v in doc.items() if k != "cells"}))


if __name__ == "__main__":
    main()
