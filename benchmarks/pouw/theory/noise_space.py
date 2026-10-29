"""The noise and de-noise design space as a grammar, enumerated, checked and scored.

A design is one choice per axis:

- side: where the noise sits. `A` (a row's lane against a public weight in B), `B` (a column's lane against a public
  weight in A), `AorB` (whichever of the two is cheaper, per linear), `A+B` (a row lane and a column lane each step,
  no column code needed), `AB` (a row lane times a column lane at one k position), `dither` (XOR'd into the low bit
  of an A data byte, against B's own odd weight), or no lane at all: `init` (the accumulator's initial value), `kord`
  (the k order), `scale` (the dequant scales), `sparsity` (2:4 metadata), `pad` (noise only in padding rows).
- inject: how a step's noise reaches its operand register, one lane a step merged with the data lanes. Fixed gap at
  byte 3: `rot1` (ssn_1l: rotate + LOP3, bit values 1), `rotB` for B in 2, 4, 8 (B adjacent bits of a rotated word at
  bits 24.., values 2^b, one rotation every B steps), `prmt` (PRMT's sign replicate: value 0 or -1 from the sign bit of
  any byte, one shift every four steps), `prmtV` for V in 1, 2, 4 (V shifted versions a coin word instead of 8: fewer
  shifts, more coins), `nib` (a 4-bit lane), `tele` (telescoping: lane t is x_t - x_{t-1}, so the
  final word's row-common noise is zero). Gap rotating over the four bytes (a data layout premise): `rgapB` for B in
  1, 2, 4, 8 (LOP3 with a mask, values 2^b, one shift every 4B steps), `lop1` (no shifts, four bits a coin word).
  `byte1`: one coin byte a step, PRMT'd into lane 3 from a coin word that serves four steps. Dither only: `xor1`.
  Or the step's whole 4-lane register is a raw coin word: `raw1` (fresh every step), `raw4` (one word for four steps,
  a byte a step), `rawS` (one word for the call).
- code: what tells the n columns apart at step 1. `b13s4` (ssn_1l's shifted base 13), `b13b4`, `b13b5` (balanced),
  `svL` (the n shortest nonzero vectors of Z^L, a public table, L up to 28 lanes), `rawc` (a raw coin word's 4 bytes
  against sv4's table), `initc` (the code added as each accumulator's initial value, an IDP4A an output), `colbits`
  (a row bit times a fresh column bit), `none`.
- denoise: `served` (O' = A·Bᵀ + noise), `epi_row` (the row-common part subtracted in the dequant epilogue: the
  row-scale FMUL becomes an FFMA with a per-row constant, premise `epilogue-ffma`), `epi_full` (also the code part,
  an IDP4A and an add an output), `cancel` (negated lanes in a last step, so the final word is exact), `mirror` (a lane
  at every step; the second half replays the first half's coins against negated weights, so the lanes' noise sums to
  zero and only the step-1 code is served; with an odd lane count one step carries a second lane).
- format: `s8` (s8 × s8 at ±127), `s8h126`, `odd` (odd int8 weights), `int7`, `u8s8`.
- c1: whether the formed operand registers are committed. model: M0 or M1 (only `cancel` reads it).
- tag: `tag` puts four public owner tags T(o) (balanced base-13 digits of the owner's index) in step 1 against
  coin-drawn weights G in [-3, 3] on the other side: in B per window for side A (no cost a call), in A per call for
  side B (two writes and a coin word a row). They stay served. nz: a static bit in the noised side's layout beside the
  coin bit keeps the lane nonzero (`NZ_INJECTS` only): `lo` 1 + 2^(b+1)·x, odd, so a sign folds into the same LOP3;
  `hi` 2^(b+1) + 2^b·x, at the plain lane's variance. weights: the public weight against the noise lane, per column
  and step, in B's layout: `ones`, `sign` (±1 drawn from the window's coins after B is committed; B's own lane for
  side B, which needs `lo`) or `wide` (nonzero int8 drawn the same way, side A); window weights cost nothing a call.

Three readings of the charging model are scored. Functional (the calculator's): a word is charged when it is pairwise
distinct from every other word and from every free value as a function on the salt grid. Salt-adaptive, natural data
(`dup_adaptive`): a prover that reads the coins before it computes also saves each word or row vector equal to another
at those coins, α · (zero dot share · μ₀ + identical owner share · κ), μ₀ the chance a step's noise increment is zero
and κ `a1_collapse` at the identical owners' group (noise_data.py's shares). Salt-adaptive, chosen data
(`dup_chosen`), with accumulators regrouped for free: α · max(κ over all owners identical (N-RED's A1), p, r), where
p is `cancel_share`, the mean over steps of the largest point mass of a step's increment: a data dot fixed before the
call's coins at minus that value leaves the word unchanged on p of the entries (N-RED's A2 and A2c, N-LEAN's §2b).
p is the bound over the call's coins alone, which holds however the data reads the window's weights; `p_fixed`, the
mass with the weights drawn too, is reported beside it. r is `revisit_share` (N-RED's A2r): data fixed before the
coins walks each entry so that its words revisit values it already holds, and an entry pays only for its distinct
values; `a2r_equal` reads it with every weight 1. The closed classes also read `tile_saving`: owners identical
with zero data, the words a prover saves by copying a TILE-word tile equal to another (one address write a tile),
from any owner and step, or skipping a zero word, by Monte Carlo.

`symbolic_check` decides data-independent distinctness exactly on a small instance: every committed word (and C1's
formed registers) is a multilinear polynomial in the coin bits with coefficients polynomial in the data, and a pair is
distinct for every input only if their difference has a coin monomial whose coefficient no input can zero.
`lower_bounds` states the bounds the frontier is measured against and checks them exhaustively at small sizes.
Output: a JSON report (`--json`), the frontier as a text table (`--table`); `--data` reads noise_data.py's output.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import math
import multiprocessing
import os
import sys
import zlib
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import noise_sweep as ns  # noqa: E402

H32 = ns.H32
BYTE_VAR = (256 ** 2 - 1) / 12
NIB_VAR = (16 ** 2 - 1) / 12
FIELD = 127 * 127

SIDES = ("A", "B", "AorB", "A+B", "AB", "dither", "init", "kord", "scale", "sparsity", "pad")
LANE_SIDES = ("A", "B", "AorB", "A+B", "AB", "dither")
INJECTS = ("rot1", "rot2", "rot4", "rot8", "prmt", "prmt1", "prmt2", "prmt4", "nib", "tele", "rgap1", "rgap2", "rgap4",
           "rgap8", "lop1", "xor1", "byte1", "raw1", "raw4", "rawS")
CODES = ("b13s4", "b13b4", "b13b5", "sv3", "sv4", "sv5", "sv6", "sv8", "sv12", "sv16", "sv28", "rawc", "initc",
         "colbits", "none")
DENOISES = ("served", "epi_row", "epi_full", "cancel", "mirror")
WEIGHTS = ("ones", "sign", "wide")
WIDE = np.array([v for v in range(-127, 128) if v])
FORMATS = ("s8", "s8h126", "odd", "int7", "u8s8")
RAW = {"raw1": 1, "raw4": 4, "rawS": None}
#: bit-lane injections: (bits used a byte before a shift, slots a version, versions a word, value of slot b, gap fixed)
BITS = {"rot1": (1, 1, 32, 1, True), "rot2": (2, 2, 16, None, True), "rot4": (4, 4, 8, None, True),
        "rot8": (8, 8, 4, None, True), "prmt": (1, 4, 8, -1, True), "prmt1": (1, 4, 1, -1, True),
        "prmt2": (1, 4, 2, -1, True), "prmt4": (1, 4, 4, -1, True), "nib": (4, 1, 8, "nib", True),
        "tele": (1, 4, 8, -1, True), "rgap1": (1, 4, 8, 1, False), "rgap2": (2, 8, 4, None, False),
        "rgap4": (4, 16, 2, None, False), "rgap8": (8, 32, 1, None, False), "lop1": (1, 4, 1, 1, False),
        "xor1": (1, 4, 8, 1, True)}
FORMAT_COST = {"s8": 0, "s8h126": 1, "odd": 1, "int7": 1, "u8s8": 1}
FORMAT_T3 = {"s8": "", "s8h126": "activations at +-126", "odd": "odd weights: a 7-bit weight grid",
             "int7": "7-bit operands", "u8s8": "unsigned activations"}
#: noise_data.py's shares: zero k32 dot products, identical activation rows, identical weight columns (by row scope)
SHARES = {64: {"zero_dot": 0.0014888, "id_rows": 0.0016164, "id_cols": 0.0},
          512: {"zero_dot": 0.0003587, "id_rows": 0.0044229, "id_cols": 0.0},
          8192: {"zero_dot": 0.0003601, "id_rows": 0.0050583, "id_cols": 0.0}}
SHARES_SOURCE = "built-in"


def load_shares(path: str) -> None:
    global SHARES_SOURCE
    with open(path) as fh:
        d = json.load(fh)
    s = d["summary"]
    ident = any(w.get("identical_wrows", 0) for w in d.get("weights", {}).values())
    for rows, scope in ((64, "m64"), (512, "ctx"), (8192, "all")):
        SHARES[rows] = {"zero_dot": s[scope]["zero_dot_share"], "id_rows": s[scope]["identical_row_share"],
                        "id_cols": float(ident)}
    SHARES_SOURCE = path


def _shares(m: int) -> dict:
    return SHARES[min(SHARES, key=lambda r: abs(math.log(max(m, 1) / r)))]


# ---------------------------------------------------------------- column codes

def _balanced_digits(z: np.ndarray, lanes: int) -> np.ndarray:
    out, r = [], z.copy()
    for _ in range(lanes):
        d = ((r + 6) % 13) - 6
        out.append(d)
        r = (r - d) // 13
    assert (r == 0).all(), "code too small for n"
    return np.stack(out)


def _shell(lanes: int, w: int) -> list[tuple[int, ...]]:
    """Every vector of Z^lanes with squared norm w."""
    out: list = []

    def rec(prefix: list, left: int, rem: int) -> None:
        if rem == 0:
            if left == 0:
                out.append(tuple(prefix))
            return
        b = math.isqrt(left)
        for x in range(-b, b + 1):
            prefix.append(x)
            rec(prefix, left - x * x, rem - 1)
            prefix.pop()

    rec([], w, lanes)
    return out


@lru_cache(maxsize=None)
def _sv_explicit(lanes: int, n: int) -> np.ndarray:
    """The n shortest nonzero vectors of Z^lanes, by norm then lexicographically."""
    out, w = [], 1
    while len(out) < n:
        out += sorted(_shell(lanes, w))
        w += 1
    return np.array(out[:n], dtype=np.int64).T


@lru_cache(maxsize=None)
def _theta(lanes: int, top: int) -> np.ndarray:
    """r_L(w): how many vectors of Z^lanes have squared norm w, for w < top (as floats: only compared with n)."""
    base = np.zeros(top)
    for x in range(-math.isqrt(top - 1), math.isqrt(top - 1) + 1):
        base[x * x] += 1
    out = np.zeros(top)
    out[0] = 1
    for _ in range(lanes):
        out = np.convolve(out, base)[:top]
    return out


def code_lanes(code: str) -> int:
    if code.startswith("sv"):
        return int(code[2:])
    return {"b13s4": 4, "b13b4": 4, "b13b5": 5, "rawc": 4, "initc": 0, "colbits": 1, "none": 0}[code]


def code_capacity(code: str) -> int:
    if code == "b13s4":
        return 13 ** 4 - 2
    if code.startswith("b13b"):
        return 13 ** code_lanes(code) - 1
    return 10 ** 12


@lru_cache(maxsize=None)
def code_matrix(code: str, n: int) -> np.ndarray:
    """lanes × n public weights of step 1's code lanes (rawc: sv4's table; initc: the init's weights)."""
    if code == "b13s4":
        return ns.ssn_a_w0(n, ns.DATA_PER_STEP)[0]
    if code.startswith("b13b"):
        j = np.arange(n, dtype=np.int64)
        return _balanced_digits(np.where(j % 2 == 0, j // 2 + 1, -(j // 2 + 1)), code_lanes(code))
    if code.startswith("sv"):
        return _sv_explicit(code_lanes(code), n)
    if code in ("rawc", "initc"):
        return _sv_explicit(4, n)
    if code == "colbits":
        return np.ones((1, n), dtype=np.int64)
    return np.zeros((0, n), dtype=np.int64)


@lru_cache(maxsize=None)
def code_norms(code: str, n: int) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """The distribution of |c_j|^2 over the n columns: (values, counts)."""
    if code in ("none", "colbits"):
        return (1.0,), (n,)
    if code.startswith("sv") or code in ("rawc", "initc"):
        lanes, top = (code_lanes(code) if code.startswith("sv") else 4), 64
        while _theta(lanes, top)[1:].sum() < n:
            top *= 2
        assert top <= 127 ** 2, "shortest vectors past int8"
        th = _theta(lanes, top)
        vals, cnts, left = [], [], n
        for w in range(1, len(th)):
            take = min(left, int(th[w]))
            if take:
                vals.append(float(w))
                cnts.append(take)
                left -= take
            if not left:
                break
        assert not left, "theta table too short"
        return tuple(vals), tuple(cnts)
    c = code_matrix(code, n)
    v, k = np.unique((c.astype(np.int64) ** 2).sum(0), return_counts=True)
    return tuple(float(x) for x in v), tuple(int(x) for x in k)


def code_unit_var(code: str) -> float:
    return {"rawc": BYTE_VAR, "colbits": 3 / 16, "none": 0.0}.get(code, 0.25)


def code_mu_pair(code: str, s: int) -> float:
    """The chance two owners' (rows') step-1 codes agree at one column: a lane's bit, byte, or the product's."""
    return {"rawc": 1 / 256, "colbits": 0.75, "none": 0.0}.get(code, 0.5)


def code_writes(code: str) -> tuple[int, int]:
    """(writes, formed registers) a row a call for step 1's code: one LOP3 for each full 4-lane register (a shift
    before each after the first), a rotate and a LOP3 for lanes sharing a data register."""
    if code in ("none", "initc", "rawc"):
        return (1 if code == "initc" else 0), 0
    if code == "colbits":
        return 1, 1
    lanes = code_lanes(code)
    full, part = divmod(lanes, 4)
    return full + max(0, full - 1) + (2 if part else 0), full + (1 if part else 0)


# ---------------------------------------------------------------- per-step injection

@dataclass(frozen=True)
class Inject:
    writes: int
    formed: int
    words: int
    var: float
    mu: float
    lev_steps: int
    epi: int
    rmax: float
    layout: bool


def _slot_vals(inj: str, count: int) -> list[float]:
    b_bits, _, _, val, _ = BITS[inj]
    if val == "nib":
        return [15.0] * count
    if val is not None:
        return [float(abs(val))] * count
    return [float(2 ** (t % b_bits)) for t in range(count)]


def lane_steps(inj: str, s: int, code: str, dn: str = "served") -> int:
    """Steps carrying a noise lane: 2..S, and step 1 too when there is no code (and for tele, whose path x_1 .. x_{S-1},
    0 must start away from the code's own noise, and for mirror, so that no step's increment is the code's alone)."""
    return max(0, s - 1) + (1 if code == "none" or inj == "tele" or dn == "mirror" else 0)


def lane_plan(count: int, mirror: bool) -> list[list[tuple[int, int]]]:
    """For each lane step, the (coin lane, sign) pairs it carries: lane u alone, or for mirror the first half's lanes
    replayed negated in order, the unpaired middle lane negated in the last step."""
    if not mirror:
        return [[(u, 1)] for u in range(count)]
    h = math.ceil(count / 2)
    plan = [[(u, 1)] for u in range(h)] + [[(u, -1)] for u in range(count - h)]
    if count % 2:
        plan[-1].append((h - 1, -1))
    return plan


@lru_cache(maxsize=None)
def inject_cost(inj: str, s: int, code: str, nz: str = "none", dn: str = "served") -> Inject:
    """Writes, coin words and served row-common noise of the noise lanes for one owner (a row; a column for side B)."""
    steps = lane_steps(inj, s, code, dn)
    mirror = dn == "mirror"
    coined = math.ceil(steps / 2) if mirror else steps
    code_words = 0 if code == "none" else 1
    if inj == "byte1":
        words = code_words + math.ceil(coined / 4)
        return Inject(steps + (steps % 2 if mirror else 0), coined, words, coined * BYTE_VAR, 1 / 256, 1,
                      2 * math.ceil(coined / 4), 128.0 * steps, False)
    if inj in RAW:
        reuse = RAW[inj] or max(1, steps)
        raws = math.ceil(steps / reuse) if steps else 0
        if inj == "rawS":
            counts = [len(range(e, steps, 4)) for e in range(4)]
            var = sum(c * c for c in counts) * BYTE_VAR
            lev = max(counts) if steps else 0
        else:
            var, lev = steps * BYTE_VAR, 1
        rmax = 128.0 * steps
        return Inject(0, 0, code_words + raws, var, 1 / 256, lev, 2 * raws, rmax, False)
    b_bits, per_version, versions_word, val, fixed = BITS[inj]
    if inj == "rot1" and code in ("b13s4", "b13b4") and not mirror:
        writes = ns.ssn_1l_forming_writes(s) - 1
        words = ns.ssn_1l_words(s)
    else:
        code_bits = {"none": 0, "rawc": 0, "initc": 4}.get(code, code_lanes(code))
        width = 4 if val == "nib" else 1
        slots = max(0, coined - (1 if inj == "tele" and coined else 0))
        versions = math.ceil(slots / per_version) if slots else 0
        lane_words = math.ceil(versions / versions_word) if versions else 0
        words = max(lane_words, math.ceil((code_bits + slots * width) / 32)) + (code == "rawc")
        writes = (steps + (2 if mirror else 1) * max(0, versions - lane_words)
                  + (2 * max(0, steps - 1) if inj == "tele" else 0) + (steps % 2 if mirror else 0))
    vals = _slot_vals(inj, coined)
    if val == "nib":
        var, mu, epi = steps * NIB_VAR, 1 / 16, 3 * words
    else:
        var = sum(v * v for v in vals) / 4 * (4 if nz == "lo" else 1)
        mu = 0.0 if nz != "none" else 0.5
        epi = 2 * words if b_bits in (1, 8) else (3 * (8 // b_bits) + 1) * words
    if inj == "tele":
        return Inject(writes, steps, words, 0.0, 0.5, 2, 0, 1.0, not fixed)
    rmax = {"none": sum(vals), "lo": 2 * sum(vals) + steps, "hi": 3 * sum(vals)}[nz]
    return Inject(writes, coined, words, var, mu, 1, epi, rmax, not fixed)


#: LOP3 injections with a spare bit beside the coin bit, which `nz` sets in the noised side's layout
NZ_INJECTS = ("rot1", "rot2", "rot4", "rgap1", "rgap2", "rgap4", "lop1")
NZ_FORMS = ("none", "lo", "hi")
TAG_LANES = 4


def steps_for(k: int, code: str, lanes_per_step: int, inj: str = "", extra_first: int = 0) -> int:
    first = 32 - code_lanes(code) - extra_first - (lanes_per_step if code == "none" or inj == "tele" else 0)
    return 1 + math.ceil(max(0, k - first) / (32 - lanes_per_step))


TAG_MAX = 13 ** TAG_LANES - 1


def owner_tags(owners: int) -> np.ndarray:
    """TAG_LANES × owners public tags: balanced base-13 digits of +-1, +-2, … (distinct and nonzero up to TAG_MAX)."""
    j = np.arange(owners, dtype=np.int64)
    return _balanced_digits(np.where(j % 2 == 0, j // 2 + 1, -(j // 2 + 1)), TAG_LANES)


@lru_cache(maxsize=None)
def tag_norm2(owners: int) -> float:
    """Mean |T(o)|^2 of the owners' tags."""
    return float((owner_tags(owners).astype(float) ** 2).sum(0).mean())


def _code_states(code: str) -> float:
    """How many equally likely step-1 code values an owner draws (distinct codes give distinct word vectors)."""
    return {"rawc": 2.0 ** 32, "none": 1.0, "colbits": 2.0, "initc": 16.0}.get(code, 2.0 ** code_lanes(code))


def _occupied(p: np.ndarray, weight: float, q: float, m: int) -> float:
    """E[distinct states] among m draws from q equally likely codes times a walk of point masses p (each standing for
    `weight` states of mass p)."""
    return float((weight * q * -np.expm1(m * np.log1p(-np.minimum(p / q, 1 - 1e-16)))).sum())


def _coin_states(inj: str) -> float:
    return 256.0 if inj in RAW or inj == "byte1" else (16.0 if inj == "nib" else 2.0)


def _active(plan: list[list[tuple[int, int]]]) -> list[list[int]]:
    """The coin lanes whose noise is still in the accumulator after each lane step."""
    coef: dict = {}
    out = []
    for lanes in plan:
        for c, sg in lanes:
            coef[c] = coef.get(c, 0) + sg
        out.append(sorted(c for c, v in coef.items() if v))
    return out


def _gauss_occupied(var: float, q: float, m: int) -> float:
    if not var:
        return _occupied(np.ones(1), 1.0, q, m)
    sd = math.sqrt(var)
    step = max(1.0, sd / 400)
    x = np.arange(-8 * sd, 8 * sd + step, step)
    return _occupied(np.exp(-x * x / (2 * var)) / math.sqrt(2 * math.pi * var), step, q, m)


@lru_cache(maxsize=None)
def a1_collapse(inj: str, s: int, code: str, m: int, history: bool = False, dn: str = "served") -> float:
    """N-RED's A1 at the vector level: m rows with equal data, the mean over steps of the share of row vectors (a row's
    words at one step) that equal another row's at that step, which an adaptive prover copies instead of computing.
    A row's vector at step t is fixed by its code coins and its owner-common cumulative noise D_t; with `history` (AB,
    dither, window weights) the noise varies per column, so the state is the row's coin history."""
    if m < 2:
        return 0.0
    q, count = _code_states(code), lane_steps(inj, s, code, dn)
    start, out = s - count, []
    cap = 60 / math.log2(_coin_states(inj))
    if dn == "mirror":
        active = _active(lane_plan(count, True))
        vals = _slot_vals(inj, len(active)) if inj in BITS else []
        for t in range(s):
            lanes = active[t - start] if t >= start else []
            if history:
                states = q * _coin_states(inj) ** min(cap, len(lanes))
                out.append(1 - states * -math.expm1(m * math.log1p(-1 / states)) / m)
                continue
            if inj == "byte1":
                dist = _gauss_occupied(len(lanes) * BYTE_VAR, q, m)
            else:
                pmf = np.ones(1)
                for c in lanes:
                    pmf = (np.convolve(pmf, np.full(16, 1 / 16)) if inj == "nib"
                           else np.convolve(pmf, np.r_[0.5, np.zeros(int(vals[c]) - 1), 0.5]))
                dist = _occupied(pmf[pmf > 0], 1.0, q, m)
            out.append(1 - min(dist, m) / m)
        return float(np.mean(out))
    if history:
        for t in range(s):
            states = q * _coin_states(inj) ** min(cap, max(0, t - start + 1))
            out.append(1 - states * -math.expm1(m * math.log1p(-1 / states)) / m)
        return float(np.mean(out))
    pmf, var = np.ones(1), 0.0
    vals = _slot_vals(inj, count) if inj in BITS else []
    for t in range(s):
        lane = t - start
        if lane >= 0:
            if inj == "byte1":
                var = (lane + 1) * BYTE_VAR
            elif inj in RAW:
                var = (sum(c * c for c in (len(range(e, lane + 1, 4)) for e in range(4))) if inj == "rawS"
                       else lane + 1) * BYTE_VAR
            elif inj == "tele":
                pmf = np.ones(1) if t == s - 1 else np.array([0.5, 0.5])
            elif inj == "nib":
                pmf = np.convolve(pmf, np.full(16, 1 / 16))
            else:
                v = int(vals[lane])
                nxt = np.zeros(len(pmf) + v)
                nxt[:len(pmf)] += pmf / 2
                nxt[v:] += pmf / 2
                pmf = nxt
        dist = _gauss_occupied(var, q, m) if var else _occupied(pmf[pmf > 0], 1.0, q, m)
        out.append(1 - min(dist, m) / m)
    return float(np.mean(out))


TILE = 4
TILE_POSITIONS = 8


def _lane_draws(inj: str, count: int, owners: int, nz: str, rng: np.random.Generator) -> np.ndarray:
    """count × owners lane values drawn as `_lane_values` builds them."""
    if count <= 0:
        return np.zeros((0, owners), dtype=np.int64)
    if inj == "byte1":
        return rng.integers(-128, 128, (count, owners))
    if inj in RAW:
        reuse, t = RAW[inj] or count, np.arange(count)
        words = rng.integers(0, 2 ** 32, (math.ceil(count / reuse), owners), dtype=np.int64)
        byte = (words[t // reuse] >> (8 * (t % 4))[:, None]) & 255
        return np.where(byte >= 128, byte - 256, byte)
    if inj == "nib":
        return rng.integers(0, 16, (count, owners))
    x = rng.integers(0, 2, (count, owners))
    if inj == "tele":
        if count == 1:
            return x
        out = np.empty((count, owners), dtype=np.int64)
        out[0], out[1:-1], out[-1] = -x[0], x[:count - 2] - x[1:count - 1], x[count - 2]
        return out
    v = np.array(_slot_vals(inj, count), dtype=np.int64)[:, None]
    if nz == "lo":
        return 1 + 2 * v * x
    if nz == "hi":
        return 2 * v + v * x
    return (-1 if BITS[inj][3] == -1 else 1) * v * x


def _code_part(code: str, owners: int, cols: np.ndarray, others: int, rng: np.random.Generator) -> np.ndarray:
    """owners × len(cols) step-1 code noise: each owner's code coins against the code's public weights."""
    if code == "none":
        return np.zeros((owners, len(cols)), dtype=np.int64)
    if code == "colbits":
        return rng.integers(0, 2, (owners, 1)) * rng.integers(0, 2, (1, len(cols)))
    cm = code_matrix(code, others)[:, cols].astype(np.int64)
    lo, hi = (-128, 128) if code == "rawc" else (0, 2)
    return rng.integers(lo, hi, (owners, cm.shape[0])) @ cm


def _weight_draws(weights: str, count: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    if weights == "sign":
        return rng.choice(np.array([-1, 1]), (count, cols))
    if weights == "wide":
        return rng.choice(WIDE, (count, cols))
    return np.ones((count, cols), dtype=np.int64)


@lru_cache(maxsize=None)
def tile_saving(side: str, inj: str, s: int, code: str, owners: int, others: int, nz: str = "none",
                tag: str = "none", weights: str = "ones", dn: str = "served") -> float:
    """N-RED's tile-level A1 with zero tails, by Monte Carlo: `owners` owners with equal, zero data, at TILE_POSITIONS
    tiles of TILE consecutive others; the share of all (owner, step) words a prover saves by copying a tile equal to
    one already written (one address write for TILE words) or by leaving a zero word unwritten."""
    rng = np.random.default_rng(zlib.crc32(repr((side, inj, s, code, owners, others, nz, tag, weights, dn)).encode()))
    count = lane_steps(inj, s, code, dn)
    start = s - count
    plan = lane_plan(count, dn == "mirror")
    coined = 1 + max((c for lanes in plan for c, _ in lanes), default=-1)
    mult = rng.integers(1, 2 ** 63, TILE, dtype=np.uint64) | np.uint64(1)
    saved = total = 0.0
    for pos in rng.choice(others // TILE, size=min(TILE_POSITIONS, others // TILE), replace=False):
        cols = np.arange(pos * TILE, (pos + 1) * TILE)
        cur = _code_part(code, owners, cols, others, rng)
        if tag != "none":
            cur = cur + owner_tags(owners).T @ rng.integers(-3, 4, (TAG_LANES, TILE))
        lanes = _lane_draws(inj, coined, owners, nz, rng)
        w = _lane_draws(inj, coined, TILE, nz, rng) if side == "AB" else _weight_draws(weights, coined, TILE, rng)
        keys, zeros = [], []
        for t in range(s):
            if t >= start:
                for c, sg in plan[t - start]:
                    cur = cur + sg * lanes[c][:, None] * w[c][None, :]
            keys.append((cur.astype(np.uint64) * mult).sum(1))
            zeros.append((cur == 0).sum(1))
        key, zero = np.concatenate(keys), np.concatenate(zeros)
        _, first, cnt = np.unique(key, return_index=True, return_counts=True)
        zu = zero[first]
        saved += float(np.where(zu == TILE, cnt * TILE, zu + (cnt - 1) * (TILE - 1)).sum())
        total += key.size * TILE
    return saved / total if total else 0.0


Pmf = tuple  # ((value, mass), ...)


def _pmf(values, masses=None) -> Pmf:
    v = np.asarray(values, dtype=np.int64).ravel()
    w = np.full(v.size, 1 / v.size) if masses is None else np.asarray(masses, dtype=float).ravel()
    u, inv = np.unique(v, return_inverse=True)
    return tuple(zip(u.tolist(), np.bincount(inv, weights=w).tolist()))


def _conv(a: Pmf, b: Pmf) -> Pmf:
    va, ma = np.array([x for x, _ in a]), np.array([y for _, y in a])
    vb, mb = np.array([x for x, _ in b]), np.array([y for _, y in b])
    return _pmf(va[:, None] + vb[None, :], ma[:, None] * mb[None, :])


def _times(a: Pmf, b: Pmf) -> Pmf:
    va, ma = np.array([x for x, _ in a]), np.array([y for _, y in a])
    vb, mb = np.array([x for x, _ in b]), np.array([y for _, y in b])
    return _pmf(va[:, None] * vb[None, :], ma[:, None] * mb[None, :])


def _top(a: Pmf) -> float:
    return max(y for _, y in a)


def lane_pmfs(inj: str, count: int, nz: str = "none") -> list[Pmf]:
    """Each coin lane's value over the call's coins (tele: the per-step difference, its middle steps' form)."""
    if inj in RAW or inj == "byte1":
        return [_pmf(range(-128, 128))] * count
    if inj == "nib":
        return [_pmf(range(16))] * count
    if inj == "tele":
        return [_pmf([-1, 0, 0, 1])] * count
    if inj == "xor1":
        return [_pmf([0, 1])] * count
    sign = -1 if BITS[inj][3] == -1 else 1
    out = []
    for v in _slot_vals(inj, count):
        v = int(v)
        out.append(_pmf({"none": [0, sign * v], "lo": [1, 1 + 2 * v], "hi": [2 * v, 3 * v]}[nz]))
    return out


WEIGHT_PMF = {"ones": _pmf([1]), "sign": _pmf([-1, 1]), "wide": _pmf(WIDE)}
CANCEL_COLUMNS = 64


@lru_cache(maxsize=None)
def _code_top(code: str, others: int) -> float:
    """The step-1 code part's largest point mass, averaged over CANCEL_COLUMNS evenly spread columns."""
    if code in ("none", "initc"):
        return 1.0
    if code == "colbits":
        return 0.75
    cols = np.unique(np.linspace(0, others - 1, min(others, CANCEL_COLUMNS)).astype(int))
    cm = code_matrix(code, others)[:, cols].astype(np.int64)
    coin = _pmf(range(-128, 128)) if code == "rawc" else _pmf([0, 1])
    tops = []
    for j in range(cm.shape[1]):
        acc: Pmf = ((0, 1.0),)
        for c in cm[:, j]:
            if c:
                acc = _conv(acc, _times(coin, ((int(c), 1.0),)))
        tops.append(_top(acc))
    return float(np.mean(tops))


@lru_cache(maxsize=None)
def _term_tops(side: str, lane: Pmf, sg: int, weights: str) -> tuple[float, float]:
    """(fixed, bound) largest masses of one lane's increment: a row lane times a column lane (AB), plus one (A+B), or
    times its window weight."""
    if side == "AB":
        t = _top(_times(lane, lane))
        return t, t
    if side == "A+B":
        t = _top(_conv(lane, lane))
        return t, t
    wt = WEIGHT_PMF[weights]
    return _top(_times(lane, _pmf([sg * v for v, _ in wt], [y for _, y in wt]))), _top(lane)


@lru_cache(maxsize=None)
def cancel_share(side: str, inj: str, s: int, code: str, others: int, nz: str = "none", weights: str = "ones",
                 dn: str = "served") -> tuple[float, float]:
    """N-LEAN's §2b: the mean over the S steps of the largest point mass of one entry's step increment, which a data
    dot fixed before the call's coins cancels (the word repeats) on that share of the entries. Returns (fixed, bound):
    `fixed` with the window's weights drawn too, `bound` over the call's coins alone, for any known nonzero weight.
    A step with two lanes, or step 1's code beside a lane, takes the smaller lane's mass: P(X + Y = c) is at most
    either's largest mass for independent X, Y."""
    if side == "dither":
        return 0.5, 0.5
    count = lane_steps(inj, s, code, dn)
    start = s - count
    plan = lane_plan(count, dn == "mirror")
    coined = 1 + max((c for lanes in plan for c, _ in lanes), default=-1)
    lanes = lane_pmfs(inj, coined, nz)
    fixed, bound = [], []
    for t in range(s):
        tops = [_term_tops(side, lanes[c], sg, weights) for c, sg in (plan[t - start] if t >= start else [])]
        ft, b = min([x for x, _ in tops], default=1.0), min([y for _, y in tops], default=1.0)
        if t == 0:
            ct = _code_top(code, others)
            ft, b = min(ft, ct), min(b, ct)
        fixed.append(ft)
        bound.append(b)
    return float(np.mean(fixed)), float(np.mean(bound))


REVISIT_ENTRIES = 4096


def _repeat_share(inc: np.ndarray) -> float:
    """The mean over entries (columns) of the share of an entry's words, the running sums of `inc` (steps × entries),
    that equal an earlier word of the entry."""
    pos = np.sort(np.cumsum(inc, axis=0), axis=0)
    return float(1 - (1 + (np.diff(pos, axis=0) != 0).sum(0)).mean() / inc.shape[0])


def _walk_share(lane_inc: np.ndarray, lo: np.ndarray, hi: np.ndarray, mu: np.ndarray, plan: list, start: int,
                s: int) -> float:
    """The best data strategy's `_repeat_share`. `lane_inc` (lanes × entries) is each coin lane's increment and lo, hi,
    mu its least, largest and mean value as the data knows them. The strategies: no data, each step's increment
    centred on its mean, and N-RED's lazy walk, which cancels the increment's least value at odd steps and its largest
    at even ones."""
    shape = (s, lane_inc.shape[1])
    inc, lo_s, hi_s, mu_s = (np.zeros(shape) for _ in range(4))
    lo, hi, mu = (np.broadcast_to(a, lane_inc.shape) for a in (lo, hi, mu))
    for t in range(start, s):
        for c, sg in plan[t - start]:
            inc[t] += sg * lane_inc[c]
            lo_s[t] += lo[c] if sg > 0 else -hi[c]
            hi_s[t] += hi[c] if sg > 0 else -lo[c]
            mu_s[t] += sg * mu[c]
    odd = (np.arange(s) % 2 == 0)[:, None]
    return max(_repeat_share(inc), _repeat_share(inc - np.rint(mu_s)), _repeat_share(inc - np.where(odd, lo_s, hi_s)))


@lru_cache(maxsize=None)
def revisit_share(side: str, inj: str, s: int, count: int, nz: str = "none", weights: str = "ones",
                  dn: str = "served", entries: int = REVISIT_ENTRIES) -> tuple[float, float]:
    """N-RED's A2r under model P, by Monte Carlo: data fixed before the call's coins, knowing the window's weights, sets
    each entry's data increment a step (`_walk_share`'s strategies), and an entry pays only for its distinct words.
    `count` is `lane_steps`. Returns (drawn, equal), the share of an entry's S words that repeat an earlier word of the
    entry: with the window's weights drawn, and with every weight 1, the lattice walk that weights the data chooses
    would allow. Step 1's code and tags are constant within an entry and drop out; dither is not modelled and is
    charged in full."""
    if side == "dither":
        return 1.0, 1.0
    start = s - count
    plan = lane_plan(count, dn == "mirror")
    coined = 1 + max((c for lanes in plan for c, _ in lanes), default=-1)
    rng = np.random.default_rng(zlib.crc32(repr(("revisit", side, inj, s, count, nz, weights, dn, entries)).encode()))
    pm = lane_pmfs(inj, coined, nz)
    lo = np.array([min(v for v, _ in q) for q in pm], dtype=float)[:, None]
    hi = np.array([max(v for v, _ in q) for q in pm], dtype=float)[:, None]
    mu = np.array([sum(v * y for v, y in q) for q in pm], dtype=float)[:, None]
    rows = _lane_draws(inj, coined, entries, nz, rng)
    if side in ("AB", "A+B"):
        cols = _lane_draws(inj, coined, entries, nz, rng)
        if side == "A+B":
            r = _walk_share(rows + cols, 2 * lo, 2 * hi, 2 * mu, plan, start, s)
        else:
            ends = np.stack([lo * lo, lo * hi, hi * hi])
            r = _walk_share(rows * cols, ends.min(0), ends.max(0), mu * mu, plan, start, s)
        return r, r
    out = []
    for w in (_weight_draws(weights, coined, entries, rng), np.ones((coined, entries), dtype=np.int64)):
        a, b = lo * w, hi * w
        out.append(_walk_share(rows * w, np.minimum(a, b), np.maximum(a, b), mu * w, plan, start, s))
    return out[0], out[1]


def entropy_revisits(steps: tuple = (67, 265), bits: tuple = tuple(range(1, 13)), entries: int = REVISIT_ENTRIES) -> list:
    """The least coin entropy a step that survives A2r: one uniform e-bit coin lane a step on every step, against weight
    1 or a drawn wide window weight, served or mirrored; each row is A2r's share of an entry's words (`_walk_share`)."""
    out = []
    for s in steps:
        for e in bits:
            rng = np.random.default_rng(zlib.crc32(repr(("entropy", s, e, entries)).encode()))
            half = 2 ** (e - 1)
            row = {"steps": s, "bits": e}
            for dn in ("served", "mirror"):
                plan = lane_plan(s, dn == "mirror")
                coined = 1 + max(c for lanes in plan for c, _ in lanes)
                lanes = rng.integers(-half, half, (coined, entries)) if e > 1 else rng.integers(0, 2, (coined, entries))
                lo, hi = (0, 1) if e == 1 else (-half, half - 1)
                mu = (lo + hi) / 2
                for wname in ("ones", "wide"):
                    w = _weight_draws(wname, coined, entries, rng)
                    a, b = lo * w, hi * w
                    row[f"{dn}_{wname}"] = _walk_share(lanes * w, np.minimum(a, b), np.maximum(a, b), mu * w, plan, 0, s)
            out.append(row)
    return out


# ---------------------------------------------------------------- compiled design

@dataclass(frozen=True)
class Point:
    side: str
    inject: str
    code: str
    denoise: str
    fmt: str
    c1: bool
    model: str
    tag: str = "none"
    nz: str = "none"
    weights: str = "ones"

    @property
    def name(self) -> str:
        base = f"{self.side}/{self.inject}/{self.code}/{self.denoise}/{self.fmt}/{'c1' if self.c1 else 'noc1'}/{self.model}"
        return (base + ("+tag" if self.tag != "none" else "") + {"none": "", "lo": "+nz", "hi": "+nzh"}[self.nz]
                + {"ones": "", "sign": "+ws", "wide": "+wv"}[self.weights])


def premises(p: Point) -> list[str]:
    """Open premises a design's guarantee or price reads, beyond the calculator's shared model."""
    out = []
    if p.denoise == "cancel" and p.model == "M1":
        out.append("M1")
    if p.denoise in ("epi_row", "epi_full"):
        out.append("epilogue-ffma")
    return out


def kernel_assumptions(p: Point) -> list[str]:
    out = []
    if p.inject in BITS and not BITS[p.inject][4] and p.side != "dither":
        out.append("gap-layout")
    if p.side in ("B", "A+B", "AB") or p.code == "colbits":
        out.append("b-forming")
    if p.weights != "ones":
        out.append("b-window")
    return out


@lru_cache(maxsize=None)
def _side_terms(side: str, inj: str, code: str, dn: str, c1: bool, m: int, n: int, k: int, nz: str = "none",
                tag: str = "none", weights: str = "ones") -> dict:
    """A design's terms before its format's sign fix and its model's final word."""
    if inj == "xor1" and side != "dither" or side == "dither" and inj != "xor1":
        return {"invalid": "xor1 is dither's injection, and dither's only"}
    if side in ("AB", "A+B") and (inj in RAW or inj == "byte1"):
        return {"invalid": "a byte lane a side is AB's or A+B's only with a byte weight a call; not in the grammar"}
    if nz != "none" and (inj not in NZ_INJECTS or side == "dither"):
        return {"invalid": "nz needs a LOP3 lane with a spare bit"}
    if tag != "none" and side not in ("A", "B"):
        return {"invalid": "owner tags sit in the noised side's step 1, against coin-drawn weights on the other"}
    if weights != "ones" and (side not in ("A", "B") or nz == "none" and inj != "byte1"):
        return {"invalid": "window weights scale one noise lane a side in B's layout, and move only a lane that is "
                           "never zero"}
    if side == "B" and (weights == "wide" or weights == "sign" and nz != "lo"):
        return {"invalid": "side B's weight sits in its own lane: a sign folds into one LOP3 only on an odd (nz lo) "
                           "lane, and a wide weight needs a multiply"}
    if weights != "ones" and dn in ("epi_row", "epi_full", "cancel"):
        return {"invalid": "window weights make the lanes' noise rank S, which only mirror removes"}
    if dn == "mirror" and (side not in ("A", "B") or inj in RAW or inj == "tele" or code in ("none", "initc")):
        return {"invalid": "mirror replays one lane a side against negated weights after a served step-1 code"}
    own, other = (n, m) if side == "B" else (m, n)
    if other > code_capacity(code):
        return {"invalid": f"code {code} holds {code_capacity(code)} < {other}"}
    if tag != "none" and own > TAG_MAX:
        return {"invalid": f"tags hold {TAG_MAX} owners"}
    lanes = 4 if inj in RAW else (0 if side == "dither" else (2 if side == "A+B" else 1))
    s = steps_for(k, code, lanes, inj, (TAG_LANES if tag != "none" else 0) + (lanes if dn == "mirror" else 0))
    cost = inject_cost(inj, s, code, nz, "mirror" if dn == "mirror" else "served")
    cw, cformed = code_writes(code)
    alpha = H32 * s / k
    per_owner = cost.writes + cw
    form = per_owner * H32 / (other * k)
    form_other, forced_other, coins_other = 0.0, 0.0, 0.0
    if code == "colbits":
        form_other += H32 / (own * k)
        forced_other += H32 / (own * k)
        coins_other += other * 32
    if code == "initc":
        form_other += H32 / k
    if side in ("A+B", "AB"):
        form_other += cost.writes * H32 / (own * k)
        forced_other += cost.formed * H32 / (own * k)
        coins_other += other * 32 * cost.words
    if tag != "none" and side == "B":
        form_other += 2 * H32 / (own * k)
        forced_other += H32 / (own * k)
        coins_other += other * 32
    forced = ((cost.formed + cformed) * H32 / (other * k) + forced_other) if c1 else 0.0
    coin_bits = own * 32 * cost.words + coins_other
    norms, counts = code_norms(code, other)
    var_code = np.repeat(np.array(norms) * code_unit_var(code), counts)
    if side == "AB":
        row_var = (s - 1) * 3 / 16
    elif side == "A+B":
        row_var = 2 * cost.var
    elif side == "dither":
        row_var = (s - 1) * ns.WEIGHT_SIGMA_INT8 ** 2 / 4
    else:
        row_var = cost.var * float((WIDE.astype(float) ** 2).mean() if weights == "wide" else 1.0)
    if tag != "none":
        var_code = var_code + tag_norm2(own) * 4.0
    mu_step = {"AB": 0.0 if nz != "none" else 0.75, "A+B": cost.mu ** 2}.get(side, cost.mu)
    p_fixed, p_bound = cancel_share(side, inj, s, code, other, nz, weights, dn)
    adn = "mirror" if dn == "mirror" else "served"
    r_drawn, r_equal = revisit_share(side, inj, s, lane_steps(inj, s, code, adn), nz, weights, adn)
    denoise, final, extra_lanes = 0.0, 0.0, 0
    lcode = code_lanes(code) if code not in ("initc",) else 4
    if dn == "epi_row":
        if side in ("AB", "A+B", "dither"):
            return {"invalid": "no owner-common part to fold"}
        if inj == "tele":
            return {"invalid": "tele's final owner-common noise is already zero"}
        denoise = (cost.epi + 2) * H32 / (other * k)
        row_var = 0.0
    elif dn == "epi_full":
        per_out = 1 + math.ceil(lcode / 4) if code != "none" else 0
        if side == "AB":
            per_out += 2 * cost.words + 1
        elif side == "A+B":
            per_out += 1
        elif side == "dither":
            per_out += 1
            denoise += (s - 1) / k
        denoise += per_out * H32 / k + (cost.epi + 2) * H32 / (other * k)
        if side == "A+B":
            denoise += (cost.epi + 2) * H32 / (own * k)
        row_var, var_code = 0.0, var_code * 0
    elif dn == "cancel":
        if side in ("AB", "A+B", "dither"):
            return {"invalid": "cancel needs the noise as an owner sum plus a code"}
        rmax = cost.rmax if inj != "tele" else 0.0
        extra_lanes = lcode + math.ceil(rmax / FIELD) + (1 if rmax else 0)
        if code == "rawc":
            extra_lanes += 3
        denoise = (2 + cost.epi + 2 * math.ceil(extra_lanes / 4) + 1) * H32 / (other * k)
        row_var, var_code = 0.0, var_code * 0
    elif dn == "mirror":
        extra_lanes = lane_steps(inj, s, code, dn) % 2
        row_var = 0.0
    alpha_tot = alpha + extra_lanes / k
    gap = max(0.0, form + form_other - forced) + denoise
    coins = coin_bits / 8 * ns.COIN_PRICED_BYTE / (m * n * k)
    sh = _shares(m)
    history = side in ("AB", "dither") or weights != "ones"
    if tag != "none":
        kappa_chosen = 0.0
    else:
        kappa_chosen = a1_collapse(inj, s, code, own, history, adn)
    group = max(2, round(sh["id_rows"] * m))
    kappa_nat = 0.0 if side == "B" or tag != "none" else a1_collapse(inj, s, code, group, history, adn)
    dup_nat = sh["zero_dot"] * mu_step + sh["id_rows"] * kappa_nat
    if side == "B":
        dup_nat += sh["id_cols"] * a1_collapse(inj, s, code, max(2, round(sh["id_cols"] * n)), history, adn)
    dup_chosen = max(kappa_chosen, p_bound, r_drawn)
    std = float(np.sqrt(var_code + row_var).mean())
    qerr = math.sqrt(k / 12) * ns.WEIGHT_SIGMA_INT8
    return {"alpha": alpha_tot, "s": s, "forming": form + form_other, "forced": forced, "denoise": denoise,
            "final_word": 0.0, "sign_fix": 0.0, "gap": gap, "coins": coins, "dup_adaptive": alpha_tot * dup_nat,
            "dup_chosen": alpha_tot * dup_chosen, "a1_chosen": alpha_tot * kappa_chosen,
            "a2_chosen": alpha_tot * p_bound, "a2_fixed": alpha_tot * p_fixed, "p_bound": p_bound,
            "p_fixed": p_fixed, "a2r": alpha_tot * r_drawn, "a2r_equal": alpha_tot * r_equal, "std": std, "row_std": math.sqrt(row_var),
            "code_std": float(np.sqrt(var_code).mean()), "t3_ratio": std / qerr, "mu_step": mu_step,
            "coin_bits_owner": 32 * cost.words, "coin_bits_per_mac": coin_bits / (m * n * k),
            "leverage_words_per_bit": cost.lev_steps * other,
            "c1_bytes": forced / H32 * 4 if c1 else 0.0, "writes_owner_per_step": per_owner / max(1, s - 1),
            "side_B": float(side == "B"), "side": side, "owners": own, "others": other}


@lru_cache(maxsize=None)
def terms(p: Point, m: int, n: int, k: int) -> dict:
    """One linear's T1 terms (shares of stock's m·n·K MACs), coins, α, the served noise and the adaptive dup."""
    if p.side not in LANE_SIDES:
        return {"invalid": "no data-independent noise in the committed words"}
    sides = ("A", "B") if p.side == "AorB" else (p.side,)
    cands = [_side_terms(sd, p.inject, p.code, p.denoise, p.c1, m, n, k, p.nz, p.tag, p.weights) for sd in sides]
    good = [c for c in cands if "invalid" not in c]
    if not good:
        return cands[0]
    t = dict(min(good, key=lambda c: c["gap"]))
    t["final_word"] = H32 / k if p.denoise == "cancel" and p.model == "M0" else 0.0
    t["sign_fix"] = H32 / k if p.fmt == "u8s8" else 0.0
    t["gap"] += t["final_word"] + t["sign_fix"]
    t["gap_coins_uncredited"] = t["gap"] + t["coins"]
    t["gap_adaptive"] = t["gap"] + t["dup_adaptive"]
    t["gap_chosen"] = t["gap"] + t["dup_chosen"]
    return t


WORKLOADS = {"llama_T512_64tok": (ns.LLAMA, 64), "llama_T512_512tok": (ns.LLAMA, 512), "llama_T8192": (ns.LLAMA, 8192),
             "v4dense_T512_64tok": (ns.V4_DENSE, 64), "v4dense_T8192": (ns.V4_DENSE, 8192),
             "v4expert_T8192": (ns.V4_EXPERT, round(8192 * ns.V4_EXPERT_SHARE))}


class _Duck:
    """What noise_sweep.t2_estimate reads from a design."""

    def __init__(self, p: Point):
        self.p = p

    def alpha(self, k: int) -> float:
        return terms(self.p, 8192, 2048, k)["alpha"]

    def c1_bytes(self, m: int, n: int, k: int) -> float:
        return terms(self.p, m, n, k)["c1_bytes"]

    def coin_bits(self, m: int, n: int, k: int) -> float:
        return terms(self.p, m, n, k)["coins"] * (m * n * k) * 8 / ns.COIN_PRICED_BYTE


def score(p: Point) -> dict:
    out = {"point": dict(p.__dict__), "name": p.name, "premises": premises(p), "kernel": kernel_assumptions(p)}
    for wl, (shapes, m) in WORKLOADS.items():
        acc, tot = {}, 0.0
        for n, k in shapes.values():
            t = terms(p, m, n, k)
            if "invalid" in t:
                out["invalid"] = t["invalid"]
                return out
            w = m * n * k
            for key, v in t.items():
                if key not in ("side", "owners", "others"):
                    acc[key] = acc.get(key, 0.0) + w * v
            tot += w
        out[wl] = {key: v / tot for key, v in acc.items()}
    for point in ns.STEP:
        out[f"t2_{point}"] = ns.t2_estimate(_Duck(p), point, p.c1)
    return out


# ---------------------------------------------------------------- exact symbolic distinctness on a small instance

Poly = dict  # frozenset(coin vars) -> {tuple(data syms): int}


def _add(a: Poly, b: Poly, sign: int = 1) -> Poly:
    out = {mo: dict(c) for mo, c in a.items()}
    for mono, coef in b.items():
        tgt = out.setdefault(mono, {})
        for d, v in coef.items():
            tgt[d] = tgt.get(d, 0) + sign * v
            if tgt[d] == 0:
                del tgt[d]
        if not tgt:
            del out[mono]
    return out


def _mul(a: Poly, b: Poly) -> Poly:
    out: Poly = {}
    for m1, c1 in a.items():
        for m2, c2 in b.items():
            for d1, v1 in c1.items():
                for d2, v2 in c2.items():
                    out = _add(out, {m1 | m2: {tuple(sorted(d1 + d2)): v1 * v2}})
    return out


def _scale(a: Poly, c: int) -> Poly:
    return {mo: {d: v * c for d, v in cf.items()} for mo, cf in a.items()} if c else {}


def _data(sym: str) -> Poly:
    return {frozenset(): {(sym,): 1}}


def _bit(v: int) -> Poly:
    return {frozenset({v}): {(): 1}}


def _sure(poly: Poly, odd: set) -> bool:
    """A non-constant coin monomial whose coefficient no input can zero: a nonzero constant or one odd weight."""
    for mono, coef in poly.items():
        if not mono or len(coef) != 1:
            continue
        (d, v), = coef.items()
        if v and (d == () or (len(d) == 1 and d[0] in odd)):
            return True
    return False


class _Coins:
    def __init__(self):
        self.n = 0
        self.words: list[list[int]] = []
        self.pools: dict = {}

    def word(self) -> list[int]:
        w = list(range(self.n, self.n + 32))
        self.n += 32
        self.words.append(w)
        return w

    def bits(self, owner, count: int) -> list[int]:
        pool, out = self.pools.setdefault(owner, []), []
        for _ in range(count):
            if not pool or pool[-1][1] >= 32:
                pool.append([self.word(), 0])
            out.append(pool[-1][0][pool[-1][1]])
            pool[-1][1] += 1
        return out

    def byte(self, word: list[int], e: int) -> Poly:
        out: Poly = {}
        for b in range(8):
            out = _add(out, _scale(_bit(word[8 * e + b]), 2 ** b if b < 7 else -128))
        return out


def _lane_values(inj: str, coins: _Coins, owner, count: int, nz: str = "none") -> list[Poly]:
    """Noise lane values for steps 2..s of one owner."""
    if count <= 0:
        return []
    if nz != "none":
        out = []
        for x, v in zip(coins.bits(owner, count), _slot_vals(inj, count)):
            static, coin = (1, 2 * int(v)) if nz == "lo" else (2 * int(v), int(v))
            out.append(_add({frozenset(): {(): static}}, _scale(_bit(x), coin)))
        return out
    if inj == "byte1":
        vals, word = [], None
        for t in range(count):
            if t % 4 == 0:
                word = coins.word()
            vals.append(coins.byte(word, t % 4))
        return vals
    if inj in RAW:
        reuse, vals, word = RAW[inj] or count, [], None
        for t in range(count):
            if t % reuse == 0:
                word = coins.word()
            vals.append(coins.byte(word, t % 4))
        return vals
    if inj == "nib":
        out = []
        for _ in range(count):
            b = coins.bits(owner, 4)
            v: Poly = {}
            for e in range(4):
                v = _add(v, _scale(_bit(b[e]), 2 ** e))
            out.append(v)
        return out
    if inj == "tele":
        if count == 1:
            return [_bit(coins.bits(owner, 1)[0])]
        xs = coins.bits(owner, count - 1)
        vals = [_scale(_bit(xs[0]), -1)]
        for t in range(1, count - 1):
            vals.append(_add(_scale(_bit(xs[t]), -1), _scale(_bit(xs[t - 1]), -1), -1))
        vals.append(_bit(xs[-1]))
        return vals
    xs = coins.bits(owner, count)
    sv = _slot_vals(inj, count)
    sign = -1 if BITS[inj][3] == -1 else 1
    return [_scale(_bit(x), sign * int(v)) for x, v in zip(xs, sv)]


def _public_weight(weights: str, x: int, c: int) -> int:
    """A window weight for (other, coin lane) in the symbolic instance: fixed and mixed, as a window's draw is once
    known."""
    if weights == "sign":
        return 1 - 2 * ((x * 7 + c * 3 + (x * c) % 5) % 2)
    if weights == "wide":
        w = (x * 37 + c * 11 + (x * c) % 7) % 254 + 1
        return w if w <= 127 else 127 - w
    return 1


def symbolic_check(p: Point, m: int = 2, n: int = 3, s: int = 5) -> dict:
    """Committed words (and C1's formed registers) of an m × n × s-step instance as polynomials; count those that some
    input makes equal to another or to a free value (a constant, a salt-independent function, a raw coin word)."""
    side, inj, code = p.side, p.inject, p.code
    if side == "AorB":
        a = symbolic_check(replace(p, side="A"), m, n, s)
        b = symbolic_check(replace(p, side="B"), m, n, s)
        return {k: (a[k] and b[k]) if k == "valid" else a[k] + b[k] for k in a}
    coins = _Coins()
    odd = {f"b{j}_{t}" for j in range(n) for t in range(s)} if p.fmt == "odd" else set()
    data_a = {(i, t): _data(f"a{i}_{t}") for i in range(m) for t in range(s)}
    data_b = {(j, t): _data(f"b{j}_{t}") for j in range(n) for t in range(s)}
    inc_extra: dict = {}
    init: dict = {}
    formed: list[Poly] = []

    def add_inc(i, j, t, poly):
        inc_extra[(i, j, t)] = _add(inc_extra.get((i, j, t), {}), poly)

    if side in ("A", "B", "A+B", "AB", "dither"):
        own_rows = side != "B"
        owners, others = (range(m), range(n)) if own_rows else (range(n), range(m))

        def at(o, x):
            return (o, x) if own_rows else (x, o)

        col_bits = {x: _bit(coins.bits(("oth", x), 1)[0]) for x in others} if code == "colbits" else {}
        if p.c1:
            formed += [_add(_data(f"w{x}_0"), _scale(v, 2 ** 24)) for x, v in col_bits.items()]
        for o in owners:
            if code == "none":
                pass
            elif code == "colbits":
                rb = _bit(coins.bits(("own", o), 1)[0])
                for x in others:
                    add_inc(*at(o, x), 0, _mul(rb, col_bits[x]))
                if p.c1:
                    formed.append(_add(_data(f"r{o}_0"), _scale(rb, 2 ** 24)))
            else:
                lanes = code_lanes(code) if code != "initc" else 4
                cm = code_matrix(code, len(others))
                if code == "rawc":
                    wd = coins.word()
                    vals = [coins.byte(wd, e) for e in range(4)]
                else:
                    vals = [_bit(b) for b in coins.bits(("own", o), lanes)]
                for x in others:
                    tot: Poly = {}
                    for e, v in enumerate(vals):
                        tot = _add(tot, _scale(v, int(cm[e, x])))
                    if code == "initc":
                        init[at(o, x)] = tot
                    else:
                        add_inc(*at(o, x), 0, tot)
                if p.c1 and code not in ("rawc", "initc"):
                    reg: Poly = {}
                    for e, v in enumerate(vals[:4]):
                        reg = _add(reg, _scale(v, 256 ** e))
                    formed.append(reg)
        if side == "dither":
            for o in owners:
                for t, x in enumerate(coins.bits(("own", o), s - 1), start=1):
                    data_a[(o, t)] = _add(data_a[(o, t)], _bit(x))
                    if p.c1:
                        formed.append(data_a[(o, t)])
        else:
            count = lane_steps(inj, s, code, p.denoise)
            start = s - count
            plan = lane_plan(count, p.denoise == "mirror")
            coined = 1 + max((c for lanes in plan for c, _ in lanes), default=-1)
            own_vals = {o: _lane_values(inj, coins, ("own", o), coined, p.nz) for o in owners}
            oth_vals = ({x: _lane_values(inj, coins, ("oth", x), coined, p.nz) for x in others}
                        if side in ("A+B", "AB") else {})
            if p.tag != "none":
                tags = owner_tags(len(owners))
                for x in others:
                    gb = coins.bits(("tag", x), 3 * TAG_LANES)
                    g = [_add(_add(_bit(gb[3 * e]), _scale(_bit(gb[3 * e + 1]), 2)),
                              _add(_scale(_bit(gb[3 * e + 2]), 4), {frozenset(): {(): -3}})) for e in range(TAG_LANES)]
                    for o in owners:
                        tot: Poly = {}
                        for e in range(TAG_LANES):
                            tot = _add(tot, _scale(g[e], int(tags[e, o])))
                        add_inc(*at(o, x), 0, tot)
            for o in owners:
                for u, step_lanes in enumerate(plan):
                    t = start + u
                    for c, sg in step_lanes:
                        v = own_vals[o][c]
                        if p.c1 and inj not in RAW and sg > 0:
                            formed.append(_add(_data(f"r{o}_{t}_{c}"), _scale(v, 2 ** 24)))
                        for x in others:
                            if side == "AB":
                                add_inc(*at(o, x), t, _mul(v, oth_vals[x][c]))
                            else:
                                add_inc(*at(o, x), t, _scale(v, sg * _public_weight(p.weights, x, c)))
            if side == "A+B":
                for x in others:
                    for t, v in enumerate(oth_vals[x], start=start):
                        for o in owners:
                            add_inc(*at(o, x), t, v)
    elif side == "init":
        for i in range(m):
            for j in range(n):
                init[(i, j)] = _bit(coins.bits(("init", i, j), 1)[0])
    elif side in ("kord", "sparsity"):
        for t in range(s):
            x = _bit(coins.bits(("perm", t), 1)[0])
            for i in range(m):
                for j in range(n):
                    swap = _add(_mul(_data(f"c{i}_{t}"), _data(f"d{j}_{t}")), _mul(data_a[(i, t)], data_b[(j, t)]), -1)
                    add_inc(i, j, t, _mul(x, swap))
    elif side == "pad":
        coins.bits(("pad",), s)

    words, cum = [], {}
    for t in range(s):
        for i in range(m):
            for j in range(n):
                inc = _add(_mul(data_a[(i, t)], data_b[(j, t)]), inc_extra.get((i, j, t), {}))
                cum[(i, j)] = _add(cum.get((i, j), init.get((i, j), {})), inc)
                words.append(((i, j, t), cum[(i, j)]))
    if p.denoise == "cancel":
        words = [((i, j, t), {mo: c for mo, c in w.items() if not mo} if t == s - 1 else w) for (i, j, t), w in words]
    free = []
    for w in coins.words:
        h: Poly = {}
        for b, v in enumerate(w):
            h = _add(h, _scale(_bit(v), 2 ** b if b < 31 else -(2 ** 31)))
        free.append(h)
    allw = [w for _, w in words] + formed
    bad = set()
    for a, wa in enumerate(allw):
        if not _sure(wa, odd) or any(not _sure(_add(wa, h, -1), odd) for h in free):
            bad.add(a)
        for b in range(a + 1, len(allw)):
            if not _sure(_add(wa, allw[b], -1), odd):
                bad.update((a, b))
    final_bad = sum(1 for a in bad if a < len(words) and words[a][0][2] == s - 1)
    return {"words": len(words), "formed": len(formed), "bad": len(bad), "bad_final": final_bad,
            "valid": len(bad) == 0 or (p.denoise == "cancel" and len(bad) == final_bad == m * n)}


# ---------------------------------------------------------------- lower bounds, checked exhaustively at small sizes

def lane_bound(n: int, r: int) -> int:
    """The fewest step-1 noise lanes whose public weights in [-r, r] give n distinct nonzero columns."""
    lanes = 0
    while (2 * r + 1) ** lanes - 1 < n:
        lanes += 1
    return lanes


def _compositions(total: int, parts: int, cap: int):
    if parts == 1:
        if total <= cap:
            yield (total,)
        return
    for x in range(min(total, cap) + 1):
        for rest in _compositions(total - x, parts - 1, cap):
            yield (x,) + rest


def exhaustive_lanes(n: int, r: int, s: int, cap: int) -> dict:
    """Over every assignment of fresh-bit noise lanes (public weights in [-r, r] a column, at most `cap` a step) to s
    steps, in order of total lanes: the fewest total for which every word of a row is distinct for every input. A
    word's noise is the cumulative weight vector of its column, so distinctness is exactly: every (column, step) vector
    nonzero and all pairwise different. Every total below the bound is searched in full; at the bound the search stops
    at the first assignment that works, trying step 1 at L(n, r) lanes first (fewer is impossible by pigeonhole)."""
    weights = list(itertools.product(range(-r, r + 1), repeat=n))
    bound, l1 = lane_bound(n, r) + (s - 1), lane_bound(n, r)
    configs, found = 0, None
    for total in range(bound + 1):
        for comp in sorted(_compositions(total, s, cap), key=lambda c: c[0] != l1):
            for combo in itertools.product(*[list(itertools.combinations_with_replacement(range(len(weights)), c))
                                             for c in comp]):
                configs += 1
                prefix, vecs = [], []
                for c in combo:
                    prefix = prefix + [weights[w] for w in c]
                    for j in range(n):
                        vecs.append(tuple(lane[j] for lane in prefix) + (0,) * (bound + 2 - len(prefix)))
                if all(any(v) for v in vecs) and len(set(vecs)) == len(vecs):
                    found = (total, comp)
                    break
            if found:
                break
        if found:
            break
    return {"n": n, "r": r, "s": s, "min_total": found and found[0], "witness_lanes_a_step": found and found[1],
            "bound_total": bound, "bound_step1": l1, "configs": configs,
            "pass": bool(found) and found[0] == bound and found[1][0] == l1}


def variance_floor_check(max_support: int = 5, grid: int = 8) -> dict:
    """Var X >= (1 - mu)/2 for an integer X whose largest point mass is mu, checked over every distribution on
    {0..max_support-1} with masses in multiples of 1/grid; and the bound is met by a fair bit."""
    worst = math.inf
    count = 0
    for masses in itertools.product(range(grid + 1), repeat=max_support):
        if sum(masses) != grid:
            continue
        count += 1
        p = np.array(masses) / grid
        x = np.arange(max_support)
        var = float((p * x * x).sum() - (p * x).sum() ** 2)
        mu = float(p.max())
        if mu < 1:
            worst = min(worst, var - (1 - mu) / 2)
    return {"distributions": count, "min_slack": worst, "pass": bool(worst >= -1e-12),
            "tight_at_bit": 0.25 == (1 - 0.5) / 2}


def mass_floor_check(max_support: int = 7, grid: int = 12) -> dict:
    """Var X >= (1/p^2 - 1)/12 for an integer X whose largest point mass is p, checked over every distribution on
    {0..max_support-1} with masses in multiples of 1/grid; equality at a uniform on 1/p consecutive integers."""
    worst, count = math.inf, 0
    x = np.arange(max_support)
    for masses in itertools.product(range(grid + 1), repeat=max_support):
        if sum(masses) != grid:
            continue
        count += 1
        p = np.array(masses) / grid
        var = float((p * x * x).sum() - (p * x).sum() ** 2)
        worst = min(worst, var - (1 / p.max() ** 2 - 1) / 12)
    return {"distributions": count, "min_slack": worst, "pass": bool(worst >= -1e-12)}


def cancel_floor(var_served: float, s: int) -> float:
    """§2b's floor for a served design whose S step increments are independent: their variances sum to the served
    variance V, each step's largest mass is >= f(v) = (12v + 1)^(-1/2), and f is convex, so the mean is >= f(V/S)."""
    return 1 / math.sqrt(12 * var_served / s + 1)


def cancel_floors() -> dict:
    """§2b's floor at T3_SAFE, MAC-weighted over each workload's linears, with ssn_1l's steps."""
    out = {}
    for wl, (shapes, m) in WORKLOADS.items():
        num = den = 0.0
        for n, k in shapes.values():
            v = (T3_SAFE * math.sqrt(k / 12) * ns.WEIGHT_SIGMA_INT8) ** 2
            num += m * n * k * cancel_floor(v, ns.steps1l(k))
            den += m * n * k
        out[wl] = num / den
    return out


def coin_bound(m: int, n: int, s: int, mu: float, q: int) -> float:
    """Coin bits >= m·n·(S-1)·log2(1/mu)/q: each step increment needs min-entropy log2(1/mu), its entropy is at most
    the number of coin bits it reads, and each bit is read by at most q increments."""
    return m * n * (s - 1) * math.log2(1 / mu) / q


def lower_bounds() -> dict:
    lanes = [exhaustive_lanes(2, 1, 4, 1), exhaustive_lanes(3, 1, 3, 2), exhaustive_lanes(4, 1, 2, 2)]
    s = ns.steps1l(8192)
    return {
        "lanes": lanes,
        "variance": variance_floor_check(),
        "mass": mass_floor_check(),
        "cancel_floor_at_t3_safe": cancel_floors(),
        "coins_ssn_1l_llama_down_T8192": {"bound_bits_row": coin_bound(1, 2048, s, 0.5, 2048),
                                          "ssn_1l_bits_used_row": s + 3,
                                          "ssn_1l_bits_expanded_row": 32 * ns.ssn_1l_words(s)},
        "forming_floor_share": {wl: sum(m * n * k * (H32 * (ns.steps1l(k) - 1) / (max(m, n) * k))
                                        for n, k in shapes.values()) / sum(m * n * k for n, k in shapes.values())
                                for wl, (shapes, m) in WORKLOADS.items()},
        "statements": {
            "lanes": "every k32 step t >= 2 needs a noise lane whose public counterpart is nonzero for every dot "
                     "product (else some input makes w(i,j,t) = w(i,j,t-1)); step 1 needs L lanes with "
                     "(2r+1)^L - 1 >= n for public weights in [-r, r]. So noise lanes >= L(n, r) + (S - 1).",
            "forming": "the coins are derived from the operands' committed roots, so a register holding both data and "
                       "noise is written after A and B are committed and is no free value: >= 1 write an owner a step "
                       "on the side with fewer owners, min(m, n)·(S - 1) a call, unless the noise register is a raw "
                       "coin word (0 writes) at alpha >= 32/28.",
            "coins": "coin bits >= m·n·(S - 1)·log2(1/mu)/q for step increments of largest point mass mu when no bit "
                     "is read by more than q increments (entropy counting). ssn_1l meets it at mu = 1/2, q = n: one "
                     "bit a row a step. Functional distinctness alone needs only one bit a call (noise (index)·x), "
                     "which leaves every word free on the salts where x = 0.",
            "t3": "a served step increment of largest point mass mu has variance >= (1 - mu)/2 (Var X = E(X - X')^2/2 "
                  ">= P(X != X')/2), so a fair bit's 1/4 is the floor at mu = 1/2; the owner-common sum can be zero "
                  "(tele, epi_row); the column part of an L-lane bit code is >= the mean |c|^2/4 of the n shortest "
                  "vectors of Z^L, which tends to 1/4 as L -> n/2.",
            "cancel": "a data dot fixed before the coins at minus a step increment's likeliest value leaves the word "
                      "unchanged with that value's mass p (§2b). An integer X with largest mass p has Var X >= "
                      "(1/p^2 - 1)/12 (put mass p on the integers nearest the mean). So a served design with "
                      "independent step increments and served variance V gives up a mean share >= (12V/S + 1)^(-1/2) "
                      "of its words when regrouping is free: cancel_floor_at_t3_safe at V = (T3_SAFE·qerr)^2. Only "
                      "dependent increments (mirror, cancel) or noise removed in the epilogue go below it.",
        },
    }


# ---------------------------------------------------------------- enumeration and frontier

def enumerate_points() -> list[Point]:
    pts = []
    for side, inj, code, dn, fmt, c1, model in itertools.product(SIDES, INJECTS, CODES, DENOISES, FORMATS,
                                                                 (False, True), ("M0", "M1")):
        if side not in LANE_SIDES and (inj != "rot1" or code != "none" or dn != "served" or c1 or fmt != "s8"):
            continue
        if (side == "dither") != (inj == "xor1"):
            continue
        if model == "M1" and dn != "cancel":
            continue
        nzs = NZ_FORMS if inj in NZ_INJECTS and side in LANE_SIDES and side != "dither" else ("none",)
        tags = ("none", "tag") if side in ("A", "B", "AorB") else ("none",)
        for nz, tag in itertools.product(nzs, tags):
            moves = (nz != "none" or inj == "byte1") and side in ("A", "B", "AorB")
            for weights in (WEIGHTS if moves else ("ones",)):
                pts.append(Point(side, inj, code, dn, fmt, c1, model, tag, nz, weights))
    return pts


OBJECTIVES = (("llama_T512_64tok", "gap"), ("llama_T8192", "gap"), ("llama_T512_64tok", "gap_coins_uncredited"),
              ("llama_T8192", "gap_coins_uncredited"), ("llama_T512_64tok", "gap_adaptive"),
              ("llama_T8192", "gap_chosen"), ("t2_T8192", "t2"), ("llama_T8192", "std"), ("premises", None),
              ("format", None))


def _obj(r: dict) -> tuple:
    out = []
    for a, b in OBJECTIVES:
        if a == "premises":
            out.append(len(r["premises"]))
        elif a == "format":
            out.append(FORMAT_COST[r["point"]["fmt"]])
        else:
            out.append(r[a][b])
    return tuple(out)


def frontier(rows: list[dict]) -> list[dict]:
    """The Pareto-optimal rows, one per distinct objective vector (the first in enumeration order)."""
    first: dict = {}
    for i, r in enumerate(rows):
        first.setdefault(tuple(round(float(x), 12) for x in _obj(r)), i)
    keys = list(first)
    objs = np.array(keys)
    keep = []
    for i, o in enumerate(objs):
        le = (objs <= o + 1e-15).all(1)
        lt = (objs < o - 1e-15).any(1)
        if not (le & lt).any():
            keep.append(i)
    return [rows[first[keys[i]]] for i in sorted(keep, key=lambda i: tuple(objs[i]))]


def _structural(p: Point) -> tuple:
    return (p.side, p.inject, p.code, p.denoise if p.denoise in ("cancel", "mirror") else "served",
            p.fmt == "odd" and p.side == "dither", p.c1, p.tag, p.nz, p.weights)


SSN_1L = "A/rot1/b13s4/served/s8/noc1/M0"
#: N-RED's first fix: row tags against coin-drawn weights, and later lanes never zero
SSN_1L_TAG = SSN_1L + "+tag+nz"
#: N-RED's second: the later lanes' public weights signed by window coins drawn after B
SSN_1L_HARD = SSN_1L + "+tag+nzh+ws"
#: a coin byte a step against wide window weights, replayed negated in the second half, with N-RED's owner tags (the
#: code-only first and last steps otherwise repeat across owners)
MIRROR = "A/byte1/b13s4/mirror/s8/noc1/M0+tag+wv"
#: N-PREM's ssn_1l_hw as the grammar holds it: odd later lanes {1, 3} against signs; its code is b13s4's 4 bits (hw
#: reads nibbles, 16 bits) and its signs are window draws (hw's are public), neither of which the bound reading sees
SSN_1L_HW = SSN_1L + "+nz+ws"
REFERENCES = {"ssn_1l": SSN_1L, "ssn_1l_tag": SSN_1L_TAG, "ssn_1l_hw": SSN_1L_HW, "ssn_1l_hard": SSN_1L_HARD,
              "mirror": MIRROR}
#: how many designs each densest-coins list keeps
DENSE = 30
#: (C1, premises, chosen-reading T1 within T1_BAR)
CLASSES = {"noc1, no premise": (False, False, False), "noc1, premises": (False, True, False),
           "c1, no premise": (True, False, False), "c1, premises": (True, True, False),
           "noc1, no premise, chosen <= 1%": (False, False, True), "noc1, premises, chosen <= 1%": (False, True, True),
           "c1, no premise, chosen <= 1%": (True, False, True)}
#: the class frontiers keep designs whose served noise std is at most this share of the GEMM's own activation
#: rounding error, sqrt(K/12)·sigma_w (an estimate standing in for T3 until perplexity is measured)
T3_SAFE = 0.05
#: T1's bar: a design is closed when its chosen-reading T1 (gap plus the chosen saving) is within it
T1_BAR = 1e-2
CLOSED_WORKLOADS = ("llama_T512_64tok", "llama_T8192")


def _closed(r: dict) -> bool:
    return all(r[wl]["gap_chosen"] <= T1_BAR for wl in CLOSED_WORKLOADS)


def tile_update(point: dict) -> dict:
    """A scored design's chosen-data saving with `tile_saving` read beside the vector level, at CLOSED_WORKLOADS."""
    p = Point(**point)
    out = {}
    for wl in CLOSED_WORKLOADS:
        shapes, m = WORKLOADS[wl]
        dup = tile = tot = 0.0
        for n, k in shapes.values():
            t = terms(p, m, n, k)
            sv = t["alpha"] * tile_saving(t["side"], p.inject, t["s"], p.code, t["owners"], t["others"], p.nz, p.tag,
                                          p.weights, "mirror" if p.denoise == "mirror" else "served")
            dup, tile, tot = dup + m * n * k * max(t["dup_chosen"], sv), tile + m * n * k * sv, tot + m * n * k
        out[wl] = {"dup_chosen": dup / tot, "tile_saving": tile / tot}
    return out


def _apply_tile(r: dict, upd: dict) -> None:
    for wl, u in upd.items():
        row = r[wl]
        row["dup_chosen_vector"] = row["dup_chosen"]
        row.update(u)
        row["gap_chosen"] = row["gap"] + row["dup_chosen"]
    r["tile_checked"] = True


def tile_check(rows: list[dict], pmap) -> None:
    todo = [r for r in rows if not r.get("tile_checked")]
    for r, upd in zip(todo, pmap(tile_update, [r["point"] for r in todo])):
        _apply_tile(r, upd)


def tile_frontier(rows: list[dict], pmap) -> list[dict]:
    """The frontier of the rows that stay closed once tile-checked. Only frontier rows are checked, until the frontier
    holds checked rows alone: a check can only raise a row's chosen saving, so a row dominated by a checked row in
    its unchecked (lower) values stays dominated."""
    live = [r for r in rows if _closed(r)]
    while True:
        front = frontier(live)
        if all(r.get("tile_checked") for r in front):
            return front
        tile_check(front, pmap)
        live = [r for r in live if _closed(r)]


def _in_class(r: dict, cls: str) -> bool:
    c1, prem, closed = CLASSES[cls]
    return r["point"]["c1"] == c1 and bool(r["premises"]) == prem and (not closed or _closed(r))


def dominates(a: dict, b: dict) -> bool:
    oa, ob = _obj(a), _obj(b)
    return all(x <= y + 1e-15 for x, y in zip(oa, ob)) and any(x < y - 1e-15 for x, y in zip(oa, ob))


def _beating(rows: list[dict], ref: dict, pmap) -> list[dict]:
    """The s8 rows that dominate `ref` with both tile-checked (dominators checked until the list is stable)."""
    tile_check([ref], pmap)
    while True:
        beat = [r for r in rows if r["point"]["fmt"] == "s8" and dominates(r, ref)]
        if all(r.get("tile_checked") for r in beat):
            return beat
        tile_check(beat, pmap)


@contextlib.contextmanager
def _mapper(jobs: int):
    if jobs <= 1:
        yield lambda f, xs: list(map(f, xs))
        return
    with multiprocessing.get_context("fork").Pool(jobs) as pool:
        yield lambda f, xs: pool.map(f, xs, chunksize=max(1, len(xs) // (8 * jobs)))


def _density(r: dict) -> float:
    return r["llama_T8192"]["coin_bits_per_mac"]


def densest(rows: list[dict], pmap) -> dict:
    """The designs reading the most coin bits per MAC at llama_T8192: over all, over those within T3_SAFE, and over
    those within T3_SAFE that stay closed once tile-checked."""
    safe = sorted((r for r in rows if r["llama_T8192"]["t3_ratio"] <= T3_SAFE), key=_density, reverse=True)
    closed, i = [], 0
    while len(closed) < DENSE and i < len(safe):
        batch = [r for r in safe[i:i + 4 * DENSE] if _closed(r)]
        tile_check(batch, pmap)
        closed += [r for r in batch if _closed(r)]
        i += 4 * DENSE
    return {"all": [r["name"] for r in sorted(rows, key=_density, reverse=True)[:DENSE]],
            "t3_safe": [r["name"] for r in safe[:DENSE]], "closed": [r["name"] for r in closed[:DENSE]]}


def run(jobs: int = 1) -> dict:
    pts = enumerate_points()
    with _mapper(jobs) as pmap:
        scored = pmap(score, pts)
        reps: dict = {}
        for p, r in zip(pts, scored):
            if "invalid" not in r:
                reps.setdefault(_structural(p), p)
        for side in SIDES[len(LANE_SIDES):]:
            p = Point(side, "rot1", "none", "served", "s8", False, "M0")
            reps.setdefault(_structural(p), p)
        keys = list(reps)
        checks = dict(zip(keys, pmap(symbolic_check, [reps[k] for k in keys])))
        rows, invalid = [], {}
        for p, r in zip(pts, scored):
            if "invalid" in r:
                invalid[p.name] = r["invalid"]
            elif not checks[_structural(p)]["valid"]:
                invalid[p.name] = "not distinct for every input"
            else:
                r["check"] = checks[_structural(p)]
                r["t3_format"] = FORMAT_T3[p.fmt]
                rows.append(r)
        front = frontier(rows)
        vector_closed = sum(map(_closed, rows))
        safe = [r for r in rows if r["llama_T8192"]["t3_ratio"] <= T3_SAFE]
        classes = {cls: (tile_frontier if CLASSES[cls][2] else lambda rs, _: frontier(rs))(
            [r for r in safe if _in_class(r, cls)], pmap) for cls in CLASSES}
        by_name = {r["name"]: r for r in rows}
        refs = {key: by_name[nm] for key, nm in REFERENCES.items()}
        beats = {key: _beating(rows, ref, pmap) for key, ref in refs.items()}
        dense = densest(rows, pmap)
    reasons: dict = {}
    for v in invalid.values():
        reasons[v] = reasons.get(v, 0) + 1
    checked = [r for r in rows if r.get("tile_checked")]
    return {"counts": {"points": len(pts), "scored": len(rows), "invalid": len(invalid), "frontier": len(front),
                       "structural_checks": len(checks), **{f"frontier {c}": len(v) for c, v in classes.items()},
                       **{f"dominate_{key}": len(v) for key, v in beats.items()},
                       "vector_closed": vector_closed, "tile_checked": len(checked),
                       "tile_closed": sum(map(_closed, checked))},
            "invalid_reasons": reasons, "shares_source": SHARES_SOURCE, **refs, "frontier": front,
            "class_frontiers": {c: [r["name"] for r in v] for c, v in classes.items()},
            **{f"dominate_{key}": [r["name"] for r in v] for key, v in beats.items()},
            "densest": dense, "rows": rows,
            "checks": {"|".join(map(str, k)): v for k, v in checks.items()}, "lower_bounds": lower_bounds(),
            "revisit_entropy": entropy_revisits(),
            "objectives": [list(o) for o in OBJECTIVES], "tile": {"words": TILE, "positions": TILE_POSITIONS}}


def _row(r: dict) -> str:
    L, L64 = r["llama_T8192"], r["llama_T512_64tok"]
    return " | ".join([r["name"], f"{100 * L64['gap']:.4f}%", f"{100 * L['gap']:.4f}%",
                       f"{100 * L64['gap_coins_uncredited']:.3f}%", f"{100 * L['gap_coins_uncredited']:.3f}%",
                       f"{100 * L64['gap_adaptive']:.4f}%",
                       f"{100 * L64['gap_chosen']:.2f}%/{100 * L['gap_chosen']:.2f}%",
                       f"{100 * L64['tile_saving']:.2f}%/{100 * L['tile_saving']:.2f}%" if r.get("tile_checked")
                       else "-",
                       f"{100 * L['p_fixed']:.3f}%/{100 * L['p_bound']:.3f}%",
                       f"{100 * L['a2r']:.2f}%/{100 * L['a2r_equal']:.2f}%",
                       f"{1 / L['coin_bits_per_mac']:.0f}" if L["coin_bits_per_mac"] else "-",
                       f"{100 * r['v4dense_T8192']['gap']:.4f}%",
                       f"{r['t2_T8192']['t2']:.2f}x", f"{L['std']:.2f} ({L['row_std']:.2f}/{L['code_std']:.2f})",
                       f"{L['t3_ratio']:.4f}", f"{L['alpha']:.4f}", f"{L['writes_owner_per_step']:.3f}",
                       f"{L['coin_bits_owner']:.0f}", f"{L['mu_step']:.4f}", f"{L['leverage_words_per_bit']:.0f}",
                       ",".join(r["premises"]) or "-", ",".join(r["kernel"]) or "-"])


HEADER = ("design | T1 64tok | T1 8192 | +coins 64tok | +coins 8192 | adaptive 64tok | chosen 64tok/8192 | "
          "tile 64tok/8192 | p fixed/bound 8192 | A2r drawn/equal 8192 | MACs/coin bit 8192 | V4d 8192 | T2 8192 | "
          "std (row/code) | std/qerr | alpha | writes/step | coin bits/owner | mu_step | leverage/bit | premises | kernel")


def table(rep: dict, limit: int = 40) -> str:
    """Each class's frontier by its best design for each numeric objective, then the designs dominating ssn_1l."""
    by_name = {r["name"]: r for r in rep["rows"]}
    lines = []
    for cls, names in rep["class_frontiers"].items():
        rows = [by_name[nm] for nm in names]
        heads: dict = {}
        for a, b in OBJECTIVES:
            if b is not None and rows:
                best = min(rows, key=lambda r: (r[a][b], _obj(r)))
                heads.setdefault(best["name"], []).append(f"{a.replace('llama_', '')}:{b}")
        lines += [f"## {cls}, std/qerr <= {T3_SAFE}: {len(rows)} on its frontier; the best for each objective",
                  HEADER + " | best at"]
        lines += [_row(by_name[nm]) + " | " + ", ".join(objs) for nm, objs in heads.items()]
        lines.append("")
    for key, nm in REFERENCES.items():
        beat = rep[f"dominate_{key}"]
        lines += [f"## s8 designs that dominate {nm} on every objective, tile-checked ({len(beat)}; first {limit})",
                  HEADER]
        lines += [_row(by_name[b]) for b in sorted(beat, key=lambda b: _obj(by_name[b]))[:limit]]
        lines.append("")
    for key, names in rep["densest"].items():
        lines += [f"## the most coin bits per MAC at llama_T8192, {key}", HEADER] + [_row(by_name[nm]) for nm in names]
        lines.append("")
    lines += ["## references", HEADER] + [_row(rep[key]) for key in REFERENCES]
    lines += ["", "## A2r's share of an entry's words, one uniform e-bit coin lane a step",
              "steps | bits | served, weight 1 | served, wide | mirror, weight 1 | mirror, wide"]
    lines += [f"{r['steps']} | {r['bits']} | " + " | ".join(f"{100 * r[c]:.2f}%" for c in
                                                         ("served_ones", "served_wide", "mirror_ones", "mirror_wide"))
              for r in rep["revisit_entropy"]]
    lines += ["", "cancel floor at T3_SAFE: " + json.dumps(rep["lower_bounds"]["cancel_floor_at_t3_safe"]),
              json.dumps(rep["counts"]), json.dumps(rep["invalid_reasons"])]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--data", help="noise_data.py's output, for the adaptive reading's shares")
    ap.add_argument("--jobs", type=int, default=1, help="worker processes")
    args = ap.parse_args(argv)
    if args.data:
        load_shares(args.data)
    rep = run(args.jobs)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rep, fh, separators=(",", ":"), default=float)
    if args.table:
        print(table(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
