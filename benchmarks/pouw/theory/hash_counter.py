"""HASH-COUNTER: the cheapest prover for a commitment rule, in W1, on exact small instances.

A rule says which accumulator words of a unit are committed (`pos`: every step, every L-th step, or the final one),
how many bits of each (`bits`: all 32, the low k or the high k) and how they are aggregated (`agg`: SHA-512 over the
fields, or a 32-bit sketch Σ r_t g(f_t) mod 2³² with r public before the window and g the identity `lin`, the square
`sq` or a bytewise public permutation `sbox`). The sketch is then SHA-512'd.

The instance is a noised chain an entry (i, j) of an m×n tile: S k32 steps, q coin lanes a step carrying raw int8
coins U[i, t, l]·V[j, t, l], the other 32 − q lanes the data A·B, and the word w_t = Σ_{τ≤t} c_τ (`add`) or
w_t = ρ_t·w_{t−1} + c_t with public odd ρ (`mul`), all mod 2³². `zero` data is M0's choice (the prover picks the
data); `random` data stands for operands the prover can't pick.

Prices (W1, in register writes an entry; a write is 32 units, and a k32 MMA step writes one accumulator):

- stock: the data's GEMM, (32 − q)·S/32 writes; it is the credit every design claims;
- an IMMA chain of depth D over d products: ⌈d/D⌉, with D 32 at s8/u8, 64 at u4 and 256 at b1 (AND + POPC);
- an ALU instruction: 1; SHA-512: `SHA_REGS` registers a 1,024-bit compression, so `WB` writes a committed bit.

Every strategy computes the committed fields (or sketch) itself and is checked bit for bit against the honest
commitment; a strategy that doesn't match fails the run. The hash of the committed bytes is the same for every
strategy, so the gap is `stock − (cheapest − common hash)`, as a share of stock; T1 fails above 1%. Work shared by a
row or a column (a folded or bit-packed coin operand) is amortized over a CALL_M × CALL_N call, not the CPU tile.

Strategies:
- `honest`: every step's MMA (and the `mul` scale), then the field extraction and the sketch's instructions.
- `skip` (zero data): only the committed words, each segment's coin products packed into the narrowest IMMA the
  field's bits allow (low 1 bit: b1; low ≤ 4: u4), with `mul`'s coefficients folded into the column operand.
- `fold` (`lin` over 32-bit words, zero data): Σ r_t w_t = Σ_τ c_τ R_τ, with R_τ V split into four u8 digits.
- `bitslice` (low k ≤ 4 bits, zero data): k bit planes of 32 rows a register; at k 1, acc ^= U-bits & V-mask, one
  LOP3 a lane step for 32 entries. Operand loads are amortized over a BLOCK × BLOCK register tile of row blocks and
  columns, and the fields pay a 32×32 bit transpose (entry-major, as the honest prover packs them) a 1,024 bits.
- `four_russians`: `bitslice` with each 32-row block's increment planes tabulated over a lane group's column bits
  (shared by every column of the call), so a lane step is k loads and a k-bit ripple add whatever q is.
- `bsketch` (a sketch over 1-bit fields): the bitslice's bits, transposed, against r's bit planes in one b1 IMMA.
- `copy`: entries whose inputs repeat are computed once (exact distinct count).
- `table`: a field whose inputs span at most 34 bits (2³⁴ entries, `counter_tile`'s MEM) is one indexed load,
  after gathering its index bits.

Run: `uv run --no-project --with numpy --with-editable benchmarks/pouw/exact_kit python -B hash_counter.py OUT.json
[--quick] [--mutations]` (from this directory).
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from dataclasses import asdict, dataclass

import numpy as np

from exact_kit.checks import Checks

M32 = (1 << 32) - 1
SHA_REGS = 3515.38
WB = SHA_REGS / 1024
MEM_BITS = 34
TRANSPOSE_WB = (2 * 2 + 3 * 5) * 16 / 1024
T1 = 0.01
CALL_M = CALL_N = 4096
BLOCK = 8

MAC_S_H200 = 0.4 * 9.89e14
WORLD_MAC_YR = 4.68e30
AND_PER_MAC = 2079 / 16
DC = 1e5
PERIODS = {"window": 600.0, "week": 7 * 86400.0, "month": 30 * 86400.0, "year": 365.25 * 86400.0}


@dataclass(frozen=True)
class Rule:
    pos: str
    bits: str
    agg: str

    def positions(self, S: int) -> np.ndarray:
        if self.pos == "all":
            return np.arange(S)
        if self.pos == "final":
            return np.array([S - 1])
        L = int(self.pos.split(":")[1])
        return np.arange(L - 1, S, L)

    @property
    def k(self) -> int:
        return 32 if self.bits == "32" else int(self.bits.split(":")[1])

    @property
    def low(self) -> bool:
        return self.bits.startswith("low")

    def field(self, w):
        if self.bits == "32":
            return w
        if self.low:
            return w & ((1 << self.k) - 1)
        return w >> (32 - self.k)

    def __str__(self):
        return f"{self.pos}/{self.bits}/{self.agg}"


@dataclass(frozen=True)
class Inst:
    m: int
    n: int
    S: int
    q: int
    data: str
    rec: str
    seed: int
    coins: str = "bilinear"

    def __str__(self):
        return f"{self.m}x{self.n} S{self.S} q{self.q} {self.data} {self.rec} {self.coins} s{self.seed}"


def mulmod(a, b):
    """a·b mod 2³² for int64 arrays with 0 ≤ a, b < 2³²."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    lo, hi = b & 0xFFFF, b >> 16
    return ((a * lo) + (((a * hi) & 0xFFFF) << 16)) & M32


class Draw:
    def __init__(self, inst: Inst, rule: Rule):
        g = np.random.default_rng(inst.seed)
        m, n, S, q = inst.m, inst.n, inst.S, inst.q
        self.U = g.integers(-128, 128, (m, S, q), dtype=np.int64)
        self.V = g.integers(-128, 128, (n, S, q), dtype=np.int64)
        if inst.coins == "sep":
            self.V[:, :, : q // 2] = 1
            self.U[:, :, q // 2:] = 1
        shape_a, shape_b = (m, S, 32 - q), (n, S, 32 - q)
        if inst.data == "zero":
            self.A, self.B = np.zeros(shape_a, np.int64), np.zeros(shape_b, np.int64)
        else:
            self.A = g.integers(-128, 128, shape_a, dtype=np.int64)
            self.B = g.integers(-128, 128, shape_b, dtype=np.int64)
        self.rho = (g.integers(0, 1 << 31, S, dtype=np.int64) * 2 + 1) if inst.rec == "mul" else np.ones(S, np.int64)
        self.P = rule.positions(S)
        self.r = g.integers(0, 1 << 32, len(self.P), dtype=np.int64)
        self.sbox = g.permutation(256).astype(np.int64)
        c = np.einsum("itl,jtl->ijt", self.U, self.V) + np.einsum("itl,jtl->ijt", self.A, self.B)
        self.c = c & M32
        if inst.rec == "add":
            self.w = np.cumsum(c, axis=2) & M32
        else:
            w = np.zeros((m, n, S), np.int64)
            acc = np.zeros((m, n), np.int64)
            for t in range(S):
                acc = (mulmod(acc, self.rho[t]) + self.c[:, :, t]) & M32
                w[:, :, t] = acc
            self.w = w


def g_of(rule: Rule, d: Draw, f):
    if rule.agg == "lin":
        return f
    if rule.agg == "sq":
        return mulmod(f, f)
    nb = max(1, math.ceil(rule.k / 8))
    out = np.zeros_like(f)
    for b in range(nb):
        out |= d.sbox[(f >> (8 * b)) & 0xFF] << (8 * b)
    return out


def aggregate(rule: Rule, d: Draw, fields):
    if rule.agg == "hash":
        return fields
    return mulmod(g_of(rule, d, fields), d.r[None, None, :]).sum(axis=2) & M32


def common_hash(rule: Rule, nP: int) -> float:
    return nP * rule.k * WB if rule.agg == "hash" else 32 * WB


def commit_ops(rule: Rule, nP: int) -> float:
    e = 0 if rule.k == 32 else 1
    if rule.agg == "hash":
        return nP * e
    if rule.agg == "lin":
        return nP * (e + 1)
    if rule.agg == "sq":
        return nP * (e + 2)
    nb = max(1, math.ceil(rule.k / 8))
    return nP * (e + 2 * nb + (nb - 1) + 1)


def segments(P):
    prev = -1
    for p in P:
        yield prev, int(p)
        prev = int(p)


def depth(rule: Rule) -> int:
    if rule.low and rule.k == 1:
        return 256
    if rule.low and rule.k <= 4:
        return 64
    return 32


def honest(rule, inst, d):
    fields = rule.field(d.w[:, :, d.P])
    cost = inst.S * (1 if inst.rec == "add" else 2) + commit_ops(rule, len(d.P))
    return cost, aggregate(rule, d, fields), "MMA every step"


def skip(rule, inst, d):
    """Only the committed words, from the segment's coin products; zero data only (M0)."""
    if inst.data != "zero":
        return None
    m, n, q = inst.m, inst.n, inst.q
    D = depth(rule)
    narrow = rule.low and rule.k <= 4
    mod = (1 << rule.k) - 1 if narrow else M32
    U = (d.U & 1) if D == 256 else (d.U & 15) if narrow else d.U
    acc = np.zeros((m, n), np.int64)
    fields = np.zeros((m, n, len(d.P)), np.int64)
    cost = 0.0
    for idx, (prev, p) in enumerate(segments(d.P)):
        span = p - prev
        ts = np.arange(prev + 1, p + 1)
        if inst.rec == "add" or D == 256:
            coef = np.ones(span, np.int64)
            scale = 1
        else:
            coef = np.ones(span, np.int64)
            for a in range(span - 2, -1, -1):
                coef[a] = mulmod(coef[a + 1], d.rho[ts[a + 1]])
            scale = int(mulmod(coef[0], d.rho[ts[0]])) if prev >= 0 else 0
        Vc = mulmod(d.V[:, ts, :] & M32, coef[None, :, None]) & mod
        if D == 256:
            Vc = d.V[:, ts, :] & 1
        if narrow or D == 256:
            seg = np.einsum("itl,jtl->ij", U[:, ts, :], Vc)
            cost += math.ceil(q * span / D) + (inst.rec == "mul" and D != 256 and prev >= 0)
            if inst.rec == "mul" and D != 256:
                cost += q * span / CALL_M
        elif inst.rec == "add":
            seg = np.einsum("itl,jtl->ij", U[:, ts, :], d.V[:, ts, :])
            cost += math.ceil(q * span / D)
        else:
            seg = sum(np.einsum("itl,jtl->ij", U[:, ts, :], (Vc >> (8 * b)) & 0xFF) << (8 * b) for b in range(4))
            cost += 4 * math.ceil(q * span / 32) + 3 + (prev >= 0) + 4 * q * span / CALL_M
        acc = (mulmod(acc, scale) + seg) & mod
        fields[:, :, idx] = acc
    if not narrow and D != 256:
        fields = rule.field(fields)
    cost += commit_ops(rule, len(d.P))
    return cost, aggregate(rule, d, fields), f"pack D{D}"


def fold(rule, inst, d):
    """Σ r_t w_t = Σ_τ c_τ R_τ over a linear sketch of whole words; zero data only."""
    if inst.data != "zero" or rule.agg != "lin" or rule.k != 32:
        return None
    S, q = inst.S, inst.q
    inP = np.zeros(S, np.int64)
    inP[d.P] = d.r
    R = np.zeros(S + 1, np.int64)
    for t in range(S - 1, -1, -1):
        nxt = mulmod(d.rho[t + 1], R[t + 1]) if t + 1 < S else 0
        R[t] = (inP[t] + nxt) & M32
    VR = mulmod(d.V & M32, R[None, :S, None])
    s = sum(np.einsum("itl,jtl->ij", d.U, (VR >> (8 * b)) & 0xFF) << (8 * b) for b in range(4)) & M32
    cost = 4 * math.ceil(q * S / 32) + 3 + 4 * q * S / CALL_M
    return cost, s, "Σ c_τ R_τ, u8 digits"


def bitslice_ops(k: int) -> int:
    """LOP3s a lane step for 32 entries: add (a << y)·b_y mod 2^k for each column bit y, a ripple adder an add."""
    return sum(1 if k - y == 1 else 3 * (k - y) - 2 for y in range(k))


def bitslice(rule, inst, d):
    """Low k ≤ 4 bit fields as k bit planes of 32 rows a register: column coin bits are masks, row coin bits are
    planes, each lane step a ripple add of the masked shifted planes; zero data only, `mul` only at k 1 (ρ is odd)."""
    if inst.data != "zero" or not (rule.low and rule.k <= 4) or inst.m % 32:
        return None
    if inst.rec == "mul" and rule.k > 1:
        return None
    m, n, S, q, k = inst.m, inst.n, inst.S, inst.q, rule.k
    rb = m // 32
    weights = (np.int64(1) << np.arange(32, dtype=np.int64))
    planes = [(((d.U >> x) & 1).reshape(rb, 32, S, q) * weights[None, :, None, None]).sum(axis=1) for x in range(k)]
    masks = [-((d.V >> y) & 1) & M32 for y in range(k)]
    acc = [np.zeros((rb, n), np.int64) for _ in range(k)]
    snaps = {}
    Pset = set(int(p) for p in d.P)
    for t in range(S):
        for lane in range(q):
            for y in range(k):
                M = masks[y][None, :, t, lane]
                c = None
                for x in range(y, k):
                    p = planes[x - y][:, None, t, lane] & M
                    if c is None:
                        c, acc[x] = acc[x] & p, acc[x] ^ p
                    else:
                        c, acc[x] = (acc[x] & p) | (acc[x] & c) | (p & c), acc[x] ^ p ^ c
        if t in Pset:
            snaps[t] = [a.copy() for a in acc]
    fields = np.zeros((m, n, len(d.P)), np.int64)
    for idx, p in enumerate(d.P):
        for x in range(k):
            bits = (snaps[int(p)][x][:, None, :] >> np.arange(32)[None, :, None]) & 1
            fields[:, :, idx] |= bits.reshape(m, n) << x
    per_lane_step = bitslice_ops(k) / 32 + k * (2 / CALL_M + 12 / (32 * CALL_N)) + k * 2 / (32 * BLOCK)
    cost = S * q * per_lane_step + len(d.P) * k * TRANSPOSE_WB
    if rule.agg != "hash":
        cost += commit_ops(rule, len(d.P))
    return cost, aggregate(rule, d, fields), f"{bitslice_ops(k)} LOP3 a lane step a 32-row register"


def four_russians_cost(k: int, q: int) -> tuple[float, int]:
    """Writes an entry a step when each (32-row block, step) tabulates, for every value of g lanes' low column bits,
    the k planes of Σ_l a_l·b_l mod 2^k (built in Gray order, one k-plane add an entry, shared by CALL_N columns), and
    each (block, column, step) a group loads the column's table offset and k planes and ripple-adds them: 1 + k loads
    and 2k − 1 LOP3s."""
    best = None
    for g in range(1, q + 1):
        groups = math.ceil(q / g)
        per = groups * (1 + k + 2 * k - 1) / 32 + groups * (1 << (k * g)) * (3 * k) / (32 * CALL_N)
        if best is None or per < best[0]:
            best = (per, g)
    return best


def four_russians(rule, inst, d):
    """`bitslice`'s fields, with a lane group's increment planes looked up instead of computed."""
    res = bitslice(rule, inst, d)
    if res is None:
        return None
    _, out, _ = res
    per, g = four_russians_cost(rule.k, inst.q)
    cost = inst.S * per + len(d.P) * rule.k * TRANSPOSE_WB
    if rule.agg != "hash":
        cost += commit_ops(rule, len(d.P))
    return cost, out, f"tables of {g} lanes' column bits a 32-row block"


def bsketch(rule, inst, d):
    """A sketch over 1-bit fields is affine in the bits: Σ r_t g(b_t) = g(0)·Σ r_t + (g(1) − g(0))·Σ_β 2^β popc(r_β & b),
    one b1 IMMA (K 256) against r's 32 bit planes after the bitslice's fields are transposed entry-major."""
    if rule.agg == "hash" or not (rule.low and rule.k == 1):
        return None
    sub = Rule(rule.pos, rule.bits, "hash")
    res = bitslice(sub, inst, d)
    if res is None:
        return None
    cost, bits, _ = res
    cost = min(cost, inst.S * four_russians_cost(1, inst.q)[0] + len(d.P) * TRANSPOSE_WB)
    g0, g1 = (0, 1) if rule.agg in ("lin", "sq") else (int(d.sbox[0]), int(d.sbox[1]))
    rbits = (d.r[:, None] >> np.arange(32)[None, :]) & 1
    pop = np.einsum("ijt,tb->ijb", bits, rbits)
    dot = (pop << np.arange(32)[None, None, :]).sum(axis=2) & M32
    s = (mulmod(g0, int(d.r.sum() & M32)) + mulmod((g1 - g0) & M32, dot)) & M32
    cost += 32 * math.ceil(len(d.P) / 256) + 32 + 2
    return cost, s, "bitslice + b1 IMMA on r's bit planes"


def expand(rule, inst, d):
    """A squared sketch over separable words w_t = R_t(i) + C_t(j): Σ r R² per row, Σ r C² per column, and the cross
    term 2 Σ_t (r_t R_t)·C_t, one u8 IMMA chain a digit weight (digit pairs a + b ≤ 3) of depth (a + b + 1)·|P|."""
    if inst.coins != "sep" or inst.data != "zero" or inst.rec != "add" or rule.agg != "sq" or rule.k != 32:
        return None
    h = inst.q // 2
    R = np.cumsum(d.U[:, :, :h].sum(axis=2), axis=1)[:, d.P] & M32
    C = np.cumsum(d.V[:, :, h:].sum(axis=2), axis=1)[:, d.P] & M32
    rr = d.r[None, :]
    A = mulmod(rr, mulmod(R, R)).sum(axis=1) & M32
    B = mulmod(rr, mulmod(C, C)).sum(axis=1) & M32
    rR = mulmod(rr, R)
    X = np.zeros((inst.m, inst.n), np.int64)
    for a, b in itertools.product(range(4), range(4)):
        if a + b <= 3:
            X += np.einsum("it,jt->ij", (rR >> (8 * a)) & 0xFF, (C >> (8 * b)) & 0xFF) << (8 * (a + b))
    s = (A[:, None] + B[None, :] + 2 * X) & M32
    nP = len(d.P)
    cost = sum(math.ceil((w + 1) * nP / 32) for w in range(4)) + 3 + 2
    cost += (3 * nP + 4 * nP) / CALL_N + (2 * nP + 4 * nP) / CALL_M
    return cost, s, "row and column squares + cross-term IMMA"


def copy(rule, inst, d):
    """Entries whose inputs repeat: computed once, then moved."""
    rows = np.concatenate([d.U.reshape(inst.m, -1), d.A.reshape(inst.m, -1)], axis=1)
    cols = np.concatenate([d.V.reshape(inst.n, -1), d.B.reshape(inst.n, -1)], axis=1)
    distinct = len(np.unique(rows, axis=0)) * len(np.unique(cols, axis=0))
    if distinct == inst.m * inst.n:
        return None
    base, out, _ = honest(rule, inst, d)
    return base * distinct / (inst.m * inst.n), out, f"{distinct} distinct entries"


def table(rule, inst, d):
    """A field whose inputs fit in 2^MEM_BITS: one load after gathering the index."""
    best = None
    for prev, p in segments(d.P):
        span = p - prev
        kin = rule.k if rule.low and rule.k <= 8 else 8
        bits = 2 * inst.q * span * kin + ((rule.k if rule.low else 32) if prev >= 0 else 0)
        if inst.data != "zero":
            bits += 2 * (32 - inst.q) * span * kin
        if bits > MEM_BITS:
            return None
        best = (best or 0) + math.ceil(bits / 8) + 1
    _, out, _ = honest(rule, inst, d)
    return best + commit_ops(rule, len(d.P)), out, "index gather + load (exact by construction)"


STRATEGIES = {"honest": honest, "skip": skip, "fold": fold, "bitslice": bitslice, "four_russians": four_russians,
              "bsketch": bsketch, "expand": expand, "copy": copy, "table": table}


def evaluate(rule: Rule, inst: Inst, strategies=STRATEGIES):
    d = Draw(inst, rule)
    stock = (32 - inst.q) * inst.S / 32
    hc = common_hash(rule, len(d.P))
    h_cost, ref, _ = honest(rule, inst, d)
    rows = {}
    for name, fn in strategies.items():
        res = fn(rule, inst, d)
        if res is None:
            continue
        cost, out, how = res
        rows[name] = {"writes": round(cost, 4), "exact": bool(np.array_equal(out, ref)), "how": how}
    exact_rows = {k: v for k, v in rows.items() if v["exact"]}
    best = min(exact_rows, key=lambda k: exact_rows[k]["writes"])
    cheapest = exact_rows[best]["writes"]
    gap = max(0.0, (stock - cheapest) / stock)
    return {
        "rule": str(rule),
        "inst": asdict(inst),
        "stock": stock,
        "honest_total": round(h_cost + hc, 4),
        "common_hash": round(hc, 4),
        "h": round((h_cost - inst.S * (1 if inst.rec == "add" else 2) + hc) / stock, 4),
        "strategies": rows,
        "cheapest": best,
        "cheapest_writes": cheapest,
        "gap_share": round(gap, 6),
        "breaks_T1": gap > T1,
    }


def per_period(share: float) -> dict:
    out = {}
    for name, sec in PERIODS.items():
        mac = MAC_S_H200 * sec
        out[name] = {
            "h200_AND": share * mac * AND_PER_MAC,
            "dc_AND": share * mac * DC * AND_PER_MAC,
            "world_AND": share * WORLD_MAC_YR * sec / PERIODS["year"] * AND_PER_MAC,
        }
    return out


RULES = [Rule(p, b, a) for p, b, a in itertools.product(
    ["all", "every:2", "every:4", "every:8", "every:64", "final"],
    ["32", "low:1", "low:2", "low:3", "low:4", "low:8", "high:8", "high:1"],
    ["hash", "lin", "sq", "sbox"])]


def instances(quick: bool):
    S_set = [64] if quick else [64, 256, 1024]
    q_set = [1, 4] if quick else [1, 2, 4, 8]
    for S, q, data, rec, seed in itertools.product(S_set, q_set, ["zero", "random"], ["add", "mul"], [0, 1]):
        if not (quick and seed) and not (S == 1024 and seed):
            yield Inst(64, 16, S, q, data, rec, seed)
    for S, q in itertools.product(S_set, [2, 4] if quick else [2, 4, 8]):
        yield Inst(64, 16, S, q, "zero", "add", 0, "sep")


def mutations(checks: Checks):
    rule, inst = Rule("all", "low:1", "hash"), Inst(64, 16, 64, 4, "zero", "add", 7)
    d = Draw(inst, rule)
    _, ref, _ = honest(rule, inst, d)

    def bs_drop(r, i, dd):
        dd.U = dd.U.copy()
        dd.U[:, :, -1] = 0
        return bitslice(r, i, dd)

    _, out, _ = bs_drop(rule, inst, d)
    checks.mutation("bitslice dropping a coin lane", not np.array_equal(out, ref))
    rule_k, inst_k = Rule("all", "low:3", "hash"), Inst(64, 16, 64, 4, "zero", "add", 7)
    dk = Draw(inst_k, rule_k)
    _, refk, _ = honest(rule_k, inst_k, dk)
    dk.V = dk.V & ~2
    _, outk, _ = bitslice(rule_k, inst_k, dk)
    checks.mutation("bitslice k 3 with column coin bit 1 cleared", not np.array_equal(outk, refk))
    rule2, inst2 = Rule("all", "32", "lin"), Inst(64, 16, 64, 4, "zero", "add", 7)
    d2 = Draw(inst2, rule2)
    _, ref2, _ = honest(rule2, inst2, d2)
    d2.r = d2.r.copy()
    d2.r[0] ^= 1
    _, out2, _ = fold(rule2, inst2, d2)
    checks.mutation("fold with one sketch coin flipped", not np.array_equal(out2, ref2))
    rule3, inst3 = Rule("every:8", "32", "hash"), Inst(64, 16, 64, 4, "random", "add", 7)
    d3 = Draw(inst3, rule3)
    _, ref3, _ = honest(rule3, inst3, d3)
    inst3z = Inst(64, 16, 64, 4, "zero", "add", 7)
    _, out3, _ = skip(rule3, inst3z, d3)
    checks.mutation("skip on random data (data term dropped)", not np.array_equal(out3, ref3))
    rule4, inst4 = Rule("all", "32", "sq"), Inst(64, 16, 64, 4, "zero", "add", 7, "sep")
    d4 = Draw(inst4, rule4)
    _, ref4, _ = honest(rule4, inst4, d4)
    d4.V = d4.V.copy()
    d4.V[:, :, 2:] = 0
    _, out4, _ = expand(rule4, inst4, d4)
    checks.mutation("expand with the column term dropped", not np.array_equal(out4, ref4))


def main() -> int:
    checks = Checks()
    args = [a for a in checks.args if not a.startswith("--")]
    quick = "--quick" in checks.args
    out_path = args[0] if args else "hash_counter.json"
    results = []
    for inst in instances(quick):
        for rule in RULES:
            results.append(evaluate(rule, inst))
    for name in STRATEGIES:
        ran = [r for r in results if name in r["strategies"]]
        bad = [f"{r['rule']} at {r['inst']}" for r in ran if not r["strategies"][name]["exact"]]
        checks.check(f"{name} matches the honest commitment bit for bit", not bad,
                     f"{len(ran) - len(bad)}/{len(ran)} (rule, instance) pairs" + (f"; first miss {bad[0]}" if bad else ""))
    breaks = [r for r in results if r["breaks_T1"]]
    survive = [r for r in results if not r["breaks_T1"]]
    known = {
        "fold": [r for r in results if r["rule"] == "all/32/lin" and r["inst"]["data"] == "zero"],
        "random": [r for r in results if r["inst"]["data"] == "random"],
    }
    checks.check("fold beats stock on every zero-data all/32/lin instance with q ≤ 4",
                 all(r["cheapest"] == "fold" and r["breaks_T1"] for r in known["fold"] if r["inst"]["q"] <= 4))
    checks.check("no strategy beats honest on random data", all(not r["breaks_T1"] for r in known["random"]))
    sep_sq = [r for r in results if r["rule"] == "all/32/sq" and r["inst"]["coins"] == "sep"]
    checks.check("expand is cheapest on every separable all/32/sq instance",
                 bool(sep_sq) and all(r["cheapest"] == "expand" for r in sep_sq),
                 [(r["inst"]["S"], r["inst"]["q"], r["gap_share"]) for r in sep_sq])
    checks.check("bitslice op counts", [bitslice_ops(k) for k in (1, 2, 3, 4)] == [1, 5, 12, 22],
                 [bitslice_ops(k) for k in (1, 2, 3, 4)])
    hash32 = [r for r in results if r["rule"] == "all/32/hash"]
    checks.check("all/32/hash costs SHA_REGS/32 writes a word (h · stock)",
                 all(abs(r["common_hash"] - r["inst"]["S"] * 32 * WB) < 1e-6 for r in hash32))
    if checks.mutations:
        mutations(checks)
    by_rule = {}
    for r in results:
        b = by_rule.setdefault(r["rule"], {"breaks": 0, "of": 0, "max_gap": 0.0, "h": [], "cheapest": set()})
        b["of"] += 1
        b["breaks"] += r["breaks_T1"]
        b["max_gap"] = max(b["max_gap"], r["gap_share"])
        b["h"].append(r["h"])
        if r["breaks_T1"]:
            b["cheapest"].add(r["cheapest"])
    summary = {k: {"breaks": v["breaks"], "of": v["of"], "max_gap": v["max_gap"], "h_min": min(v["h"]),
                   "h_max": max(v["h"]), "by": sorted(v["cheapest"])} for k, v in by_rule.items()}
    json.dump({"schema": "hash-counter/v1", "rules": len(RULES), "instances": len(results) // len(RULES),
               "breaks": len(breaks), "survive": len(survive), "summary": summary, "results": results,
               "per_period_at_1pct": per_period(T1)}, open(out_path, "w"), indent=1, default=list)
    for k, v in summary.items():
        print(f"{k:24s} breaks {v['breaks']:3d}/{v['of']:3d} max gap {v['max_gap']:.4f} h {v['h_min']:.2f}–"
              f"{v['h_max']:.2f} by {','.join(v['by'])}")
    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
