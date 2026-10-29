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

`agg` may also be `pow:D`, the sketch Σ r_t f_t^D. `expand` takes a power sketch over separable words apart, and
`--aggs=A,B` and `--bits=A,B` restrict the sweep to those rules.

`agg` `xor` is the unkeyed fold σ = ⊕_t f_t, one LOP3 a pair of words, sealed with the final word (64 bits hashed);
`xor:m` commits the XOR of each m consecutive words (⌈(m − 1)/2⌉ LOP3s and 32 hashed bits a block).
`lead:e7`, `lead:e4`, `lead:w127` and `lead:rot` instances have the lead's zero-data word form,
w_t = d0 + rpos_t(i) + cpos_t(j): a −1 lane and a 1 + b lane (b ∈ {0, 1}) on the row, and three odd e-bit coin lanes
(one at e4; weights 1, 127, 1 at w127, and (127, 1, 1) or (1, 127, 1) by row parity at rot) on the column, with d0 a
per-cell offset of the lead's size. Only `honest`, `pack`, `copy` and `table` run on them. `pack` (an `xor` fold over words that split into a row and a column term) puts L = ⌊32g/b⌋ cells in b-bit
lanes of g registers, b the narrowest signed width holding every word, so g IADD3s write L words and g LOP3s fold 2L;
σ's bits above a lane are its sign bit. `gap_claimed` is (honest − cheapest)/stock: the gap against a design that
credits everything its honest prover does (C₀ = cost(H)), beside `gap_share` against stock.

`agg` `nh2:p` is NH over adjacent column pairs, S += w(j)·w(j+1), one IMAD a pair and one 32-bit value a p pairs of a
row over the committed steps; `nh:c` pairs consecutive committed words of an entry, c entries a value. `decomp` takes
either apart over separable words: row-only and column-only sums are shared (column terms by the rows of a class in a
`TILE`-row coin tile, as the lead's column coins are drawn a tile), and each value's cross term Σ_t R_t(i)·Ĉ_t(g) is
u8-digit IMMA chains, so its products are batched 32 a write.

Run: `uv run --no-project --with numpy --with-editable benchmarks/pouw/exact_kit python -B hash_counter.py OUT.json
[--quick] [--mutations] [--aggs=...] [--bits=...]` (from this directory).
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
TILE = 64

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
        self.w0 = np.zeros((m, n), np.int64)
        if inst.coins.startswith("lead:"):
            e = 4 if inst.coins == "lead:e4" else 7
            lanes = q - 2
            weights = (1, 127, 1) if inst.coins == "lead:w127" else (1,) * lanes
            odd = np.arange(-(2 ** e - 1), 2 ** e, 2)
            self.U[:] = 1
            self.V[:] = 1
            self.V[:, :, 0] = -1
            self.U[:, :, 1] = 1 + g.integers(0, 2, (m, S))
            self.V[:, :, 2:] = g.choice(odd, (n, S, lanes)) * np.array(weights)[None, None, :]
            if inst.coins == "lead:rot":
                self.U[:, :, 2:] = np.array([(127, 1, 1), (1, 127, 1)])[np.arange(m) % 2][:, None, :]
            self.w0 = g.integers(-2 * S, 2 * S + 1, (m, n))
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
            self.w = (self.w0[:, :, None] + np.cumsum(c, axis=2)) & M32
        else:
            w = np.zeros((m, n, S), np.int64)
            acc = self.w0 & M32
            for t in range(S):
                acc = (mulmod(acc, self.rho[t]) + self.c[:, :, t]) & M32
                w[:, :, t] = acc
            self.w = w


def degree(rule: Rule) -> int | None:
    """g's degree when g is the power f^d (`sq` is d 2), else None."""
    if rule.agg == "sq":
        return 2
    if rule.agg.startswith("pow:"):
        return int(rule.agg.split(":")[1])
    return None


def g_of(rule: Rule, d: Draw, f):
    if rule.agg == "lin":
        return f
    if degree(rule):
        out = f
        for _ in range(degree(rule) - 1):
            out = mulmod(out, f)
        return out
    nb = max(1, math.ceil(rule.k / 8))
    out = np.zeros_like(f)
    for b in range(nb):
        out |= d.sbox[(f >> (8 * b)) & 0xFF] << (8 * b)
    return out


def aggregate(rule: Rule, d: Draw, fields):
    if rule.agg == "hash":
        return fields
    if rule.agg.startswith("xor"):
        return xor_blocks(rule, fields)
    if nh_kind(rule):
        return nh_sketch(rule, fields)
    return mulmod(g_of(rule, d, fields), d.r[None, None, :]).sum(axis=2) & M32


def nh_kind(rule: Rule) -> tuple[str, int] | None:
    """`nh2:p`: S += w(j)·w(j+1) over adjacent column pairs of a row, p pairs (2p entries) a value; `nh:c`:
    S += w_(2k)·w_(2k+1) over consecutive committed words of an entry (an odd last word added), c entries a value."""
    kind, _, size = rule.agg.partition(":")
    return (kind, int(size)) if kind in ("nh2", "nh") else None


def nh_sketch(rule: Rule, fields):
    kind, size = nh_kind(rule)
    m, n, nP = fields.shape
    if kind == "nh2":
        per = mulmod(fields[:, 0::2], fields[:, 1::2]).sum(axis=2)
        return per.reshape(m, n // (2 * size), size).sum(axis=2) & M32
    k2 = nP // 2
    per = mulmod(fields[:, :, 0:2 * k2:2], fields[:, :, 1:2 * k2:2]).sum(axis=2) + (fields[:, :, -1] if nP % 2 else 0)
    return (per & M32).reshape(m, n // size, size).sum(axis=2) & M32


def entries_a_value(rule: Rule) -> int:
    kind, size = nh_kind(rule)
    return 2 * size if kind == "nh2" else size


def xor_block(rule: Rule) -> int | None:
    """`xor:m` commits the XOR of each m consecutive committed words; `xor` one XOR over all of them."""
    return int(rule.agg.split(":")[1]) if rule.agg.startswith("xor:") else None


def xor_blocks(rule: Rule, fields):
    m = xor_block(rule)
    if m is None:
        return np.bitwise_xor.reduce(fields, axis=-1)
    return np.stack([np.bitwise_xor.reduce(fields[..., b:b + m], axis=-1) for b in range(0, fields.shape[-1], m)], -1)


def xor_lop3s(rule: Rule, nP: int) -> float:
    m = xor_block(rule)
    return nP * 0.5 if m is None else sum(math.ceil((min(m, nP - b) - 1) / 2) for b in range(0, nP, m))


def common_hash(rule: Rule, nP: int) -> float:
    if rule.agg == "hash":
        return nP * rule.k * WB
    if xor_block(rule):
        return -(-nP // xor_block(rule)) * 32 * WB
    if nh_kind(rule):
        return 32 * WB / entries_a_value(rule)
    return (64 if rule.agg == "xor" else 32) * WB


def commit_ops(rule: Rule, nP: int) -> float:
    e = 0 if rule.k == 32 else 1
    if rule.agg == "hash":
        return nP * e
    if rule.agg.startswith("xor"):
        return nP * e + xor_lop3s(rule, nP)
    if nh_kind(rule):
        return nP * e + (nP / 2 if nh_kind(rule)[0] == "nh2" else nP // 2 + nP % 2)
    if rule.agg == "lin":
        return nP * (e + 1)
    if degree(rule):
        dg = degree(rule)
        return nP * (e + dg.bit_length() - 1 + bin(dg).count("1") - 1 + 1)
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
    if rule.agg == "hash" or rule.agg.startswith("xor") or nh_kind(rule) or not (rule.low and rule.k == 1):
        return None
    sub = Rule(rule.pos, rule.bits, "hash")
    res = bitslice(sub, inst, d)
    if res is None:
        return None
    cost, bits, _ = res
    cost = min(cost, inst.S * four_russians_cost(1, inst.q)[0] + len(d.P) * TRANSPOSE_WB)
    g0, g1 = (0, 1) if rule.agg == "lin" or degree(rule) else (int(d.sbox[0]), int(d.sbox[1]))
    rbits = (d.r[:, None] >> np.arange(32)[None, :]) & 1
    pop = np.einsum("ijt,tb->ijb", bits, rbits)
    dot = (pop << np.arange(32)[None, None, :]).sum(axis=2) & M32
    s = (mulmod(g0, int(d.r.sum() & M32)) + mulmod((g1 - g0) & M32, dot)) & M32
    cost += 32 * math.ceil(len(d.P) / 256) + 32 + 2
    return cost, s, "bitslice + b1 IMMA on r's bit planes"


def power(x, e):
    out = np.ones_like(x)
    for _ in range(e):
        out = mulmod(out, x)
    return out


def expand(rule, inst, d):
    """A power sketch Σ r_t w_t^D over separable words w_t = R_t(i) + C_t(j): Σ r R^D per row, Σ r C^D per column, and
    each cross term binom(D, a)·Σ_t (r_t R_t^a)·C_t^(D−a), 0 < a < D, as u8 IMMA chains, one a digit weight (digit pairs
    of weight ≤ 3) of depth (weight + 1)·|P|."""
    D = degree(rule)
    if inst.coins != "sep" or inst.data != "zero" or inst.rec != "add" or not D or rule.k != 32:
        return None
    h = inst.q // 2
    R = np.cumsum(d.U[:, :, :h].sum(axis=2), axis=1)[:, d.P] & M32
    C = np.cumsum(d.V[:, :, h:].sum(axis=2), axis=1)[:, d.P] & M32
    rr = d.r[None, :]
    s = (mulmod(rr, power(R, D)).sum(axis=1)[:, None] + mulmod(rr, power(C, D)).sum(axis=1)[None, :]) & M32
    for a in range(1, D):
        left = mulmod(mulmod(rr, power(R, a)), math.comb(D, a) & M32)
        right = power(C, D - a)
        for x, y in itertools.product(range(4), range(4)):
            if x + y <= 3:
                s += np.einsum("it,jt->ij", (left >> (8 * x)) & 0xFF, (right >> (8 * y)) & 0xFF) << (8 * (x + y))
        s &= M32
    nP = len(d.P)
    chains = sum(math.ceil((w + 1) * nP / 32) for w in range(4)) + 3
    cost = (D - 1) * chains + D
    cost += ((D + 4) * nP * (D - 1) + 2 * nP) / CALL_N + ((D + 3) * nP * (D - 1) + 2 * nP) / CALL_M
    return cost, s, f"row and column powers + {D - 1} cross-term IMMA chains"


def separable(inst, d):
    """At zero data under `add`, w_t = w0 + R_t(i) + C_t(class(i), j) when each coin lane's V is constant across the
    columns (a row lane) or its U takes at most four row patterns (the classes): (R, C, class, patterns, row lanes)."""
    m, S = inst.m, inst.S
    rowlane = np.all(d.V == d.V[:1], axis=(0, 1))
    pats, cls = np.unique(d.U[:, :, ~rowlane].reshape(m, -1), axis=0, return_inverse=True)
    if len(pats) > 4:
        return None
    R = np.cumsum(np.einsum("itl,tl->it", d.U[:, :, rowlane], d.V[0][:, rowlane]), axis=1)
    C = np.stack([np.cumsum(np.einsum("tl,jtl->jt", p.reshape(S, -1), d.V[:, :, ~rowlane]), axis=1) for p in pats])
    return R, C, cls.reshape(-1), pats, rowlane


def digit_dot(Rd, Cd, cls, nR):
    """Σ_t R(i, t)·Ĉ(class(i), g, t) mod 2³² from u8 digits, as IMMA would: one chain a digit pair of weight ≤ 3.
    Returns the product and the chains' writes an output."""
    m, nP = Rd.shape
    out = np.zeros((m, Cd.shape[1]), np.int64)
    for c in range(Cd.shape[0]):
        rows = cls == c
        for x, y in itertools.product(range(nR), range(4)):
            if x + y <= 3:
                out[rows] += np.einsum("it,gt->ig", (Rd[rows] >> (8 * x)) & 0xFF, (Cd[c] >> (8 * y)) & 0xFF) << (8 * (x + y))
    pairs = [sum(1 for x in range(nR) if 0 <= w - x < 4) for w in range(4)]
    return out & M32, sum(math.ceil(p * nP / 32) for p in pairs) + 3


def decomp(rule, inst, d):
    """An `nh2` or `nh` sketch over separable words, w = x + R_t(i) + C_t(class, j), taken apart. Each product is
    x·x′ + x·(R + C′) + x′·(R + C) + R² + R·(C + C′) + C·C′, summed:
    - row-only sums (Σ R, Σ R², Σ R_u R_v), shared by the call's columns;
    - column-only sums (Σ C, Σ C·C′ over the value's pairs, Ĉ_t = Σ over its entries of C_t), shared by the rows of a
      class in a coin tile (`TILE`);
    - per value, the cross term Σ_t R_t(i)·Ĉ_t(g) as u8-digit IMMA chains, and per entry a few IMADs with x.
    Zero data, `add`, 32-bit words."""
    kind = nh_kind(rule)
    if not kind or rule.k != 32 or inst.data != "zero" or inst.rec != "add":
        return None
    parts = separable(inst, d)
    if parts is None:
        return None
    R, C, cls, pats, rowlane = parts
    (kind, size), m, n, P = kind, inst.m, inst.n, d.P
    nP, npat, e = len(P), len(pats), entries_a_value(rule)
    x = d.w0 & M32
    if kind == "nh2":
        Rp, Cp = R[:, P] & M32, C[:, :, P] & M32
        Ca, Cb = Cp[:, 0::2], Cp[:, 1::2]
        G = n // e
        Chat = (Ca + Cb).reshape(npat, G, size, nP).sum(axis=2) & M32
        K = mulmod(Ca, Cb).sum(axis=2).reshape(npat, G, size).sum(axis=2) & M32
        colsum = Cp.sum(axis=2) & M32
        cs = colsum[cls]
        xa, xb = x[:, 0::2], x[:, 1::2]
        per = mulmod(xa, (mulmod(xb, nP) + cs[:, 1::2]) & M32) + mulmod(xb, cs[:, 0::2])
        sx = (xa + xb).reshape(m, G, size).sum(axis=2) & M32
        rowterm = mulmod(mulmod(Rp, Rp).sum(axis=1) & M32, size)
        rowsum = Rp.sum(axis=1) & M32
        Rcat, Ccat = Rp, Chat
        alu = 4 * size + 3
    else:
        k2 = nP // 2
        u, v = P[0:2 * k2:2], P[1:2 * k2:2]
        Ru, Rv, Cu, Cv = R[:, u] & M32, R[:, v] & M32, C[:, :, u] & M32, C[:, :, v] & M32
        G = n // size

        def grp(a):
            return a.reshape(npat, G, size, -1).sum(axis=2) & M32

        Chu, Chv = grp(Cu), grp(Cv)
        K = mulmod(Cu, Cv).sum(axis=2).reshape(npat, G, size).sum(axis=2)
        cs = (Cu.sum(axis=2) + Cv.sum(axis=2)) & M32
        rowterm = mulmod(mulmod(Ru, Rv).sum(axis=1) & M32, size)
        rowsum = (Ru.sum(axis=1) + Rv.sum(axis=1)) & M32
        if nP % 2:
            K = K + grp(C[:, :, P[-1:]] & M32)[:, :, 0]
            rowterm = rowterm + mulmod(R[:, P[-1]] & M32, size)
            rowsum = rowsum + 1
        K, rowterm, rowsum = K & M32, rowterm & M32, rowsum & M32
        per = mulmod(x, (mulmod(x, k2) + cs[cls]) & M32)
        sx = x.reshape(m, G, size).sum(axis=2) & M32
        Rcat, Ccat = np.concatenate([Ru, Rv], axis=1), np.concatenate([Chv, Chu], axis=2)
        alu = 2.5 * size + 3
    nR = max(1, math.ceil(int(Rcat.max(initial=0)).bit_length() / 8))
    cross, chains = digit_dot(Rcat, Ccat, cls, nR)
    s = (per.reshape(m, G, -1).sum(axis=2) + mulmod(sx, rowsum[:, None]) + rowterm[:, None] + K[cls] + cross) & M32
    tile = TILE if inst.coins.startswith("lead:") else CALL_M
    col_ops = int((~rowlane).sum()) + 3 + 4 / e
    cost = (chains + alu) / e + col_ops * nP * npat / tile + (nR + 4) * nP / CALL_N
    return cost, s, f"row and column sums + {chains} IMMA writes a value for Σ R·Ĉ ({nR}×4 digits)"


def pack(rule, inst, d):
    """An `xor` fold over words w = d0(i, j) + R_t(i) + C_t(j) at zero data: each lane's V is constant across the
    columns (a row term), or its U takes at most four row patterns (a column term a row class, as the lead's
    row-parity weights). L cells go in b-bit lanes of a 32·g-bit integer of g registers (g 1 or 2, whichever is
    cheaper, and only when L > g), the low lanes biased by 2^(b−1) so no lane borrows. A committed step costs g IADD3s
    (the second with carry) of the packed d0, row and column terms and 2g/BLOCK operand loads, and the fold is the
    honest LOP3 count a packed register. Unpacking (6 writes a cell for `xor` with its final word, 2 a block for
    `xor:m`) undoes the bias's parity and sign-extends from bit b − 1."""
    if not rule.agg.startswith("xor") or rule.k != 32 or inst.data != "zero" or inst.rec != "add":
        return None
    m, n, S = inst.m, inst.n, inst.S
    parts = separable(inst, d)
    if parts is None:
        return None
    R, C, cls, pats, rowlane = parts
    W = d.w0[:, :, None] + R[:, None, :] + C[cls]
    need = int(np.abs(W).max()).bit_length() + 1
    nrow, ncol, P = int(rowlane.sum()), int((~rowlane).sum()), d.P
    mb = xor_block(rule) or len(P)
    ranges = [(s0, min(s0 + mb, len(P))) for s0 in range(0, len(P), mb)]
    unpack = 6 if xor_block(rule) is None else 2 * len(ranges)
    best = None
    for g in (1, 2):
        L = 32 * g // need
        if L <= g:
            continue
        groups = -(-n // L)
        cost = (groups / n * g * (len(P) * (1 + 2 / BLOCK) + xor_lop3s(rule, len(P))) + unpack
                + (nrow + 1) * S / CALL_N + len(pats) * (ncol + L) * g * S / CALL_M)
        if best is None or cost < best[0]:
            best = (cost, g, L)
    if best is None:
        return None
    cost, g, L = best
    width_all, b = 32 * g, 32 * g // L
    u = np.uint64
    out = np.zeros((m, n, len(ranges)), np.int64)
    for gi in range(-(-n // L)):
        cols = list(range(gi * L, min(n, (gi + 1) * L)))
        Dp = sum((d.w0[:, j] + (1 << (b - 1)) * (a < len(cols) - 1)).astype(u) << u(a * b) for a, j in enumerate(cols))
        Rp = sum(R[:, P].astype(u) << u(a * b) for a in range(len(cols)))
        Cp = sum(C[:, j][:, P].astype(u) << u(a * b) for a, j in enumerate(cols))
        Pk = (Dp[:, None] + Rp + Cp[cls]) & u((1 << width_all) - 1)
        for bi, (s0, s1) in enumerate(ranges):
            sp = np.bitwise_xor.reduce(Pk[:, s0:s1], axis=1)
            for a, j in enumerate(cols):
                top = a == len(cols) - 1
                width = width_all - a * b if top else b
                x = ((sp >> u(a * b)) & u((1 << width) - 1)).astype(object)
                if not top and (s1 - s0) % 2:
                    x ^= 1 << (b - 1)
                if width < 32:
                    x = np.where(x >> (width - 1) & 1, x | (M32 ^ ((1 << width) - 1)), x)
                out[:, j, bi] = (x & M32).astype(np.int64)
    how = f"{L} cells in {b}-bit lanes of {g} register(s)" + (f", {len(pats)} row classes" if len(pats) > 1 else "")
    return cost, (out if xor_block(rule) else out[:, :, 0]), how


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
              "bsketch": bsketch, "expand": expand, "pack": pack, "decomp": decomp, "copy": copy, "table": table}
LEAD_STRATEGIES = ("honest", "pack", "decomp", "copy", "table")


def evaluate(rule: Rule, inst: Inst, strategies=STRATEGIES):
    if inst.coins.startswith("lead:"):
        strategies = {k: v for k, v in strategies.items() if k in LEAD_STRATEGIES}
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
        "gap_claimed": round(max(0.0, (h_cost - cheapest) / stock), 6),
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


POSITIONS = ["all", "every:2", "every:4", "every:8", "every:64", "final"]
BITS = ["32", "low:1", "low:2", "low:3", "low:4", "low:8", "high:8", "high:1"]
AGGS = ["hash", "lin", "sq", "sbox"]


def make_rules(bits=BITS, aggs=AGGS):
    return [Rule(p, b, a) for p, b, a in itertools.product(POSITIONS, bits, aggs)]


RULES = make_rules()


def instances(quick: bool):
    S_set = [64] if quick else [64, 256, 1024]
    q_set = [1, 4] if quick else [1, 2, 4, 8]
    for S, q, data, rec, seed in itertools.product(S_set, q_set, ["zero", "random"], ["add", "mul"], [0, 1]):
        if not (quick and seed) and not (S == 1024 and seed):
            yield Inst(64, 16, S, q, data, rec, seed)
    for S, q in itertools.product(S_set, [2, 4] if quick else [2, 4, 8]):
        yield Inst(64, 16, S, q, "zero", "add", 0, "sep")
    lead = [("lead:e7", 5), ("lead:e4", 3), ("lead:w127", 5), ("lead:rot", 5)]
    for S, (coins, q), data, seed in itertools.product([67] if quick else [67, 265], lead, ["zero", "random"], [0, 1]):
        if not (quick and seed):
            yield Inst(64, 48, S, q, data, "add", seed, coins)


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
    rule5, inst5 = Rule("all", "32", "xor"), Inst(64, 16, 67, 5, "zero", "add", 7, "lead:e7")
    d5 = Draw(inst5, rule5)
    _, ref5, _ = honest(rule5, inst5, d5)
    d5.w0 = d5.w0.copy()
    d5.w0[0, 1] += 1
    _, out5, _ = pack(rule5, inst5, d5)
    checks.mutation("pack with one cell's offset off by one", not np.array_equal(out5, ref5))
    d5 = Draw(inst5, rule5)
    d5.P = d5.P[:-1]
    _, out5, _ = pack(rule5, inst5, d5)
    checks.mutation("pack folding one word fewer", not np.array_equal(out5, ref5))
    for agg in ("nh2:2", "nh:4"):
        rule6, inst6 = Rule("all", "32", agg), Inst(64, 16, 67, 5, "zero", "add", 7, "lead:rot")
        d6 = Draw(inst6, rule6)
        _, ref6, _ = honest(rule6, inst6, d6)
        d6.w0 = d6.w0.copy()
        d6.w0[3, 5] += 1
        _, out6, _ = decomp(rule6, inst6, d6)
        checks.mutation(f"decomp {agg} with one cell's offset off by one", not np.array_equal(out6, ref6))
        d6 = Draw(inst6, rule6)
        d6.U = d6.U.copy()
        d6.U[1::2, :, 2:] = d6.U[0::2, :, 2:]
        _, out6, _ = decomp(rule6, inst6, d6)
        checks.mutation(f"decomp {agg} with the row-parity classes merged", not np.array_equal(out6, ref6))


def main() -> int:
    checks = Checks()
    args = [a for a in checks.args if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in checks.args if a.startswith("--") and "=" in a)
    quick = "--quick" in checks.args
    out_path = args[0] if args else "hash_counter.json"
    rules = make_rules(opts["bits"].split(",") if "bits" in opts else BITS,
                       opts["aggs"].split(",") if "aggs" in opts else AGGS)
    full = rules == RULES
    results = []
    for inst in instances(quick):
        for rule in rules:
            results.append(evaluate(rule, inst))
    for name in STRATEGIES:
        ran = [r for r in results if name in r["strategies"]]
        bad = [f"{r['rule']} at {r['inst']}" for r in ran if not r["strategies"][name]["exact"]]
        checks.check(f"{name} matches the honest commitment bit for bit", not bad,
                     f"{len(ran) - len(bad)}/{len(ran)} (rule, instance) pairs" + (f"; first miss {bad[0]}" if bad else ""))
    breaks = [r for r in results if r["breaks_T1"]]
    survive = [r for r in results if not r["breaks_T1"]]
    known = {
        "fold": [r for r in results if r["rule"] == "all/32/lin" and r["inst"]["data"] == "zero"
                 and not r["inst"]["coins"].startswith("lead:")],
        "random": [r for r in results if r["inst"]["data"] == "random"],
    }
    checks.check("no strategy beats honest on random data", all(not r["breaks_T1"] for r in known["random"]))
    checks.check("bitslice op counts", [bitslice_ops(k) for k in (1, 2, 3, 4)] == [1, 5, 12, 22],
                 [bitslice_ops(k) for k in (1, 2, 3, 4)])
    if full:
        checks.check("fold beats stock on every zero-data all/32/lin instance with q ≤ 4",
                     all(r["cheapest"] == "fold" and r["breaks_T1"] for r in known["fold"] if r["inst"]["q"] <= 4))
        sep_sq = [r for r in results if r["rule"] == "all/32/sq" and r["inst"]["coins"] == "sep"]
        checks.check("expand is cheapest on every separable all/32/sq instance",
                     bool(sep_sq) and all(r["cheapest"] == "expand" for r in sep_sq),
                     [(r["inst"]["S"], r["inst"]["q"], r["gap_share"]) for r in sep_sq])
        hash32 = [r for r in results if r["rule"] == "all/32/hash"]
        checks.check("all/32/hash costs SHA_REGS/32 writes a word (h · stock)",
                     all(abs(r["common_hash"] - r["inst"]["S"] * 32 * WB) < 1e-6 for r in hash32))
    xor = [r for r in results if r["rule"] == "all/32/xor" and r["inst"]["data"] == "zero"]
    if xor:
        narrow = [r for r in xor if r["inst"]["coins"] in ("lead:e7", "lead:e4")]
        checks.check("pack is cheapest on every narrow lead all/32/xor instance",
                     bool(narrow) and all(r["cheapest"] == "pack" for r in narrow),
                     [(r["inst"]["coins"], r["inst"]["S"], r["strategies"]["pack"]["how"], r["gap_claimed"])
                      for r in narrow])
        wide = [r for r in xor if r["inst"]["coins"] in ("lead:w127", "lead:rot")]
        checks.check("pack at lead:w127 and lead:rot (words past 16 bits) uses two-register lanes",
                     bool(wide) and all("2 register" in r["strategies"].get("pack", {}).get("how", "") for r in wide),
                     [(r["inst"]["S"], r["strategies"].get("pack", {}).get("how"), r["gap_claimed"]) for r in wide])
    nh = [r for r in results if nh_kind(Rule(*r["rule"].split("/"))) and r["inst"]["data"] == "zero"
          and r["rule"].startswith("all/32/") and (r["inst"]["coins"].startswith("lead:") or r["inst"]["coins"] == "sep")]
    if nh:
        checks.check("decomp is cheapest and breaks T1 on every zero-data separable all/32/nh instance",
                     all(r["cheapest"] == "decomp" and r["breaks_T1"] for r in nh),
                     sorted({(r["rule"], r["inst"]["coins"], r["inst"]["S"], round(r["gap_share"], 3)) for r in nh})[:8])
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
    json.dump({"schema": "hash-counter/v1", "rules": len(rules), "instances": len(results) // len(rules),
               "breaks": len(breaks), "survive": len(survive), "summary": summary, "results": results,
               "per_period_at_1pct": per_period(T1)}, open(out_path, "w"), indent=1, default=list)
    for k, v in summary.items():
        print(f"{k:24s} breaks {v['breaks']:3d}/{v['of']:3d} max gap {v['max_gap']:.4f} h {v['h_min']:.2f}–"
              f"{v['h_max']:.2f} by {','.join(v['by'])}")
    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
