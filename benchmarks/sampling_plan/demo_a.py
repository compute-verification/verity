"""Demo A's frontier on one H200: plain INT8 weights, no P2 decode, a block of 16,384 requests at a 6.35 s step (its
commit 1.0 s of it), on the map of every subcircuit to a Definition C-Flock stages (`examples/h200/e2e_h200/plan.py`'s
`Map`) with PoUW's tiles as one more subcircuit, planned by `optimize.frontier` under the demo's objective (its work half
circuit size): half skipped work, half leaked output bits.

**Harm.** A unit's work is its subcircuit's own ANDs (`Candidate.ands`, not its rows' openings), every subcircuit alike,
the PoUW tiles included, with no PoUW work table and no influence. Every unit carries its output bits. At each budget B
the exchange rate divides each half by the least bound B certifies for that half alone,
`Objective(delta, work=0.5/H_work(B), bits=0.5/H_bits(B))`, which is scale-free, and the plan reports the certificate's
two bounds from its one draw, `free_compute` and `leak`, at delta = 1% and 1e-6, and the tile stratum's epsilon at delta
(`pouw.window.eps_at`).

**Units** (`families`). Each subcircuit is the map's Definition at its measured ANDs and C-Flock's staged k_log (at most
27, its statement cap `min(2^(35 - k_log), 2^20)`), its units counted from the shape: the s8 coordinates as
`ScaledMmS8CoordinateRows_v1` (x in the quantizer's 8 KiB chunk rows), the residual stream as 128 chunk rows of 256 B
(`Bf16AddRow_v2{128}` x 128 an add, RMSNorm as `RmsSumSqPart_v1` and `RmsNormOutSums_v1` a chunk, the quantizer as
`RowAbsMax_v1` and `QuantChunkS8_v1`), the gathers as `AllGather_v1` (the O projection's input `{2,64}`), attention as
`QkSoftmaxRow_v1` and `PvRow_v1` a head and position, and the embedding and argmax a coordinate and a token. PoUW's tiles
are J10-R's per-call lane units, `PouwH1LaneZy_v2` over `PouwH1Tile_v5` (a 32 x 32 leaf over the call's K), one
subcircuit at each K, at their traced ANDs (`TILE_TRACED`): a statement for each 131,072
of its MACs (`PouwH1LaneZyChunk_v2`) and one for the combine, each statement's output the one bit `ok`,
and about 257 bits of receipt committed a chunk. Drawn lanes share their chunks' statements, at the chunk's k_log.

**Budget.** Proving is within 1% of the block's compute. Serving's overhead is beside it, never in it: the commits
(1.0 s a step, `COMMIT_STEP_S`) and PoUW's per-call noise. One serving
block's 1% is less than one statement's fixed cost, so the plan of record is given for sampling blocks (the proof blocks
one draw covers, committed and registered before it) of the least length whose 1% affords a plan, a week, a month and a
year: one draw over all their units, its statements shared per subcircuit. M3's live draw (`m3`) is a uniform 8 of one
block's units (and, for its reservation, 8 of its tile lanes), given as its cost against the block's GPU time and its 1%
budget, and what it certifies.

**Certificate.** `certify`'s: one stratified harm bound over every subcircuit (core's `Stratified.harm_bound`), with no
delta split by subcircuit, its leak's location term counting the wrong units by the unit-weighted harm bound in place of
the exact `count_bound` (`certificate`). The bound of record is `Flock.Guarantees.JointStratified`
(Security/Proofs/Flock/EndToEnd/Joint/Statement.lean): per sampling block, one detection budget
`sum_s k_s x_s / n_s <= ln(1/delta)` over every subcircuit, and the ANDs and output bits of an accepted block's wrong
units each at most `StratifiedBlock.maxHarm`, their exact supremum over the sets escaping with probability at least
delta, from one draw at one delta. A difftest holds `certify`'s fields to that supremum.

**Cost.** `h200.Rates`' fit with statements shared (`CostModel.shared`: one session a subcircuit a sampling block,
`ceil(k / cap)` statements of drawn units), its feature a unit's ANDs with its rows' openings, every unit's outputs
committed at the GPU row committer's 225 GB/s, and a drawn request replayed at its share of the block's
GPU time (a stand-in). A lane unit's 32 x 32 outputs span 32 requests, so each lane subcircuit is its own replay domain,
a drawn unit replayed at 32 requests' share (`TILE_REQUESTS`), unshared with the other subcircuits' replays.

    uv run python benchmarks/sampling_plan/demo_a.py --out DIR [--spans NAME ...] [--no-frontier]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from fractions import Fraction
from math import ceil, comb, floor, fsum, inf, log, log2, nextafter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h200 as H  # noqa: E402

from verity.core.protocols.pouw.window import eps_at  # noqa: E402
from verity.core.protocols.profile import MARGIN, Stratified, Stratum, neg_log_ratio  # noqa: E402
from verity.core.protocols.sampled_proofs.optimize import (  # noqa: E402
    Candidate, Certificate, CostModel, Objective, Plan, ReplayDomain, TemplateCost, _plan, density_bound, for_budget,
    frontier, row_ands)

#: Demo A's block: 16,384 requests, a 6.35 s step (the commit 1.0 s of it), 32 steps a request
SHAPE = H.Shape(batch=16_384, step_s=6.35, steps=32)
#: the step's GPU time by what runs, seconds: the INT8 linears, and attention with the row ops
STEP_SPLIT = {"ScaledMmS8Coordinate_v1": 5.08, "attention and row ops": 0.25}
#: committing a step's outputs, GPU-s: serving's overhead, beside the proving budget
COMMIT_STEP_S = 1.0
#: the GPU row committer's measured rate, bytes a second
COMMIT_BPS = 225e9
DELTAS = H.DELTAS
PERIODS = H.PERIODS
#: the proving budget: proving at this share of the sampling block's GPU time
BUDGET_SHARE = 0.01
#: the budgets swept, in multiples of one serving block's GPU time, past the cheapest plan's
GRID = (1, 8.5, 10, 12.5, 15, 20, 30, 50, 100, 300, 1000)
S8 = "ScaledMmS8CoordinateRows_v1"
TILE = "PouwH1LaneZy_v2"
#: C-Flock's largest statement (it refuses k_log 28)
K_LOG_MAX = 27
#: M3's live draw: a uniform subset of this many of one block's units
M3_UNITS = 8
#: PoUW's proof block: J10-R's per-call lane unit `PouwH1LaneZy_v2` over
#: `PouwH1Tile_v5`, 1,024 outputs over the call's whole K, traced on its Boolean form: at each K its ANDs and its
#: statements, one `PouwH1LaneZyChunk_v2` of 131,072 MACs each and the combine, each statement's output the one bit `ok`
TILE_OUTPUTS = 1024
TILE_CHUNK_MACS = 131_072
TILE_TRACED = {16_384: (5_540_007_551, 129), 65_536: (22_095_177_215, 513)}
#: serving commits only the receipt, about 257 bits a chunk
TILE_COMMIT_BITS = 257
#: a chunk's rows, PouwH1Tile_v1's measured lane's (a stand-in until the v2 chunk's are staged), and its staged k_log
TILE_ROWS = (4096,) * 12 + (64,) * 3
TILE_K_LOG = 27
#: a unit is a 32 x 32 leaf (`h1zy`: 32 lanes of 4 x 8 and their combine), its rows 32 tokens, in decode 32 requests
TILE_REQUESTS = 32


def tile_parts() -> tuple[int, int]:
    """A chunk's ANDs and the combine's, from the two traced lanes (each lane its chunks and one combine)."""
    (a0, s0), (a1, s1) = (TILE_TRACED[k] for k in sorted(TILE_TRACED))
    chunk, rest = divmod(a1 - a0, s1 - s0)
    combine = a0 - (s0 - 1) * chunk
    if rest or combine <= 0 or any(s - 1 != k * TILE_OUTPUTS // TILE_CHUNK_MACS for k, (_, s) in TILE_TRACED.items()):
        raise ValueError(f"TILE_TRACED is not chunks of {TILE_CHUNK_MACS} MACs plus one combine: {TILE_TRACED}")
    return chunk, combine
#: `PvRow_v1{T}`'s staged ANDs, linear between
PV_ANDS = ((1, 111_104), (8, 321_024), (31, 1_009_152), (32, 1_039_168))


@dataclass(frozen=True)
class Unit(H.Template):
    """A subcircuit of the block at C-Flock's staged `k_log`, serving committing `commit` bits a unit (its `out_bits`
    when None). A unit of `chunks` chunks (PoUW's lanes) is proved as its chunks at `k_log`, a statement holding `cap`
    whole units in `groups` statements, and one combine."""

    k_log: int = 0
    commit: int | None = None
    chunks: int = 0

    @property
    def cap(self) -> int:
        base = min(1 << (35 - self.k_log), 1 << 20)
        return max(1, base // self.chunks) if self.chunks else base

    @property
    def groups(self) -> int:
        return max(1, ceil(self.chunks / min(1 << (35 - self.k_log), 1 << 20))) if self.chunks else 1


def qk_ands(t: int) -> int:
    """`QkSoftmaxRow_v1{T=t}`'s ANDs, exact at T = 8, 16 and 32, pinned at T = 1."""
    return 39_924 if t == 1 else 37_692 * t + 2_218


def pv_ands(t: int) -> int:
    for (a, x), (b, y) in zip(PV_ANDS, PV_ANDS[1:]):
        if a <= t <= b:
            return x + (y - x) * (t - a) // (b - a)
    raise ValueError(f"PvRow_v1 is staged for 1 <= T <= 32, not {t}")


def attention_k_log(t: int) -> int:
    """Both attention units' staged k_log at T keys (22 to T = 6, 23 to 14, 24 to 30, 25 at 31)."""
    return max(22, 19 + ceil(log2(t + 2)))


def families(m: H.Shape = SHAPE, *, tiles: bool = True) -> list[Unit]:
    """The map's partition of one block, each subcircuit's units counted per token (R T a block) and
    per generated token (R G), and with `tiles` PoUW's tile lanes over every s8 multiply-accumulate."""
    R, L, T, Gn, d, f, V = m.batch, m.layers, m.positions, m.generate, m.width, m.mlp, m.vocab
    q, kv, heads = m.heads * m.head_dim, m.kv_heads * m.head_dim, m.heads
    out: list[Unit] = []

    def put(tid: str, token: int, gen: int, ands: int, rows: tuple[int, ...], out_bits: int, k: int,
            commit: int | None = None) -> None:
        out.append(Unit(tid, R * (T * token + Gn * gen), ands, rows, out_bits, True, k, commit))

    chunks = (256,) * (d // 128)
    for k, r, token, gen, ands, kl in ((d, 2, L * (q + 2 * kv + d + f), V, 2_150_483, 25),
                                       (f, 8, L * d, 0, 8_586_323, 27)):
        put(f"{S8}{{K={k},R={r},DOT=BlackwellS8ImmaDot32_v1}}", token, gen, ands,
            (k // r,) * r + (4,) + (k // r,) * r + (4, 2), 16, kl)
    put("AllGather_v1{WORLD=64,N=1}", L * (heads + 2 * m.kv_heads), 0, 0, (2,) * 64 + (128,), 1024, 26)
    put("AllGather_v1{WORLD=128,N=1}", L * (2 * d + f) // 128 + d // 128, V // 128, 0, (2,) * 128 + (256,), 2048, 27)
    put("AllGather_v1{WORLD=2,N=64}", L * heads // 2, 0, 0, (128, 128, 256), 2048, 22)
    put("RowAbsMax_v1{N=16384,RELU=True}", L * (f // d - 1), 0, 1_228_770, chunks + (2,), 16, 27)
    put("RowAbsMax_v1{N=16387,RELU=True}", L, 0, 1_228_995, chunks + (2,) * 4, 16, 27)
    put("QuantChunkS8_v1{N=8192,RELU=True}", L * f // 8192, 0, 22_602_634, (256,) * 64 + (2, 8192, 4), 65_568, 27)
    put("RowAbsMax_v1{N=16384,RELU=False}", 3 * L, 1, 491_490, chunks + (2,), 16, 27)
    put("QuantChunkS8_v1{N=8192,RELU=False}", 3 * L * d // 8192, d // 8192, 22_225_802, (256,) * 64 + (2, 8192, 4),
        65_568, 27)
    put("RmsSumSqPart_v1{C=128}", 2 * L * d // 128, d // 128, 128 * (2406 + 823), (256, 4), 32, 22)
    put("RmsNormOutSums_v1{C=128,S=128,G=2,N=16384,EPS=1e-05}", 2 * L * d // 128, d // 128, 360_356,
        (256, 256) + (4,) * 128 + (256,), 2048, 27)
    for t in range(1, T + 1):
        out.append(Unit(f"QkSoftmaxRow_v1{{T={t},D=64,DOT=HopperBF16WgmmaDot16_v2}}", R * L * heads, qk_ands(t),
                        (128,) * (t + 1) + (2 * t,), 16 * t, True, attention_k_log(t)))
        out.append(Unit(f"PvRow_v1{{T={t},K=32,D=64,DOT=HopperBF16WgmmaDot16_v1}}", R * L * heads, pv_ands(t),
                        (2 * t,) + (128,) * (t + 1), 1024, True, attention_k_log(t)))
    put("Bf16AddRow_v2{N=128}", 2 * L * d // 128, 0, 91_776, (256, 256, 256), 2048, 22)
    put("EmbeddingCoordinate_v1{V=256}", d, 0, 4_820, (1, 2 * V, 2, 2), 16, 22)
    put("Argmax_v2{V=256}", 0, 1, 15_557, (2 * 128, 2 * 128, 2), 16, 22)
    if tiles:
        for k, outputs in ((d, R * (T * L * (q + 2 * kv + d + f) + Gn * V)), (f, R * T * L * d)):
            ands, statements = TILE_TRACED[k]
            chunks = statements - 1
            if outputs % TILE_OUTPUTS:
                raise ValueError(f"{outputs} outputs at K={k} are not whole lanes of {TILE_OUTPUTS}")
            out.append(Unit(f"{TILE}{{H1=v5,K={k},N={TILE_OUTPUTS}}}", outputs // TILE_OUTPUTS, ands,
                            TILE_ROWS * chunks, statements, True, TILE_K_LOG, TILE_COMMIT_BITS * chunks, chunks))
    return out


def candidates(m: H.Shape = SHAPE, *, blocks: int = 1, tiles: bool = True) -> list[Candidate]:
    """One candidate a subcircuit over a sampling block of `blocks` serving blocks, its unit's work its own ANDs, its
    output bits committed, and its statement cap its staged k_log's."""
    return [Candidate(t.id, t.id, t.units * blocks, t.ands, t.out_bits, ands=t.ands, rows=t.rows,
                      commit_bits=t.out_bits if t.commit is None else t.commit, cap=t.cap,
                      replay=t.id if t.chunks else "request", per_replay=1 if t.chunks else t.units // m.batch)
            for t in families(m, tiles=tiles)]


def costs(m: H.Shape = SHAPE, blocks: int = 1) -> CostModel:
    """The fit's rates, each subcircuit's costs derived from them once (`CostModel.of`'s arithmetic, which otherwise
    reopens every row at each of the sweep's million lookups)."""
    r = H.Rates.fitted()
    chunk, combine = tile_parts()
    chunk += sum(row_ands(b) for b in TILE_ROWS)

    def fixed(t: Unit) -> float:
        if not t.chunks:
            return r.statement_fixed + r.statement_per_and * t.proved
        return t.groups * (r.statement_fixed + r.statement_per_and * chunk) + r.statement_fixed \
            + r.statement_per_and * combine

    each = {t.id: TemplateCost(statement_fixed=fixed(t), statement_per_unit=0.0, unit=r.per_and * t.proved)
            for t in families(m)}
    return CostModel(templates=each, per_and=r.per_and, statement_fixed=r.statement_fixed,
                     statement_per_and=r.statement_per_and,
                     commit_per_bit=1 / (8 * COMMIT_BPS),
                     replay={"request": ReplayDomain(m.batch * blocks, m.block_s / m.batch)}
                     | {t.id: ReplayDomain(t.units * blocks, TILE_REQUESTS * m.block_s / m.batch)
                        for t in families(m) if t.chunks},
                     shared=True, source="h200.Rates.fitted")


_RISK: dict[tuple[int, int, int], float] = {}


class _Memo(Stratum):
    """Core's `Stratum`, its `_risk` remembered: `harm_bound`'s bisection asks the same m of each stratum many times, and
    `_risk` sums k terms while k <= 2^14. Every value is core's, but where core's fails:

    - past 2^53 units m / (n - i) rounds to 1 at m = n - k, and `log1p(-1)` raises: there each term is core's
      `neg_log_ratio` of (n - i - m) / (n - i) in integers;
    - past 2^14 draws core takes lgamma's differences, which at a year's 1e17 units keep no digit (lgamma(1e17) is
      4e18, its ulp 512): past 2^25 units this sums core's own log1p terms instead, pairwise in binary64."""

    def _risk(self, m: int) -> float:
        key = (self.size, self.k, m)
        if key in _RISK:
            return _RISK[key]
        n, k = self.size, self.k
        if not (k > 1 << 14 and n > 1 << 25 and 0 < m <= self._finite()):
            try:
                v = Stratum._risk(self, m)
            except ValueError:
                v = fsum(neg_log_ratio(n - i - m, n - i) for i in range(k))
        elif 2 * m < n - k + 1:
            v = -float(np.log1p(-m / (n - np.arange(k, dtype=np.float64))).sum())
        else:
            v = fsum(neg_log_ratio(n - i - m, n - i) for i in range(k))
        _RISK[key] = v
        return v


def law(sizes: dict[str, int], cands: list[Candidate]) -> Stratified:
    return Stratified(tuple(_Memo(c.template, c.units, sizes[c.template]) for c in cands if c.template in sizes))


@dataclass(frozen=True)
class Point:
    budget: float
    h_work: float
    h_bits: float
    plan: Plan
    at: dict[str, dict[str, float]]


def location(n: int, count: int) -> float:
    """`consumers.location_bits` at a count bound `count` of n units: log2 of sum_{j <= count} C(n, j), rounded up."""
    if count <= 0 or n == 0:
        return 0.0
    if 2 * count >= n:
        return float(n)
    x = count / n
    return min(float(n), nextafter(n * (-x * log2(x) - (1 - x) * log2(1 - x)) * (1 + MARGIN), inf))


def certificate(sizes: dict[str, int], cands: list[Candidate], objective: Objective) -> Certificate:
    """`certify`'s certificate, its location term's count the floor of core's `harm_bound` with every unit weighing 1
    instead of the exact `count_bound`, whose exact binomials take hours at 1e12 units. Any set with more units escapes
    with less than delta, so that floor is `audit_exfiltration`'s k (`L.miss (k + 1) < delta`), and as it is at least
    `count_bound` the leak is at least `certify`'s."""
    chosen = [c for c in cands if c.template in sizes]
    d = objective.delta
    st = law(sizes, chosen)
    harm = st.harm_bound({c.template: objective.harm(c) for c in chosen}, d)
    work = st.harm_bound({c.template: float(c.work + c.influence) for c in chosen}, d)
    count = floor(st.harm_bound({c.template: 1.0 for c in chosen}, d))
    leak = st.harm_bound({c.template: float(c.out_bits) for c in chosen}, d) + location(st.population, count)
    dens = density_bound(((c.units, sizes[c.template], objective.harm(c)) for c in chosen), d)
    return Certificate(d, harm, dens, work, leak)


def _best(cands, cm, objective, budget, grid) -> Plan | None:
    """The least-harm plan within `budget`, the sweep screened by `density_bound` and only the plan taken certified
    (`certificate`)."""
    plans = frontier(cands, cm, objective, grid=grid, exact=False)
    p = for_budget(plans, budget, candidates=cands, costs=cm, objective=objective, exact=False)
    return None if p is None else replace(p, certificate=certificate(p.sizes, cands, objective))


def commit(cands: list[Candidate], cm: CostModel) -> float:
    return sum(cm.commit(c) for c in cands)


def cheapest(cands: list[Candidate], cm: CostModel) -> float:
    """The proving cost of the least plan, every subcircuit at its floor of one drawn unit."""
    p = _plan(cands, cm, Objective(DELTAS[0], work=0.0, bits=1.0), 0.0, frozenset(c.template for c in cands),
              frozenset())
    return p.cost["total"] - commit(cands, cm)


def point(cands: list[Candidate], cm: CostModel, proving: float, *, grid: int = 41) -> Point | None:
    """At proving budget `proving` each half's least bound alone, then the plan under the exchange rate there, certified at
    both deltas from its one draw; None once a half is proved whole (its least bound below 1)."""
    d, total = DELTAS[0], proving + commit(cands, cm)
    hw = _best(cands, cm, Objective(d, work=1.0), total, grid).certificate.free_compute
    hb = _best(cands, cm, Objective(d, work=0.0, bits=1.0), total, grid).certificate.leak
    if not (hw >= 1 and hb >= 1):
        return None
    mixed = Objective(d, work=0.5 / hw, bits=0.5 / hb)
    p = _best(cands, cm, mixed, total, grid)
    at = {}
    for delta in DELTAS:
        c = p.certificate if delta == d else certificate(p.sizes, cands, replace(mixed, delta=delta))
        at[str(delta)] = {"free_compute": c.free_compute, "leak": c.leak}
    return Point(proving, hw, hb, p, at)


def tile_eps(sizes: dict[str, int], cands: list[Candidate]) -> dict[str, dict[str, float]] | None:
    """Each tile stratum's epsilon at each delta from the plan's draw (`eps_at`, every lane credited its ANDs), by K."""
    out = {}
    for c in cands:
        if c.template.startswith(TILE) and sizes.get(c.template):
            k, n, w = sizes[c.template], c.units, c.ands
            out[c.template] = {str(d): float(eps_at(k, Fraction(d), n * w, n, w)) for d in DELTAS}
    return out or None


def tile_eps_max(eps: dict[str, dict[str, float]] | None) -> dict[str, float]:
    return {str(d): max(e[str(d)] for e in eps.values()) for d in DELTAS} if eps else {}


def sweep(cands: list[Candidate], cm: CostModel, m: H.Shape = SHAPE, *, grid: int = 41,
          budgets: tuple[float, ...] = GRID) -> dict:
    """The frontier over one serving block's proving budgets (`budgets`, in multiples of its GPU time)."""
    low = cheapest(cands, cm)
    points, skipped, complete = [], [], None
    for g in budgets:
        proving = g * m.block_s
        if proving < low:
            skipped.append(proving)
            continue
        p = point(cands, cm, proving, grid=grid)
        if p is None:
            complete = proving
            break
        points.append(p)
    return {"commit": commit(cands, cm), "cheapest": low, "skipped": skipped, "points": points, "complete": complete}


def least_blocks(m: H.Shape = SHAPE, *, share: float = BUDGET_SHARE) -> int:
    """The fewest serving blocks a sampling block spans for `share` of its GPU time to afford the cheapest plan. The
    cheapest plan's cost only grows with the span (its drawn units meet more requests to replay), so no span shorter
    than one block's cheapest over `share` of a block affords it, and past the first span that does every span does."""
    def affords(n: int) -> bool:
        return cheapest(candidates(m, blocks=n), costs(m, n)) <= share * n * m.block_s

    lo = max(1, ceil(cheapest(candidates(m), costs(m)) / (share * m.block_s)))
    if affords(lo):
        return lo
    hi = lo
    while not affords(hi):
        lo, hi = hi, 2 * hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if affords(mid) else (mid, hi)
    return hi


def spans(m: H.Shape = SHAPE, *, share: float = BUDGET_SHARE, periods: dict[str, int] = PERIODS) -> dict[str, int]:
    """The sampling blocks the plan of record is given for, in serving blocks."""
    return ({"one block": 1, "least": least_blocks(m, share=share)}
            | {name: floor(seconds / m.block_s) for name, seconds in periods.items()})


def plan_at(m: H.Shape, name: str, n: int, *, share: float = BUDGET_SHARE, grid: int = 41) -> dict:
    """The plan at `share` of a sampling block of n serving blocks: its draw a subcircuit, its cost a serving block, and
    its two bounds at each delta, each also as a share of the sampling block's ANDs and output bits, with the tile
    stratum's epsilon and serving's commits beside it."""
    cands, cm = candidates(m, blocks=n), costs(m, n)
    ands, bits = sum(c.units * c.ands for c in cands), sum(c.units * c.out_bits for c in cands)
    budget, low = share * n * m.block_s, cheapest(cands, cm)
    row = {"span": name, "blocks": n, "hours": n * m.block_s / 3600, "budget": budget, "cheapest": low,
           "commit": commit(cands, cm), "commit_hash20": n * m.steps * COMMIT_STEP_S, "ands": ands, "bits": bits,
           "affordable": budget >= low}
    p = point(cands, cm, budget, grid=grid) if budget >= low else None
    if p is not None:
        proving = p.plan.cost["total"] - row["commit"]
        row |= {"cost": dict(p.plan.cost), "proving": proving, "per_block": proving / n,
                "weights": {"work": 0.5 / p.h_work, "bits": 0.5 / p.h_bits}, "H_work": p.h_work, "H_bits": p.h_bits,
                "sizes": dict(p.plan.sizes), "drawn": sum(p.plan.sizes.values()),
                "statements": dict(p.plan.statement_size),
                "draw": [{"subcircuit": c.template, "units": c.units, "drawn": p.plan.sizes.get(c.template, 0),
                          "ands": c.ands, "proved_ands": cm.ands(c), "k_log": t.k_log, "cap": c.cap,
                          "out_bits": c.out_bits} for c, t in zip(cands, families(m))],
                "at": {d: v | {"work_share": v["free_compute"] / ands, "bits_share": v["leak"] / bits}
                       for d, v in p.at.items()},
                "tile_eps": tile_eps(p.plan.sizes, cands)}
    return row


def m3(m: H.Shape = SHAPE, k: int = M3_UNITS, *, tiles: bool = False, only: str | None = None) -> dict:
    """M3's live draw: a uniform k of one block's units (`subset:k`, one stratum; with `only`, of the subcircuits whose
    id starts with it), its expected proving cost, `ceil(x / cap)` statements of a subcircuit it draws x units of, against
    the block's GPU time and its 1% budget, and what it certifies by core's `harm_bound` (the heaviest units first), the
    leak's location term at the unit-weighted bound."""
    cands, cm = candidates(m, tiles=tiles or only is not None), costs(m)
    if only is not None:
        cands = [c for c in cands if c.template.startswith(only)]
    N = sum(c.units for c in cands)
    cost, total = 0.0, comb(N, k)
    for c in cands:
        a0, a1 = cm.statement(c)
        statements = sum(comb(c.units, x) * comb(N - c.units, k - x) * ceil(x / c.cap) for x in range(1, k + 1))
        requests = TILE_REQUESTS if c.template.startswith(TILE) else 1
        cost += statements / total * a0 + k * c.units / N * (a1 + cm.unit(c) + cm.of(c).replay
                                                             + requests * m.block_s / m.batch)
    st = Stratified((Stratum("unit", N, k),))
    ands, bits = sum(c.units * c.ands for c in cands), sum(c.units * c.out_bits for c in cands)
    at = {}
    for d in DELTAS:
        work = st.harm_bound({"unit": [(float(c.ands), c.units) for c in cands]}, d)
        count = floor(st.harm_bound({"unit": 1.0}, d))
        leak = st.harm_bound({"unit": [(float(c.out_bits), c.units) for c in cands]}, d) + location(N, count)
        at[str(d)] = {"free_compute": work, "leak": leak, "work_share": work / ands, "bits_share": leak / bits,
                      "wrong_units": count}
    return {"k": k, "tiles": tiles, "only": only, "units": N, "ands": ands, "bits": bits, "cost": cost,
            "of_block": cost / m.block_s, "of_budget": cost / (BUDGET_SHARE * m.block_s), "at": at}


def ands_against_gpu(m: H.Shape = SHAPE) -> dict:
    """Each Definition's share of the block's own ANDs, of its proved ANDs (with the rows' openings, what proving costs)
    and of its units, beside the step's GPU split (`STEP_SPLIT`): where circuit size misstates GPU time. The PoUW tiles
    are left out: they recompute the s8 linears."""
    by: dict[str, dict[str, float]] = {}
    for t in families(m, tiles=False):
        g = by.setdefault(t.id.split("{")[0], {"units": 0, "ands": 0, "proved": 0})
        g["units"] += t.units
        g["ands"] += t.units * t.ands
        g["proved"] += t.units * t.proved
    ands, proved = sum(g["ands"] for g in by.values()), sum(g["proved"] for g in by.values())
    gpu = sum(STEP_SPLIT.values())
    s8 = sum(g["ands"] for k, g in by.items() if S8 in k)
    return {"definitions": {k: g | {"ands_share": g["ands"] / ands, "proved_share": g["proved"] / proved}
                            for k, g in sorted(by.items(), key=lambda kv: -kv[1]["ands"])},
            "step_split": STEP_SPLIT,
            "s8": {"ands_share": s8 / ands, "gpu_share": STEP_SPLIT["ScaledMmS8Coordinate_v1"] / gpu},
            "rest": {"ands_share": 1 - s8 / ands, "gpu_share": 1 - STEP_SPLIT["ScaledMmS8Coordinate_v1"] / gpu}}


def knee(points: list[Point], ands: float, bits: float) -> int:
    """The point farthest below the chord between the ends of log(cost) against the log of the geometric mean of the two
    certified shares at delta = 1%, `free_compute` of the block's ANDs and `leak` of its output bits (`kneedle` on the
    log-log frontier). Each point's own rate normalizes each half by that point's best, so it cannot rank points."""
    xs = [log(p.budget) for p in points]
    ys = [0.5 * log(p.at["0.01"]["free_compute"] / ands) + 0.5 * log(p.at["0.01"]["leak"] / bits) for p in points]
    if len(points) < 3:
        return 0
    (x0, y0), (x1, y1) = (xs[0], ys[0]), (xs[-1], ys[-1])
    return max(range(len(points)), key=lambda i: (y0 + (y1 - y0) * (xs[i] - x0) / (x1 - x0)) - ys[i])


def frontier_report(m: H.Shape = SHAPE) -> dict:
    cands, cm = candidates(m), costs(m)
    s = sweep(cands, cm, m)
    pts = s["points"]
    ands, bits = sum(c.units * c.ands for c in cands), sum(c.units * c.out_bits for c in cands)

    def row(p: Point) -> dict:
        return {"proving": p.budget, "of_block": p.budget / m.block_s, "cost": dict(p.plan.cost),
                "weights": {"work": 0.5 / p.h_work, "bits": 0.5 / p.h_bits}, "H_work": p.h_work, "H_bits": p.h_bits,
                "at": p.at, "drawn": sum(p.plan.sizes.values()), "sizes": dict(p.plan.sizes),
                "statements": dict(p.plan.statement_size), "tile_eps": tile_eps(p.plan.sizes, cands)}

    return {"shape": vars(m) | {"block_s": m.block_s}, "templates": len(cands),
            "units": sum(c.units for c in cands), "ands": ands, "bits": bits, "commit": s["commit"],
            "commit_hash20": m.steps * COMMIT_STEP_S, "cheapest": s["cheapest"],
            "unaffordable": s["skipped"], "complete": s["complete"], "share": BUDGET_SHARE,
            "frontier": [row(p) for p in pts], "knee": knee(pts, ands, bits) if pts else None,
            "ands_against_gpu": ands_against_gpu(m)}


def render(o: dict) -> str:
    b = o["shape"]["block_s"]
    lines = [f"# Demo A's frontier: {o['templates']} subcircuits, {o['units']:.3g} units, {o['ands']:.3g} ANDs and "
             f"{o['bits']:.3g} output bits a block of {b:g} GPU-s", "",
             f"Committing every unit's outputs: {o['commit_hash20']:.4g} GPU-s a block, beside every plan. "
             f"The cheapest plan proves for {o['cheapest']:.4g} GPU-s ({o['cheapest'] / b:.3g} of the block).", "",
             "| proving GPU-s | of the block | H_work(B) | H_bits(B) | free_compute at 1% | leak at 1% (bits) | "
             "free_compute at 1e-6 | leak at 1e-6 | drawn | |", "|---|---|---|---|---|---|---|---|---|---|"]
    for i, f in enumerate(o["frontier"]):
        a, z = f["at"]["0.01"], f["at"]["1e-06"]
        lines.append(f"| {f['proving']:.4g} | {f['of_block']:g} | {f['H_work']:.3g} | {f['H_bits']:.3g} | "
                     f"{a['free_compute']:.3g} | {a['leak']:.3g} | {z['free_compute']:.3g} | {z['leak']:.3g} | "
                     f"{f['drawn']:,} | {'knee' if i == o['knee'] else ''} |")
    g = o["ands_against_gpu"]
    lines += ["", f"ANDs against GPU time: the INT8 linears hold {g['s8']['ands_share']:.4g} of the ANDs and "
              f"{g['s8']['gpu_share']:.4g} of the step's GPU time outside the commit; the rest "
              f"{g['rest']['ands_share']:.3g} and {g['rest']['gpu_share']:.3g}.", "",
              "| Definition | units | own ANDs | share | proved ANDs share |", "|---|---|---|---|---|"]
    for k, d in g["definitions"].items():
        lines.append(f"| {k} | {d['units']:.3g} | {d['ands']:.3g} | {d['ands_share']:.2g} | {d['proved_share']:.2g} |")
    return "\n".join(lines)


def render_record(rows: list[dict], share: float = BUDGET_SHARE) -> str:
    lines = [f"Plan of record: proving within {share:g} of the sampling block's GPU time.", "",
             "| sampling block | blocks | budget GPU-s | cheapest | proving a block | drawn | free_compute at 1% | "
             "leak at 1% | free_compute at 1e-6 | leak at 1e-6 | tile eps at 1% / 1e-6 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "at" not in r:
            lines.append(f"| {r['span']} | {r['blocks']:,} | {r['budget']:.4g} | {r['cheapest']:.4g} | no plan | | | | "
                         "| | |")
            continue
        a, z, e = r["at"]["0.01"], r["at"]["1e-06"], tile_eps_max(r.get("tile_eps"))
        lines.append(f"| {r['span']} | {r['blocks']:,} | {r['budget']:.4g} | {r['cheapest']:.4g} | "
                     f"{r['per_block']:.4g} | {r['drawn']:,} | {a['free_compute']:.3g} ({a['work_share']:.2g}) | "
                     f"{a['leak']:.3g} ({a['bits_share']:.2g}) | {z['free_compute']:.3g} ({z['work_share']:.2g}) | "
                     f"{z['leak']:.3g} ({z['bits_share']:.2g}) | {e.get('0.01', 0):.3g} / {e.get('1e-06', 0):.3g} |")
    for r in rows:
        for t, eps in (r.get("tile_eps") or {}).items():
            lines.append(f"- {r['span']}: k_T = {r['sizes'][t]:,} of {t}, eps {eps['0.01']:.3g} / {eps['1e-06']:.3g}")
    return "\n".join(lines)


def render_m3(rows: list[dict]) -> str:
    lines = ["| M3 draw | units | cost GPU-s | of the block | of its 1% budget | free_compute at 1% | leak at 1% |",
             "|---|---|---|---|---|---|---|"]
    for o in rows:
        name = f"{o['k']} of {o['only']}" if o["only"] else f"uniform {o['k']}{' with tiles' if o['tiles'] else ''}"
        a = o["at"]["0.01"]
        lines.append(f"| {name} | {o['units']:.4g} | {o['cost']:.4g} | {o['of_block']:.3g} | {o['of_budget']:.3g} | "
                     f"{a['free_compute']:.3g} ({a['work_share']:.2g}) | {a['leak']:.3g} ({a['bits_share']:.2g}) |")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--spans", nargs="*", help="only these sampling blocks (`spans`' names)")
    ap.add_argument("--no-frontier", action="store_true")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    draws = [m3(), m3(tiles=True), m3(only=TILE)] + [m3(only=f"{TILE}{{H1=v5,K={k},") for k in sorted(TILE_TRACED)]
    (a.out / "m3.json").write_text(json.dumps(draws, indent=1, default=str))
    print(render_m3(draws), flush=True)
    tag = "-".join(s.replace(" ", "_") for s in a.spans) if a.spans else "all"
    rows = []
    for name, n in spans(SHAPE).items():
        if a.spans and name not in a.spans:
            continue
        rows.append(plan_at(SHAPE, name, n))
        (a.out / f"plan_of_record-{tag}.json").write_text(json.dumps(rows, indent=1, default=str))
        print(json.dumps({k: v for k, v in rows[-1].items() if k not in ("draw", "sizes", "statements")},
                         default=str), flush=True)
    print(render_record(rows), flush=True)
    if not a.no_frontier:
        o = frontier_report() | {"m3": draws}
        (a.out / "frontier.json").write_text(json.dumps(o, indent=1, default=str))
        print(render(o), flush=True)


if __name__ == "__main__":
    main()
