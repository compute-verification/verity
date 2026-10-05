"""Costing the lookup argument alternatives for the frozen target, through the accountant.

The 16-bit range lookups (``T_range16``, 284 of the 306 queries per unit, 65 536 rows) are the largest census line and
set A-GKR's sequential depth (the fractional GKR over a hypercube of dimension ``ceil(log2(N + |T|))``).  This
module costs, at ``B in {1, 64, 4096}``, ``K = 1536``, Goldilocks with the cubic extension (``2^-128``) and the
quadratic extension (``2^-80``):

1. **Baseline** logUp / fractional-sumcheck GKR (``backends/gkr/PROTOCOL.md``): ``account_candidate_a`` /
   ``census_estimate_a`` unchanged; the numbers of PROTOCOL.md 9.4 are reproduced by :func:`cost_a` with the
   ``logup16`` option.
2. **Lasso / Jolt-style decomposable lookups** for ``T_range16`` (Setty-Thaler-Wahby, *Unlocking the lookup
   singularity with Lasso*, ePrint 2023/1216; Arun-Setty-Thaler, *Jolt*, ePrint 2023/1217, the Surge instantiation;
   sparse commitment = Spark: dense commitments to ``dim``, ``E``, ``read_cts`` (``N`` each per chunk) and
   ``final_cts`` (``M^(1/c)`` per chunk), so ``3cN + c M^(1/c)`` committed elements, all through the hash-based
   Ligero tableau, no other PCS assumption).  The memory-checking grand products are proved by Thaler's product-tree
   GKR (arXiv:1304.3812 section 5), whose sequential chain has the same shape as the fractional GKR
   (``accounting._fractional_chain``).  Also **Shout** (Thaler, *Twist and Shout*, ePrint 2025/105): one-hot
   ``ra`` with ``d``-way decomposition, ``d N M^(1/d)`` committed 0/1 elements, sumchecks of ``log N + log M``
   rounds run in parallel -- the low-depth extreme, priced at Ligero's one symbol per committed element.
   The four fixed tables (``ALIGN`` 512, ``POW`` 512, ``LEAD`` 31, ``NORM`` 32 rows, 22 queries per unit) stay logUp
   in every option: a 512-row table needs no decomposition (Lasso's ``c = 1`` *is* logUp-with-counters) and its
   multiplicity vector is 512 elements.
3. **Table-width trade**: range checks in ``W``-bit chunks (``W = 8, 12, 16``; a ``w``-bit check is ``ceil(w/W)``
   queries and ``ceil(w/W) - 1`` linear splits with the extra chunk digits as committed hint columns), with or
   without dedicated small tables ``T_range7/8/9`` for the 7/8/9-bit checks (instead of ``x 2^(16-w)`` into the
   16-bit table).  :func:`range_width_sweep` gives the crossovers for A (hypercube dimension) and B (helper cells).
4. **cq-like multiplicity-free lookups** (Eagen-Fiore-Gabizon, ePrint 2022/1763) need a pairing-based KZG
   commitment to the table and cached quotients: excluded (hash-based only, ``backends/AGENTS.md``).
   **Batched small-table lookups**: one grand product per table (plookup, Gabizon-Williamson ePrint 2020/315:
   sorted concatenation ``s`` of length ``N + |T|`` committed) versus one logUp over all five tables with the
   public table id folded into the row compression (``rows = sum |T|``, ``N = 306 U``).
5. **Depth**: every option through ``rounds_and_depth_a``-style chains (:func:`chain_depth`), with the
   ``c``-variables-per-message and table-sharding proposals of PROTOCOL.md A 5.4 (``DepthOptions``).

Nothing in ``accounting.py`` or ``census.py`` is edited; every per-unit count is read from ``CHECKER_UNIT`` and every
logUp option is a ``CandidateA`` / ``CandidateB`` with a different ``tables`` tuple (and the chunk-column /
split deltas), costed by ``census_estimate_a`` / ``census_estimate_b``.  The non-logUp arguments (Lasso, Shout,
plookup grand products) are costed additively on top of the accountant's checker-plus-fixed-tables figure with the
same constants the accountant uses (``EXT_MULT_COST``, ``_ntt_ops``, 15 F_c multiplications per tree leaf,
``HM_LEAF_BYTES``), so the columns are comparable; they are ESTIMATES from the stated model, not measurements.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from verity_numerical.security import accounting as acc
from verity_numerical.security import census as cen

__all__ = [
    "BATCHES",
    "FIXED_TABLES",
    "OPTIONS_A",
    "OPTIONS_B",
    "POINTS",
    "OperatingPoint",
    "RangePlan",
    "RangeScheme",
    "candidate_a",
    "candidate_b",
    "chain_depth",
    "cost_a",
    "cost_b",
    "killing_table",
    "lasso_extra",
    "main",
    "plookup_extra",
    "range_plan",
    "range_width_sweep",
    "report",
    "shout_extra",
    "tables_for",
]

GL = acc.Field.prime("Goldilocks", acc.GOLDILOCKS)
K = 1536
BATCHES: tuple[int, ...] = (1, 64, 4096)
HASH_QUERIES = 2**60          # the reference adversary of PROTOCOL.md A section 9 (hash term 2^-136)
UNIT = acc.CHECKER_UNIT
FIXED_TABLES: tuple[acc.LookupTable, ...] = tuple(t for t in acc.CENSUS_TABLES if not t.name.startswith("T_range"))
BASELINE_RANGE: acc.LookupTable = next(t for t in acc.CENSUS_TABLES if t.name == f"T_range{cen.RANGE_CHUNK}")
CENSUS_COLUMNS = ("census.witness_elements", "census.committed_elements", "census.bytes_encoded", "census.hash_bytes",
                  "census.field_ops", "census.memory_passes", "census.bytes_moved")
EXTRA_COLUMNS = ("rounds.protocol", "rounds.sequential_depth", "transcript.bytes", "estimate.peak_bytes")
NO_DEPTH_OPTIONS = acc.DepthOptions()          # c = 1 variable per message, no sharding (PROTOCOL.md baseline)
DEPTH_PROFILES: dict[str, acc.DepthOptions] = {
    "c1s0": NO_DEPTH_OPTIONS,
    "c2s5": acc.DepthOptions(variables_per_round=2, table_shard_bits=5),
    "c3s5": acc.DepthOptions(variables_per_round=3, table_shard_bits=5),
}


@dataclass(frozen=True)
class OperatingPoint:
    """Reference configuration of PROTOCOL.md A section 9 for one operating point."""

    name: str
    target_log2: int
    extension: int
    ligero: acc.LigeroParams

    @property
    def challenge_field(self) -> acc.Field:
        return GL.extension(self.extension)


POINTS: dict[str, OperatingPoint] = {
    "2^-128": OperatingPoint("2^-128", -128, 3, acc.LigeroParams(n=2**14, k=2**12, l=3840, t=192)),
    "2^-80": OperatingPoint("2^-80", -80, 2, acc.LigeroParams(n=2**14, k=2**12, l=3840, t=121)),
}


# ---------------------------------------------------------------------------------------------------------------------
# Range-table schemes (the table-width trade)
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RangeScheme:
    """Range checks in ``chunk_bits``-bit chunks; ``dedicated_small``: the widths in ``small_widths`` that are below
    ``chunk_bits`` get their own ``T_range{w}`` (``2^w`` rows) instead of ``x 2^(chunk_bits - w)`` into the chunk table.
    ``chunk_bits = 16`` without dedicated tables is the census convention (``census.lookups_by_table``)."""

    chunk_bits: int = cen.RANGE_CHUNK
    dedicated_small: bool = False
    small_widths: tuple[int, ...] = (7, 8, 9)

    @property
    def label(self) -> str:
        return f"logup{self.chunk_bits}" + ("+small" if self.dedicated_small else "")


@dataclass(frozen=True)
class RangePlan:
    """Per-unit consequences of a :class:`RangeScheme` on the census unit: queries per table, table rows, and the
    deltas against the 16-bit convention (extra chunk digits committed as hint columns, extra linear splits)."""

    scheme: RangeScheme
    queries: dict[str, int]          # table name -> queries per unit (range tables only)
    rows: dict[str, int]             # table name -> rows
    chunk_columns_delta: int         # extra committed digit columns per unit vs the 16-bit convention
    splits_delta: int                # extra linear split identities per unit vs the 16-bit convention

    @property
    def range_queries(self) -> int:
        return sum(self.queries.values())

    def tables(self) -> tuple[acc.LookupTable, ...]:
        return tuple(acc.LookupTable(name, self.rows[name], q, 1) for name, q in self.queries.items())


def _chunks(w: int, W: int) -> int:
    return -(-w // W)


def range_plan(scheme: RangeScheme | None = None, unit: cen.UnitCensus = UNIT) -> RangePlan:
    """Realise the unit's ``range_by_width`` under ``scheme``.  A ``w``-bit check with ``w`` in the dedicated set goes
    to ``T_range{w}`` as one query; otherwise it is ``ceil(w/W)`` queries into ``T_range{W}`` (``w < W``: the
    constant-scaled value, no column; ``w > W``: the chunk digits are ``ceil(w/W) - 1`` extra committed columns, one
    split identity)."""
    scheme = scheme or RangeScheme()
    W = scheme.chunk_bits
    queries: Counter = Counter()
    rows: dict[str, int] = {}
    cols = splits = 0
    for w, n in sorted(unit.range_by_width.items()):
        if scheme.dedicated_small and w in scheme.small_widths and w < W:
            name = f"T_range{w}"
            queries[name] += n
            rows[name] = 2**w
            k = 1
        else:
            name = f"T_range{W}"
            k = _chunks(w, W)
            queries[name] += k * n
            rows[name] = 2**W
        base = _chunks(w, cen.RANGE_CHUNK)
        cols += n * (k - base)
        splits += n * (k - base)
    return RangePlan(scheme, dict(queries), rows, cols, splits)


def tables_for(scheme: RangeScheme | None = None) -> tuple[acc.LookupTable, ...]:
    """The range tables of ``scheme`` followed by the four fixed tables (unchanged)."""
    return range_plan(scheme or RangeScheme()).tables() + FIXED_TABLES


def _merged_table(tables: Sequence[acc.LookupTable]) -> acc.LookupTable:
    """One logUp over all tables: rows = sum |T|, queries = sum q_T, columns = max (the public table id rides in the
    row compression as one more public coefficient; the beta-degree bound is charged at the widest table)."""
    return acc.LookupTable("ALL(selector)", sum(t.rows for t in tables), sum(t.queries_per_unit for t in tables),
                           max(t.columns for t in tables))


# ---------------------------------------------------------------------------------------------------------------------
# Candidates per option (logUp family goes straight through the accountant)
# ---------------------------------------------------------------------------------------------------------------------

def candidate_a(B: int, point: OperatingPoint, tables: Sequence[acc.LookupTable], chunk_columns_delta: int = 0,
                depth: acc.DepthOptions = NO_DEPTH_OPTIONS, K: int = K) -> acc.CandidateA:
    return acc.CandidateA(K=K, B=B, trace_field=GL, challenge_field=point.challenge_field, ligero=point.ligero,
                          tables=tuple(tables), inputs_per_unit=UNIT.columns_committed_a + chunk_columns_delta,
                          depth=depth, hash_queries=HASH_QUERIES)


def candidate_b(B: int, point: OperatingPoint, tables: Sequence[acc.LookupTable], chunk_columns_delta: int = 0,
                splits_delta: int = 0, K: int = K) -> acc.CandidateB:
    return acc.CandidateB(K=K, B=B, trace_field=GL, challenge_field=point.challenge_field, ligero=point.ligero,
                          tables=tuple(tables), witness_per_unit=UNIT.columns_committed_b + chunk_columns_delta,
                          linear_per_unit=UNIT.lin_total + 2 * UNIT.mul + splits_delta, hash_queries=HASH_QUERIES)


# ---------------------------------------------------------------------------------------------------------------------
# Depth chains (rounds_and_depth_a-style) for the non-logUp arguments
# ---------------------------------------------------------------------------------------------------------------------

def chain_depth(n_dim: int, c: int = 1) -> int:
    """Sequential depth of one product/fractional tree GKR of dimension ``n_dim`` after its challenge message
    (``accounting._fractional_chain``: layer ``k`` = ``ceil(k/c)`` sumcheck messages + one 2-to-1 message)."""
    return acc._fractional_chain(n_dim, c)[1]


def chain_slots(n_dim: int) -> int:
    return acc._fractional_chain(n_dim, 1)[0]


def _sharded(N: int, rows: int, s: int) -> tuple[int, int]:
    """(shards, entries per shard) under the sharding rule of ``accounting._shards`` (shard while N/2^s > rows)."""
    while s > 0 and N // 2**s <= rows:
        s -= 1
    shards = 2**s
    return shards, -(-N // shards)


# ---------------------------------------------------------------------------------------------------------------------
# Additive lookup-argument models (replace T_range16's logUp in A-GKR)
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class LookupExtra:
    """What an argument for the ``T_range16`` queries adds on top of the checker + fixed-table figure:
    committed F_t elements, tree leaves (15 F_c mults each), extra sumcheck mults, pads (F_c), sequential chain,
    challenge slots, sumcheck tables in memory (bytes), and how many memory passes."""

    name: str
    committed: int
    tree_leaves: int
    extra_mults_fc: int
    pads: int
    chain: int
    slots: int
    table_bytes: int
    passes: int
    hypercube_dim: int
    note: str = ""


def lasso_extra(N: int, table_bits: int, chunks: int, ext: int, depth: acc.DepthOptions = NO_DEPTH_OPTIONS,
                name: str | None = None) -> LookupExtra:
    """Lasso/Surge for ``N`` lookups into a ``2^table_bits`` range table decomposed into ``chunks`` subtables of
    ``2^(table_bits/chunks)`` rows (range tables decompose exactly: ``T(x) = sum_i 2^(i b) T_b(x_i)``, ``g`` linear).

    Committed (Spark, dense, Ligero): per chunk ``dim_i``, ``E_i``, ``read_cts_i`` (``N`` each) + ``final_cts_i``
    (``2^(b)``): ``3 c N + c 2^b``.  Sumchecks: primary (``log N`` rounds, degree 2 for linear ``g``) and, per chunk,
    four grand products (read/write of size ``N``, init/final of size ``2^b``) by product-tree GKR; all independent
    and run in parallel, so the chain is ``max(ceil(log N / c_vars), chain_depth(log N))``.  Sharding splits ``N``
    (the counters are per shard, ``+ c 2^b`` per extra shard)."""
    if table_bits % chunks:
        raise acc.AccountingError("chunks must divide the table width")
    b = table_bits // chunks
    m = 2**b
    shards, per = _sharded(N, m, depth.table_shard_bits)
    n = max(1, math.ceil(math.log2(per)))
    c = depth.variables_per_round
    committed = 3 * chunks * N + shards * chunks * m
    tree_leaves = shards * chunks * (2 * per + 2 * m)
    primary = shards * (chunks + 2) * per
    pads_per_tree = 3 * (n * (n - 1) // 2) + 4 * n
    pads = shards * (4 * chunks * pads_per_tree + 2 * n)
    chain = max(-(-n // c), chain_depth(n, c))
    slots = shards * (1 + 4 * chunks * chain_slots(n) + n)
    return LookupExtra(name or f"lasso16_c{chunks}", committed, tree_leaves, primary, pads, chain, slots,
                       2 * tree_leaves * ext * 8, 2 * 4 * chunks, n,
                       note=f"3cN + c 2^b committed (c={chunks}, b={b}); 4c grand products of dim {n}; primary {n} rounds")


def shout_extra(N: int, table_bits: int, d: int, ext: int, depth: acc.DepthOptions = NO_DEPTH_OPTIONS) -> LookupExtra:
    """Shout (ePrint 2025/105) read-only lookup: one-hot ``ra_i`` (``i < d``) over ``N x 2^(table_bits/d)``; committed
    ``d N 2^(table_bits/d)`` 0/1 elements (Ligero prices every one as a symbol).  Read-checking sumcheck
    ``log N + table_bits`` rounds (degree ``d + 1``), booleanity and Hamming-weight-one sumchecks in parallel:
    chain ``ceil((log N + table_bits)/c_vars) + 1``.  Sharding is pointless (no tree) and ignored."""
    if table_bits % d:
        raise acc.AccountingError("d must divide the table width")
    m = 2 ** (table_bits // d)
    n = max(1, math.ceil(math.log2(N)))
    rounds = n + table_bits
    c = depth.variables_per_round
    committed = d * N * m
    return LookupExtra(f"shout16_d{d}", committed, 0, (d + 2) * committed, 3 * rounds * (d + 1), -(-rounds // c) + 1,
                       3 * rounds + 1, 2 * committed * ext * 8, 3, n,
                       note=f"d N 2^(16/d) one-hot committed (d={d}); {rounds}-round sumchecks in parallel")


def plookup_extra(N: int, rows: int, ext: int, depth: acc.DepthOptions = NO_DEPTH_OPTIONS) -> LookupExtra:
    """One grand product per table (plookup, ePrint 2020/315): the sorted concatenation ``s`` (``N + |T|``) is
    committed; the two grand products over ``N + |T|`` factors are proved by product-tree GKR (same chain as the
    fractional GKR of the same dimension).  Committed ``+ (N + |T|)`` per (shard of the) table."""
    shards, per = _sharded(N, rows, depth.table_shard_bits)
    L = per + rows
    n = max(1, math.ceil(math.log2(L)))
    c = depth.variables_per_round
    pads = 3 * (n * (n - 1) // 2) + 4 * n
    return LookupExtra("plookup_gp", shards * L, shards * 2 * L, 0, shards * 2 * pads, chain_depth(n, c),
                       shards * (1 + 2 * chain_slots(n)), 2 * shards * 2 * L * ext * 8, 2, n,
                       note="sorted vector s (N+|T|) committed; 2 product trees of dim n per table")


def _add_extra(base: dict[str, float], p: acc.CandidateA, extra: LookupExtra, point: OperatingPoint) -> dict[str, float]:
    """Add a :class:`LookupExtra` to ``census_estimate_a``'s figure for the checker + fixed tables."""
    ext = point.extension
    ec = acc.EXT_MULT_COST[ext]
    n, k, l, t = point.ligero.n, point.ligero.k, point.ligero.l, point.ligero.t
    rows_extra = -(-(extra.committed + extra.pads * ext) // l)
    bytes_extra = rows_extra * n * 8
    d = acc.rounds_and_depth_a(p)
    parallel = max(d["depth.lookup_chain"], d["depth.checker_chain"], extra.chain)
    depth = 1 + parallel + 2 + (1 if p.coin_commitment_bits is not None else 0)
    out = dict(base)
    out["census.witness_elements"] += extra.committed + 4 * extra.tree_leaves * ext
    out["census.committed_elements"] += extra.committed + extra.pads * ext
    out["census.bytes_encoded"] += bytes_extra
    out["census.hash_bytes"] += bytes_extra
    out["census.field_ops"] += (acc._ntt_ops(rows_extra, n, k) + 3 * rows_extra * n * ext
                                + 15 * ec * extra.tree_leaves + ec * extra.extra_mults_fc)
    out["census.memory_passes"] += extra.passes
    out["census.bytes_moved"] += 3 * bytes_extra + extra.committed * 8 + 2 * extra.table_bytes
    out["rounds.protocol"] += extra.slots
    out["rounds.sequential_depth"] = depth
    out["transcript.bytes"] += rows_extra * t * 8 + extra.pads * ext * 8 + extra.slots * ext * 8
    out["estimate.peak_bytes"] += bytes_extra + extra.committed * 8
    for r in acc.RTTS_MS:
        out[f"t.e2e.rtt_{r}ms"] = depth * float(r) / 1000.0
    out["lookup.hypercube_dim"] = extra.hypercube_dim
    out["lookup.chain"] = extra.chain
    out["lookup.committed_extra"] = extra.committed
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------------------------------------------------

OPTIONS_A: tuple[str, ...] = ("logup16", "logup16+small", "logup8", "logup8+small", "logup12", "logup12+small",
                              "logup_merged", "plookup_gp", "lasso16_c2", "lasso16_c4", "shout16_d4", "shout16_d8")
OPTIONS_B: tuple[str, ...] = ("logup16", "logup16+small", "logup8", "logup8+small", "logup12", "logup12+small",
                              "logup_merged", "plookup_gp", "lasso16_c2")

_SCHEMES = {s.label: s for s in (RangeScheme(W, d) for W in (8, 12, 16) for d in (False, True))}


def _range_N(B: int) -> int:
    return BASELINE_RANGE.queries_per_unit * B * (K // 16)


def cost_a(option: str, B: int, point: OperatingPoint = POINTS["2^-128"],
           depth: acc.DepthOptions = NO_DEPTH_OPTIONS) -> dict[str, float]:
    """The contract columns of A-GKR under ``option`` (one of :data:`OPTIONS_A`), plus ``soundness.log2_total``,
    ``soundness.meets``, ``lookup.hypercube_dim`` (of the range-table argument), ``lookup.chain``,
    ``lookup.committed_extra`` (committed elements the lookup argument adds beyond multiplicities)."""
    ext = point.extension
    if option in _SCHEMES or option == "logup_merged":
        plan = range_plan(_SCHEMES.get(option, RangeScheme()))
        tables = plan.tables() + FIXED_TABLES
        if option == "logup_merged":
            tables = (_merged_table(tables),)
        p = candidate_a(B, point, tables, plan.chunk_columns_delta, depth)
        out = acc.census_estimate_a(p)
        bud = acc.account_candidate_a(p)
        big = max(tables, key=lambda t: t.queries_per_unit)
        shards, _, n_dim = acc._shards(p, big)
        out["lookup.hypercube_dim"] = n_dim
        out["lookup.chain"] = acc.rounds_and_depth_a(p)["depth.lookup_chain"]
        out["lookup.committed_extra"] = plan.chunk_columns_delta * p.units + (shards - 1) * big.rows
    else:
        p = candidate_a(B, point, FIXED_TABLES, 0, depth)
        base = acc.census_estimate_a(p)
        bud = acc.account_candidate_a(p)
        N = _range_N(B)
        if option == "plookup_gp":
            extra = plookup_extra(N, BASELINE_RANGE.rows, ext, depth)
            # plookup's Schwartz-Zippel term (degree N+|T| in the product challenge) is of logUp's order: charged alike.
            bud.add("lookup", "plookup product identity", acc.schwartz_zippel_error(N + BASELINE_RANGE.rows, point.challenge_field))
        elif option.startswith("lasso16_c"):
            extra = lasso_extra(N, cen.RANGE_CHUNK, int(option.split("_c")[1]), ext, depth)
            bud.add("lookup", "Lasso memory-checking fingerprint (Schwartz-Zippel, degree N per grand product)",
                    acc.schwartz_zippel_error(4 * extra.committed // 3, point.challenge_field))
        elif option.startswith("shout16_d"):
            extra = shout_extra(N, cen.RANGE_CHUNK, int(option.split("_d")[1]), ext, depth)
        else:
            raise acc.AccountingError(f"unknown option {option!r}")
        out = _add_extra(base, p, extra, point)
    out["soundness.log2_total"] = acc.log2(bud.total())
    out["soundness.meets"] = float(bud.meets(point.target_log2))
    return out


def cost_b(option: str, B: int, point: OperatingPoint = POINTS["2^-128"]) -> dict[str, float]:
    """B-Ligero (``helper_columns`` mode, depth 3) under ``option`` (one of :data:`OPTIONS_B`).  The non-logUp
    arguments have no sumcheck in B: their grand products become committed running-product columns (``L`` F_c
    elements per product of ``L`` factors), priced like helpers."""
    ext = point.extension
    if option in _SCHEMES or option == "logup_merged":
        plan = range_plan(_SCHEMES.get(option, RangeScheme()))
        tables = plan.tables() + FIXED_TABLES
        if option == "logup_merged":
            tables = (_merged_table(tables),)
        p = candidate_b(B, point, tables, plan.chunk_columns_delta, plan.splits_delta)
        out = acc.census_estimate_b(p)
        bud = acc.account_candidate_b(p)
        out["lookup.helper_cells"] = sum(t.queries_per_unit * p.units + t.rows for t in tables) * ext
        out["lookup.committed_extra"] = plan.chunk_columns_delta * p.units
    else:
        p = candidate_b(B, point, FIXED_TABLES)
        out = acc.census_estimate_b(p)
        bud = acc.account_candidate_b(p)
        N = _range_N(B)
        rows = BASELINE_RANGE.rows
        if option == "plookup_gp":
            L = N + rows
            committed = L + 2 * L * ext          # sorted s + two running products in F_c
            quadratic = 2 * L
        elif option == "lasso16_c2":
            m = 2**8
            committed = 3 * 2 * N + 2 * m + 2 * (2 * (N + m)) * ext   # Spark vectors + 4 running products per chunk
            quadratic = 2 * 4 * (N + m)
        else:
            raise acc.AccountingError(f"option {option!r} needs a sumcheck; Candidate B has none")
        n, k, l, t = point.ligero.n, point.ligero.k, point.ligero.l, point.ligero.t
        rows_extra = -(-committed // l)
        bytes_extra = rows_extra * n * 8
        out["census.witness_elements"] += committed
        out["census.committed_elements"] += committed
        out["census.bytes_encoded"] += bytes_extra
        out["census.hash_bytes"] += bytes_extra
        out["census.field_ops"] += acc._ntt_ops(rows_extra, n, k) + 3 * rows_extra * n * ext + 3 * acc.EXT_MULT_COST[ext] * quadratic
        out["census.bytes_moved"] += 3 * bytes_extra + committed * 8
        out["transcript.bytes"] += rows_extra * t * 8
        out["estimate.peak_bytes"] += bytes_extra + committed * 8
        out["lookup.helper_cells"] = committed
        out["lookup.committed_extra"] = committed
    out["soundness.log2_total"] = acc.log2(bud.total())
    out["soundness.meets"] = float(bud.meets(point.target_log2))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Table-width crossover
# ---------------------------------------------------------------------------------------------------------------------

def range_width_sweep(B: int, widths: Sequence[int] = tuple(range(4, 25)), ext: int = 3,
                      dedicated_small: bool = False) -> list[dict[str, int]]:
    """For each chunk width ``W``: queries per unit, table rows, A's hypercube dimension and fractional chain (``c=1``,
    no sharding), A's committed delta vs the 16-bit convention (chunk digit columns x U + multiplicity rows delta),
    and B's lookup cells (helpers ``(N + |T|) ext`` + chunk columns) with its delta vs 16-bit."""
    U = B * (K // 16)
    rows16 = range_plan(RangeScheme(16, dedicated_small))
    base_cells = sum((q * U + rows16.rows[t]) * ext for t, q in rows16.queries.items()) + rows16.chunk_columns_delta * U
    base_rows = sum(rows16.rows.values())
    out = []
    for W in widths:
        plan = range_plan(RangeScheme(W, dedicated_small))
        big = f"T_range{W}"
        N = plan.queries[big] * U
        n_dim = max(1, math.ceil(math.log2(N + 2**W)))
        cells = sum((q * U + plan.rows[t]) * ext for t, q in plan.queries.items()) + plan.chunk_columns_delta * U
        out.append({
            "W": W, "queries_per_unit": plan.range_queries, "chunk_table_queries_per_unit": plan.queries[big],
            "rows": sum(plan.rows.values()), "chunk_columns_delta": plan.chunk_columns_delta,
            "A.hypercube_dim": n_dim, "A.chain": chain_depth(n_dim),
            "A.committed_delta": plan.chunk_columns_delta * U + sum(plan.rows.values()) - base_rows,
            "B.lookup_cells": cells, "B.cells_delta": cells - base_cells,
        })
    return out


def crossover_batch(W_small: int = 8, W_ref: int = 16, ext: int = 3) -> float:
    """Batch size at which B's lookup cells under ``W_small``-bit chunks equal those under ``W_ref``: solves
    ``U (q_small - q_ref) ext + U cols_small = (2^W_ref - 2^W_small) ext`` for ``U`` and returns ``B = U / (K/16)``."""
    ps, pr = range_plan(RangeScheme(W_small)), range_plan(RangeScheme(W_ref))
    per_unit = (ps.range_queries - pr.range_queries) * ext + (ps.chunk_columns_delta - pr.chunk_columns_delta)
    if per_unit <= 0:
        return math.inf
    U = (2**W_ref - 2**W_small) * ext / per_unit
    return U / (K // 16)


# ---------------------------------------------------------------------------------------------------------------------
# Tables and report
# ---------------------------------------------------------------------------------------------------------------------

def killing_table(point: OperatingPoint = POINTS["2^-128"], depth_profile: str = "c1s0") -> dict:
    """Per option and batch: A's committed elements and sequential depth, B's committed elements, and whether the
    option beats the logUp baseline on A committed AND A depth, and on B committed."""
    depth = DEPTH_PROFILES[depth_profile]
    rows = {}
    for B in BATCHES:
        base_a = cost_a("logup16", B, point, depth)
        base_b = cost_b("logup16", B, point)
        rows[str(B)] = {}
        for opt in OPTIONS_A:
            a = cost_a(opt, B, point, depth)
            b = cost_b(opt, B, point) if opt in OPTIONS_B else None
            rows[str(B)][opt] = {
                "A.committed": a["census.committed_elements"], "A.depth": a["rounds.sequential_depth"],
                "A.field_ops": a["census.field_ops"], "A.transcript": a["transcript.bytes"],
                "A.hypercube_dim": a["lookup.hypercube_dim"], "A.meets": bool(a["soundness.meets"]),
                "B.committed": None if b is None else b["census.committed_elements"],
                "B.meets": None if b is None else bool(b["soundness.meets"]),
                "beats_A_both": bool(a["census.committed_elements"] < base_a["census.committed_elements"]
                                     and a["rounds.sequential_depth"] < base_a["rounds.sequential_depth"]),
                # depth strictly better at a committed-element cost within 0.1 % (multiplicities / pads only)
                "beats_A_depth_within_0.1pct_committed": bool(
                    a["census.committed_elements"] <= 1.001 * base_a["census.committed_elements"]
                    and a["rounds.sequential_depth"] < base_a["rounds.sequential_depth"]),
                "beats_B_committed": None if b is None else bool(b["census.committed_elements"] < base_b["census.committed_elements"]),
            }
    return rows


CITATIONS = {
    "Hab22": "Haboeck, Multivariate lookups based on logarithmic derivatives, ePrint 2022/1530 (rev. 2025-02-20): logUp, Lemma 5, Theorem 4, 3.4",
    "PH23": "Papini-Haboeck, Improving logarithmic derivative lookups using GKR, ePrint 2023/1284 (rev. 2025-02-20): fractional-sum GKR, Prop. 1-2",
    "STW23": "Setty-Thaler-Wahby, Unlocking the lookup singularity with Lasso, ePrint 2023/1216: decomposable tables, Spark/Surge, 3cN + c M^(1/c) committed",
    "AST23": "Arun-Setty-Thaler, Jolt: SNARKs for virtual machines via lookups, ePrint 2023/1217: Surge instantiation, offline memory checking",
    "Tha13": "Thaler, Time-optimal interactive proofs for circuit evaluation, arXiv:1304.3812: product-tree GKR (grand products), data-parallel GKR",
    "Tha25": "Thaler, Twist and Shout: faster memory checking arguments via one-hot addressing, ePrint 2025/105: Shout, d N M^(1/d) one-hot commitments",
    "GW20": "Gabizon-Williamson, plookup, ePrint 2020/315: sorted-concatenation grand-product lookup",
    "EFG22": "Eagen-Fiore-Gabizon, cq: cached quotients for fast lookups, ePrint 2022/1763: pairing-based, excluded",
    "libzk": "draft-google-cfrg-libzk-02 section 6.3: layered quad circuits (checker shape)",
}

ASSUMPTIONS = [
    ("Every committed element costs one Goldilocks symbol in the Ligero tableau (rows = ceil(elements/l), n symbols per row):"
     "a hash-based PCS has no cheap small-value or 0/1 commitments, which is what Lasso/Shout's savings rest on."),
    ("Grand products (Lasso memory checking, plookup) are proved by Thaler's product-tree GKR; its sequential chain has the same"
     "layer structure as the fractional GKR (k-variable degree-3 sumcheck + one 2-to-1 message per layer)."),
    ("Extra chunk digits of a range check split into more chunks than the 16-bit convention are committed hint columns (A) /"
     "slots (B); the 16-bit baseline's own chunk digits are inside the census's 321 committed columns (delta accounting)."),
    ("Lasso's primary sumcheck has degree 2 (range decomposition g is linear); its 4c grand products, the primary sumcheck and"
     "the fixed tables' fractional GKRs run in parallel; sharding splits N and duplicates the counters."),
    ("Shout: read-checking (log N + 16 rounds), booleanity and Hamming-weight-one sumchecks in parallel; no tree, so sharding"
     "does not apply; field ops (d+2) per committed one-hot element."),
    ("One logUp over all tables folds the public table id into the beta-compression of the row (no committed selector: the id"
     "is fixed per query position by the circuit); the beta-degree term is charged at the widest table (4 columns)."),
    ("The checker's layer shape (DEFAULT_LAYERS_A) is left unchanged under every option: the extra split identities are"
     "< 1% of the assertion layer width and do not change its log2 size."),
    ("Non-logUp options add their argument additively to the accountant's checker + fixed-tables figure with the accountant's"
     "own constants (EXT_MULT_COST, _ntt_ops, 15 F_c mults per tree leaf); their soundness terms are Schwartz-Zippel of the"
     "product degree over F_c (same order as the logUp identity)."),
    ("Candidate B has no sumcheck: a grand product there is a committed running-product column (L F_c elements), so Lasso and"
     "plookup in B are priced as committed cells; Shout in B is not applicable."),
]


def _fmt(v) -> str:
    if isinstance(v, bool) or v is None:
        return str(v)
    if isinstance(v, float) and not v.is_integer():
        return f"{v:.1f}"
    v = int(v)
    for unit, div in (("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(v) >= div:
            return f"{v / div:.2f}{unit}"
    return str(v)


def option_table(point: OperatingPoint, depth_profile: str = "c1s0") -> dict:
    depth = DEPTH_PROFILES[depth_profile]
    out = {"A": {}, "B": {}}
    for B in BATCHES:
        out["A"][str(B)] = {opt: cost_a(opt, B, point, depth) for opt in OPTIONS_A}
        out["B"][str(B)] = {opt: cost_b(opt, B, point) for opt in OPTIONS_B}
    return out


def report() -> dict:
    """Everything ``reports/lookups.json`` holds: option tables per operating point and depth profile, the killing
    table, the width sweep with crossovers, citations and assumptions."""
    plans = {s.label: {"queries": range_plan(s).queries, "rows": range_plan(s).rows,
                       "chunk_columns_delta": range_plan(s).chunk_columns_delta,
                       "splits_delta": range_plan(s).splits_delta} for s in _SCHEMES.values()}
    return {
        "K": K, "batches": list(BATCHES),
        "points": {k: {"target_log2": v.target_log2, "extension": v.extension, "ligero_t": v.ligero.t} for k, v in POINTS.items()},
        "depth_profiles": {k: {"variables_per_round": v.variables_per_round, "table_shard_bits": v.table_shard_bits}
                           for k, v in DEPTH_PROFILES.items()},
        "range_plans": plans,
        "options": {pt: {dp: option_table(point, dp) for dp in DEPTH_PROFILES} for pt, point in POINTS.items()},
        "killing": {pt: {dp: killing_table(point, dp) for dp in DEPTH_PROFILES} for pt, point in POINTS.items()},
        "width_sweep": {str(B): range_width_sweep(B) for B in BATCHES},
        "width_sweep_dedicated_small": {str(B): range_width_sweep(B, dedicated_small=True) for B in BATCHES},
        "crossover_batch_B_8_vs_16": crossover_batch(8, 16),
        "crossover_batch_B_12_vs_16": crossover_batch(12, 16),
        "citations": CITATIONS, "assumptions": ASSUMPTIONS,
        "summary": SUMMARY,
    }


SUMMARY = [
    ("Killing question: no option beats logUp on both committed elements and depth for Candidate A at any B, and none beats "
    "logUp-16 on committed cells for Candidate B at B >= 64.  LogUp with fractional GKR is committed-optimal (multiplicities "
    "only); every argument that avoids the quadratic-depth tree (Shout) or changes the tree (Lasso, plookup) commits O(N) "
    "or O(N M^(1/d)) extra elements, x4.7 (Lasso c=2) to x40 (Shout d=4) at B=4096."),
    ("Depth lever inside logUp: dedicated T_range7/8/9 tables move 128 of the 284 range queries per unit off T_range16 "
    "(N: 111.7M -> 61.3M, hypercube 27 -> 26): depth 381 -> 354 at B=4096 for +9.2k committed elements (+0.005%); "
    "with the c=2 / s=5 proposals both sit on the checker-chain floor (158), so the lever is worth 27 trips only if the "
    "proposals are not adopted."),
    ("Table width: 16-bit chunks are optimal for B at B >= 4 (every 16 < w <= 32 check is two chunks whatever W in 16..22; "
    "narrower chunks add 127-175 queries and 127-175 digit columns per unit).  At B = 1 (U = 96) the 65 536 multiplicity "
    "rows dominate: 8/12-bit chunks halve A's and B's committed elements (116k -> 67k; 457k -> 260k) at equal depth "
    "(checker chain 225 is the floor); crossover B ~ 3 (8-bit) / ~ 4 (12-bit)."),
    ("Fixed tables (ALIGN/POW/LEAD/NORM, <= 512 rows, 22 queries per unit) need no Lasso: their logUp trees have dimension "
    "20-23 at B=4096 (chains 210-276, under the checker chain 297); their multiplicity vectors total 1 087 elements."),
    ("One logUp over all tables (public table id in the row compression) is depth-neutral (hypercube 27 either way) and saves "
    "only the four extra trees' pads (-8.4k committed, -0.1 MB transcript at B=4096) at the price of a single challenge "
    "z for all tables; one grand product per table (plookup) costs +N+|T| committed per table (+112M at B=4096) at equal depth."),
    "cq-like multiplicity-free lookups are excluded: they need a pairing-based KZG commitment to the table (hash-based only).",
]


def _print_option_table(cand: str, point: OperatingPoint, depth_profile: str) -> None:
    cols = ("census.committed_elements", "census.witness_elements", "census.bytes_encoded", "census.field_ops",
            "census.memory_passes", "census.bytes_moved", "rounds.sequential_depth", "rounds.protocol",
            "transcript.bytes", "estimate.peak_bytes", "soundness.log2_total")
    short = ("committed", "witness", "encoded", "field_ops", "passes", "moved", "depth", "slots", "transcript", "peak", "log2err")
    print(f"\n== Candidate {cand}, {point.name}, depth profile {depth_profile} ==")
    for B in BATCHES:
        print(f"-- B={B}")
        print(f"{'option':16}" + "".join(f"{s:>11}" for s in short))
        for opt in (OPTIONS_A if cand == "A" else OPTIONS_B):
            r = cost_a(opt, B, point, DEPTH_PROFILES[depth_profile]) if cand == "A" else cost_b(opt, B, point)
            print(f"{opt:16}" + "".join(f"{_fmt(r[c]):>11}" for c in cols))


def main(argv: Sequence[str] | None = None) -> None:
    import sys

    args = list(argv if argv is not None else sys.argv[1:])
    if args and args[0] == "--json":
        print(json.dumps(report(), indent=1, default=str))
        return
    for pt in POINTS.values():
        for dp in ("c1s0", "c2s5"):
            _print_option_table("A", pt, dp)
        _print_option_table("B", pt, "c1s0")
    print("\n== Table-width sweep (2^-128, B=4096 / B=1) ==")
    for B in (4096, 1):
        print(f"-- B={B}")
        print(f"{'W':>3} {'q/unit':>7} {'rows':>8} {'A.dim':>6} {'A.chain':>8} {'A.dcommit':>10} {'B.cells':>10} {'B.delta':>10}")
        for r in range_width_sweep(B):
            print(f"{r['W']:>3} {r['queries_per_unit']:>7} {r['rows']:>8} {r['A.hypercube_dim']:>6} {r['A.chain']:>8} "
                  f"{_fmt(r['A.committed_delta']):>10} {_fmt(r['B.lookup_cells']):>10} {_fmt(r['B.cells_delta']):>10}")
    print(f"\nB crossover 8-bit vs 16-bit chunks: B = {crossover_batch(8, 16):.2f}; 12 vs 16: B = {crossover_batch(12, 16):.2f}")
    print("\n== Killing table (2^-128, c1s0) ==")
    for B, rows in killing_table().items():
        print(f"-- B={B}")
        print(f"{'option':16}{'A.commit':>11}{'A.depth':>9}{'B.commit':>11}{'A both':>8}{'B commit':>9}")
        for opt, r in rows.items():
            print(f"{opt:16}{_fmt(r['A.committed']):>11}{r['A.depth']:>9}{_fmt(r['B.committed']):>11}"
                  f"{r['beats_A_both']!s:>8}{r['beats_B_committed']!s:>9}")


if __name__ == "__main__":       # pragma: no cover
    main()
