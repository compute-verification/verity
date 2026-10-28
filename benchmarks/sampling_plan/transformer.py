"""The sampling plan for the end-to-end demo's transformer under C-Flock: the demo's decoder block, small (the CPU
reference) or scaled onto one RTX PRO 6000 (the Phase 2 model), every operation a proof block.

The Definition DAG is one server-time block of serving: `requests` requests of `context` positions each (`prompt` given,
the rest generated greedily, the last one never fed back, so `positions` = context - 1 run), through `layers` pre-norm
blocks of width `width` (`heads` heads of 64, an MLP of `mlp` with ReLU), learned absolute positions, a byte vocabulary
and a W8A8 output head. Every op is a unit of one request's (`cuts` puts every subcircuit in the `request` replay domain),
and its units, from the finest up:

- the static matmuls (QKV, the output projection, MLP up and down, the head): `ScaledMmS8Coordinate_v1{K}` on sm_120's
  s8 step, one output coordinate a unit, or a tile of one request's positions by 32 output channels (`stub:S8Tile`, a
  row of activations and a row of weights opened once for 32 coordinates), the weights registered reads
  (1/sqrt(head_dim) folded into W_q's scales, a power of two);
- attention's QK^T and PV: `GemmCoordinate_v2{K}` on sm_120's BF16 step (`GEMM_HOPPER`: `HopperBF16WgmmaDot16_v1`
  carries the parameters `silicon` pins for sm_120's mma.m16n8k16.bf16), one score or output coordinate a unit, or one
  request's head (`stub:QkHead`, `stub:PvHead`); every row padded to `context` keys (masked keys read as zero), so
  every position has one shape, as the GPU loop runs it;
- the residual and position adds: `Bf16Add_v1`, a coordinate or a row (`stub:Bf16AddRow`);
- RMSNorm, the per-row s8 quantizer, softmax, ReLU, the embedding's select and argmax: Definitions not on main
  yet, so stubs here (every id `stub:`), each one's ANDs composed from pinned pieces. A row op is a unit whole
  when its outputs fit a size class, or commits its row statistic as a unit of its own (`shared`: RMSNorm's 1/rms, the
  quantizer's scale, softmax's max and sum) read by coordinate units or chunks (`chunked`: the quantizer at N > 16,384,
  whose whole row of N/2 code pairs and its scale is over `UNIT_CLASS_MAX`), or recomputes it in each reader
  (`recomputed`); ReLU is a coordinate or a chunk of at most 16,384. The planner chooses (`candidates.cuts`).

The harm is faulty ANDs: each unit is credited its ANDs (a recomputed statistic once, in the op it belongs to), so a
plan's certificate means the same whatever the cut, where a count of faulty proof blocks falls as the cut coarsens; the
faulty proof blocks (`IntegrityPeriod`'s W) are reported beside it. The cost is C-Flock's, fitted from its zero-knowledge
sessions on sm_120 (`inputs.observations`, `costfit`), in GPU-seconds of an RTX PRO 6000, with every serve-time
commitment of the block: every unit's outputs, the requests' bytes, the carried KV cache at the block's close, PoUW's
CP-W receipts (h1, assumed) and the bandwidth audit's receipts (`serve`), and the replay of each drawn request
(`Serving.replay`). A plan is a stratified law, k units of each subcircuit; the chosen one is the cheapest certifying at
most `TARGET_SHARE` of the block's ANDs wrong. `recertify` recomputes its certificate from the sizes alone through
core's `Stratified.harm_bound` and `count_bound`, and `pooled` bounds the faulty proof blocks over a monitoring period of many
blocks. `handoff` is the plan as the GPU loop registers it: one `IntegrityPeriod` window a subcircuit, or the block as one
window under the `work:1` law.

    uv run python benchmarks/sampling_plan/transformer.py --out DIR [--model small|large] [--share S] [--delta D]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from math import ceil, log, log2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inputs import I, KEY, observations  # noqa: E402

from verity.core.protocols.pous import wipe_monitoring
from verity.core.protocols.profile import Stratified, Stratum  # noqa: E402
from verity.core.protocols.sampled_proofs.candidates import Shape, cuts, shape_of  # noqa: E402
from verity.core.protocols.sampled_proofs.costfit import Fitted, fit  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage.draw import work_k, work_table  # noqa: E402
from verity.core.protocols.sampled_proofs.optimize import (Candidate, Commitment, CostModel, Objective,  # noqa: E402
                                                           Plan, ReplayDomain, certify, density_bound, for_target,
                                                           frontier, ln_upper, row_ands)
from verity.ml.circuits.codec import descriptor_id  # noqa: E402
from verity.ml.circuits.defs import bind  # noqa: E402
from verity.ml.circuits.ops import fp32 as F, gemm as G, prims as P  # noqa: E402

#: core's reporting deltas (`wipe_monitoring`), as floats
DELTAS = tuple(map(float, wipe_monitoring.DELTAS))
#: the server-time block, assumed: the Phase 2 default, 20 minutes, which a `gpu-lease`'s 30-minute ceiling sets; no
#: wipe has been timed, and the census has no block length
BLOCK_S = 1200
#: the datacenter, assumed: the census has no datacenter size
SERVERS = 10_000
#: core's monitoring periods (`wipe_monitoring`), seconds by name
PERIODS = dict(wipe_monitoring.PERIODS)
#: the target, a share of the block's ANDs certified at most wrong: a share of the work scales with the model where a
#: count of units does not (a coarser cut has fewer units); 1e-3 is the small model's knee (halving it costs about 8%
#: more, each halving past it over 10%), held fixed for every size
TARGET_SHARE = 1e-3
#: F32Rint_v1's ANDs (a Definition not on main), the quantizer's round half to even
F32_RINT = 275
#: the widest ReLU chunk and quantizer code chunk the Phase 2 model plans
CHUNK = 16_384
#: output channels a static-matmul tile spans
TILE_COLS = 32


def pieces() -> dict[str, int]:
    """Every pinned piece's ANDs (circuit-check's pins beside `verity.ml.circuits.ops`)."""
    out: dict[str, int] = {}
    for f in sorted(Path(P.__file__).parent.glob("*.circuit-check.json")):
        out |= {k: v["and"] for k, v in json.loads(f.read_text()).get("pieces", {}).items() if "and" in v}
    return out


@dataclass(frozen=True)
class Model:
    """The demo's block at a size, and one server-time block's serving: `requests` requests of `context` positions, the
    first `prompt` given, `sequences` of them live at once (staggered, so the KV cache's occupancy is constant);
    `inflight` carry their KV cache across the block's close."""

    layers: int = 2
    width: int = 128
    heads: int = 2
    mlp: int = 512
    vocab: int = 256
    context: int = 32
    prompt: int = 8
    requests: int = 4
    sequences: int = 4
    inflight: int = 0

    def __post_init__(self) -> None:
        if self.width % self.heads or self.head_dim % 16 or self.width % TILE_COLS or self.mlp % TILE_COLS \
                or self.vocab % TILE_COLS or self.context % 16 or not 0 < self.prompt < self.context \
                or min(self.layers, self.requests, self.sequences) < 1 or not 0 <= self.inflight <= self.requests \
                or self.mlp % min(self.mlp, CHUNK):
            raise ValueError(f"{self}: needs heads | width, head_dim and context multiples of 16, width, mlp and vocab "
                             f"multiples of {TILE_COLS}, mlp in whole chunks of {CHUNK}, 0 < prompt < context, and "
                             f"0 <= inflight <= requests")

    @property
    def head_dim(self) -> int:
        return self.width // self.heads

    @property
    def positions(self) -> int:
        """Positions a request runs through the stack: the last generated token is never fed back."""
        return self.context - 1

    @property
    def generate(self) -> int:
        return self.context - self.prompt

    @property
    def linear_params(self) -> int:
        return self.layers * (4 * self.width ** 2 + 2 * self.width * self.mlp) + self.vocab * self.width

    @property
    def weight_bytes(self) -> int:
        """INT8 codes and one f32 scale an output channel for every linear, the bf16 token and position embeddings and
        RMSNorm gains."""
        channels = self.layers * (4 * self.width + self.mlp) + self.vocab
        return self.linear_params + 4 * channels + 2 * (self.vocab + self.context) * self.width \
            + 2 * self.width * (2 * self.layers + 1)

    @property
    def kv_bytes(self) -> int:
        """One position's K and V over every layer, bf16."""
        return self.layers * 2 * self.width * 2

    @property
    def kv_resident(self) -> int:
        """The KV cache held while serving: `sequences` staggered over the positions, so on average each holds half
        (the Phase 2 model's 16,368 entries a layer at 1,023 sequences)."""
        return self.sequences * ((self.positions + 1) // 2) * self.kv_bytes


def requests_per_block(sequences: int, step_s: float, positions: int, block_s: float = BLOCK_S) -> int:
    """Requests a block serves when `sequences` decode one token each a step of `step_s`, each request running
    `positions` steps."""
    return round(block_s / step_s * sequences / positions)


def _cap(ands: int) -> int:
    """A statement's units under m = k_log + nbl <= 35 (`int8_llama.cap`)."""
    return min(1 << max(0, 35 - (ceil(log2(max(ands, 2))) + 1)), 1 << 20)


def _counted(s: Shape, memo: dict[str, Shape]) -> Shape:
    """`s` credited its ANDs as work (unless it names its own), and its statement cap from its ANDs and rows."""
    if s.id in memo:
        return memo[s.id]
    out = replace(s, work=s.work or s.ands, cap=_cap(s.ands + sum(row_ands(r) for r in s.rows)),
                  parts=tuple((_counted(c, memo), k) for c, k in s.parts),
                  ways=tuple((label, tuple((_counted(c, memo), k) for c, k in ps)) for label, ps in s.ways))
    memo[s.id] = out
    return out


def _stub(name: str, ands: int, out_bits: int, rows: tuple[int, ...], out_gates: int = 1, work: int = 0,
          **kw) -> Shape:
    return Shape(f"stub:{name}", ands, work, out_bits, out_gates, rows, out_bits, **kw)


def _group(name: str, child: Shape, count: int, rows: tuple[int, ...]) -> Shape:
    """`count` of `child` as one unit opening `rows`, or cut into them."""
    return _stub(name, count * child.ands, count * child.out_bits, rows, count * child.out_gates,
                 parts=((child, count),))


class DefinitionGraph:
    """The block's Definition DAG: `ops` maps each op to its root shape and its instances a block, and `ids` each real
    Definition's shape id to its descriptor id (the circuit record's subcircuit id); a stub has none."""

    def __init__(self, m: Model, pins: dict[str, int] | None = None) -> None:
        self.m, self.pins = m, pins if pins is not None else pieces()
        self.ids: dict[str, str] = {}
        self.memo: dict[str, Shape] = {}
        self.ops = self._ops()

    def template_id(self, c: Candidate) -> str:
        """A candidate's subcircuit in the circuit record: its unit's descriptor id, or the stub's id (the planner's
        subcircuit is `{op}#{cut}:{shape id}`, and a stub's op and shape ids both hold a colon)."""
        sid = c.template[len(c.op) + 1:].split(":", 1)[1]
        return self.ids.get(sid, sid)

    def real(self, fn) -> Shape:
        s = _counted(shape_of(fn, self.pins), self.memo)
        self.ids[s.id] = descriptor_id(fn)
        return s

    def s8(self, k: int) -> Shape:
        return self.real(bind(G.ScaledMmS8Coordinate, K=k, **G.GEMM_SM120_S8))

    def gemm(self, k: int) -> Shape:
        return self.real(bind(G.GemmCoordinate, K=k, **G.GEMM_HOPPER))

    def tile(self, k: int, rows: int) -> Shape:
        """`rows` positions of one request by `TILE_COLS` output channels of the K = k linear: a row of activations
        each position and a row of weights each channel, opened once, their scales and the outputs a row each."""
        c, n = self.s8(k), rows * TILE_COLS
        return _counted(_stub(f"S8Tile{{K={k},{rows}x{TILE_COLS}}}", n * c.ands, 16 * n,
                              (k,) * (rows + TILE_COLS) + (4 * rows, 4 * TILE_COLS, 2 * n), n, parts=((c, n),)),
                        self.memo)

    def _linears(self, k: int, tiles: tuple[tuple[Shape, int], ...]) -> Shape:
        """One request's static matmuls of K = k: never a unit (no size class holds it), only its tiles."""
        return _stub(f"S8Linears{{K={k}}}", sum(t.ands * n for t, n in tiles), 1 << 40, (), parts=tiles)

    def _row_op(self, name: str, n: int, stat: Shape, reader: Shape, fused: Shape, per_row: int,
                rows: tuple[int, ...], chunk: Shape | None = None) -> Shape:
        """A row op of `per_row` readers over a row of `n` bf16, whole, or its statistic committed once (`shared`,
        read by each reader or by chunks of them, `chunked`) or recomputed in each reader (`recomputed`)."""
        ways = [("shared", ((stat, 1), (reader, per_row)))]
        if chunk is not None:
            ways.append(("chunked", ((stat, 1), (chunk, per_row * reader.out_gates // chunk.out_gates))))
        ways.append(("recomputed", ((fused, per_row),)))
        return _counted(Shape(f"stub:{name}{{N={n}}}", stat.ands + per_row * reader.ands, 0,
                              stat.out_bits + per_row * reader.out_bits, 1 + per_row * reader.out_gates,
                              rows, ways=tuple(ways)), self.memo)

    def rmsnorm(self, n: int) -> Shape:
        p = self.pins
        inv = _stub(f"RmsInv{{N={n}}}", n * p["F32Fma_v2"] + p["F32Mul_v2"] + p["F32Add_v2"] + p["RsqrtApprox_v1"], 32,
                    (2 * n, 4))
        coord = _stub("RmsCoord", 2 * p["F32Mul_v2"] + p["F2fpBf16_v1"], 16, (2, 4, 2, 2))
        fused = _stub(f"RmsCoordRecomputed{{N={n}}}", inv.ands + coord.ands, 16, (2 * n, 2, 2), work=coord.ands)
        out = self._row_op("RmsNorm", n, inv, coord, fused, n, (2 * n, 2 * n, 2 * n))
        return replace(out, out_bits=16 * n, out_gates=n, commit_bits=16 * n)

    def quant(self, n: int) -> Shape:
        """The per-row s8 quantizer over `n` bf16: `n`/2 code pairs (16 bits each) and the row's f32 scale."""
        p = self.pins
        scale = _stub(f"QuantScale{{N={n}}}", n * p["F32Max_v1"] + p["F32Div_v2"] + p["SelectF32_v1"], 32, (2 * n, 4))
        pair = _stub(f"QuantPair{{N={n}}}", 2 * (p["F32Div_v2"] + F32_RINT + p["S32ClampS8_v1"]), 16, (4, 4, 2))
        fused = _stub(f"QuantPairRecomputed{{N={n}}}", scale.ands + pair.ands, 16, (2 * n, 2), work=pair.ands)
        chunk = None
        if n > CHUNK:
            chunk = _group(f"QuantChunk{{N={n},C={CHUNK}}}", pair, CHUNK // 2, (2 * CHUNK, 4, CHUNK))
        out = self._row_op("Quant", n, scale, pair, fused, n // 2, (2 * n, n, 4), chunk)
        return replace(out, commit_bits=out.out_bits)

    def softmax_row(self) -> Shape:
        """One query position's softmax over `context` keys, the masked ones -inf so their probabilities are +0."""
        p, L = self.pins, self.m.context
        exp = p["F32Add_v2"] + p["F32Mul_v2"] + p["MufuEx2Ftz_v1"]
        prob = _stub("SoftmaxProb", exp + p["MufuRcpFtz_v1"] + p["F32Mul_v2"] + p["F2fpBf16_v1"], 16, (2, 4, 4, 2))
        mx = _stub(f"SoftmaxMax{{N={L}}}", L * p["F32Max_v1"], 32, (2 * L, 4))
        sm = _stub(f"SoftmaxSum{{N={L}}}", L * (exp + p["F32Add_v2"]), 32, (2 * L, 4, 4))
        fused = _stub(f"SoftmaxProbRecomputed{{N={L}}}", mx.ands + sm.ands + prob.ands, 16, (2 * L, 2), work=prob.ands)
        return _counted(Shape(f"stub:SoftmaxRow{{N={L}}}", mx.ands + sm.ands + L * prob.ands, 0, 16 * L, L,
                              (2 * L, 2 * L), 16 * L,
                              ways=(("shared", ((mx, 1), (sm, 1), (prob, L))), ("recomputed", ((fused, L),)))),
                        self.memo)

    def _ops(self) -> dict[str, tuple[Shape, int]]:
        m, p = self.m, self.pins
        R, T, Ly, Gn, H, d, f, V, C = (m.requests, m.positions, m.layers, m.generate, m.heads, m.width, m.mlp,
                                       m.vocab, m.context)
        hd, mc = m.head_dim, self.memo
        embed = _stub(f"EmbedSelect{{V={V}}}", V * (p["SelectBf16_v1"] + 8), 16, (1, 2 * V, 2))
        relu = _stub("ReluBf16", p["SelectBf16_v1"] + 1, 16, (2, 2))
        argmax = _stub(f"Argmax{{V={V}}}", V * (p["Bf16GtStrict_v1"] + p["SelectBf16_v1"] + 8), 8, (2 * V, 1))
        add = self.real(F.Bf16Add)
        cr = min(f, CHUNK)
        lin_d = self._linears(d, ((self.tile(d, T), Ly * (4 * d + f) // TILE_COLS), (self.tile(d, Gn), V // TILE_COLS))
                              + (((self.tile(f, T), Ly * d // TILE_COLS),) if f == d else ()))
        lin_f = self._linears(f, ((self.tile(f, T), Ly * d // TILE_COLS),))
        ops: dict[str, list] = {}

        def put(shape: Shape, n: int) -> None:
            if n:
                shape = _counted(shape, mc)
                ops.setdefault(shape.id, [shape, 0])[1] += n

        put(_group(f"EmbedRow{{N={d}}}", embed, d, (1, 2 * V * d, 2 * d)), R * T)
        put(self.rmsnorm(d), R * (2 * T * Ly + Gn))
        put(self.quant(d), R * (3 * T * Ly + Gn))
        put(self.quant(f), R * T * Ly)
        put(lin_d, R)
        if f != d:
            put(lin_f, R)
        put(_group(f"QkHead{{T={T}}}", self.gemm(hd), T * (T + 1) // 2, (2 * hd * T, 2 * hd * T, T * (T + 1))),
            R * Ly * H)
        put(_group(f"SoftmaxHead{{T={T}}}", self.softmax_row(), T, (2 * C * T, 2 * C * T)), R * Ly * H)
        put(_group(f"PvHead{{T={T}}}", self.gemm(C), T * hd, (2 * C * T, 2 * C * hd, 2 * T * hd)), R * Ly * H)
        put(_group(f"ReluRow{{N={cr}}}", relu, cr, (2 * cr, 2 * cr)), R * T * Ly * f // cr)
        put(_group(f"Bf16AddRow{{N={d}}}", add, d, (2 * d, 2 * d, 2 * d)), R * T * (2 * Ly + 1))
        put(argmax, R * Gn)
        return {sid: (s, n) for sid, (s, n) in ops.items()}

    @property
    def work(self) -> int:
        """The block's credited work: every op's ANDs (a recomputed statistic credited once)."""
        return sum(s.work * n for s, n in self.ops.values())

    @property
    def units_dense(self) -> int:
        """The block's units when every op is cut at its finest form (one unit a coordinate, statistics shared)."""
        return sum(n for n in self.unit_counts().values())

    def unit_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}

        def walk(s: Shape, n: int) -> None:
            if s.ways and s.ways[0][0] == "shared":
                for c, k in s.ways[0][1]:
                    walk(c, n * k)
            elif s.parts and s.id.startswith("stub:"):
                for c, k in s.parts:
                    walk(c, n * k)
            else:
                out[s.id] = out.get(s.id, 0) + n

        for s, n in self.ops.values():
            walk(s, n)
        return out


@dataclass(frozen=True)
class Serving:
    """The block's serving, for its serve-time commitments and replay: `block_s` seconds, `audit_store` bytes under
    the bandwidth audit with `audit_passes` passes a block, `replay` GPU-s to rebuild one drawn request's witnesses
    (its positions re-run from its committed bytes; the loop is batch-invariant), and `step_s`, the decode step when
    the requests a block are derived from it."""

    block_s: float = BLOCK_S
    audit_store: float = 0.0
    audit_passes: float = 1.0
    replay: float = 0.0
    step_s: float | None = None


def serve(m: Model, s: Serving) -> tuple[Commitment, ...]:
    """Every serve-time commitment of the block beside the units' own outputs (which `Candidate.commit_bits` carries):
    the requests' bytes in, the in-flight requests' KV cache at the close (the carried state; also the QKV units'
    outputs, so a copy `overlaps` would price), PoUW's CP-W receipts under h1 (2 (K/32 + m) u32 a 32 x 32 tile, the
    scales table's; the demo's PoUW scheme on this workload is assumed) and the bandwidth audit's receipts (64 B a 4 KiB
    block of its store a pass)."""
    rate = I["commit_per_byte"].value
    d, f = m.width, m.mlp
    calls = [(d, 3 * d), (d, d), (d, f), (f, d)]
    per_token = sum(cells * 2 * (k // 32 + (2 if k <= 4096 else 3)) * 4 / 1024 for k, cells in calls) * m.layers
    head = m.vocab * 2 * (d // 32 + (2 if d <= 4096 else 3)) * 4 / 1024
    rows = [("requests", "request bytes in", m.requests * m.prompt),
            ("carry", "KV cache of the in-flight requests at the close", m.inflight * m.context * m.kv_bytes),
            ("pouw:h1", "CP-W receipts", m.requests * (m.positions * per_token + m.generate * head)),
            ("bandwidth", "receipts over the audited store", s.audit_passes * s.audit_store / 4096 * 64)]
    return tuple(Commitment(scale, tensor, b, cost=b * rate) for scale, tensor, b in rows if b)


def costs(m: Model, s: Serving) -> CostModel:
    """C-Flock's fitted model (int8_llama's fit) with the commitment rate, the block's serve-time commitments and the
    `request` replay domain."""
    obs = observations()
    fitted = Fitted(fit(obs, "prove", ["one", "ands", "ands*staged", "ands*drawn"]),
                    fit(obs, "verify", ["one", "ands", "ands*staged", "ands*drawn"]),
                    fit(obs, "stage", ["one", "ands", "ands*staged"]), I["cpu"].value, KEY)
    return replace(fitted.model(), commit_per_bit=I["commit_per_byte"].value / 8, serve=serve(m, s),
                   replay={"request": ReplayDomain(m.requests, s.replay)})


def candidates(dag: DefinitionGraph, cm: CostModel, objective: Objective, grid: int = 21) -> tuple[list[Candidate], dict]:
    """`candidates.cuts` on each op, every subcircuit in the `request` replay domain: its alternatives, each its units
    by Definition."""
    out, alts = [], {}
    for op, (root, n) in dag.ops.items():
        cs, a, _ = cuts(op, root, n, cm, objective, grid=grid, replay=("request", dag.m.requests))
        out += cs
        alts[op] = {label: [list(u) for u in units] for label, units in a.items()}
    return out, alts


def plans(m: Model, s: Serving, delta: float = DELTAS[0], *, grid: int = 61, cut_grid: int = 21):
    """The block's DAG, its cost model, objective and candidates, and the frontier of expected cost against certified
    faulty proof blocks (screened by `density_bound`)."""
    dag = DefinitionGraph(m)
    objective = Objective(delta)
    cm = costs(m, s)
    cands, alts = candidates(dag, cm, objective, cut_grid)
    return dag, cm, objective, cands, alts, frontier(cands, cm, objective, grid=grid, exact=False)


def chosen(cands: list[Candidate], cm: CostModel, objective: Objective, front: list[Plan],
           target: float) -> Plan | None:
    """The cheapest plan certifying at most `target` faulty ANDs (`for_target` on the frontier)."""
    return for_target(cands, cm, objective, target, plans=front, exact=False)


def law(plan: Plan, cands: list[Candidate]) -> Stratified:
    """The plan's stratified law: one stratum a subcircuit, k_s of n_s drawn."""
    return Stratified(tuple(Stratum(c.template, c.units, plan.sizes[c.template]) for c in cands
                            if c.template in plan.sizes))


def recertify(plan: Plan, cands: list[Candidate], delta: float, *, exact_count: bool = True) -> dict:
    """The plan's certificate from its sizes alone: faulty ANDs (each unit's credited work) and faulty proof blocks by core's
    `Stratified.harm_bound` (and units by `count_bound`, exact, when `exact_count`), and the optimizer's own `certify`
    and `density_bound` of the ANDs."""
    chosen_ = [c for c in cands if c.template in plan.sizes]
    st = law(plan, cands)
    out = {"delta": delta, "ands": st.harm_bound({c.template: float(c.work) for c in chosen_}, delta),
           "units": st.harm_bound({c.template: 1.0 for c in chosen_}, delta),
           "certify": certify(plan.sizes, cands, Objective(delta), exact=True).harm,
           "density_bound": density_bound(((c.units, plan.sizes[c.template], float(c.work)) for c in chosen_), delta)}
    if exact_count:
        out["count_bound"] = st.count_bound(delta)
    return out


def pooled(plan: Plan, cands: list[Candidate], delta: float, blocks: float, *, ands: bool = False) -> float:
    """Faulty proof blocks (or their ANDs) over `blocks` blocks, each drawn by the plan's law on its own live coins and all
    accepted, except with probability `delta`: a set of x_{b,s} faulty proof blocks escapes with at most
    prod exp(-k_s x_{b,s} / n_s), never once x_{b,s} > n_s - k_s, so the total is the knapsack `density_bound` over
    `blocks` copies of each stratum, which the period's length doesn't move (each stratum's rate k_s / n_s is a
    block's).  This relaxation is the optimizer's, not a Lean guarantee; `IntegrityHorizon` proves the
    uniform case, ln(1/delta) / q (`horizon`)."""
    return density_bound(((c.units * blocks, plan.sizes[c.template] * blocks, float(c.work) if ands else 1.0)
                          for c in cands if c.template in plan.sizes), delta)


def horizon(plan: Plan, cands: list[Candidate], delta: float) -> float:
    """`IntegrityHorizon`'s bound on the faulty proof blocks over any number of windows, each a subcircuit's window of a block
    (`windows`): ln(1/delta) / q at q the least share k_s / n_s any window draws, ln(1/delta) the planner's `ln_upper`."""
    return float(ln_upper(delta) / min(Fraction(plan.sizes[c.template], c.units) for c in cands
                                       if c.template in plan.sizes))


def monitoring(plan: Plan, cands: list[Candidate], s: Serving, *, servers: int = SERVERS) -> list[dict]:
    """Per monitoring period (one server, and `servers` of them), at each delta: the pooled bound on faulty proof blocks and
    their ANDs, and beside it each block's own at delta / blocks, so the period's total error is delta."""
    rows = []
    for fleet, count in (("1 server", 1), (f"{servers:,} servers", servers)):
        for period, seconds in PERIODS.items():
            blocks = count * seconds / s.block_s
            rows.append({"period": period, "fleet": fleet, "blocks": blocks,
                         "pooled": {str(d): pooled(plan, cands, d, blocks) for d in DELTAS},
                         "pooled_ands": {str(d): pooled(plan, cands, d, blocks, ands=True) for d in DELTAS},
                         "per_block": {str(d): pooled(plan, cands, d / blocks, 1) for d in DELTAS}})
    return rows


def handoff(plan: Plan, cands: list[Candidate], dag: DefinitionGraph) -> dict:
    """The plan as a GPU loop registers it (a `circuit_record(..., law, cap)`), each subcircuit
    a Definition's descriptor id (a stub's id until its Definition lands) with its units:

    * `windows`, the form `IntegrityPeriod` reads: one window a subcircuit, its units under `subset:k_s` in
      members of its statement size (the loop's `--law` and `--cap`), and at each delta its `K0`, `IntegrityPeriodLog`'s
      ln(1/delta_w)/q at delta_w = delta over the block's windows and q = k/n (or n - k + 1, which no draw misses), so
      outside delta every window has fewer than its K0 units wrong;
    * `record`, the block as one window: every subcircuit under the `work:1` law whose work table's floors are the plan's
      k, which `draw.derive` expands to exactly those k (`work_k`). It needs one `cap` for every subcircuit, so it carries
      the least statement size; `IntegrityPeriod` covers only a uniform subset, not this law.

    The planner's strata are its ops' subcircuits, so two ops cut to one Definition (attention's QK and PV matmuls both to
    `F2fpBf16_v1`, say) are two of its strata and one of the record's: merged, at the higher of their rates, which only
    raises each one's draw. A merged subcircuit's units are credited one work, as the work law needs."""
    groups: dict[str, list[Candidate]] = {}
    for c in cands:
        if c.template in plan.sizes:
            groups.setdefault(dag.template_id(c), []).append(c)
    templates, table, statement, windows = [], {}, {}, []
    for tid, cs in groups.items():
        if len({c.work for c in cs}) > 1:
            raise ValueError(f"{tid}: one template credited two works")
        n = sum(c.units for c in cs)
        k = min(n, max(ceil(plan.sizes[c.template] * n / c.units) for c in cs))
        templates.append([tid, n])
        table[tid] = {"work": cs[0].work, "floor": k}
        statement[tid] = min(plan.statement_size[c.template] for c in cs)
        windows.append({"templates": [[tid, n]], "law": f"subset:{k}", "cap": statement[tid],
                        "K0": {str(d): min(ceil(log(len(groups) / d) * n / k), n - k + 1) for d in DELTAS}})
    total = sum(v["work"] * n for (_, n), v in zip(templates, table.values()))
    for (tid, n), v in zip(templates, table.values()):
        if (k := work_k(1, v["work"], n, total, v["floor"])) != v["floor"]:
            raise ValueError(f"{tid}: the work:1 law draws {k} of its units, the plan {v['floor']}")
    work_table(table)
    return {"format": "verity/sampling-plan/v1", "windows": windows,
            "record": {"templates": templates, "law": "work:1", "work_table": table, "cap": min(statement.values())},
            "statement_units": statement, "stubs": sorted(t for t, _ in templates if t.startswith("stub:"))}


def summary(plan: Plan, cands: list[Candidate], dag: DefinitionGraph) -> dict:
    rows = [{"template": dag.template_id(c), "op": c.op, "units": c.units, "k": plan.sizes[c.template],
             "statement": plan.statement_size[c.template], "work": c.work,
             "ands": c.ands + sum(row_ands(r) for r in c.rows)} for c in cands if c.template in plan.sizes]
    return {"cost": dict(plan.cost), "faulty": plan.certificate.harm if plan.certificate else None,
            "units": sum(v["units"] for v in rows), "drawn": sum(v["k"] for v in rows),
            "templates": rows, "choice": dict(plan.choice), "floor_only": sorted(plan.floor_only),
            "full": sorted(plan.full)}


def budget(front: list[Plan], gpu_s: float) -> Plan | None:
    """The frontier's most assuring plan costing at most `gpu_s` a block."""
    return min((p for p in front if p.cost["total"] <= gpu_s), key=lambda p: p.certificate.harm, default=None)


SMALL = Model()
#: the Phase 2 model: the demo's block at
#: width 8,192, 70 layers, 1,023 sequences staggered; 52.5 GiB of INT8 linears and 35.0 GiB of KV, 91% of 96 GiB
LARGE_SEQUENCES = 1023
#: its decode step, estimated by their sizing script, not measured: the INT8 GEMMs at 85% of peak (86% of the step)
#: and decode attention's KV read
LARGE_STEP_S = 0.195
#: its replay of a drawn request, estimated: under 10 s for about 1,000 requests batched on one GPU
LARGE_REPLAY_S = 0.01
LARGE = Model(layers=70, width=8192, heads=128, mlp=32768, context=32, prompt=8, sequences=LARGE_SEQUENCES,
              requests=requests_per_block(LARGE_SEQUENCES, LARGE_STEP_S, 31))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", choices=("small", "large"), default="small")
    shape = ("layers", "width", "heads", "mlp", "context", "prompt", "requests", "sequences", "inflight")
    for f in shape:
        ap.add_argument(f"--{f}", type=int)
    ap.add_argument("--step-s", type=float, help="decode step (s): derive the requests a block from it")
    ap.add_argument("--block-s", type=float, default=BLOCK_S)
    ap.add_argument("--replay", type=float, help="GPU-s to replay one drawn request (large: LARGE_REPLAY_S)")
    ap.add_argument("--delta", type=float, default=DELTAS[0])
    ap.add_argument("--share", type=float, default=TARGET_SHARE,
                    help="the share of the block's ANDs the chosen plan certifies at most wrong")
    ap.add_argument("--grid", type=int, default=61)
    args = ap.parse_args(argv)
    large = args.model == "large"
    m = replace(LARGE if large else SMALL, **{f: v for f in shape if (v := getattr(args, f)) is not None})
    step = args.step_s if args.step_s is not None else (LARGE_STEP_S if large else None)
    if step is not None and args.requests is None:
        m = replace(m, requests=requests_per_block(m.sequences, step, m.positions, args.block_s))
    replay = args.replay if args.replay is not None else (LARGE_REPLAY_S if large else 0.0)
    s = Serving(block_s=args.block_s, audit_store=m.weight_bytes, replay=replay, step_s=step)
    dag, cm, objective, cands, alts, front = plans(m, s, args.delta, grid=args.grid)
    target = args.share * dag.work
    plan = chosen(cands, cm, objective, front, target)
    b = budget(front, s.block_s)
    out = {"model": asdict(m), "serving": asdict(s), "delta": args.delta, "target": target, "share": args.share,
           "work": dag.work, "units_dense": dag.units_dense, "parameters": parameters(m, s, args.share),
           "budget": None if b is None else {"cost": b.cost["total"], "faulty": b.certificate.harm},
           "serve": [asdict(c) for c in cm.serve], "serve_cost": cm.serve_cost(),
           "rates": {"statement_fixed": cm.statement_fixed, "statement_per_and": cm.statement_per_and,
                     "staged_per_and": cm.staged_per_and, "per_and": cm.per_and, "commit_per_bit": cm.commit_per_bit},
           "alternatives": alts,
           "frontier": [{"cost": p.cost["total"], "faulty": p.certificate.harm, "drawn": sum(p.sizes.values()),
                         "choice": dict(p.choice), "floor_only": sorted(p.floor_only)} for p in front]}
    if plan is not None:
        out["chosen"] = summary(plan, cands, dag)
        out["recertified"] = [recertify(plan, cands, d, exact_count=not large) for d in DELTAS]
        out["horizon"] = {str(d): horizon(plan, cands, d) for d in DELTAS}
        out["monitoring"] = monitoring(plan, cands, s)
        out["handoff"] = handoff(plan, cands, dag)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "plan.json").write_text(json.dumps(out, indent=1, default=str))
    print(render(out))


def parameters(m: Model, s: Serving, share: float) -> list[list[str]]:
    """Every parameter the plan rests on, with its value and its source or the statement that it is assumed."""
    tok = None if s.step_s is None else m.sequences / s.step_s
    return [
        ["shape", f"L {m.layers}, d {m.width}, {m.heads} heads of {m.head_dim}, MLP {m.mlp}, V {m.vocab}, "
                  f"C {m.context} ({m.positions} run, {m.prompt} prompt)", "the demo's block; large: the Phase 2 model's"],
        ["memory", f"{m.weight_bytes / 2**30:.4g} GiB weights + {m.kv_resident / 2**30:.4g} GiB KV",
         "derived from the shape and `sequences`"],
        ["sequences live", f"{m.sequences:,}", "large: the Phase 2 model's"],
        ["step", "-" if s.step_s is None else f"{s.step_s * 1e3:.0f} ms, {tok:,.0f} tokens/s",
         "large: the Phase 2 model's estimate (a sizing script, not measured)"],
        ["block", f"{s.block_s:.0f} s, {m.requests:,} requests",
         "assumed (`BLOCK_S`)" + ("; requests from the step" if s.step_s is not None else "; requests the demo's")],
        ["datacenter", f"{SERVERS:,} servers", "assumed (`SERVERS`); not in benchmarks/census"],
        ["replay", f"{s.replay:g} GPU-s a drawn request", "large: the Phase 2 model's estimate"],
        ["serve-time commitments", "units' outputs, request bytes, KV carry, PoUW h1 receipts, bandwidth receipts",
         "h1 assumed; commitment rate `inputs.commit_per_byte`"],
        ["cost model", "C-Flock zk sessions of ScaledMmS8Coordinate at K = 4096, 14336 on sm_120",
         "`inputs.observations`; extrapolated past them"],
        ["target", f"{share:g} of the block's ANDs wrong at most",
         "`TARGET_SHARE`: the small model's knee, held fixed" if share == TARGET_SHARE else "given (`--share`)"],
    ]


def render(out: dict) -> str:
    m = out["model"]
    lines = [f"# Demo block, {m['layers']} layers x {m['width']} ({m['heads']} heads, MLP {m['mlp']}), "
             f"{m['requests']:,} requests a block: the sampling plan at delta {out['delta']:g}", "",
             "| parameter | value | source |", "|---|---|---|"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in out["parameters"]]
    lines += ["", f"a block: {out['work']:.4g} ANDs, {out['units_dense']:,} units in the finest form; serve-time "
              f"commitments {out['serve_cost']:.3g} GPU-s", "", "| expected cost (GPU-s) | certified faulty ANDs | "
              "share of the block's | drawn | floor-only templates |", "|---|---|---|---|---|"]
    for f in out["frontier"]:
        lines.append(f"| {f['cost']:.4g} | {f['faulty']:.4g} | {f['faulty'] / out['work']:.3g} | {f['drawn']:,} | "
                     f"{len(f['floor_only'])} |")
    b = out["budget"]
    lines += ["", f"the block's own GPU time, {out['serving']['block_s']:g} GPU-s, buys "
              + ("no plan" if b is None else f"at most {b['faulty'] / out['work']:.3g} of its ANDs wrong ({b['cost']:.4g} "
                 f"GPU-s)")]
    c = out.get("chosen")
    if c is None:
        lines += ["", f"no plan certifies {out['target']:.4g} faulty ANDs"]
        return "\n".join(lines)
    lines += ["", f"chosen (at most {out['share']:g} of the block's ANDs wrong, {out['target']:.4g}): cost "
              f"{c['cost']['total']:.4g} GPU-s a block "
              f"({', '.join(f'{k} {v:.3g}' for k, v in c['cost'].items() if k != 'total')}), {c['drawn']:,} drawn of "
              f"{c['units']:,} units", "", "| template | op | units | k | statement | ANDs a unit | proved, rows opened |",
              "|---|---|---|---|---|---|---|"]
    for v in c["templates"]:
        lines.append(f"| {v['template']} | {v['op'].split('{')[0]} | {v['units']:,} | {v['k']:,} | {v['statement']:,} | "
                     f"{v['work']:,} | {v['ands']:,} |")
    lines += ["", "a block, recertified from the sizes (`Stratified.harm_bound`; `count_bound` exact, small model only):", "",
              "| delta | faulty ANDs | share | faulty proof blocks | count_bound | certify (ANDs) |", "|---|---|---|---|---|---|"]
    for r in out["recertified"]:
        lines.append(f"| {r['delta']:g} | {r['ands']:.4g} | {r['ands'] / out['work']:.3g} | {r['units']:.4g} | "
                     f"{r.get('count_bound', '-')} | {r['certify']:.4g} |")
    h = out["horizon"]
    lines += ["", f"pooled (`density_bound` over the period's blocks; `IntegrityHorizon`'s ln(1/delta)/q_min, "
              f"{h['0.01']:.4g} units at 1% and {h['1e-06']:.4g} at 1e-6, holds over any period):", "",
              "| period | fleet | blocks | units at 1% | at 1e-6 | ANDs at 1% | at 1e-6 | each block, units at 1%/blocks | "
              "at 1e-6/blocks |", "|---|---|---|---|---|---|---|---|---|"]
    for r in out["monitoring"]:
        p, a, b = r["pooled"], r["pooled_ands"], r["per_block"]
        lines.append(f"| {r['period']} | {r['fleet']} | {r['blocks']:,.0f} | {p['0.01']:.4g} | {p['1e-06']:.4g} | "
                     f"{a['0.01']:.4g} | {a['1e-06']:.4g} | {b['0.01']:.4g} | {b['1e-06']:.4g} |")
    lines += ["", "hand-off, `IntegrityPeriod`'s windows (one a template; `compile --law L --cap N`):", "",
              "| template | units | law | cap | K0 at 1% | K0 at 1e-6 |", "|---|---|---|---|---|---|"]
    for w in out["handoff"]["windows"]:
        (t, n), = w["templates"]
        lines.append(f"| {t} | {n:,} | {w['law']} | {w['cap']:,} | {w['K0']['0.01']:,} | {w['K0']['1e-06']:,} |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
