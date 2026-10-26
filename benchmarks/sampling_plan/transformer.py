"""The sampling plan for the end-to-end demo's transformer under C-Flock (PRF-166): CMP-92's decoder block, small (the
CPU reference) or scaled up (the large random model on one RTX PRO 6000), every operation a proof unit.

The Definition DAG is one server-time block of serving: `requests` requests of `context` positions each (`prompt` given,
the rest generated greedily), through `layers` pre-norm blocks of width `width` (`heads` heads, an MLP of `mlp`), a byte
vocabulary and a W8A8 output head. Its ops, and what each is a unit of:

- the static matmuls (QKV, the output projection, MLP up and down, the head): `ScaledMmS8Coordinate_v1{K}` on sm_120's
  s8 step, one output coordinate a unit, the weights registered reads (1/sqrt(head_dim) folded into W_q's scales, a
  power of two);
- attention's QK^T and PV: `GemmCoordinate_v2{K}` (BF16, fp32 accumulation), one score or one output coordinate a unit,
  PV's K the key count padded to a multiple of 16 (masked keys read as zero), its step `AmpereBF16TcDot16_v2`: no sm_120
  BF16 step is registered, and Ampere's mma.sync m16n8k16 is the instruction sm_120 runs (a stand-in until kernels fixes
  the attention's accumulation order);
- the residual adds: `Bf16Add_v1`, one coordinate a unit;
- RMSNorm, the per-row s8 quantizer, softmax, ReLU, the embedding's select and argmax: CIR-100's Definitions, not on
  main yet, so stubs here (every id `stub:`), each one's ANDs composed from pinned pieces. A row op either commits its
  row statistic as a unit of its own (`shared`: RMSNorm's 1/rms, the quantizer's scale, softmax's max and sum) or
  recomputes it inside each reader (`recomputed`), and the planner chooses (`candidates.cuts`).

The harm is the integrity leg's: faulty proof units (`IntegrityPeriod`'s W), each unit counting one whatever its size.
The cost is C-Flock's, fitted from its zero-knowledge sessions on sm_120 (`inputs.observations`, `costfit`), in
GPU-seconds of an RTX PRO 6000, with every serve-time commitment of the block: every unit's outputs, the requests'
bytes, the carried KV cache at the block's close, PoUW's CP-W receipts (h1, assumed) and the bandwidth audit's receipts
(`serve`). A plan is a stratified law, k units of each template; `recertify` recomputes its certificate from the sizes
alone through core's `Stratified.harm_bound` and `count_bound`, and `pooled` bounds the faulty units over a monitoring
period of many blocks. `handoff` is the plan as the circuit record's templates and the `work:1` law the window
registers.

    uv run python benchmarks/sampling_plan/transformer.py --out DIR [--model small|large] [--target W] [--delta D]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from math import ceil, log2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inputs import I, KEY, observations  # noqa: E402

from verity.core.protocols.profile import Stratified, Stratum  # noqa: E402
from verity.core.protocols.sampled_proofs.candidates import Shape, cuts, shape_of  # noqa: E402
from verity.core.protocols.sampled_proofs.costfit import Fitted, fit  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage.draw import work_k, work_table  # noqa: E402
from verity.core.protocols.sampled_proofs.optimize import (Candidate, Commitment, CostModel, Objective,  # noqa: E402
                                                           Plan, TemplateCost, certify, density_bound, for_target,
                                                           frontier, row_ands, work_law)
from verity.ml.circuits.codec import descriptor_id  # noqa: E402
from verity.ml.circuits.defs import bind  # noqa: E402
from verity.ml.circuits.ops import fp32 as F, gemm as G, prims as P  # noqa: E402

DELTAS = (1e-2, 1e-6)
#: an assumed server-time block (no wipe has been timed; #4358's table took the same) and an assumed datacenter: neither
#: is in benchmarks/census
BLOCK_S = 600
SERVERS = 10_000
PERIODS = {"week": 7 * 86400, "month": 30 * 86400, "year": 365 * 86400}
#: the chosen plan's default target, a share of the block's units: a choice, at the small frontier's knee (from 1% of
#: the units to 0.1% costs 8% more, the next halving 9% more)
TARGET = 1e-3
#: F32Rint_v1's ANDs from #2487 (not on main), the quantizer's round half to even
F32_RINT = 275


def pieces() -> dict[str, int]:
    """Every pinned piece's ANDs (circuit-check's pins beside `verity.ml.circuits.ops`)."""
    out: dict[str, int] = {}
    for f in sorted(Path(P.__file__).parent.glob("*.circuit-check.json")):
        out |= {k: v["and"] for k, v in json.loads(f.read_text()).get("pieces", {}).items() if "and" in v}
    return out


@dataclass(frozen=True)
class Model:
    """CMP-92's block at a size, and one server-time block's serving: `requests` requests of `context` positions, the
    first `prompt` given; `inflight` of them carry their KV cache across the block's close."""

    layers: int = 2
    width: int = 128
    heads: int = 2
    mlp: int = 512
    vocab: int = 256
    context: int = 32
    prompt: int = 8
    requests: int = 4
    inflight: int = 0

    def __post_init__(self) -> None:
        if self.width % self.heads or self.head_dim % 16 or self.width % 32 or self.mlp % 32 \
                or not 0 < self.prompt < self.context or min(self.layers, self.requests, self.vocab) < 1 \
                or not 0 <= self.inflight <= self.requests:
            raise ValueError(f"{self}: needs heads | width, head_dim and the K's multiples of 16 and 32 (the steps), "
                             f"0 < prompt < context, and 0 <= inflight <= requests")

    @property
    def head_dim(self) -> int:
        return self.width // self.heads

    @property
    def generate(self) -> int:
        return self.context - self.prompt

    @property
    def linear_params(self) -> int:
        return self.layers * (4 * self.width ** 2 + 2 * self.width * self.mlp) + self.vocab * self.width

    @property
    def weight_bytes(self) -> int:
        """INT8 codes and one f32 scale an output channel for every linear, the bf16 embedding and RMSNorm gains."""
        channels = self.layers * (4 * self.width + self.mlp) + self.vocab
        return self.linear_params + 4 * channels + 2 * self.vocab * self.width + 2 * self.width * (2 * self.layers + 1)

    @property
    def kv_bytes(self) -> int:
        """One position's K and V over every layer, bf16."""
        return self.layers * 2 * self.width * 2

    @property
    def macs_per_token(self) -> int:
        """Static MACs of one position through the stack (the head at every position)."""
        return self.linear_params


def _cap(ands: int) -> int:
    """A statement's units under m = k_log + nbl <= 35 (`int8_llama.cap`)."""
    return min(1 << max(0, 35 - (ceil(log2(max(ands, 2))) + 1)), 1 << 20)


def _counted(s: Shape, memo: dict[str, Shape]) -> Shape:
    """`s` with every node one unit of harm and its statement cap from its ANDs and rows."""
    if s.id in memo:
        return memo[s.id]
    out = replace(s, work=1, cap=_cap(s.ands + sum(row_ands(r) for r in s.rows)),
                  parts=tuple((_counted(c, memo), k) for c, k in s.parts),
                  ways=tuple((label, tuple((_counted(c, memo), k) for c, k in ps)) for label, ps in s.ways))
    memo[s.id] = out
    return out


def _stub(name: str, ands: int, out_bits: int, rows: tuple[int, ...], **kw) -> Shape:
    return Shape(f"stub:{name}", ands, 1, out_bits, 1, rows, out_bits, **kw)


class Dag:
    """The block's Definition DAG: `ops` maps each op to its root shape and its instances a block, and `ids` each real
    Definition's shape id to its descriptor id (the circuit record's template id); a stub has none."""

    def __init__(self, m: Model, pins: dict[str, int] | None = None) -> None:
        self.m, self.pins = m, pins if pins is not None else pieces()
        self.ids: dict[str, str] = {}
        self.memo: dict[str, Shape] = {}
        self.ops = self._ops()

    def real(self, fn) -> Shape:
        s = _counted(shape_of(fn, self.pins), self.memo)
        self.ids[s.id] = descriptor_id(fn)
        return s

    def template_id(self, c: Candidate) -> str:
        """A candidate's template in the circuit record: its unit's descriptor id, or the stub's id (the planner's
        template is `{op}#{cut}:{shape id}`, and a stub's op and shape ids both hold a colon)."""
        sid = c.template[len(c.op) + 1:].split(":", 1)[1]
        return self.ids.get(sid, sid)

    def s8(self, k: int) -> Shape:
        return self.real(bind(G.ScaledMmS8Coordinate, K=k, **G.GEMM_SM120_S8))

    def gemm(self, k: int) -> Shape:
        return self.real(bind(G.GemmCoordinate, K=k, **G.GEMM_AMPERE))

    def _row_op(self, name: str, n: int, stat: Shape, reader: Shape, fused: Shape, per_row: int) -> Shape:
        """A row op of `per_row` 16-bit outputs over a row of `n` bf16: its statistic committed once (`shared`) or
        recomputed in each reader (`recomputed`)."""
        return _counted(Shape(f"stub:{name}{{N={n}}}", stat.ands + per_row * reader.ands, 1, 16 * per_row, per_row,
                              ways=(("shared", ((stat, 1), (reader, per_row))), ("recomputed", ((fused, per_row),)))),
                        self.memo)

    def rmsnorm(self, n: int) -> Shape:
        p = self.pins
        inv = _stub(f"RmsInv{{N={n}}}", n * p["F32Fma_v2"] + p["F32Mul_v2"] + p["F32Add_v2"] + p["RsqrtApprox_v1"], 32,
                    (2 * n, 4))
        coord = _stub("RmsCoord", 2 * p["F32Mul_v2"] + p["F2fpBf16_v1"], 16, (2, 4, 2, 2))
        fused = _stub(f"RmsCoordRecomputed{{N={n}}}", inv.ands + coord.ands, 16, (2 * n, 2, 2))
        return self._row_op("RmsNorm", n, inv, coord, fused, n)

    def quant(self, n: int) -> Shape:
        p = self.pins
        scale = _stub(f"QuantScale{{N={n}}}", n * p["F32Max_v1"] + p["F32Div_v2"] + p["SelectF32_v1"], 32, (2 * n, 4))
        pair = _stub(f"QuantPair{{N={n}}}", 2 * (p["F32Div_v2"] + F32_RINT + p["S32ClampS8_v1"]), 16, (4, 4, 2))
        fused = _stub(f"QuantPairRecomputed{{N={n}}}", scale.ands + pair.ands, 16, (2 * n, 2))
        return self._row_op("Quant", n, scale, pair, fused, n // 2)

    def softmax_head(self) -> Shape:
        """One head's softmax over every query position of a request: position t's row of t scores, padded to its key
        class (a multiple of 16), its t probabilities each a unit."""
        p = self.pins
        exp = p["F32Add_v2"] + p["F32Mul_v2"] + p["MufuEx2Ftz_v1"]
        prob = _stub("SoftmaxProb", exp + p["MufuRcpFtz_v1"] + p["F32Mul_v2"] + p["F2fpBf16_v1"], 16, (2, 4, 4, 2))
        rows: dict[str, list] = {}
        for t in range(1, self.m.context + 1):
            L = 16 * ceil(t / 16)
            mx = _stub(f"SoftmaxMax{{L={L}}}", L * p["F32Max_v1"], 32, (2 * L, 4))
            sm = _stub(f"SoftmaxSum{{L={L}}}", L * (exp + p["F32Add_v2"]), 32, (2 * L, 4, 4))
            fused = _stub(f"SoftmaxProbRecomputed{{L={L}}}", mx.ands + sm.ands + prob.ands, 16, (2 * L, 2))
            row = fused if t == 1 else Shape(f"stub:SoftmaxRow{{L={L},t={t}}}", mx.ands + sm.ands + t * prob.ands, 1,
                                             16 * t, t, ways=(("shared", ((mx, 1), (sm, 1), (prob, t))),
                                                              ("recomputed", ((fused, t),))))
            r = _counted(row, self.memo)
            rows.setdefault(r.id, [r, 0])[1] += 1
        parts = tuple((r, k) for r, k in rows.values())
        return _counted(Shape(f"stub:SoftmaxHead{{T={self.m.context}}}", sum(r.ands * k for r, k in parts), 1,
                              16 * self.m.context * (self.m.context + 1) // 2, self.m.context, parts=parts), self.memo)

    def _ops(self) -> dict[str, tuple[Shape, int]]:
        m, p = self.m, self.pins
        R, T, Ly, Gn, H, d = m.requests, m.context, m.layers, m.generate, m.heads, m.width
        relu = _counted(_stub("ReluBf16", p["SelectBf16_v1"] + 1, 16, (2, 2)), self.memo)
        embed = _counted(_stub(f"EmbedSelect{{V={m.vocab}}}", m.vocab * (p["SelectBf16_v1"] + 8), 16,
                               (1, 2 * m.vocab, 2)), self.memo)
        argmax = _counted(_stub(f"Argmax{{V={m.vocab}}}", m.vocab * (p["Bf16GtStrict_v1"] + p["SelectBf16_v1"] + 8), 8,
                                (2 * m.vocab, 1)), self.memo)
        ops: dict[str, list] = {}

        def add(shape: Shape, n: int) -> None:
            if n:
                ops.setdefault(shape.id, [shape, 0])[1] += n

        add(embed, R * T * d)
        add(self.rmsnorm(d), R * (T * 2 * Ly + Gn))
        add(self.quant(d), R * (T * 3 * Ly + Gn))
        add(self.quant(m.mlp), R * T * Ly)
        add(self.s8(d), R * (T * Ly * (4 * d + m.mlp) + Gn * m.vocab))
        add(self.s8(m.mlp), R * T * Ly * d)
        add(self.gemm(m.head_dim), R * Ly * H * T * (T + 1) // 2)
        for t in range(1, T + 1):
            add(self.gemm(16 * ceil(t / 16)), R * Ly * H * m.head_dim)
        add(self.softmax_head(), R * Ly * H)
        add(relu, R * T * Ly * m.mlp)
        add(self.real(F.Bf16Add), R * T * Ly * 2 * d)
        add(argmax, R * Gn)
        return {sid: (s, n) for sid, (s, n) in ops.items()}

    @property
    def units_dense(self) -> int:
        """The block's units when every op is cut at its finest row form (shared statistics, one unit a coordinate)."""
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
    the bandwidth audit with `audit_passes` passes a block, and `replay` GPU-s to rebuild one drawn unit's witness (one
    layer of one step re-executed from its committed input)."""

    block_s: float = BLOCK_S
    audit_store: float = 0.0
    audit_passes: float = 1.0
    replay: float = 0.0


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
            ("pouw:h1", "CP-W receipts", m.requests * (m.context * per_token + m.generate * head)),
            ("bandwidth", "receipts over the audited store", s.audit_passes * s.audit_store / 4096 * 64)]
    return tuple(Commitment(scale, tensor, b, cost=b * rate) for scale, tensor, b in rows if b)


def costs(m: Model, s: Serving, cands: list[Candidate] | None = None) -> CostModel:
    """C-Flock's fitted model (int8_llama's fit) with the commitment rate, the block's serve-time commitments and, when
    `cands` are given, each template's replay per drawn unit."""
    obs = observations()
    fitted = Fitted(fit(obs, "prove", ["one", "ands", "ands*staged", "ands*drawn"]),
                    fit(obs, "verify", ["one", "ands", "ands*staged", "ands*drawn"]),
                    fit(obs, "stage", ["one", "ands", "ands*staged"]), I["cpu"].value, KEY)
    base = replace(fitted.model(), commit_per_bit=I["commit_per_byte"].value / 8, serve=serve(m, s))
    if cands and s.replay:
        base = replace(base, templates={c.template: TemplateCost(replay=s.replay) for c in cands})
    return base


def candidates(dag: Dag, cm: CostModel, objective: Objective, grid: int = 21) -> tuple[list[Candidate], dict]:
    """`candidates.cuts` on each op: its alternatives, each its units by Definition."""
    out, alts = [], {}
    for op, (root, n) in dag.ops.items():
        cs, a, _ = cuts(op, root, n, cm, objective, grid=grid)
        out += cs
        alts[op] = {label: [list(u) for u in units] for label, units in a.items()}
    return out, alts


def plans(m: Model, s: Serving, delta: float = DELTAS[0], *, grid: int = 61, cut_grid: int = 21):
    """The block's DAG, its cost model, objective and candidates, and the frontier of expected cost against certified
    faulty units (screened by `density_bound`)."""
    dag = Dag(m)
    objective = Objective(delta)
    cm = costs(m, s)
    cands, alts = candidates(dag, cm, objective, cut_grid)
    cm = costs(m, s, cands)
    return dag, cm, objective, cands, alts, frontier(cands, cm, objective, grid=grid, exact=False)


def chosen(cands: list[Candidate], cm: CostModel, objective: Objective, front: list[Plan],
           target: float) -> Plan | None:
    """The cheapest plan certifying at most `target` faulty units (`for_target` on the frontier)."""
    return for_target(cands, cm, objective, target, plans=front, exact=False)


def law(plan: Plan, cands: list[Candidate]) -> Stratified:
    """The plan's stratified law: one stratum a template, k_s of n_s drawn."""
    return Stratified(tuple(Stratum(c.template, c.units, plan.sizes[c.template]) for c in cands
                            if c.template in plan.sizes))


def recertify(plan: Plan, cands: list[Candidate], delta: float, *, exact_count: bool = True) -> dict:
    """The plan's certificate from its sizes alone: faulty units by core's `Stratified.harm_bound` (and `count_bound`,
    exact, when `exact_count`), the optimizer's own `certify`, and the faulty units' ANDs (`harm_bound` of each unit's
    ANDs, its rows' openings left out)."""
    chosen_ = [c for c in cands if c.template in plan.sizes]
    st = law(plan, cands)
    out = {"delta": delta, "harm_bound": st.harm_bound({c.template: 1.0 for c in chosen_}, delta),
           "certify": certify(plan.sizes, cands, Objective(delta), exact=True).harm,
           "density_bound": density_bound(((c.units, plan.sizes[c.template], 1.0) for c in chosen_), delta),
           "ands_bound": st.harm_bound({c.template: float(c.ands) for c in chosen_}, delta)}
    if exact_count:
        out["count_bound"] = st.count_bound(delta)
    return out


def pooled(plan: Plan, cands: list[Candidate], delta: float, blocks: float) -> float:
    """Faulty units over `blocks` blocks, each drawn by the plan's law on its own live coins and all accepted, except
    with probability `delta`: a set of x_{b,s} faulty units escapes with at most prod exp(-k_s x_{b,s} / n_s), never
    once x_{b,s} > n_s - k_s, so the total is the knapsack `density_bound` over `blocks` copies of each stratum, which
    the period's length doesn't move (each stratum's rate k_s / n_s is a block's).  The uniform-subset case is
    `IntegrityHorizon`'s ln(1/delta) / q (#4358); the stratified one is this relaxation's."""
    return density_bound(((c.units * blocks, plan.sizes[c.template] * blocks, 1.0) for c in cands
                          if c.template in plan.sizes), delta)


def monitoring(plan: Plan, cands: list[Candidate], s: Serving, *, servers: int = SERVERS) -> list[dict]:
    """Per monitoring period (one server, and `servers` of them), at each delta: the pooled bound, and beside it each
    block's own at delta / blocks, so the period's total error is delta."""
    rows = []
    for fleet, count in (("1 server", 1), (f"{servers:,} servers", servers)):
        for period, seconds in PERIODS.items():
            blocks = count * seconds / s.block_s
            rows.append({"period": period, "fleet": fleet, "blocks": blocks,
                         "pooled": {str(d): pooled(plan, cands, d, blocks) for d in DELTAS},
                         "per_block": {str(d): pooled(plan, cands, d / blocks, 1) for d in DELTAS}})
    return rows


def handoff(plan: Plan, cands: list[Candidate], dag: Dag) -> dict:
    """The plan as the GPU loop registers it (`integrations/torch_loop`'s circuit record): its templates in proof units,
    each a Definition's descriptor id (a stub's id until CIR-100 lands) with its units, and the `work:1` law whose work
    table's floors are the plan's k per template, which `draw.derive` expands to exactly those k (`work_k`)."""
    rule = work_law(plan, cands)
    templates, table, statement = [], {}, {}
    for c in cands:
        if c.template not in plan.sizes:
            continue
        tid = dag.template_id(c)
        if tid in table:
            raise ValueError(f"{tid} is a template of two ops: one template is one stratum, so merge the ops")
        templates.append([tid, c.units])
        table[tid] = rule["table"][c.template]
        statement[tid] = plan.statement_size[c.template]
    total = sum(v["work"] * n for (_, n), v in zip(templates, table.values()))
    for (tid, n), v in zip(templates, table.values()):
        if (k := work_k(1, v["work"], n, total, v["floor"])) != v["floor"]:
            raise ValueError(f"{tid}: the work:1 law draws {k} of its units, the plan {v['floor']}")
    work_table(table)
    return {"format": "verity/sampling-plan/v0", "templates": templates, "law": rule["law"], "work_table": table,
            "statement_units": statement, "stubs": sorted(t for t, _ in templates if t.startswith("stub:"))}


def summary(plan: Plan, cands: list[Candidate], dag: Dag) -> dict:
    by: dict[str, dict] = {}
    for c in cands:
        if c.template in plan.sizes:
            by[dag.template_id(c)] = {"op": c.op, "units": c.units, "k": plan.sizes[c.template],
                                      "statement": plan.statement_size[c.template],
                                      "ands": c.ands + sum(row_ands(r) for r in c.rows)}
    return {"cost": dict(plan.cost), "faulty": plan.certificate.harm if plan.certificate else None,
            "units": sum(v["units"] for v in by.values()), "drawn": sum(v["k"] for v in by.values()),
            "templates": by, "choice": dict(plan.choice), "floor_only": sorted(plan.floor_only),
            "full": sorted(plan.full)}


SMALL = Model()
#: the large random model's shape until compute-accounting's: CMP-92's block at width 8,192 (heads of 64 as the small
#: one's, an MLP of 4 x width), 100 layers: 80.6 GB of INT8 linears; a batch of 128 at context 32 holds 13.4 GB of KV
LARGE = Model(layers=100, width=8192, heads=128, mlp=32768, context=32, prompt=8, requests=128, inflight=128)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", choices=("small", "large"), default="small")
    for f in ("layers", "width", "heads", "mlp", "context", "prompt", "requests", "inflight"):
        ap.add_argument(f"--{f}", type=int)
    ap.add_argument("--delta", type=float, default=DELTAS[0])
    ap.add_argument("--target", type=float, help=f"faulty units the chosen plan certifies at most (default: {TARGET:g} "
                                                  "of the block's units)")
    ap.add_argument("--grid", type=int, default=61)
    args = ap.parse_args(argv)
    m = SMALL if args.model == "small" else LARGE
    m = replace(m, **{f: v for f in ("layers", "width", "heads", "mlp", "context", "prompt", "requests", "inflight")
                      if (v := getattr(args, f)) is not None})
    s = Serving(audit_store=m.weight_bytes)
    dag, cm, objective, cands, alts, front = plans(m, s, args.delta, grid=args.grid)
    units = dag.units_dense
    target = args.target if args.target is not None else TARGET * units
    plan = chosen(cands, cm, objective, front, target)
    out = {"model": asdict(m), "serving": asdict(s), "delta": args.delta, "target": target, "units_dense": units,
           "weight_bytes": m.weight_bytes, "params": m.linear_params,
           "serve": [asdict(c) for c in cm.serve], "serve_cost": cm.serve_cost(),
           "rates": {"statement_fixed": cm.statement_fixed, "statement_per_and": cm.statement_per_and,
                     "staged_per_and": cm.staged_per_and, "per_and": cm.per_and, "commit_per_bit": cm.commit_per_bit},
           "alternatives": alts,
           "frontier": [{"cost": p.cost["total"], "faulty": p.certificate.harm, "drawn": sum(p.sizes.values()),
                         "choice": dict(p.choice), "floor_only": sorted(p.floor_only)} for p in front]}
    if plan is not None:
        out["chosen"] = summary(plan, cands, dag)
        out["recertified"] = [recertify(plan, cands, d) for d in DELTAS]
        out["monitoring"] = monitoring(plan, cands, s)
        out["handoff"] = handoff(plan, cands, dag)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "plan.json").write_text(json.dumps(out, indent=1, default=str))
    print(render(out))


def render(out: dict) -> str:
    m = out["model"]
    lines = [f"# CMP-92 block, {m['layers']} layers x {m['width']} ({m['heads']} heads, MLP {m['mlp']}), "
             f"{m['requests']} requests of {m['context']} positions a block: the sampling plan at delta "
             f"{out['delta']:g}",
             "", f"units a block (finest row form): {out['units_dense']:,}; serve-time commitments "
             f"{out['serve_cost']:.3g} GPU-s a block", "", "| expected cost (GPU-s) | certified faulty units | drawn | "
             "floor-only templates |", "|---|---|---|---|"]
    for f in out["frontier"]:
        lines.append(f"| {f['cost']:.4g} | {f['faulty']:.4g} | {f['drawn']:,} | {len(f['floor_only'])} |")
    c = out.get("chosen")
    if c is None:
        lines += ["", f"no plan certifies {out['target']:.4g} faulty units"]
        return "\n".join(lines)
    lines += ["", f"chosen (target {out['target']:.4g}): cost {c['cost']['total']:.4g} GPU-s "
              f"({', '.join(f'{k} {v:.3g}' for k, v in c['cost'].items() if k != 'total')}), {c['drawn']:,} drawn of "
              f"{c['units']:,}", "", "| template | op | units | k | statement | ANDs a proved unit |",
              "|---|---|---|---|---|---|"]
    for t, v in c["templates"].items():
        lines.append(f"| {t} | {v['op'].split('{')[0]} | {v['units']:,} | {v['k']:,} | {v['statement']:,} | "
                     f"{v['ands']:,} |")
    lines += ["", "| delta | Stratified.harm_bound | count_bound | certify | faulty ANDs |", "|---|---|---|---|---|"]
    for r in out["recertified"]:
        lines.append(f"| {r['delta']:g} | {r['harm_bound']:.4g} | {r.get('count_bound', '-')} | {r['certify']:.4g} | "
                     f"{r['ands_bound']:.4g} |")
    lines += ["", "| period | fleet | blocks | pooled at 1% | pooled at 1e-6 | each block at 1% / blocks | "
              "at 1e-6 / blocks |",
              "|---|---|---|---|---|---|---|"]
    for r in out["monitoring"]:
        p, b = r["pooled"], r["per_block"]
        lines.append(f"| {r['period']} | {r['fleet']} | {r['blocks']:,.0f} | {p['0.01']:.4g} | {p['1e-06']:.4g} | "
                     f"{b['0.01']:.4g} | {b['1e-06']:.4g} |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
