"""The sampling plan for the end-to-end demo on one H200: the frozen MQA shape on the demo's partition into
Definitions, every op a proof block, one unit a row for row ops.

A block is one lockstep window: `batch` requests of `context` bytes (`prompt` given, the rest generated greedily, the
last never fed back), each from its first byte to its last, so a block runs `positions` = context - 1 decode steps and
nothing carries across it. Its subcircuits are Definitions (`templates`): the W8A8 coordinates
(`ScaledMmS8Coordinate_v1` on `GEMM_SM90_S8`), attention's BF16 coordinates (`GemmCoordinate_v2` on `GEMM_HOPPER`),
and the row units (RMSNorm, the quantizer, softmax, the residual add, the gathers, the embedding and
argmax), each id `stub:` until its Definition lands, its ANDs composed from pinned pieces.

**The law is what Lean proves.** Each subcircuit is one sampling block of the block (`IntegrityPeriod`): one C-Flock `--zk`
session drawing a uniform `subset:k` of its n units, split over `tables` tables of at most `_cap` drawn units each
(m = k_log + nbl <= 35). Every subcircuit draws `k = ceil(ln(S/delta)/share)`, so by `IntegrityPeriodLog` each sampling block has
fewer than `K0 = ln(S/delta) n/k` <= share n units wrong outside delta (the S sampling blocks of the block split delta), and by
`IntegrityHorizon` a subcircuit's sampling blocks over any period have fewer than ln(S/delta) n/k in all. The faulty ANDs beside
them (`Stratified.harm_bound`, each unit weighted by its ANDs) are informative: the law-level bound is proved
(`FlockSoundness.Audit.stratified_isHarmBound_exact`), but no guarantee in the lock composes it with acceptance.

**Cost** is C-Flock's fit (`transformer.costs`: zero-knowledge sessions on sm_120, in RTX PRO 6000 GPU-seconds, taken as
H200 GPU-seconds), per table `statement_fixed + statement_per_and A` and per drawn unit `(per_and + staged_per_and) A`,
A a unit's ANDs with its rows' openings (`row_ands`); staging is taken to scale with the drawn units, not n. The fit
measured units of 5.1e6 to 3.0e7 ANDs, so the row subcircuits' (up to 2e8) are extrapolated. Replaying
the drawn requests (`REPLAY_BLOCKS` of the block's own run) and committing every unit's outputs (a design rule, not a
choice) are priced beside the proving, not traded against it.

    uv run python benchmarks/sampling_plan/h200.py --out DIR [--share S] [--batch B] [--step-s T] [--tiles] [--pad]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field, replace
from math import ceil, exp, log
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import transformer as X  # noqa: E402
from inputs import I  # noqa: E402

from verity.core.protocols.profile import Stratified, Stratum  # noqa: E402
from verity.core.protocols.sampled_proofs.candidates import shape_of  # noqa: E402
from verity.core.protocols.sampled_proofs.optimize import row_ands  # noqa: E402
from verity.ml.circuits.codec import descriptor_id  # noqa: E402
from verity.ml.circuits.defs import bind  # noqa: E402
from verity.ml.circuits.ops import fp32 as F, gemm as G  # noqa: E402

DELTAS = (1e-2, 1e-6)
#: each subcircuit's sampling block certifies at most this share of its units wrong at delta = 1% (`transformer.TARGET_SHARE`)
SHARE = X.TARGET_SHARE
#: replaying the drawn requests, in blocks of the block's own run: stand-in, the drawn units touch nearly every request
#: and a step's cost is its weights' decode, not its batch, which is not yet priced
REPLAY_BLOCKS = 1.0
#: output channels a static-matmul tile spans (`--tiles`, the option of a 2,048-bit tile unit)
TILE = 128
PERIODS = X.PERIODS


@dataclass(frozen=True)
class Shape:
    """The frozen H200 shape. `batch` and `step_s` are stand-ins until P2's decode is timed: about 20,480 sequences,
    and 7.5 s a step at 40 GB/s (3.7 s with the decode-to-staging kernel)."""

    layers: int = 52
    width: int = 16_384
    heads: int = 256
    head_dim: int = 64
    kv_heads: int = 1
    mlp: int = 65_536
    vocab: int = 256
    context: int = 32
    prompt: int = 8
    batch: int = 20_480
    step_s: float = 7.5
    #: the block's steps when they differ from its positions (a request's lifetime as the demo's shape counts it)
    steps: int = 0

    @property
    def positions(self) -> int:
        return self.context - 1

    @property
    def generate(self) -> int:
        return self.context - self.prompt

    @property
    def block_s(self) -> float:
        """One lockstep window: every request from its first byte to its last."""
        return (self.steps or self.positions) * self.step_s


@dataclass(frozen=True)
class Template:
    """A subcircuit of the block: `units` a block, each `ands` ANDs opening `rows` (bytes each) and writing `out_bits`."""

    id: str
    units: int
    ands: int
    rows: tuple[int, ...]
    out_bits: int
    real: bool = False

    @property
    def proved(self) -> int:
        """A drawn unit's ANDs with its rows' openings, the fit's feature."""
        return self.ands + sum(row_ands(r) for r in self.rows)


def templates(m: Shape, pins: dict[str, int] | None = None, *, tiles: bool = False,
              pad: bool = False) -> list[Template]:
    """The partition of one block, one subcircuit a Definition, its units a block. `pad`:
    softmax and PV padded to the context (one subcircuit each) instead of one per key count p + 1; `tiles`: the static matmuls as
    tiles of `TILE` output channels writing one row, which removes the gathers of their outputs."""
    p = pins if pins is not None else X.pieces()
    R, L, T, Gn, d, f, V = m.batch, m.layers, m.positions, m.generate, m.width, m.mlp, m.vocab
    q, kv = m.heads * m.head_dim, m.kv_heads * m.head_dim
    out: list[Template] = []

    def put(tid: str, units: int, ands: int, rows: tuple[int, ...], out_bits: int, real: bool = False) -> None:
        out.append(Template(tid, units, ands, rows, out_bits, real))

    def s8(k: int, units: int) -> None:
        fn = bind(G.ScaledMmS8Coordinate, K=k, **G.GEMM_SM90_S8)
        s = shape_of(fn, p)
        if tiles:
            put(f"stub:S8Tile{{K={k},N={TILE}}}", units // TILE, TILE * s.ands,
                (k,) + (k,) * TILE + (4, 4 * TILE, 2 * TILE), 16 * TILE)
        else:
            put(descriptor_id(fn), units, s.ands, s.rows, s.out_bits, True)

    rms = d * p["F32Fma_v2"] + p["F32Mul_v2"] + p["F32Add_v2"] + p["RsqrtApprox_v1"] + d * (2 * p["F32Mul_v2"]
                                                                                            + p["F2fpBf16_v1"])
    put(f"stub:RmsNormRow_v1{{N={d}}}", R * (2 * L * T + Gn), rms, (2 * d,) * 3, 16 * d)
    quant = d * p["F32Max_v1"] + p["F32Div_v2"] + p["SelectF32_v1"] + d * (p["F32Div_v2"] + X.F32_RINT
                                                                           + p["S32ClampS8_v1"])
    put(f"stub:QuantRowS8_v1{{N={d}}}", R * (3 * L * T + Gn), quant, (2 * d, d + 4), 8 * d + 32)
    s8(d, R * (L * T * (q + 2 * kv + d + f) + Gn * V))
    s8(f, R * L * T * d)
    gathers = 128 + 2 if tiles else q // 128 + 2 * ceil(kv / 128) + d // 128 + 2 * (d // 128)
    put("stub:Gather_v1{N=128}", R * (L * T * gathers + (0 if tiles else 2 * Gn)), 0, (2,) * 128 + (256,), 2048)
    relu = 128 * (p["SelectBf16_v1"] + 1 + p["F32Max_v1"])
    put("stub:GatherReluAmax_v1{N=128}", R * L * T * (f // 128), relu,
        ((256,) if tiles else (2,) * 128) + (256, 4), 2080)
    put("stub:RowAbsMaxTree_v1{N=128}", R * L * T * 5, 128 * p["F32Max_v1"], (512, 4), 32)
    put(f"stub:QuantCodes_v1{{N={d}}}", R * L * T * (f // d), d * (p["F32Div_v2"] + X.F32_RINT + p["S32ClampS8_v1"]),
        (256,) * 128 + (4, d), 8 * d)
    qk = bind(G.GemmCoordinate, K=m.head_dim, **G.GEMM_HOPPER)
    s = shape_of(qk, p)
    put(descriptor_id(qk), R * L * m.heads * T * (T + 1) // 2, s.ands, s.rows, s.out_bits, True)
    exp_ = p["F32Add_v2"] + p["F32Mul_v2"] + p["MufuEx2Ftz_v1"]
    pv = bind(G.GemmCoordinate, K=m.context, **G.GEMM_HOPPER)
    pv_s = shape_of(pv, p)
    for n, count in ([(m.context, T)] if pad else [(i + 1, 1) for i in range(T)]):
        sm = n * p["F32Max_v1"] + n * (exp_ + p["F32Add_v2"]) + p["MufuRcpFtz_v1"] + n * (p["F32Mul_v2"]
                                                                                          + p["F2fpBf16_v1"])
        put(f"stub:SoftmaxRow_v1{{N={n}}}", R * L * m.heads * count, sm, (2 * n, 2 * n), 16 * n)
        put(f"stub:GemmCoordinatePad_v1{{T={n},K={m.context}}}", R * L * q * count, pv_s.ands,
            (2 * n,) + (2 * m.head_dim,) * n + (2,), 16)
    add = shape_of(F.Bf16Add, p).ands
    put(f"stub:Bf16AddRow_v1{{N={d}}}", R * 2 * L * T, d * add, (2 * d,) + ((2 * d,) if tiles else (256,) * 128)
        + (2 * d,), 16 * d)
    put(f"stub:EmbedRow_v1{{N={d}}}", R * T, d * (p["SelectBf16_v1"] + 8 + add), (1, 2 * d, 2 * d, 2 * d), 16 * d)
    put(f"stub:ArgmaxRow_v1{{V={V}}}", R * Gn, V * (p["Bf16GtStrict_v1"] + p["SelectBf16_v1"] + 8), (2 * V, 2), 8)
    return out


@dataclass(frozen=True)
class Rates:
    """C-Flock's fitted rates (`transformer.costs`) and the commitment rate."""

    statement_fixed: float
    statement_per_and: float
    per_and: float
    commit_per_byte: float

    @classmethod
    def fitted(cls) -> "Rates":
        cm = X.costs(X.SMALL, X.Serving())
        return cls(cm.statement_fixed, cm.statement_per_and, cm.per_and + cm.staged_per_and,
                   I["commit_per_byte"].value)


@dataclass(frozen=True)
class Window:
    """One subcircuit's sampling block of the block: `k` of its units drawn in one session of `tables` tables."""

    template: Template
    k: int
    tables: int
    statements: float
    drawn: float
    K0: dict[str, int] = field(default_factory=dict)
    pooled: dict[str, float] = field(default_factory=dict)

    @property
    def cost(self) -> float:
        return self.statements + self.drawn


def window(t: Template, k: int, windows: int, r: Rates) -> Window:
    """`t`'s sampling block drawing k: its tables (at most `_cap` drawn units each, k rounded up to a multiple of them, as
    `HiddenWindow.hdiv` needs), its expected proving cost, and at each delta `IntegrityPeriodLog`'s K0 (delta split over
    the block's `windows`) and `IntegrityHorizon`'s bound over any number of the subcircuit's sampling blocks."""
    a, cap = t.proved, X._cap(t.proved)
    k = min(t.units, k)
    tables = ceil(k / cap)
    k = min(t.units, tables * ceil(k / tables))
    K0 = {str(d): min(ceil(log(windows / d) * t.units / k), t.units - k + 1) for d in DELTAS}
    pooled = {str(d): min(log(windows / d) * t.units / k, t.units - k + 1) for d in DELTAS}
    return Window(t, k, tables, tables * (r.statement_fixed + r.statement_per_and * a), k * r.per_and * a, K0, pooled)


def plan(ts: list[Template], share: float, r: Rates) -> list[Window]:
    """Every subcircuit's sampling block at `k = ceil(ln(S/delta)/share)`, delta = 1%: each certifies at most `share` of its
    units wrong."""
    k = ceil(log(len(ts) / DELTAS[0]) / share)
    return [window(t, k, len(ts), r) for t in ts]


def commit_cost(ts: list[Template], r: Rates) -> float:
    """Committing every unit's outputs a block (at `inputs.commit_per_byte`), which every plan pays."""
    return sum(t.units * t.out_bits / 8 for t in ts) * r.commit_per_byte


def replay_cost(ws: list[Window], m: Shape) -> float:
    """Replaying the drawn units' requests: `REPLAY_BLOCKS` of the block's run, scaled by the expected share of requests
    the draw touches."""
    drawn = sum(w.k for w in ws)
    return REPLAY_BLOCKS * m.block_s * (1 - exp(-drawn / m.batch))


def proving(ws: list[Window], m: Shape) -> float:
    return sum(w.cost for w in ws) + replay_cost(ws, m)


def budget_share(ts: list[Template], m: Shape, r: Rates, gpu_s: float) -> float | None:
    """The least share whose plan's proving and replay cost at most `gpu_s` a block, or None when even the share 1 does
    not."""
    if proving(plan(ts, 1.0, r), m) > gpu_s:
        return None
    lo, hi = 1e-9, 1.0
    for _ in range(60):
        mid = (lo * hi) ** 0.5
        lo, hi = (lo, mid) if proving(plan(ts, mid, r), m) <= gpu_s else (mid, hi)
    return hi


def window_blocks(ws: list[Window], m: Shape) -> int:
    """The least B for which sampling blocks of B blocks prove within their own GPU time. K0 = ln(S/delta) n/k is the same share
    of a sampling block's units whatever its n, and `_cap` reads only a unit's ANDs, so a sampling block over B blocks keeps k, its
    tables and its cost, and costs a block 1/B of it."""
    return ceil(proving(ws, m) / m.block_s)


def informative_ands(ws: list[Window], delta: float) -> float:
    """Faulty ANDs (`Stratified.harm_bound`, each unit its ANDs): not a Lean guarantee."""
    st = Stratified(tuple(Stratum(w.template.id, w.template.units, w.k) for w in ws))
    return st.harm_bound({w.template.id: float(w.template.ands) for w in ws}, delta)


def report(m: Shape, share: float, *, tiles: bool = False, pad: bool = False, servers: int = X.SERVERS) -> dict:
    r = Rates.fitted()
    ts = templates(m, tiles=tiles, pad=pad)
    ws = plan(ts, share, r)
    alt = plan(templates(m, tiles=not tiles, pad=pad), share, r)
    padded = plan(templates(m, tiles=tiles, pad=not pad), share, r)
    units = sum(t.units for t in ts)
    periods = []
    for fleet, count in (("1 H200", 1), (f"{servers:,} H200s", servers)):
        for name, seconds in PERIODS.items():
            blocks = count * seconds / m.block_s
            periods.append({"period": name, "fleet": fleet, "blocks": blocks,
                            "each_block": {str(d): sum(min(ceil(log(len(ws) * blocks / d) * w.template.units / w.k),
                                                           w.template.units - w.k + 1) for w in ws) for d in DELTAS},
                            "pooled": {str(d): sum(w.pooled[str(d)] for w in ws) for d in DELTAS}})
    return {
        "shape": asdict(m), "block_s": m.block_s, "share": share, "tiles": tiles, "pad": pad, "units": units,
        "ands": sum(t.units * t.ands for t in ts), "rates": asdict(r),
        "windows": [{"template": w.template.id, "real": w.template.real, "units": w.template.units, "k": w.k,
                     "tables": w.tables, "ands": w.template.ands, "proved_ands": w.template.proved,
                     "statements": w.statements, "drawn": w.drawn, "cost": w.cost, "K0": w.K0, "pooled": w.pooled}
                    for w in ws],
        "cost": {"statements": sum(w.statements for w in ws), "drawn": sum(w.drawn for w in ws),
                 "sessions": sum(w.cost for w in ws), "replay": replay_cost(ws, m), "proving": proving(ws, m),
                 "commit": commit_cost(ts, r), "block_gpu_s": m.block_s, "floor": proving(plan(ts, 1.0, r), m)},
        "alternatives": {("coordinates" if tiles else "tiles"): {"proving": proving(alt, m), "windows": len(alt)},
                         ("per key count" if pad else "padded"): {"proving": proving(padded, m),
                                                                  "windows": len(padded)}},
        "per_drawn_and": {w.template.id: w.cost / (w.k * w.template.ands) for w in ws if w.template.ands},
        "frontier": [{"share": s, "proving": proving(plan(ts, s, r), m)} for s in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)],
        "budget_share": budget_share(ts, m, r, m.block_s),
        "window_blocks": window_blocks(ws, m),
        "informative_ands": {str(d): informative_ands(ws, d) for d in DELTAS},
        "periods": periods,
        "handoff": [{"templates": [[w.template.id, w.template.units]], "law": f"subset:{w.k}", "tables": w.tables,
                     "cap": X._cap(w.template.proved), "K0": w.K0} for w in ws],
    }


def render(o: dict) -> str:
    m, c = o["shape"], o["cost"]
    lines = [f"# The H200 plan: {m['layers']} layers, width {m['width']}, {m['heads']} query heads of {m['head_dim']} "
             f"(MQA, {m['kv_heads']} KV head), MLP {m['mlp']}, V {m['vocab']}, C {m['context']}; a block of "
             f"{m['batch']:,} requests, {o['block_s']:g} s", "",
             f"Every subcircuit's sampling block draws k = ceil(ln(S/1%)/{o['share']:g}) of its units (S = {len(o['windows'])} "
             f"sampling blocks), so each has fewer than K0 = ln(S/delta) n/k units wrong outside delta (IntegrityPeriodLog), "
             f"and its sampling blocks over any period fewer than that in all (IntegrityHorizon). {o['units']:,} units, "
             f"{o['ands']:.4g} ANDs a block.", "",
             "| template | units a block | k | statements | ANDs a unit | proved, rows opened | statements GPU-s "
             "| drawn GPU-s | K0 at 1% | K0 at 1e-6 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for w in o["windows"]:
        lines.append(f"| {w['template']} | {w['units']:,} | {w['k']:,} | {w['tables']} | {w['ands']:,} | "
                     f"{w['proved_ands']:,} | {w['statements']:.4g} | {w['drawn']:.4g} | {w['K0']['0.01']:,} | "
                     f"{w['K0']['1e-06']:,} |")
    lines += ["", f"proving a block: statements {c['statements']:.4g} + drawn units {c['drawn']:.4g} + replay "
              f"{c['replay']:.4g} = **{c['proving']:.4g} GPU-s**, against the block's own {c['block_gpu_s']:g} GPU-s "
              f"({c['proving'] / c['block_gpu_s']:.3g}x; at share 1 it is still {c['floor']:.4g}); committing every "
              f"unit's outputs {c['commit']:.4g} GPU-s beside it (a design rule, not a choice)."]
    b = o["budget_share"]
    B = o["window_blocks"]
    lines += ["", "the block's own GPU time buys " + ("no plan" if b is None else
              f"a share of {b:.3g} of each template's units wrong at 1%") +
              f"; sampling blocks of {B:,} blocks ({B * o['block_s'] / 86400:.3g} days of one H200) prove the same share "
              "within their own GPU time", "",
              "| share at 1% | proving GPU-s a block |", "|---|---|"]
    lines += [f"| {f['share']:g} | {f['proving']:.4g} |" for f in o["frontier"]]
    lines += ["", "| cut | sampling blocks | proving GPU-s a block |", "|---|---|---|",
              f"| as planned (tiles {o['tiles']}, padded {o['pad']}) | {len(o['windows'])} | {c['proving']:.4g} |"]
    lines += [f"| {k} | {v['windows']} | {v['proving']:.4g} |" for k, v in o["alternatives"].items()]
    lines += ["", "faulty proof blocks over a period, summed over the templates (Lean: pooled is IntegrityHorizon a template, "
              "delta split over the S templates; each block is IntegrityPeriodLog at delta / (S blocks)):", "",
              "| period | fleet | blocks | pooled at 1% | at 1e-6 | each block at 1% | at 1e-6 |",
              "|---|---|---|---|---|---|---|"]
    for p in o["periods"]:
        lines.append(f"| {p['period']} | {p['fleet']} | {p['blocks']:,.0f} | {p['pooled']['0.01']:.4g} | "
                     f"{p['pooled']['1e-06']:.4g} | {p['each_block']['0.01']:.4g} | {p['each_block']['1e-06']:.4g} |")
    a = o["informative_ands"]
    lines += ["", f"informative, not a Lean guarantee: faulty ANDs a block at most {a['0.01']:.4g} at 1% and "
              f"{a['1e-06']:.4g} at 1e-6 ({a['0.01'] / o['ands']:.3g} and {a['1e-06'] / o['ands']:.3g} of the "
              "block's)."]
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--share", type=float, default=SHARE)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--step-s", type=float)
    ap.add_argument("--tiles", action="store_true")
    ap.add_argument("--pad", action="store_true")
    a = ap.parse_args(argv)
    m = replace(Shape(), **{k: v for k, v in (("batch", a.batch), ("step_s", a.step_s)) if v is not None})
    o = report(m, a.share, tiles=a.tiles, pad=a.pad)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "plan.json").write_text(json.dumps(o, indent=1))
    print(render(o))


if __name__ == "__main__":
    main()
