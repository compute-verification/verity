"""Figure 1 of the SPOT post, derived from the sampled-proofs planner: what an adversary serving GLM-5.3 on all of a
scope's compute over a monitoring period can exfiltrate, infiltrate or skip while evading detection with probability
above δ, against proof overhead, at each share of compute spent on verification.

Every input is a card (`inputs.json`, the census, rho's reference, `glm53_partition.json`) marked measured or modeled
with its source, and the output carries them all beside every plotted series (`figure1.json`), which the console reads.

**Workload.**  A scope's compute is its GPUs (the world: rho's `world-h200` fleet, the census's `world-inference-2027`;
a datacenter: the census's `dc-12500-machines-600s`) over a monitoring period (`wipe_monitoring.PERIODS`), serving at
the fleet's MFU on the H200's dense FP8 peak.  It serves GLM-5.3 requests of the census's tokens each, at the circuit's
MACs a token, cut into proof blocks as `glm53_partition.json` has them.

**Costs**, in the scope's GPU-seconds: a replay re-runs its request at serving's MFU; proving a proof block takes the
proof overhead times its FLOP at the dense peak; hashing a byte takes the H200's SHA-512 rate.  A design's cost is the
planner's (`optimize.grain_costs`), and a curve's point at budget β is the least target whose cost is at most β of the
scope's GPU-seconds, by bisection, capped at what the scope outputs (exfiltration) or computes (skipped work).

**Designs.**

* `two-stage` (exfiltration's headline): serving commits only the requests' token ids; the first stage samples
  requests, whose interiors the prover then commits, and the second stage samples proof blocks in them
  (`optimize.two_stage`), an incorrect request charged at its footprint, at most the session's output cap.
* `one-stage`: every proof block sampled uniformly, each sampled one's request replayed.  Its serve-time commitment,
  every proof block's outputs, is outside the budget, and the output gives its share of compute (`serve_hash_share`).
* `request`: request-level sampling, each sampled request replayed and proved whole, its output the session's cap.

Exfiltration is in bits, `consumers.exfiltration_bound`'s form: the incorrect proof blocks' output bits plus which ones
they are.  Skipped work is the work in incorrect proof blocks, under one-stage and request-level sampling: the
two-stage design's interior commitments come after its first stage, so they don't show the work was done at serve
time.  Infiltration is a panel of its own, empty until its consumer is read here.  The horizontal lines are rho's
reference objects (`REFERENCES`) and GLM-5.3's checkpoint.

    uv run python benchmarks/spot_figures/figure1.py --out DIR [--points N]
"""
from __future__ import annotations

import argparse
import json
import tomllib
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

from verity.core.protocols.pous.wipe_monitoring import PERIODS
from verity.core.protocols.sampled_proofs.optimize import Grain, ReplayGrain, grain_costs, leak_count, loc_bits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ADVERSARY, DEPLOYMENTS, HARDWARE = ("benchmarks/census/adversary.json", "benchmarks/census/deployments.json",
                                    "benchmarks/census/hardware.json")
RHO, RHO_HARDWARE = "benchmarks/rho/data/reference.toml", "benchmarks/rho/data/hardware.toml"
WORLD, DATACENTER, CENSUS_WORLD = "world-h200", "dc-12500-machines-600s", "world-inference-2027"
#: the H200's dense FP8 peak is the H100 SXM5's (one die, one clock)
PEAK = "h100-sxm5-80gb/e4m3"
DESIGNS = {"exfiltration": ("two-stage", "one-stage", "request"), "skipped_work": ("one-stage", "request")}
HEADLINE = {"exfiltration": "two-stage", "skipped_work": "one-stage"}
SCHEMA = "spot-figure1/v1"
#: the reference objects rho's reference rules, as cards: (table, key, field)
REFERENCES = {"ref_sparseloco": ("exfil", "sparseloco-1000", "bits"), "ref_rollouts": ("exfil", "rl-rollouts", "bits"),
              "ref_pretrain": ("work", "v4-pretrain", "flops"), "ref_step": ("work", "v4-step", "flops")}


@dataclass(frozen=True)
class Card:
    value: object
    unit: str
    status: str
    source: str
    provisional: bool = False

    def __post_init__(self) -> None:
        if self.status not in ("measured", "modeled") or not self.source:
            raise ValueError(f"a card is measured or modeled, with its source: {self}")


def _entry(path: str, ident: str) -> dict:
    return next(e for e in json.loads((ROOT / path).read_text()) if e["id"] == ident)


def _toml(path: str) -> dict:
    return tomllib.loads((ROOT / path).read_text())


def cards() -> dict[str, Card]:
    """Every input: `inputs.json`'s, and those read from the census, rho's reference and the partition card."""
    out = {k: Card(**v) for k, v in json.loads((HERE / "inputs.json").read_text()).items()}
    census, dc = _entry(ADVERSARY, CENSUS_WORLD), _entry(DEPLOYMENTS, DATACENTER)
    peak = _entry(HARDWARE, PEAK)
    rho = _toml(RHO)
    fleet = rho["fleet"][WORLD]
    devices = _toml(RHO_HARDWARE)["location"][fleet["location"]]["devices"]["value"]

    def from_rho(name: str, value, unit: str, where: str, entry: dict) -> None:
        out[name] = Card(value, unit, "modeled", f"{RHO} {where}: {entry['source']}")

    out |= {
        "tokens_per_request": Card(census["tokens_per_request"], "token", "modeled",
                                   f"{ADVERSARY} {CENSUS_WORLD}: tokens a request"),
        "datacenter_gpus": Card(dc["machines"] * dc["gpus_per_machine"], "GPU", "modeled",
                                f"{DEPLOYMENTS} {DATACENTER}"),
        "peak_flops": Card(peak["flop_per_second"], "FLOP/s", "modeled",
                           f"{HARDWARE} {PEAK}, the H200's dense FP8 peak: {peak['sources'][0]['label']}"),
        "h100_hour_flop": Card(census["h100_hour_flop"], "FLOP", "modeled",
                               f"{ADVERSARY} {CENSUS_WORLD}: an H100-hour at the dense FP8 peak"),
    }
    session = rho["policy"]["v4-pro-session"]["k"]
    from_rho("output_cap", session["value"], "bit", "policy.v4-pro-session.k", session)
    from_rho("world_gpus", fleet["count"]["value"] * devices, "GPU", f"fleet.{WORLD}, its locations' devices",
             fleet["count"])
    from_rho("mfu_serve", fleet["mfu_serve"]["value"], "", f"fleet.{WORLD}.mfu_serve", fleet["mfu_serve"])
    for name, (table, key, field) in REFERENCES.items():
        entry = rho[table][key][field]
        from_rho(name, entry["value"], entry["unit"], f"{table}.{key}", entry)
    part = json.loads((HERE / "glm53_partition.json").read_text())
    out["partition"] = Card({k: part[k] for k in ("ranks", "tokens", "strata")}, "", part["status"], part["source"])
    return out


@dataclass(frozen=True)
class Partition:
    """GLM-5.3's proof blocks a token: their count, MACs and committed bytes, and the mean MACs a proof block weighted
    by count, by MACs (the work target's samples) and by output bits (the leak target's)."""

    blocks: float
    macs: float
    bytes: float
    macs_by_count: float
    macs_by_work: float
    macs_by_bits: float
    bits: float

    @classmethod
    def of(cls, strata: list[dict]) -> Partition:
        n = sum(s["per_token"] for s in strata)
        macs = sum(s["per_token"] * s["macs"] for s in strata)
        bits = sum(s["per_token"] * s["out_bits"] for s in strata)
        return cls(n, macs, bits / 8, macs / n, sum(s["per_token"] * s["macs"] ** 2 for s in strata) / macs,
                   sum(s["per_token"] * s["out_bits"] * s["macs"] for s in strata) / bits, bits / n)


@dataclass(frozen=True)
class Scope:
    """A scope's compute and its requests, in GPU-seconds and FLOP."""

    who: str
    period: str
    gpu_seconds: float
    requests: float
    flop_request: float

    @property
    def name(self) -> str:
        return f"{self.who} {self.period}"


def scopes(c: dict[str, Card], part: Partition) -> list[Scope]:
    flop_request = c["tokens_per_request"].value * 2 * part.macs
    out = []
    for who in ("world", "datacenter"):
        for period, seconds in PERIODS:
            gs = c[f"{who}_gpus"].value * seconds
            out.append(Scope(who, period, gs, gs * c["mfu_serve"].value * c["peak_flops"].value / flop_request,
                             flop_request))
    return out


@dataclass(frozen=True)
class Model:
    """One scope's grains and replay grain at one proof overhead, every cost in the scope's GPU-seconds."""

    scope: Scope
    dot: Grain
    dot_work: Grain
    request: Grain
    replay: ReplayGrain
    byte_s: float
    delta: float


def model(c: dict[str, Card], part: Partition, s: Scope, overhead: float) -> Model:
    peak, mfu = c["peak_flops"].value, c["mfu_serve"].value
    tokens = c["tokens_per_request"].value
    n = round(s.requests * tokens * part.blocks)
    work = 2 * part.macs / part.blocks
    boundary = (tokens + 1) * c["token_bytes"].value

    def dot(macs: float) -> Grain:
        return Grain("proof block", n, work, part.bits, part.bytes / part.blocks, overhead * 2 * macs / peak)

    return Model(s, dot(part.macs_by_bits), dot(part.macs_by_work),
                 Grain("request", round(s.requests), s.flop_request, c["output_cap"].value, boundary,
                       overhead * s.flop_request / peak),
                 ReplayGrain(round(s.requests), s.flop_request / (mfu * peak), boundary, c["output_cap"].value),
                 1 / c["hash_rate"].value, c["delta"].value)


def cost(m: Model, design: str, target: str, value: float) -> dict:
    """The planner's row for one design at one target: `leak` bits (exfiltration) or `work` FLOP (skipped work)."""
    kw = {target: value}
    if design == "two-stage":
        if target != "leak":
            raise ValueError("the two-stage design's interior commitments come after its first stage: no work target")
        return grain_costs([m.dot], m.delta, h=m.byte_s, weight=1.0, replay=m.replay, design="two-stage", **kw)[0]
    if design == "one-stage":
        g = m.dot if target == "leak" else m.dot_work
        return grain_costs([g], m.delta, h=m.byte_s, weight=0.0, replay=m.replay, **kw)[0]
    if design == "request":
        return grain_costs([m.request], m.delta, h=m.byte_s, weight=1.0, replay=m.replay, **kw)[0]
    raise ValueError(f"no design {design!r}")


def certified(m: Model, design: str, target: str, value: float) -> float:
    """What the design certifies at a target: for a leak, b K + locBits(n, K) at the K the target allows (at most the
    target), and for work the target itself."""
    if target == "work":
        return value
    g = m.request if design == "request" else m.dot
    k = leak_count(g, value)
    return g.out_bits * k + loc_bits(g.units, k)


def cap(m: Model, target: str) -> float:
    """The trivial bound: everything the scope outputs (bits) or computes (FLOP)."""
    return m.request.units * m.request.out_bits if target == "leak" else m.request.units * m.request.work


def point(m: Model, design: str, target: str, budget: float, rel: float = 1e-4) -> dict:
    """The least target whose cost is at most `budget` GPU-seconds, by bisection in log, capped at `cap`."""
    top = cap(m, target)

    def fits(v: float) -> dict | None:
        row = cost(m, design, target, v)
        return row if row["total"] <= budget else None

    lo, hi = 1.0, top
    best = fits(hi)
    if best is None:
        return {"bound": top, "capped": True, "row": None}
    if (low := fits(lo)) is not None:
        return {"bound": certified(m, design, target, lo), "capped": False, "row": low}
    while hi / lo > 1 + rel:
        mid = sqrt(lo * hi)
        row = fits(mid)
        lo, hi, best = (lo, mid, row) if row is not None else (mid, hi, best)
    bound = min(certified(m, design, target, hi), top)
    return {"bound": bound, "capped": bound >= top, "row": best}


def overheads(c: dict[str, Card], points: int) -> list[float]:
    lo, hi = c["overheads"].value
    grid = [lo * (hi / lo) ** (i / (points - 1)) for i in range(points)]
    return sorted(set(grid) | {float(c["proof_overhead"].value)})


def references(c: dict[str, Card]) -> dict[str, list[dict]]:
    """The panels' horizontal lines, in the panel's unit: rho's reference objects, and the served model's weights."""
    def ref(label: str, name: str, scale: float = 1.0) -> dict:
        return {"label": label, "value": c[name].value * scale, "input": name}

    return {"exfiltration": [ref("1,000 SparseLoCo syncs", "ref_sparseloco"),
                             ref("GLM-5.3's weights", "weights_bytes", 8),
                             ref("100M tokens of RL rollouts", "ref_rollouts")],
            "skipped_work": [ref("Pretraining run", "ref_pretrain"), ref("One optimizer step", "ref_step")]}


def captions(serve_hash_share: float) -> dict[str, str]:
    serve = (f"hashes every proof block's outputs at serve time, {serve_hash_share:.2g}× the scope's compute, outside "
             "the budget")
    return {
        "exfiltration": ("Two-stage (solid): serving commits only the requests' token ids, and the sampled requests' "
                         "interiors are committed after the first stage, inside the budget. One-stage (dashed) "
                         f"{serve}. Request-level (dotted) proves each sampled request whole. A bound counts the "
                         "incorrect proof blocks' output bits and which ones they are."),
        "infiltration": "Pending: the design that certifies infiltration, and its consumer.",
        "skipped_work": (f"One-stage (solid) {serve}. Request-level (dotted) proves each sampled request whole. "
                         "Two-stage is left out: its interiors are committed after the first stage, so they don't "
                         "show the work was done at serve time."),
    }


def run(points: int = 57, scope_names: tuple[str, ...] | None = None) -> dict:
    c = cards()
    part = Partition.of(c["partition"].value["strata"])
    xs = overheads(c, points)
    budgets = c["budgets"].value
    out_scopes, series = {}, {"exfiltration": [], "skipped_work": []}
    for s in scopes(c, part):
        if scope_names is not None and s.name not in scope_names:
            continue
        base = model(c, part, s, 1.0)
        serve_hash = base.dot.units * base.dot.commit_bytes * base.byte_s
        out_scopes[s.name] = {"who": s.who, "period": s.period, "gpus": c[f"{s.who}_gpus"].value,
                              "gpu_seconds": s.gpu_seconds, "requests": s.requests, "flop_request": s.flop_request,
                              "proof_blocks": base.dot.units, "output_bits": cap(base, "leak"),
                              "flop": cap(base, "work"), "serve_hash_share": serve_hash / s.gpu_seconds,
                              "boundary_hash_share": base.replay.replay_blocks * base.replay.boundary_bytes
                              * base.byte_s / s.gpu_seconds,
                              "captions": captions(serve_hash / s.gpu_seconds)}
        models = [model(c, part, s, x) for x in xs]
        for panel, target in (("exfiltration", "leak"), ("skipped_work", "work")):
            for design in DESIGNS[panel]:
                for beta in budgets:
                    pts = [point(m, design, target, beta * s.gpu_seconds) for m in models]
                    series[panel].append({
                        "scope": s.name, "design": design, "budget": beta, "headline": design == HEADLINE[panel],
                        "x": xs, "y": [p["bound"] for p in pts], "capped": [p["capped"] for p in pts],
                        "share": [p["bound"] / cap(models[0], target) for p in pts],
                        "plan": [None if p["row"] is None else {k: p["row"].get(k) for k in
                                 ("q", "k", "draws", "replay_sample", "prove", "replay", "commit", "total")}
                                 for p in pts]})
    refs = references(c)
    return {
        "schema": SCHEMA, "model": "GLM-5.3",
        "inputs": {k: v.__dict__ for k, v in c.items()},
        "partition": part.__dict__,
        "scopes": out_scopes,
        "budgets": budgets,
        "vertical": {"x": c["proof_overhead"].value, "provisional": c["proof_overhead"].provisional,
                     "label": "Measured on an H200" + (" (provisional)" if c["proof_overhead"].provisional else "")},
        "panels": {
            "exfiltration": {"unit": "bit", "target": "leak", "headline": HEADLINE["exfiltration"],
                             "references": refs["exfiltration"], "series": series["exfiltration"]},
            "infiltration": {"status": "pending", "series": []},
            "skipped_work": {"unit": "FLOP", "target": "work", "headline": HEADLINE["skipped_work"],
                             "references": refs["skipped_work"], "series": series["skipped_work"]}},
    }


def at_overhead(out: dict) -> list[dict]:
    """Each series' point at the vertical line: the figure's table."""
    x = out["vertical"]["x"]
    rows = []
    for panel in ("exfiltration", "skipped_work"):
        for s in out["panels"][panel]["series"]:
            i = s["x"].index(x)
            rows.append({"panel": panel, "scope": s["scope"], "design": s["design"], "budget": s["budget"],
                         "bound": s["y"][i], "share": s["share"][i], "capped": s["capped"][i]})
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--points", type=int, default=57, help="proof overheads on the x-axis, the vertical line's besides")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    out = run(args.points)
    out["at_overhead"] = at_overhead(out)
    (args.out / "figure1.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    from render import render

    for name in out["scopes"]:
        render(out, name, args.out)


if __name__ == "__main__":
    main()
