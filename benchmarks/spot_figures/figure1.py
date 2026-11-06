"""Figure 1 of the SPOT post, derived from the sampled-proofs planner: what an adversary serving a model on all of a
scope's compute over a monitoring period can exfiltrate, infiltrate or skip while evading detection with probability
above δ, against proof overhead, at each share of compute spent on verification.

Every input is a card marked measured or modeled with its source: the model's `spot-card/v1` cards (`cards.py`), the
census and rho's reference.  The output carries them all beside every plotted series (`figure1.json`), which the
console reads.

**Workload.**  A scope's compute is its GPUs (the world: rho's `world-h200` fleet, the census's `world-inference-2027`;
a datacenter: the census's `dc-12500-machines-600s`) over a monitoring period (`wipe_monitoring.PERIODS`), serving at
the fleet's MFU on the H200's dense FP8 peak.  It serves the model's requests of the census's tokens each, at the
circuit's MACs a token (the `workload` card's `served_macs_per_token` where its proof blocks leave some out), cut into
proof blocks as the card has them; the captions give the share of MACs in no proof block and in `modeled` strata.

**Costs**, in the scope's GPU-seconds: a replay costs the `replay` card's factor times its request served at serving's
MFU; proving a proof block takes the marginal proof overhead (the x-axis) times its FLOP at the dense peak, plus its
share of its statement's fixed cost (`Fill`); hashing a byte takes the `commitment` card's rate.  A statement holds one
replay block's sampled proof blocks of one Definition, at most the Definition's `statement_blocks`, so the fill is the
design's: one block under one-stage, the k a sampled request's under two-stage, every block of a request under
request-level sampling.  A design's cost is the planner's (`optimize.grain_costs`), and a curve's point at budget β is
the least target whose cost is at most β of the scope's GPU-seconds, by bisection, capped at what the scope outputs
(exfiltration) or computes (skipped work).  Each design's serve-time hashing is stated beside it: every proof block's
outputs under one-stage, outside the budget, and the token ids under two-stage and request-level, inside it.

**Designs.**

* `two-stage` (exfiltration's headline): serving commits only the requests' token ids; the first stage samples
  requests, whose interiors the prover then commits, and the second stage samples proof blocks in them
  (`optimize.two_stage`), an incorrect request charged at its footprint, at most the session's output cap.
* `one-stage`: every proof block sampled uniformly, each sampled one's request replayed.  Its serve-time commitment,
  every proof block's outputs, is outside the budget, and the output gives its share of compute (`serve_hash_share`).
* `request`: request-level sampling, each sampled request replayed and proved whole, its output the session's cap.

Exfiltration is in bits, `consumers.exfiltration_bound`'s form: the incorrect proof blocks' output bits plus which ones
they are.  The leak target samples proof blocks uniformly, so the adversary makes the widest ones incorrect first
(`harm_bound`'s order): every incorrect block is charged the widest block's output bits, and a sampled one costs the
count-weighted mean MACs to prove.  Skipped work is the work in incorrect proof blocks, under one-stage and request-level sampling: the
two-stage design's interior commitments come after its first stage, so they don't show the work was done at serve
time.  Infiltration is a panel of its own, plotted from the model's `infiltration` card and pending without one.  The
horizontal lines are rho's reference objects (`REFERENCES`, its reference model's; SparseLoCo's syncs scaled to the
served model's parameters where its `workload` card gives them), the model's checkpoint and every `references` card's.
`figures.py` writes it with Figures 2 and 3.
"""
from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, replace
from math import inf, sqrt
from pathlib import Path

import cards as spot_cards
from glm53_partition import _k
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
#: rho's reference model, whose pretraining run, optimizer step and SparseLoCo syncs its references are
REF_MODEL, RHO_MODEL = "DeepSeek-V4-Pro", "deepseek-v4-pro"
#: the reference objects rho's reference rules, as cards: (table, key, field)
REFERENCES = {"ref_sparseloco": ("exfil", "sparseloco-1000", "bits"), "ref_rollouts": ("exfil", "rl-rollouts", "bits"),
              "ref_pretrain": ("work", "v4-pretrain", "flops"), "ref_step": ("work", "v4-step", "flops")}
#: the circuit-check pin a MAC's AND gates are read from: the Hopper dot's FP8 MoE coordinate
AND_PIN = ("verity/ml/definitions/registry/boolean_fp8_moe.circuit-check.json",
           "MoeExpertFp8BlockCoordinateW_v4{K=128,G=128,KB=128,SWAP=True,DOT=HopperE4m3MmaV2Dot32_v2}")


@dataclass(frozen=True)
class InputCard:
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


def slot_cards(model: str) -> dict:
    """The model's `spot-card/v1` cards Figure 1 reads, the optional ones only where they exist: `references` is the
    list of every one."""
    out = {s: spot_cards.load(s, model) for s in ("workload", "proof_overhead", "commitment", "replay", "design")}
    if inf := spot_cards.load("infiltration", model, required=False):
        out["infiltration"] = inf
    return out | ({"references": refs} if (refs := spot_cards.load_all("references", model)) else {})


def _card(c: dict, value, unit: str, field: str) -> InputCard:
    art = f" ({c['art']})" if c.get("art") else ""
    return InputCard(value, unit, c["status"], f"{c['slot']} card's {field}{art}: {c['source']}", c["provisional"])


def cards(model: str = "GLM-5.3") -> dict[str, InputCard]:
    """Every input: the model's slot cards', and those read from the census and rho's reference."""
    sc = slot_cards(model)
    design, work, po = sc["design"], sc["workload"], sc["proof_overhead"]
    if po["value"]["model_overhead"] is None:
        raise ValueError(f"{model}'s proof_overhead card has no model_overhead: Figure 1's vertical line needs one")
    out = {k: _card(design, design["value"][k], unit, k) for k, unit in
           (("delta", ""), ("budgets", "share of compute"), ("overheads", "x"), ("token_bytes", "B"))}
    out |= {"proof_overhead": _card(po, po["value"]["model_overhead"], "x", "model_overhead"),
            "proof_blocks": _card(po, po["value"]["blocks"], "", "blocks"),
            "hash_rate": _card(sc["commitment"], sc["commitment"]["value"]["hash_rate"], "B/s", "hash_rate"),
            "replay_factor": _card(sc["replay"], sc["replay"]["value"]["factor"], "x", "factor"),
            "weights_bytes": _card(work, work["value"]["weights_bytes"], "B", "weights_bytes"),
            "partition": _card(work, {k: work["value"].get(k) for k in
                                      ("ranks", "tokens", "strata", "served_macs_per_token")}, "", "strata")}
    census, dc = _entry(ADVERSARY, CENSUS_WORLD), _entry(DEPLOYMENTS, DATACENTER)
    peak = _entry(HARDWARE, PEAK)
    rho = _toml(RHO)
    fleet = rho["fleet"][WORLD]
    devices = _toml(RHO_HARDWARE)["location"][fleet["location"]]["devices"]["value"]

    def from_rho(name: str, value, unit: str, where: str, entry: dict) -> None:
        out[name] = InputCard(value, unit, "modeled", f"{RHO} {where}: {entry['source']}")

    pin, coordinate = AND_PIN
    ands = json.loads((ROOT / pin).read_text())["definitions"][coordinate]["and"]
    out |= {
        "tokens_per_request": InputCard(census["tokens_per_request"], "token", "modeled",
                                        f"{ADVERSARY} {CENSUS_WORLD}: tokens a request"),
        "datacenter_gpus": InputCard(dc["machines"] * dc["gpus_per_machine"], "GPU", "modeled",
                                     f"{DEPLOYMENTS} {DATACENTER}"),
        "peak_flops": InputCard(peak["flop_per_second"], "FLOP/s", "modeled",
                                f"{HARDWARE} {PEAK}, the H200's dense FP8 peak: {peak['sources'][0]['label']}"),
        "h100_hour_flop": InputCard(census["h100_hour_flop"], "FLOP", "modeled",
                                    f"{ADVERSARY} {CENSUS_WORLD}: an H100-hour at the dense FP8 peak"),
        "ands_per_mac": InputCard(ands / _k(coordinate), "AND/MAC", "modeled",
                                  f"{pin} {coordinate}: {ands:,} ANDs over its K MACs, the pinned Boolean lowering "
                                  "nearest GLM-5.3's FP8 MoE coordinates, taken for every MAC"),
    }
    session = rho["policy"]["v4-pro-session"]["k"]
    from_rho("output_cap", session["value"], "bit", "policy.v4-pro-session.k", session)
    from_rho("world_gpus", fleet["count"]["value"] * devices, "GPU", f"fleet.{WORLD}, its locations' devices",
             fleet["count"])
    from_rho("mfu_serve", fleet["mfu_serve"]["value"], "", f"fleet.{WORLD}.mfu_serve", fleet["mfu_serve"])
    for name, (table, key, field) in REFERENCES.items():
        entry = rho[table][key][field]
        from_rho(name, entry["value"], entry["unit"], f"{table}.{key}", entry)
    basis = rho["model"][RHO_MODEL]["n_total"]
    from_rho("ref_params", basis["value"], "param", f"model.{RHO_MODEL}.n_total, the reference model's", basis)
    if work["value"].get("params") is not None:
        out["params"] = _card(work, work["value"]["params"], "param", "params")
    elif model == REF_MODEL:
        out["params"] = out["ref_params"]
    return out


@dataclass(frozen=True)
class TokenBlocks:
    """The model's proof blocks a token: their count, MACs and the bytes one-stage commits, the mean MACs a proof
    block weighted by count (a uniformly sampled block's) and by MACs (the work target's samples), and the mean and
    widest output bits; beside them the whole circuit's MACs a token and those in `modeled` strata."""

    blocks: float
    macs: float
    bytes: float
    macs_by_count: float
    macs_by_work: float
    bits: float
    widest_bits: float
    served_macs: float
    modeled_macs: float

    @classmethod
    def of(cls, strata: list[dict], served_macs: float | None = None) -> TokenBlocks:
        n = sum(s["per_token"] for s in strata)
        macs = sum(s["per_token"] * s["macs"] for s in strata)
        bits = sum(s["per_token"] * s["out_bits"] for s in strata)
        committed = sum(s["per_token"] * s.get("commit_bytes", s["out_bits"] / 8) for s in strata)
        modeled = sum(s["per_token"] * s["macs"] for s in strata if s.get("modeled"))
        return cls(n, macs, committed, macs / n, sum(s["per_token"] * s["macs"] ** 2 for s in strata) / macs,
                   bits / n, max(s["out_bits"] for s in strata if s["per_token"] > 0), served_macs or macs, modeled)

    def coverage(self) -> dict:
        """The shares of the served MACs in no proof block and in `modeled` strata."""
        return {"uncovered_mac_share": 1 - self.macs / self.served_macs,
                "modeled_mac_share": self.modeled_macs / self.served_macs}


@dataclass(frozen=True)
class Fill:
    """Proving's price over its marginal at a statement's fill, per proof-block type: its share of the proof blocks, its
    share of their MACs, its statement's fixed GPU-seconds over a block's marginal ones, and its statement's cap in
    blocks.  A type the `proof_overhead` card doesn't measure takes the measured blocks' pooled fixed cost and least
    cap.  Proving GPU-seconds go as MACs, so a design's price is the types' MAC-weighted mean."""

    types: tuple[tuple[float, float, float, int], ...]

    @classmethod
    def of(cls, strata: list[dict], blocks: list[dict]) -> Fill:
        by = {b["block"]: b for b in blocks}
        pooled = (sum(b["fixed_seconds"] for b in blocks) / sum(b["seconds_per_block"] for b in blocks),
                  min(b["statement_blocks"] for b in blocks))
        n = sum(s["per_token"] for s in strata)
        macs = sum(s["per_token"] * s["macs"] for s in strata)
        return cls(tuple((s["per_token"] / n, s["per_token"] * s["macs"] / macs,
                          *((b["fixed_seconds"] / b["seconds_per_block"], b["statement_blocks"])
                            if (b := by.get(s["block"])) else pooled)) for s in strata))

    def factor(self, blocks: float) -> float:
        """The price over the marginal when a replay block holds `blocks` sampled proof blocks, each type's in
        statements of at most its cap."""
        return sum(m * (1 + c / max(1.0, min(cap, blocks * b))) for b, m, c, cap in self.types)


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


def scopes(c: dict[str, InputCard], part: TokenBlocks) -> list[Scope]:
    flop_request = c["tokens_per_request"].value * 2 * part.served_macs
    out = []
    for who in ("world", "datacenter"):
        for period, seconds in PERIODS:
            gs = c[f"{who}_gpus"].value * seconds
            out.append(Scope(who, period, gs, gs * c["mfu_serve"].value * c["peak_flops"].value / flop_request,
                             flop_request))
    return out


@dataclass(frozen=True)
class Model:
    """One scope's grains and replay grain at one marginal proof overhead, every cost in the scope's GPU-seconds, and
    the fill that prices each design's statements."""

    scope: Scope
    dot: Grain
    dot_work: Grain
    request: Grain
    replay: ReplayGrain
    byte_s: float
    delta: float
    fill: Fill
    blocks_per_request: float


def model(c: dict[str, InputCard], part: TokenBlocks, s: Scope, overhead: float, fill: Fill) -> Model:
    peak, mfu = c["peak_flops"].value, c["mfu_serve"].value
    tokens = c["tokens_per_request"].value
    n = round(s.requests * tokens * part.blocks)
    work = 2 * part.macs / part.blocks
    boundary = (tokens + 1) * c["token_bytes"].value

    def dot(macs: float) -> Grain:
        return Grain("proof block", n, work, part.widest_bits, part.bytes / part.blocks, overhead * 2 * macs / peak)

    return Model(s, dot(part.macs_by_count), dot(part.macs_by_work),
                 Grain("request", round(s.requests), s.flop_request, c["output_cap"].value, boundary,
                       overhead * tokens * 2 * part.macs / peak),
                 ReplayGrain(round(s.requests), c["replay_factor"].value * s.flop_request / (mfu * peak), boundary,
                             c["output_cap"].value),
                 1 / c["hash_rate"].value, c["delta"].value, fill, tokens * part.blocks)


def _priced(g: Grain, fill: float) -> Grain:
    return replace(g, prove=g.prove * fill)


def cost(m: Model, design: str, target: str, value: float) -> dict:
    """The planner's row for one design at one target, `leak` bits (exfiltration) or `work` FLOP (skipped work), its
    proving priced at the design's statement fill (`fill`, over the marginal).  Two-stage's fill is its k, which the
    planner picks at a price: the price is iterated to its fixed point."""
    kw = {target: value}
    if design == "two-stage":
        if target != "leak":
            raise ValueError("the two-stage design's interior commitments come after its first stage: no work target")
        fill = m.fill.factor(inf)
        for _ in range(8):
            row = grain_costs([_priced(m.dot, fill)], m.delta, h=m.byte_s, weight=1.0, replay=m.replay,
                              design="two-stage", **kw)[0]
            fill, last = m.fill.factor(row["k"] or inf), fill
            if abs(fill - last) <= 1e-9 * last:
                break
        return row | {"fill": last}
    if design == "one-stage":
        fill = m.fill.factor(1)
        g = m.dot if target == "leak" else m.dot_work
        return grain_costs([_priced(g, fill)], m.delta, h=m.byte_s, weight=0.0, replay=m.replay, **kw)[0] | {
            "fill": fill}
    if design == "request":
        fill = m.fill.factor(m.blocks_per_request)
        return grain_costs([_priced(m.request, fill)], m.delta, h=m.byte_s, weight=1.0, replay=m.replay,
                           **kw)[0] | {"fill": fill}
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


def overheads(c: dict[str, InputCard], points: int) -> list[float]:
    lo, hi = c["overheads"].value
    grid = [lo * (hi / lo) ** (i / (points - 1)) for i in range(points)]
    return sorted(set(grid) | {float(c["proof_overhead"].value)})


def references(c: dict[str, InputCard], model: str, refs: list[dict] = ()) -> dict[str, list[dict]]:
    """The panels' horizontal lines, in the panel's unit (bits, or FLOP for skipped work): rho's reference objects, the
    served model's weights, and every `references` card's lines, a line that depends on the scope by scope.  rho's
    references are its reference model's; SparseLoCo's sync goes as the parameters, so it is scaled to the served
    model's where its `workload` card gives them."""
    def ref(label: str, name: str, scale: float = 1.0) -> dict:
        return {"label": label, "value": c[name].value * scale, "input": name}

    params, basis = c.get("params"), c["ref_params"].value
    syncs = (ref(f"1,000 SparseLoCo syncs of {model}", "ref_sparseloco", params.value / basis)
             | {"scaled_by": "params / ref_params"} if params else
             ref(f"1,000 SparseLoCo syncs of a {basis / 1e12:g}T-parameter model", "ref_sparseloco"))
    out = {"exfiltration": [syncs, ref(f"{model}'s weights", "weights_bytes", 8),
                            ref("100M tokens of RL rollouts", "ref_rollouts")],
           "infiltration": [],
           "skipped_work": [ref(f"{REF_MODEL}'s pretraining run", "ref_pretrain"),
                            ref(f"One optimizer step of {REF_MODEL}", "ref_step")]}
    flop_per_and = 2 / c["ands_per_mac"].value
    for card in refs:
        for line in card["value"]["lines"]:
            scale = flop_per_and if line["unit"] == "AND" else 1.0
            row = {"label": line["label"], "input": "references"} | ({"art": card["art"]} if card.get("art") else {})
            out[line["panel"]].append(row | ({"value": line["value"] * scale} if "value" in line else
                                             {"by_scope": {k: v * scale for k, v in line["by_scope"].items()}}))
    return out


def reference_notes(refs: list[dict]) -> dict[str, list[str]]:
    """Each panel's sentences from the `references` cards with lines on it: a card's `caption`."""
    out: dict[str, list[str]] = {}
    for card in refs:
        if note := card["value"].get("caption"):
            for panel in dict.fromkeys(line["panel"] for line in card["value"]["lines"]):
                out.setdefault(panel, []).append(note)
    return out


def zero_lines(refs: dict[str, list[dict]], scope: str) -> dict[str, list[str]]:
    """Each panel's lines at 0 in `scope`, which a log axis can't draw and its caption names."""
    return {panel: [r["label"] for r in rows if (r["value"] if "value" in r else r["by_scope"].get(scope, 1)) <= 0]
            for panel, rows in refs.items()}


def captions(serve_hash_share: float, infiltration: dict | None = None, coverage: dict | None = None,
             boundary_hash_share: float = 0.0, zeros: dict[str, list[str]] | None = None,
             notes: dict[str, list[str]] | None = None) -> dict[str, str]:
    serve = (f"hashes every proof block's outputs at serve time, {serve_hash_share:.2g}× the scope's compute, outside "
             "the budget")
    tokens = f"the token ids, {boundary_hash_share:.2g} of the scope's compute"
    fill = " Proving is priced at each design's statement fill."
    gaps = ""
    if coverage and coverage["uncovered_mac_share"] > 0:
        gaps += (f" {coverage['uncovered_mac_share']:.0%} of the served MACs are in no proof block, and no curve "
                 "bounds what is done in them.")
    if coverage and coverage["modeled_mac_share"] > 0:
        gaps += (f" {coverage['modeled_mac_share']:.0%} are in proof blocks whose Definitions have no cut yet, "
                 "modeled at the grain they would take.")
    out = {
        "exfiltration": (f"Two-stage (solid): serving commits only {tokens}, and the sampled requests' interiors are "
                         "committed after the first stage, both inside the budget. One-stage (dashed) "
                         f"{serve}. Request-level (dotted) proves each sampled request whole; serving commits only the "
                         "token ids, inside the budget. A bound counts the incorrect proof blocks' output bits, each "
                         f"the widest block's, and which ones they are.{fill}{gaps}"),
        "infiltration": (infiltration["value"]["caption"] if infiltration else
                         "Pending: the design that certifies infiltration, and its consumer."),
        "skipped_work": (f"One-stage (solid) {serve}. Request-level (dotted) proves each sampled request whole; "
                         f"serving commits only {tokens}, inside the budget. Two-stage is left out: its interiors are "
                         f"committed after the first stage, so they don't show the work was done at serve time."
                         f"{fill}{gaps}"),
    }
    for panel, sentences in (notes or {}).items():
        out[panel] += "".join(f" {s}" for s in sentences)
    for panel, labels in (zeros or {}).items():
        if labels:
            out[panel] += f" At 0, so not drawn: {'; '.join(labels)}."
    return out


def infiltration_panel(card: dict | None, scope_names: list[str]) -> dict:
    """The infiltration panel from the model's `infiltration` card: its series where it carries them, else pending."""
    if card is None:
        return {"status": "pending", "series": []}
    v = card["value"]
    series = [{"scope": s["scope"], "design": v["design"], "budget": s["budget"], "headline": True, "x": s["x"],
               "y": s["y"]} for s in v.get("series", []) if s["scope"] in scope_names]
    return {"status": "card" if series else "pending", "unit": v["unit"], "design": v["design"], "form": v["form"],
            "headline": v["design"], "consumer": v["consumer"], "series": series}


def run(model_name: str = "GLM-5.3", points: int = 57, scope_names: tuple[str, ...] | None = None) -> dict:
    sc = slot_cards(model_name)
    c = cards(model_name)
    part = TokenBlocks.of(c["partition"].value["strata"], c["partition"].value["served_macs_per_token"])
    fill = Fill.of(c["partition"].value["strata"], c["proof_blocks"].value)
    xs = overheads(c, points)
    budgets = c["budgets"].value
    and_per_flop = c["ands_per_mac"].value / 2
    refs = references(c, model_name, sc.get("references", []))
    notes = reference_notes(sc.get("references", []))
    out_scopes, series = {}, {"exfiltration": [], "skipped_work": []}
    for s in scopes(c, part):
        if scope_names is not None and s.name not in scope_names:
            continue
        base = model(c, part, s, 1.0, fill)
        serve_hash = base.dot.units * base.dot.commit_bytes * base.byte_s / s.gpu_seconds
        boundary_hash = base.replay.replay_blocks * base.replay.boundary_bytes * base.byte_s / s.gpu_seconds
        out_scopes[s.name] = {"who": s.who, "period": s.period, "gpus": c[f"{s.who}_gpus"].value,
                              "gpu_seconds": s.gpu_seconds, "requests": s.requests, "flop_request": s.flop_request,
                              "proof_blocks": base.dot.units, "output_bits": cap(base, "leak"),
                              "flop": cap(base, "work"), "ands": cap(base, "work") * and_per_flop,
                              "serve_hash_share": serve_hash, "boundary_hash_share": boundary_hash,
                              "captions": captions(serve_hash, sc.get("infiltration"), part.coverage(),
                                                   boundary_hash, zero_lines(refs, s.name), notes)}
        models = [model(c, part, s, x, fill) for x in xs]
        for panel, target in (("exfiltration", "leak"), ("skipped_work", "work")):
            for design in DESIGNS[panel]:
                for beta in budgets:
                    pts = [point(m, design, target, beta * s.gpu_seconds) for m in models]
                    series[panel].append({
                        "scope": s.name, "design": design, "budget": beta, "headline": design == HEADLINE[panel],
                        "x": xs, "y": [p["bound"] for p in pts], "capped": [p["capped"] for p in pts],
                        "share": [p["bound"] / cap(models[0], target) for p in pts],
                        "plan": [None if p["row"] is None else {k: p["row"].get(k) for k in
                                 ("q", "k", "draws", "replay_sample", "prove", "replay", "commit", "total", "fill")}
                                 for p in pts]})
    return {
        "schema": SCHEMA, "model": model_name,
        "inputs": {k: v.__dict__ for k, v in c.items()},
        "cards": sc,
        "partition": part.__dict__ | part.coverage(),
        "fill": {"statement": "one replay block's sampled proof blocks of one Definition, at most its statement_blocks",
                 "one-stage": fill.factor(1), "request": fill.factor(c["tokens_per_request"].value * part.blocks),
                 "full": fill.factor(inf)},
        "scopes": out_scopes,
        "budgets": budgets,
        "vertical": {"x": c["proof_overhead"].value, "provisional": c["proof_overhead"].provisional,
                     "label": sc["proof_overhead"]["value"]["label"]},
        "panels": {
            "exfiltration": {"unit": "bit", "target": "leak", "headline": HEADLINE["exfiltration"],
                             "references": refs["exfiltration"], "series": series["exfiltration"]},
            "infiltration": infiltration_panel(sc.get("infiltration"), list(out_scopes))
            | {"references": refs["infiltration"]},
            "skipped_work": {"unit": "FLOP", "target": "work", "headline": HEADLINE["skipped_work"],
                             "and_per_flop": and_per_flop, "references": refs["skipped_work"],
                             "series": series["skipped_work"]}},
    }


def at_overhead(out: dict) -> list[dict]:
    """Each series' point at the vertical line: the figure's table, skipped work in AND gates too, with the design's
    serve-time hashing as a share of the scope's compute (outside the budget only under one-stage) and its fill."""
    x = out["vertical"]["x"]
    rows = []
    for panel in ("exfiltration", "skipped_work"):
        p = out["panels"][panel]
        for s in p["series"]:
            i = s["x"].index(x)
            sc = out["scopes"][s["scope"]]
            one = s["design"] == "one-stage"
            rows.append({"panel": panel, "scope": s["scope"], "design": s["design"], "budget": s["budget"],
                         "bound": s["y"][i], "share": s["share"][i], "capped": s["capped"][i],
                         "serve_hash_share": sc["serve_hash_share" if one else "boundary_hash_share"],
                         "serve_hash_in_budget": not one, "fill": (s["plan"][i] or {}).get("fill")}
                        | ({"ands": s["y"][i] * p["and_per_flop"]} if "and_per_flop" in p else {}))
    return rows
