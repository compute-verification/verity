"""Figure 1's model and its bounds against proof overhead, derived from the sampled-proofs planner: what an adversary
serving a model on all of a scope's compute over a monitoring period can exfiltrate, infiltrate or skip while evading
detection with probability above δ, against proof overhead, at each share of compute spent on verification.  Figure 1
itself plots the same model against total cost (`total_cost.py`).

Every input is a card marked measured or modeled with its source: the model's `spot-card/v1` cards (`cards.py`), the
census and rho's reference.  The output carries them all beside every plotted series (`figure1_overhead.json`), which
the console reads.

**Workload.**  A scope's compute is its GPUs (the world: rho's `world-h200` fleet, the census's `world-inference-2027`;
a datacenter: the census's `dc-12500-machines-600s`) over a monitoring period (`reporting.PERIODS`), serving at
the fleet's MFU on the H200's dense FP8 peak.  It serves the model's requests of the census's tokens each, at the
circuit's MACs a token (the `workload` card's `served_macs_per_token` where its proof blocks leave some out), cut into
proof blocks as the card has them; the captions give the share of MACs in no proof block and in `modeled` strata.
That is the `assumed` reading of serving's rate, the default.  Under `serving="measured"` (`SERVING`) a scope serves
at the model's `serving_rate` card's computed tokens a GPU-second instead: its requests are its GPU-seconds times the
rate over a request's tokens, a replay takes the request's tokens over the rate, and no MACs a token enters either;
the scope's GPUs stay rho's.  The output then says which rate it read (`serving`), every caption says so too
(`reading_note`), and each exfiltration row names the rate beside the rate at which its bound would reach `SERVING_GOAL`
(`at_reading`).

**Costs**, in the scope's GPU-seconds: a replay costs the `replay` card's factor times its request served at serving's
MFU; proving a proof block takes the marginal proof overhead (the x-axis) times its FLOP at the dense peak, plus its
share of its statement's fixed cost (`Fill`); hashing a byte takes the `commitment` card's rate.  A statement holds one
replay block's sampled proof blocks of one Definition, at most the Definition's `statement_blocks`, so the fill is the
design's: one block under one-stage, a type's share of the k a sampled request's under two-stage, every block of a
request under request-level sampling.  One-stage's and request-level's cost is the planner's (`optimize.grain_costs`),
and a curve's point at budget β the least target whose cost is at most β of the scope's GPU-seconds, by bisection; a
two-stage design is the planner's at the budget (`optimize.two_stage_design`).  Each point is capped at what the scope
outputs (exfiltration) or computes (skipped work).  Each design's serve-time hashing is stated beside it: every proof
block's outputs under one-stage, outside the budget, and the token ids under two-stage and request-level, inside it.

**Designs.**

* `two-stage` (exfiltration's headline) and `two-stage-uniform`: serving commits only the requests' token ids; the
  first stage samples requests, whose interiors the prover then commits, and the second stage samples k_t proof blocks
  of each stratum in them: the public size classes up to a threshold in one stratum and each wider one in its own
  (width-stratified, the threshold the planner's), or one stratum for all a request's blocks (uniform).  The bound is
  `CompiledTwo.Exfil`'s count, each admissible set of incorrect proof blocks charged its own blocks' committed bits
  (`optimize.two_stage_exfil`), so a wholly incorrect request is charged every bit it commits.
* `two-stage-capped` and `two-stage-capped-uniform`, `two-stage-patterns` and `two-stage-patterns-uniform`: the same
  designs with each request charged at most the bits it serves (the `served_bits` input: every one of its tokens
  served as the token select's output word), in `two_stage_exfil`'s sets' form (`card_influenceTwo_le_served_min`),
  which still counts which of a request's proof blocks are incorrect, and never above the scope's served bits
  (`card_influenceTwo_le_min`), or its patterns' form (`card_influenceTwo_le_patterns`), which counts a request's sets
  of more than its served bits once, at most the served bits too (`card_influenceTwo_le_two_pow_outs`).  `CAP_STATUS`,
  `SERVED_STATUS` and `OUTS_STATUS` say what each rests on.
* `one-stage`: every proof block sampled uniformly, each sampled one's request replayed.  Its serve-time commitment,
  every proof block's outputs, is outside the budget, and the output gives its share of compute (`serve_hash_share`).
  For exfiltration it is in `consumers.exfiltration_bound`'s form, the incorrect proof blocks' output bits plus which
  ones they are: the adversary makes the widest ones incorrect first (`harm_bound`'s order), so each is charged the
  widest block's output bits, and a sampled one costs the count-weighted mean MACs to prove.
* `request`: request-level sampling, each sampled request replayed and proved whole, its output the session's cap.

Lean checks every point's design unless `run`'s `lean` is off, and a design Lean refuses is an error; a series' `lean`
says which points it checked, None at the trivial bound, which has no design to check.  A one-stage or request-level
exfiltration point is the bits Lean certifies for its design (`lean_check`, `plancheck`).  A two-stage point is the
bits Lean certifies for the planner's design (`lean_point`, `lean_two`), at most its trivial bound, at the θ Lean finds
from the planner's: in the uncapped sets' form (`TwoStageCheckSound`), the capped sets' (`TwoStageCheckServedSound`)
or the patterns' (`TwoStageCheckPatternsSound`), with the planner's bits (`planner`) and Lean's θ (`theta`) beside it.
`total_cost.py` checks its two-stage lines with the same `lean_two`.

Exfiltration is in bits: the main figure plots the width-stratified two-stage design, uncapped and in the capped
sets' form, and request-level sampling (`MAIN`), and an appendix figure every design with the reference lines the
`design` card's `main_lines` leaves out.  A request is the census's mean, or with `request="workload"` the `workload`
card's request.  Skipped work is the work in incorrect proof blocks, under one-stage and request-level sampling: the
two-stage design's interior commitments come after its first stage, so they don't show the work was done at serve
time.  Infiltration is a panel of its own, plotted from the model's `infiltration` card and pending without one.  Each
panel names the deployment it assumes (`DEPLOYMENT`).  The horizontal lines are rho's reference objects
(`REFERENCES`, its reference model's; SparseLoCo's syncs scaled to the served model's parameters where its `workload`
card gives them), the model's checkpoint and every `references` card's.  `figures.py` writes it with Figure 1 and
Figures 2 and 3.
"""
from __future__ import annotations

import json
import re
import tomllib
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from math import ceil, inf, log, sqrt
from multiprocessing import get_context
from pathlib import Path
from typing import Callable, Iterator, Sequence

import cards as spot_cards
from verity.protocols.reporting import PERIODS
from verity.protocols.spot import plancheck
from verity.protocols.spot.optimize import (Cell, Grain, ReplayGrain, grain_costs, leak_count, loc_bits,
                                                           two_stage_design, two_stage_strata)

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
ADVERSARY, DEPLOYMENTS, HARDWARE = ("benchmarks/census/adversary.json", "benchmarks/census/deployments.json",
                                    "benchmarks/census/hardware.json")
RHO, RHO_HARDWARE = "verity/protocols/inference_only/rho/data/reference.toml", "verity/protocols/inference_only/rho/data/hardware.toml"
WORLD, DATACENTER, CENSUS_WORLD = "world-h200", "dc-12500-machines-600s", "world-inference-2027"
#: the H200's dense FP8 peak is the H100 SXM5's (one die, one clock)
PEAK = "h100-sxm5-80gb/e4m3"
DESIGNS = {"exfiltration": ("two-stage", "two-stage-uniform", "two-stage-capped", "two-stage-capped-uniform",
                            "two-stage-patterns", "two-stage-patterns-uniform", "request", "one-stage"),
           "skipped_work": ("one-stage", "request")}
#: the designs the main figure plots, at most six lines with the design card's; the appendix plots every one
MAIN = {"exfiltration": ("two-stage", "two-stage-capped", "request"), "skipped_work": ("one-stage", "request")}
HEADLINE = {"exfiltration": "two-stage", "skipped_work": "one-stage"}
#: the two-stage designs, each width-stratified (a stratum per public size class) or uniform, and each request's
#: charge: every bit it commits (None), or at most the bits it serves in `optimize.two_stage_exfil`'s sets' or patterns'
#: form
TWO_STAGE = {"two-stage": (True, None), "two-stage-uniform": (False, None), "two-stage-capped": (True, "sets"),
             "two-stage-capped-uniform": (False, "sets"), "two-stage-patterns": (True, "patterns"),
             "two-stage-patterns-uniform": (False, "patterns")}
#: what each served-output cap's form rests on
CAP_STATUS = {"sets": "card_influenceTwo_le_served_min", "patterns": "card_influenceTwo_le_patterns"}
#: what puts the sets' form at most the scope's served bits: the influence set is a set of served outputs
SERVED_STATUS = "card_influenceTwo_le_min"
#: what puts the patterns' form there: on any statement the influence set has at most 2^|outs| members
OUTS_STATUS = "card_influenceTwo_le_two_pow_outs"
#: the replay grain's fan-out F that a served window's registration verdict publishes (`Verdict.fanout`), the capped
#: form's N: its program is a population program, each instance reading only its own share of each Input node
#: (`partition.fanout`)
FANOUT = 1


def returned_bits(served: float) -> int:
    """The bits a replay block returns, which a served window's two-stage registration publishes
    (`WindowCommitmentRecord.returned`) and under which no served cap prices it (`plancheck.two`'s `returned`).
    Modelled, and read from no registration: the model's replay block is one request whose only outputs are its tokens'
    select words, which is what the `served_bits` input `served` counts, so it returns that input."""
    return ceil(served)


#: a request's tokens: the census's mean, or the `workload` card's request
REQUESTS = ("census", "workload")
#: where serving's rate comes from: rho's fleet utilization on the dense peak, or the model's `serving_rate` card, the
#: engine's measured computed tokens a GPU-second, read only when asked for
SERVING = ("assumed", "measured")
#: the exfiltration capacity a row at the measured reading is held against, in bits: 300 GB
SERVING_GOAL = 8 * 300e9
#: the deployment each panel assumes: what serving commits, and the replay block
DEPLOYMENT = {"exfiltration": "Tokens only: serving commits each request's token ids, and a replay block is a request.",
              "skipped_work": "Every block: serving commits every proof block's outputs, and a replay block is a "
                              "request.",
              "layer_of_token": "Deferred interiors at a layer of a token: serving commits each layer's residual "
                                "stream and the K and V a later token reads, and a replay block is one layer of one "
                                "token."}
#: the deployment every panel prices: SPOT[Flock]'s three roles, and the network certifier a layer of its own
ROLES = ("The deployment is SPOT[Flock]'s three roles: an inference server serves and commits its values; a proof "
         "server replays each sampled replay block and proves it with flock-circuit prove --zk, with no ZK server and "
         "no recursion (the proof_overhead card's dev arm); and a verifier draws the sample on the challenger's live "
         "coins and decides with the Lean verifier. The network certifier, which stamps each request's release order "
         "and bucket at ingress, is a layer of its own.")
#: what every bound rests on
TRUST = ("Every bound is soundness alone, resting on SHA-512's collision resistance, uniform verifier randomness and an "
         "honest verifier; a bound that reads the certifier's stamp rests on the certifier's own assumptions too, "
         "stated in its layer.")
#: the share of compute the infiltration caption compares the two panels' bounds at
RATIO_BUDGET = 0.01
SCHEMA = "spot-figure1/v1"
#: rho's reference model, whose pretraining run, optimizer step and SparseLoCo syncs its references are
REF_MODEL, RHO_MODEL = "DeepSeek-V4-Pro", "deepseek-v4-pro"
#: the reference objects rho's reference rules, as cards: (table, key, field)
REFERENCES = {"ref_sparseloco": ("exfil", "sparseloco-1000", "bits"), "ref_rollouts": ("exfil", "rl-rollouts", "bits"),
              "ref_pretrain": ("work", "v4-pretrain", "flops"), "ref_step": ("work", "v4-step", "flops")}
#: the circuit-check pin a MAC's AND gates are read from: the Hopper dot's FP8 MoE coordinate
AND_PIN = ("verity/definitions/registry/boolean_fp8_moe.circuit-check.json",
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


def _k(definition: str) -> int:
    m = re.search(r"[{,]K=(\d+)", definition)
    if m is None:
        raise ValueError(f"no K in {definition!r}")
    return int(m.group(1))


def slot_cards(model: str, serving: str = "assumed") -> dict:
    """The model's `spot-card/v1` cards Figure 1 reads, the optional ones only where they exist: `references` is the
    list of every one.  The `serving_rate` card is among them only under the measured reading."""
    if serving not in SERVING:
        raise ValueError(f"serving's rate is one of {SERVING}, not {serving!r}")
    out = {s: spot_cards.load(s, model) for s in ("workload", "proof_overhead", "commitment", "replay", "design")}
    if serving == "measured":
        out["serving_rate"] = spot_cards.load("serving_rate", model)
    if inf := spot_cards.load("infiltration", model, required=False):
        out["infiltration"] = inf
    return out | ({"references": refs} if (refs := spot_cards.load_all("references", model)) else {})


def _card(c: dict, value, unit: str, field: str) -> InputCard:
    art = f" ({c['art']})" if c.get("art") else ""
    return InputCard(value, unit, c["status"], f"{c['slot']} card's {field}{art}: {c['source']}", c["provisional"])


def cards(model: str = "GLM-5.3", request: str = "census", serving: str = "assumed") -> dict[str, InputCard]:
    """Every input: the model's slot cards', those read from the census and rho's reference, and the bits a request
    serves, each of its tokens as the token select's output word, the `design` card's `token_bytes`.  A request is
    the census's mean or, under `request="workload"`, the `workload` card's.  Under `serving="measured"` the inputs
    hold the `serving_rate` card's rate (`serving_rate`), which the lines that price serving then read in place of
    rho's utilization; under the default they hold none, whatever cards exist."""
    if request not in REQUESTS:
        raise ValueError(f"a request is one of {REQUESTS}, not {request!r}")
    sc = slot_cards(model, serving)
    design, work, po = sc["design"], sc["workload"], sc["proof_overhead"]
    if po["value"]["model_overhead"] is None:
        raise ValueError(f"{model}'s proof_overhead card has no model_overhead: Figure 1's vertical line needs one")
    out = {k: _card(design, design["value"][k], unit, k) for k, unit in
           (("delta", ""), ("budgets", "share of compute"), ("overheads", "x"), ("token_bytes", "B"))}
    out |= {k: _card(design, design["value"].get(k, {}), "", k) for k in ("main_lines", "notes")}
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
        "datacenter_machines": InputCard(dc["machines"], "machine", "modeled", f"{DEPLOYMENTS} {DATACENTER}"),
        "machine_gpus": InputCard(dc["gpus_per_machine"], "GPU", "modeled",
                                  f"{DEPLOYMENTS} {DATACENTER}: a machine's GPUs"),
        "window": InputCard(dc["window_s"], "s", "modeled", f"{DEPLOYMENTS} {DATACENTER}: the audit window"),
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
    if request == "workload":
        out["tokens_per_request"] = _card(work, work["value"]["tokens"], "token", "tokens")
    if (rate := sc.get("serving_rate")) is not None:
        out["serving_rate"] = _card(rate, rate["value"]["tokens_per_gpu_second"], "token/GPU-s", "tokens_per_gpu_second")
    tokens, bits = out["tokens_per_request"].value, round(8 * out["token_bytes"].value)
    out["served_bits"] = InputCard(tokens * bits, "bit", "modeled",
                                   f"every one of a request's {tokens:,} tokens served as the token select's output "
                                   f"word, {bits} bits (the design card's token_bytes), the registered circuit's "
                                   "only output a token")
    return out


@dataclass(frozen=True)
class TokenBlocks:
    """The model's proof blocks a token: their count, MACs and the bytes one-stage commits, the mean MACs a proof
    block is proved at (`price`) weighted by count (a uniformly sampled block's) and by MACs (the work target's
    samples), and the mean and widest output bits; beside them the whole circuit's MACs a token and those in `modeled`
    strata."""

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
        return cls(n, macs, committed, sum(s["per_token"] * price(s) for s in strata) / n,
                   sum(s["per_token"] * s["macs"] * price(s) for s in strata) / macs,
                   bits / n, max(s["out_bits"] for s in strata if s["per_token"] > 0), served_macs or macs, modeled)

    def coverage(self) -> dict:
        """The shares of the served MACs in no proof block and in `modeled` strata."""
        return {"uncovered_mac_share": 1 - self.macs / self.served_macs,
                "modeled_mac_share": self.modeled_macs / self.served_macs}


def price(stratum: dict) -> float:
    """The MACs a stratum's block is proved at: its `price_macs` where its proving is priced in seconds
    (`total_cost.priced`), else its MACs."""
    return stratum.get("price_macs", stratum["macs"])


@dataclass(frozen=True)
class Fill:
    """Proving's price over its marginal at a statement's fill, per proof-block type: its share of the proof blocks, its
    share of their proving (`price`), its statement's fixed GPU-seconds over a block's marginal ones, and its
    statement's cap in blocks: the stratum's own where it is priced in seconds, else the `proof_overhead` card's block
    of its name, else the measured blocks' pooled fixed cost and least cap.  Proving GPU-seconds go as `price`, so a
    design's price is the types' mean weighted by it."""

    types: tuple[tuple[float, float, float, int], ...]

    @classmethod
    def of(cls, strata: list[dict], blocks: list[dict]) -> Fill:
        by = {b["block"]: b for b in blocks}
        pooled = (sum(b["fixed_seconds"] for b in blocks) / sum(b["seconds_per_block"] for b in blocks),
                  min(b["statement_blocks"] for b in blocks))
        n = sum(s["per_token"] for s in strata)
        macs = sum(s["per_token"] * price(s) for s in strata)
        return cls(tuple((s["per_token"] / n, s["per_token"] * price(s) / macs,
                          *((b["fixed_seconds"] / b["seconds_per_block"], b["statement_blocks"])
                            if (b := s if "fixed_seconds" in s else by.get(s["block"])) else pooled))
                         for s in strata))

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


#: the post's headline scope, whose figures carry the bare file names, and its secondary scope, plotted beside it
PRIMARY, SECONDARY = "world year", "datacenter year"


def token_seconds(c: dict[str, InputCard], served_macs: float) -> float:
    """The GPU-seconds serving takes a computed token: its FLOP at rho's fleet utilization on the dense peak, or under
    the measured reading the `serving_rate` card's rate inverted, with no MACs a token in it."""
    if "serving_rate" in c:
        return 1 / c["serving_rate"].value
    return 2 * served_macs / (c["mfu_serve"].value * c["peak_flops"].value)


def scopes(c: dict[str, InputCard], part: TokenBlocks) -> list[Scope]:
    """Each scope's GPU-seconds and the requests it serves in them: at rho's fleet utilization on the dense peak, or
    under the measured reading at the `serving_rate` card's tokens a GPU-second over a request's own tokens.  The
    GPUs are rho's and the census's under either."""
    flop_request = c["tokens_per_request"].value * 2 * part.served_macs
    rate = c.get("serving_rate")
    out = []
    for who in ("world", "datacenter"):
        for period, seconds in PERIODS:
            gs = c[f"{who}_gpus"].value * seconds
            requests = (gs * rate.value / c["tokens_per_request"].value if rate is not None
                        else gs * c["mfu_serve"].value * c["peak_flops"].value / flop_request)
            out.append(Scope(who, period, gs, requests, flop_request))
    return out


@dataclass(frozen=True)
class Model:
    """One scope's grains and replay grain at one marginal proof overhead, every cost in the scope's GPU-seconds, the
    fill that prices each design's statements, a request's proof blocks by type as the two-stage planner's cells, and
    the bits a request serves."""

    scope: Scope
    dot: Grain
    dot_work: Grain
    request: Grain
    replay: ReplayGrain
    byte_s: float
    delta: float
    fill: Fill
    blocks_per_request: float
    cells: tuple[Cell, ...]
    served: float


def model(c: dict[str, InputCard], part: TokenBlocks, s: Scope, overhead: float, fill: Fill) -> Model:
    peak, mfu = c["peak_flops"].value, c["mfu_serve"].value
    tokens = c["tokens_per_request"].value
    n = round(s.requests * tokens * part.blocks)
    work = 2 * part.macs / part.blocks
    boundary = (tokens + 1) * c["token_bytes"].value
    replay = (c["replay_factor"].value * tokens / c["serving_rate"].value if "serving_rate" in c
              else c["replay_factor"].value * s.flop_request / (mfu * peak))

    def dot(macs: float) -> Grain:
        return Grain("proof block", n, work, part.widest_bits, part.bytes / part.blocks, overhead * 2 * macs / peak)

    cells = tuple(Cell(x["per_token"] * tokens, x["out_bits"], prove=overhead * 2 * price(x) / peak,
                       fixed=ratio * overhead * 2 * price(x) / peak, cap=statement,
                       commit_bytes=x.get("commit_bytes", x["out_bits"] / 8))
                  for x, (_, _, ratio, statement) in zip(c["partition"].value["strata"], fill.types)
                  if x["per_token"] > 0)
    return Model(s, dot(part.macs_by_count), dot(part.macs_by_work),
                 Grain("request", round(s.requests), s.flop_request, c["output_cap"].value, boundary,
                       overhead * tokens * 2 * part.macs / peak),
                 ReplayGrain(round(s.requests), replay, boundary),
                 1 / c["hash_rate"].value, c["delta"].value, fill, tokens * part.blocks, cells, c["served_bits"].value)


def reading(sc: dict, c: dict[str, InputCard], part: TokenBlocks) -> dict:
    """What an output at the measured reading says of the rate it read: the `serving_rate` card's value, whether the
    card is provisional, and beside it the rate rho's utilization comes to on this circuit, which the assumed
    reading serves at."""
    card = sc["serving_rate"]
    return {"reading": "measured", **card["value"], "provisional": card["provisional"],
            "assumed_tokens_per_gpu_second": c["mfu_serve"].value * c["peak_flops"].value / (2 * part.served_macs)}


#: where a bound priced at a sampling rate holds, said in every caption that prints a bound beside the measured rate
REGISTRATION_NOTE = ("A bound priced at a sampling rate q holds at a registration of 1/q requests: the plan counts q·n "
                     "sampled of a registration's n, and the first stage draws at least ⌈q·n⌉, never fewer than one. "
                     "No bound here is what serving costs at a smaller registration.")


def outdir(out: Path, serving: str) -> Path:
    """Where a reading's files go: `out` at the assumed reading and `out/measured` at the measured one, so no file at
    the measured reading has the path the default writes."""
    return out if serving == SERVING[0] else out / serving


def reading_note(serving: dict) -> str:
    """What every caption at the measured reading carries: the rate the figure was planned at, whether it is a floor
    (and that its bounds then are), what it stands in place of, and the registration its bounds hold at
    (`REGISTRATION_NOTE`)."""
    rate, assumed = serving["tokens_per_gpu_second"], serving["assumed_tokens_per_gpu_second"]
    what = ("above which the engine computed no more" if serving["saturated"]
            else "a lower bound: no rate above it was served")
    goal = f"{SERVING_GOAL / 8e9:,.0f} GB"
    floor = ("" if serving["saturated"] else f" Each bound here is then a lower bound on the bound at the engine's own "
             f"rate: one under {goal} does not show that {goal} is met.")
    return (f"Serving is planned at the measured {rate:,.5g} computed tokens a GPU-second ({what}), not at rho's "
            f"fleet utilization, {assumed:,.5g} on this circuit: the scope's GPUs are the same, and its requests "
            f"{rate / assumed:.2g} of that reading's.{floor} {REGISTRATION_NOTE}")


def at_reading(serving: dict | None, bound: float) -> dict:
    """What an exfiltration row at the measured reading names beside its bound: the rate, whether a rate above it gave
    no more, and the rate at which the bound would reach `SERVING_GOAL`.  That rate is by proportion: a scope's requests
    go as the rate and the bound is taken to go as the requests; nothing is planned again.  Nothing under the default."""
    if not serving:
        return {}
    rate = serving["tokens_per_gpu_second"]
    return {"tokens_per_gpu_second": rate, "saturated": serving["saturated"],
            "tokens_per_gpu_second_at_goal_by_proportion": rate * SERVING_GOAL / bound if bound > 0 else None}


def _priced(g: Grain, fill: float) -> Grain:
    return replace(g, prove=g.prove * fill)


def two_stage(m: Model, design: str, budget: float) -> dict | None:
    """The two-stage design (`TWO_STAGE`) with the least exfiltration bound at `budget` GPU-seconds, serving's
    boundaries, the token ids, paid from it: `optimize.two_stage_design` on the model's cells, a request's charge
    capped at the bits it serves under a capped design."""
    stratify, form = TWO_STAGE[design]
    return two_stage_design(m.cells, m.replay.replay_blocks, m.delta, budget=budget, replay=m.replay.cost,
                            boundary=m.replay.replay_blocks * m.replay.boundary_bytes * m.byte_s, h=m.byte_s,
                            stratify=stratify, served=None if form is None else m.served, union=form == "patterns")


def cost(m: Model, design: str, target: str, value: float) -> dict:
    """The planner's row for one-stage or request-level sampling at one target, `leak` bits (exfiltration) or `work`
    FLOP (skipped work), its proving priced at the design's statement fill (`fill`, over the marginal)."""
    kw = {target: value}
    if design in TWO_STAGE:
        raise ValueError(f"a {design} design is the planner's at a budget (`point`), not at a target")
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


def served_total(m: Model) -> float:
    """The scope's served bits, |outs| of its period's audit: every request's `served`."""
    return m.request.units * m.served


def two_stage_top(m: Model, design: str, top: float | None = None) -> float:
    """A two-stage point's trivial bound: `top` (default everything the scope outputs), under a capped form at most
    the scope's served bits too (`SERVED_STATUS`, `OUTS_STATUS`)."""
    top = cap(m, "leak") if top is None else top
    return min(top, served_total(m)) if TWO_STAGE[design][1] else top


def point(m: Model, design: str, target: str, budget: float, rel: float = 1e-4, top: float | None = None) -> dict:
    """The least target whose cost is at most `budget` GPU-seconds, by bisection in log, capped at `top` (default
    `cap`); a two-stage design's planned bound at the budget, which certifies exfiltration alone (its interiors are
    committed after its first stage), at most its trivial bound (`two_stage_top`), which `lean_point` checks."""
    if design in TWO_STAGE:
        if target != "leak":
            raise ValueError("the two-stage design's interior commitments come after its first stage: no work target")
        top = two_stage_top(m, design, top)
        row = two_stage(m, design, budget)
        bound = top if row is None else min(row["bound"], top)
        return {"bound": bound, "capped": bound >= top, "row": row}
    top = cap(m, target) if top is None else top

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


def lean_design(m: Model, design: str, target: str, p: dict) -> dict | None:
    """The one-stage or request-level design behind `point`'s `p`, as Lean's checker reads it (`plancheck`), or None
    where there is none to check: at the trivial bound, or at a leak count of every unit, which samples none."""
    if p["capped"]:
        return None
    if target == "leak":
        g = m.request if design == "request" else m.dot
        return None if leak_count(g, p["bound"]) >= g.units else plancheck.leak_design(g, p["bound"], m.delta)
    return plancheck.work_design(m.request if design == "request" else m.dot_work, p["bound"], m.delta)


def lean_check(points: list[tuple[Model, str, str, dict]]) -> list[tuple[bool | None, float]]:
    """Lean's check of each (model, design, target, point)'s design (`lean_design`), in one run of the checker: whether
    it checked one (None where there is none), and the point's bound, a leak point's the bits Lean certifies rounded up
    to a float where it checked one.  A design Lean refuses is an error."""
    designs = [lean_design(*q) for q in points]
    got = iter(plancheck.check([d for d in designs if d is not None]))
    out = []
    for (m, design, target, p), d in zip(points, designs):
        if d is None:
            out.append((None, p["bound"]))
            continue
        a = next(got)
        if a is None or a is False:
            raise plancheck.LeanError(f"Lean refuses the planner's {design} design at {m.scope.name}: {d}")
        out.append((True, plancheck.upper(a) if target == "leak" else p["bound"]))
    return out


def split_cells(cells: Sequence[Cell], merged: int | None) -> list[Cell]:
    """The cells in the strata of a two-stage design whose row's `merged` is this (`optimize.two_stage_strata`)."""
    return two_stage_strata(cells, merged)


def lean_two(cells: Sequence[Cell], replay_blocks: float, row: dict, delta: float, served: float | None = None,
             union: bool = False) -> tuple[float, dict]:
    """Lean's check of the planner's two-stage design `row` (`optimize.two_stage_design`'s) on `replay_blocks` replay
    blocks of `cells`, as one class (`plancheck.two`) whose strata are the row's (`split_cells`): every bit charged
    (`TwoStageCheckSound`), each replay block charged at most the `served` bits it serves in the capped sets' form
    (`TwoStageCheckServedSound`) or, under `union`, the patterns' (`TwoStageCheckPatternsSound`).  The bits Lean
    certifies, rounded up to a float, at the θ it finds from the planner's, with the planner's bits and that θ beside
    them; a point Lean refuses is an error."""
    d = plancheck.two(split_cells(cells, row["merged"]), row["k"], replay_blocks, row["q"], delta, served, union,
                      row["theta"], fanout=FANOUT, returned=None if served is None else returned_bits(served))
    ((bits, theta),) = plancheck.check([d])
    if bits is None:
        raise plancheck.LeanError(f"Lean refuses the planner's two-stage point: {d}")
    return plancheck.upper(bits), {"lean": True, "planner": row["bound"], "theta": float(theta)}


def lean_point(m: Model, design: str, p: dict) -> dict:
    """A two-stage `point`'s `p` with Lean's check of its design (`lean_two`): its bound the bits Lean certifies, at most
    its trivial bound (`two_stage_top`), with `lean` True, the planner's bits (`planner`) and Lean's θ (`theta`); at
    the trivial bound, where there is no design to check, `lean` and `theta` None."""
    row = p["row"]
    if row is None or p["capped"]:
        return p | {"lean": None, "planner": None if row is None else row["bound"], "theta": None}
    form = TWO_STAGE[design][1]
    bits, checked = lean_two(m.cells, m.replay.replay_blocks, row, m.delta, None if form is None else m.served,
                             form == "patterns")
    top = two_stage_top(m, design)
    return p | {"bound": min(bits, top), "capped": bits >= top} | checked


def checked_point(job: tuple[Model, str, str, float, bool]) -> dict:
    """One point of a curve (`point`) and, with `lean`, Lean's check of its design: a two-stage point's `lean_point`,
    any other's `lean` and bound from `lean_check`.  A point reads only its own job, so processes may plan and check
    the points in any order (`processes`)."""
    m, design, target, budget, lean = job
    p = point(m, design, target, budget)
    if not lean:
        return p
    if design in TWO_STAGE:
        return lean_point(m, design, p)
    ((on, bound),) = lean_check([(m, design, target, p)])
    return p | {"bound": bound, "lean": on}


@contextmanager
def processes(workers: int, lean: bool) -> Iterator[Callable]:
    """`map` over `workers` forked processes, or `map` itself at one.  With `lean`, Lean's checker is built first, so
    each fork inherits it (`plancheck.executable` builds once a process, one process at a time) rather than queueing to
    build it again."""
    if workers <= 1:
        yield map
        return
    if lean:
        plancheck.executable(plancheck.EXE)
    with ProcessPoolExecutor(workers, mp_context=get_context("fork")) as pool:
        yield pool.map


PLAN = ("merged", "kappa", "q", "k", "draws", "replay_sample", "prove", "replay", "boundary", "interior", "commit",
        "total", "fill")


def _plan(row: dict) -> dict:
    """A point's planner row as the series carry it: a two-stage design's k by stratum, a wide stratum by its size
    class's bits and the merged one by the widest class it holds (`all` when it is the only one); where each width is
    a stratum (`merged` None), by `w` and the width's place among the widths."""
    out = {k: row[k] for k in PLAN if k in row}
    if isinstance(k := out.get("k"), dict):
        if "merged" in row and row["merged"] is None:
            out["k"] = {f"w{t}": v for t, v in k.items()}
        else:
            out["k"] = {str(t) if t else "all" if len(k) == 1 else f"<={row['merged']}": v for t, v in k.items()}
    return out


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
    chosen = c["main_lines"].value if "main_lines" in c else {}
    for panel, rows in out.items():
        for row in rows:
            styles = [style for prefix, style in chosen.get(panel, {}).items()
                      if row["label"].startswith(prefix.format(model=model))]
            row |= {"main": panel not in chosen or bool(styles), "style": styles[0] if styles else "dashed"}
    return out


def reference_notes(refs: list[dict], lines: dict[str, list[dict]] | None = None) -> dict[str, list[str]]:
    """Each panel's sentences from the `references` cards with lines on it: a card's `caption`, in the exfiltration
    appendix's caption when none of its exfiltration lines is on the main panel (`lines`, `references`'s)."""
    main = {(panel, r["label"]) for panel, rows in (lines or {}).items() for r in rows if r.get("main", True)}
    out: dict[str, list[str]] = {}
    for card in refs:
        if note := card["value"].get("caption"):
            for panel in dict.fromkeys(line["panel"] for line in card["value"]["lines"]):
                shown = lines is None or any((panel, line["label"]) in main for line in card["value"]["lines"]
                                             if line["panel"] == panel)
                out.setdefault(panel if shown or panel != "exfiltration" else "exfiltration_appendix",
                               []).append(note)
    return out


def _bits(v: float) -> str:
    return f"{v:.2g}".replace("e+0", "e").replace("e+", "e") + " bits"


def _interpolate(xs: list[float], ys: list[float], x: float) -> float:
    """`ys` at `x`, linearly in log-log between the grid's points."""
    pairs = sorted(zip(xs, ys))
    if x <= pairs[0][0]:
        return pairs[0][1]
    for (x0, y0), (x1, y1) in zip(pairs, pairs[1:]):
        if x <= x1:
            if y0 <= 0 or y1 <= 0 or x1 == x0:
                return y1
            return y0 * (y1 / y0) ** (log(x / x0) / log(x1 / x0))
    return pairs[-1][1]


def infiltration_ratio(infiltration: dict, exfiltration: list[dict], scope: str, x: float,
                       budget: float = RATIO_BUDGET) -> str:
    """The sentence the infiltration caption ends with: its curve at the vertical line and `budget` in `scope`, against
    the exfiltration panel's headline there; empty where either has no such point."""
    inf_s = next((s for s in infiltration.get("series", []) if s["scope"] == scope and s["budget"] == budget), None)
    head = next((s for s in exfiltration if s["scope"] == scope and s["budget"] == budget and s["headline"]), None)
    if inf_s is None or head is None:
        return ""
    a, b = _interpolate(inf_s["x"], inf_s["y"], x), _interpolate(head["x"], head["y"], x)
    return (f"At the measured overhead and {budget:.0%} of compute, this curve is {_bits(a)}, {a / b:.2g}× the "
            f"exfiltration panel's tokens-only two-stage bound, {_bits(b)}.")


def zero_lines(refs: dict[str, list[dict]], scope: str) -> dict[str, list[str]]:
    """Each panel's lines at 0 in `scope`, which a log axis has no place for: drawn along its floor and named in its
    caption."""
    return {panel: [r["label"] for r in rows if (r["value"] if "value" in r else r["by_scope"].get(scope, 1)) <= 0]
            for panel, rows in refs.items()}


def captions(serve_hash_share: float, infiltration: dict | None = None, coverage: dict | None = None,
             boundary_hash_share: float = 0.0, zeros: dict[str, list[str]] | None = None,
             notes: dict[str, list[str]] | None = None, served: tuple[float, int, int, float] | None = None
             ) -> dict[str, str]:
    """Each panel's caption.  `served` is (S, tokens, bits a token, N, total) for the capped designs' sentences: the
    bits a request serves, its tokens at their bits, a request's proof blocks and the scope's served bits."""
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
                 "modeled at the grain they would take, so the provable share is "
                 f"{1 - coverage['uncovered_mac_share'] - coverage['modeled_mac_share']:.0%}.")
    two = (f"serving commits only {tokens}, and the sampled requests' interiors are committed after the first stage, "
           "both inside the budget")
    exact = ("A two-stage bound is the count the Lean bounds: every admissible set of incorrect proof blocks, each "
             "charged its own blocks' committed bits, so a wholly incorrect request is charged every bit it commits.")
    request = "Request-level (dotted) proves each sampled request whole; serving commits only the token ids."
    capped = patterns = ""
    if served:
        cap, count, bits, blocks, total = served
        capped = (f" Capped (dash-dot; {CAP_STATUS['sets']}): two-stage with each request charged at most the "
                  f"{cap:,.0f} bits it serves ({count:,} tokens of {bits} bits), so a wholly incorrect request is "
                  f"charged which of its {blocks:.2g} proof blocks are incorrect and what it serves, about "
                  f"{blocks + cap:.2g} bits; and never above the {total:.2g} bits the scope serves ({SERVED_STATUS}), "
                  "flat where that binds.")
        patterns = (f" Patterns (dash-dot-dot; {CAP_STATUS['patterns']}): a request's incorrect sets of more than "
                    f"{cap:,.0f} bits serve at most 2^{cap:,.0f} outputs together, so they count once, at the least "
                    f"escape among them; never above the scope's served bits either ({OUTS_STATUS}).")
    inf = infiltration["value"] if infiltration else None
    out = {
        "exfiltration": (f"{DEPLOYMENT['exfiltration']} {ROLES} {TRUST} Two-stage, width-stratified (solid): {two}; the second stage "
                         "samples a request's proof blocks of the public size classes up to a threshold together and "
                         "each wider class apart, at the threshold and rates with the least bound. "
                         f"{request} {exact}{capped}{fill}{gaps}"),
        "exfiltration_appendix": (f"{DEPLOYMENT['exfiltration']} {ROLES} {TRUST} Every design, with the reference lines the main "
                                  f"figure leaves out. Two-stage, width-stratified (solid) and uniform (thin, one "
                                  f"sample over a request's proof blocks): {two}. "
                                  f"One-stage (dashed) {serve}; its bound counts the incorrect proof blocks' output "
                                  f"bits, each the widest block's, and which ones they are. {request} {exact}{capped}"
                                  f"{patterns}{fill}"),
        "infiltration": inf["caption"] if inf else "Pending: the design that certifies infiltration, and its consumer.",
        "skipped_work": (f"{DEPLOYMENT['skipped_work']} {ROLES} {TRUST} One-stage (solid) {serve}. Request-level (dotted) proves each "
                         f"sampled request whole; serving commits only {tokens}, inside the budget. Two-stage is left "
                         f"out: its interiors are committed after the first stage, so they don't show the work was done "
                         f"at serve time.{fill}{gaps}"),
    }
    for panel, sentences in (notes or {}).items():
        out[panel] += "".join(f" {s}" for s in sentences)
    for panel, labels in (zeros or {}).items():
        if labels:
            out[panel] += f" At 0, drawn along the axis's floor: {'; '.join(labels)}."
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


def run(model_name: str = "GLM-5.3", points: int = 57, scope_names: tuple[str, ...] | None = None,
        request: str = "census", workers: int = 1, designs: tuple[str, ...] | None = None, lean: bool = True,
        serving: str = "assumed") -> dict:
    """Figure 1's JSON at `points` proof overheads besides the vertical line's, over the scopes and designs named
    (default all), with Lean checking every design unless `lean` is off: every point of every curve one job
    (`checked_point`) on `workers` processes (`processes`), the JSON the same at any number.  `serving` is where
    serving's rate comes from (`SERVING`); the measured reading's JSON says so (`serving`), the default's is as it was."""
    with processes(workers, lean) as each:
        return _run(model_name, points, scope_names, request, designs, each, lean, serving)


def _run(model_name: str, points: int, scope_names: tuple[str, ...] | None, request: str,
         designs: tuple[str, ...] | None, each, lean: bool, serving: str = "assumed") -> dict:
    planned = {panel: [d for d in ds if designs is None or d in designs] for panel, ds in DESIGNS.items()}
    sc = slot_cards(model_name, serving)
    c = cards(model_name, request, serving)
    part = TokenBlocks.of(c["partition"].value["strata"], c["partition"].value["served_macs_per_token"])
    fill = Fill.of(c["partition"].value["strata"], c["proof_blocks"].value)
    xs = overheads(c, points)
    budgets = c["budgets"].value
    and_per_flop = c["ands_per_mac"].value / 2
    refs = references(c, model_name, sc.get("references", []))
    notes = reference_notes(sc.get("references", []), refs)
    for panel, sentences in c["notes"].value.items():
        notes.setdefault(panel, []).extend(sentences)
    if serving == "measured":
        for panel in ("exfiltration", "exfiltration_appendix", "infiltration", "skipped_work"):
            notes.setdefault(panel, []).append(reading_note(reading(sc, c, part)))
    shown = {panel: [r for r in rows if r["main"]] for panel, rows in refs.items()}
    shown["exfiltration_appendix"] = [r for r in refs["exfiltration"] if not r["main"]]
    chosen = [s for s in scopes(c, part) if scope_names is None or s.name in scope_names]
    infiltration = infiltration_panel(sc.get("infiltration"), [s.name for s in chosen])
    capped = {d: {"form": TWO_STAGE[d][1], "status": CAP_STATUS[TWO_STAGE[d][1]]}
                 | {"served": SERVED_STATUS if TWO_STAGE[d][1] == "sets" else OUTS_STATUS}
              for d in planned["exfiltration"] if d in TWO_STAGE and TWO_STAGE[d][1]}
    out_scopes, series = {}, {"exfiltration": [], "skipped_work": []}
    models = {s.name: [model(c, part, s, x, fill) for x in xs] for s in chosen}
    curves = [(s, panel, target, design, beta) for s in chosen
              for panel, target in (("exfiltration", "leak"), ("skipped_work", "work"))
              for design in planned[panel] for beta in budgets]
    points = iter(list(each(checked_point, [(m, design, target, beta * s.gpu_seconds, lean)
                                            for s, _, target, design, beta in curves for m in models[s.name]])))
    for s, panel, target, design, beta in curves:
        pts = [next(points) for _ in xs]
        ys = [p["bound"] for p in pts]
        series[panel].append({
            "scope": s.name, "design": design, "budget": beta, "headline": design == HEADLINE[panel],
            "main": design in MAIN[panel], "x": xs, "y": ys,
            "capped": [p["capped"] for p in pts],
            "share": [y / cap(models[s.name][0], target) for y in ys],
            **({k: [p[k] for p in pts] for k in (("lean", "planner", "theta") if design in TWO_STAGE else ("lean",))}
               if lean else {}),
            "plan": [None if p["row"] is None else _plan(p["row"]) for p in pts]})
    for s in chosen:
        base = model(c, part, s, 1.0, fill)
        serve_hash = base.dot.units * base.dot.commit_bytes * base.byte_s / s.gpu_seconds
        boundary_hash = base.replay.replay_blocks * base.replay.boundary_bytes * base.byte_s / s.gpu_seconds
        ratio = infiltration_ratio(infiltration, series["exfiltration"], s.name, c["proof_overhead"].value)
        out_scopes[s.name] = {"who": s.who, "period": s.period, "gpus": c[f"{s.who}_gpus"].value,
                              "gpu_seconds": s.gpu_seconds, "requests": s.requests, "flop_request": s.flop_request,
                              "proof_blocks": base.dot.units, "output_bits": cap(base, "leak"),
                              "served_bits": served_total(base),
                              "flop": cap(base, "work"), "ands": cap(base, "work") * and_per_flop,
                              "serve_hash_share": serve_hash, "boundary_hash_share": boundary_hash,
                              "captions": captions(serve_hash, sc.get("infiltration"), part.coverage(),
                                                   boundary_hash, zero_lines(shown, s.name),
                                                   notes | ({"infiltration": [*notes.get("infiltration", []), ratio]}
                                                            if ratio else {}),
                                                   (base.served, c["tokens_per_request"].value,
                                                    round(base.served / c["tokens_per_request"].value),
                                                    base.blocks_per_request, served_total(base)) if capped else None)}
    return ({"serving": reading(sc, c, part)} if serving == "measured" else {}) | {
        "schema": SCHEMA, "model": model_name, "request": request,
        "lean": {"exe": plancheck.EXE, "slack": str(plancheck.SLACK), "search": plancheck.SEARCH,
                 "precision": plancheck.TWO_P} if lean else None,
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
                             "deployment": DEPLOYMENT["exfiltration"], "designs": planned["exfiltration"],
                             "main": [d for d in MAIN["exfiltration"] if d in planned["exfiltration"]],
                             "references": refs["exfiltration"], "capped": capped, "series": series["exfiltration"]},
            "infiltration": infiltration | {"deployment": DEPLOYMENT.get(infiltration.get("design"), ""),
                                            "references": refs["infiltration"]},
            "skipped_work": {"unit": "FLOP", "target": "work", "headline": HEADLINE["skipped_work"],
                             "deployment": DEPLOYMENT["skipped_work"], "designs": planned["skipped_work"],
                             "main": [d for d in MAIN["skipped_work"] if d in planned["skipped_work"]],
                             "and_per_flop": and_per_flop,
                             "references": refs["skipped_work"], "series": series["skipped_work"]}},
    }


def at_overhead(out: dict) -> list[dict]:
    """Each series' point at the vertical line: the figure's table, skipped work in AND gates too, with the design's
    serve-time hashing as a share of the scope's compute (outside the budget only under one-stage) and its fill.  An
    exfiltration row at the measured reading names the rate it read (`at_reading`)."""
    x = out["vertical"]["x"]
    rows = []
    for panel in ("exfiltration", "skipped_work"):
        p = out["panels"][panel]
        for s in p["series"]:
            i = s["x"].index(x)
            sc = out["scopes"][s["scope"]]
            one = s["design"] == "one-stage"
            rows.append({"panel": panel, "scope": s["scope"], "design": s["design"], "budget": s["budget"],
                         "main": s["main"], "bound": s["y"][i], "share": s["share"][i], "capped": s["capped"][i],
                         "serve_hash_share": sc["serve_hash_share" if one else "boundary_hash_share"],
                         "serve_hash_in_budget": not one, "fill": (s["plan"][i] or {}).get("fill")}
                        | ({"ands": s["y"][i] * p["and_per_flop"]} if "and_per_flop" in p else {})
                        | (at_reading(out.get("serving"), s["y"][i]) if panel == "exfiltration" else {}))
    return rows
