"""Figure 1 of the SPOT post in its total-cost form: what an adversary serving a model on all of a scope's compute over
a monitoring period can exfiltrate or infiltrate while evading detection with probability above δ, against what the
deployment costs as a share of serving compute, one line per server configuration; and, for the one configuration
that bounds it, skipped work.

A configuration (`CONFIGURATIONS`) is what serving commits at serve time, the replay block that follows from it and the
design the audit samples with.  Its serve tax, what that change to the production server costs on every token, is one
number in the model's `serve_tax` card, priced (`priced_card`) or measured.  A line's x is its tax plus β, the share
of the scope's compute the audit spends on replays, interior commitments and proofs of the sampled proof blocks, and
its y the bound at β on Figure 1's model (`figure1.model`).  So each line starts at its own tax.  Every line prices a
proof block the same way (`priced`): at its type's seconds on the `proof_overhead` card, or computed from its staged
witness bits where the card doesn't measure the type; a line's marginal overhead is what its blocks come to.

* `tokens` (A): serving commits the token ids, the schedule record and the KV cache at each replay block's edge; a
  sampled request is replayed with its whole batch from its replay block's edge, one machine's window, since the
  model's tokens come out bit-exact only in their served batch (`REPLAYS`), and the design is the two-stage one by
  template (`STRATA`).  `tokens_bi` replays a request alone, which needs batch-invariant kernels and pays their tax, the
  comparison.  Exfiltration is its patterns' form
  (`card_influenceTwo_le_patterns`), never above the scope's served bits (`card_influenceTwo_le_two_pow_outs`), at
  a verification phase of one machine's window and, as `tokens_pod` and `tokens_datacenter`, of a pod's and a
  datacenter's (`REGISTRATIONS`); `tokens_sets` is the capped sets' form (`card_influenceTwo_le_served_min`, at most
  the served bits by `card_influenceTwo_le_min`) at one machine's, the comparison.  Infiltration is its write form (`write_form`) at a
  footprint of a request's served bits: an incorrect request is charged at most the cheapest explanation of what it
  serves.
* `layer_of_token` (B): serving also commits each layer's residual stream, K/V and indexer keys; the audit samples a
  layer of one token, holding its share of the token's proof blocks, and its replay block is its engine step's whole
  batch at that layer.  Exfiltration is the planner's design on that grain in the capped sets' form, each sampled
  layer charged at most its request's served bits, which is at least the lemma's charge because min(·, S) is
  subadditive, and never above the scope's served bits.  Infiltration is its write form at a footprint of the layer's boundary bits.
* `per_expert` (A′): (A)'s serving, tax, forms and verification phases on another cut of the circuit, the model's
  `partition` card `per-expert` (the MoE cut per expert: the router's slots, table reads at each slot's expert and
  per-expert GEMMs), labelled with the card's `label`; lines only for a model with that card.
* `head_units` (A″): the same on the model's `partition` card `head-scores-values` (a sparse-attention head's key
  block proved as its scores unit and its values units on column blocks), labelled with the card's `label`; lines
  only for a model with that card.  Each unit commits its own row of the block's keys and no unit writes that row, so
  nothing holds two units' keys equal: its lines are a cost model's, not a bound, and every label says so.
* `every_block` (C): serving commits every proof block's outputs, and the audit samples them one-stage; its tax is a
  lower bound, the hashing alone.  Exfiltration is `consumers.exfiltration_bound`'s form, at most the served bits
  (`audit_exfiltration_delivered`); infiltration is its write form (the K incorrect blocks the plan detects, each the
  widest block's bits) and skipped work the work target's plan.  Lean checks each of its designs (`figure1.lean_check`,
  unless `run`'s `lean` is off): a point's `lean` says it did, its exfiltration's count is the bits Lean certifies
  (`plancheck`), and a refused design is an error.  A point without a design (the trivial bound, or a count of every
  proof block, which samples none) has `lean` None.

Lean checks each two-stage point too (`figure1.lean_two`, unless `lean` is off): its count is the bits Lean certifies
(`TwoStageCheckPatternsSound` in the patterns' form, `TwoStageCheckServedSound` in the capped sets') at the θ Lean
finds from the planner's, with the planner's bits (`planner`) and Lean's θ beside them, and a refused point is an
error.

Each configuration's two panels read one design, the one with the least exfiltration bound at β; skipped work is
planned on its own.  The serve tax is outside β; the token ids' hashing is in the tax, not in β, for every line.  A
two-stage line's first stage is the one the verifier runs: a verification phase (`REGISTRATIONS`) samples a whole count
K of its requests, at least one (`optimize.two_stage_design`'s `registration`), so each point is priced and bounded
at K over the phase's requests, and a β that does not pay for one sampled request a verification phase has no design:
the line is at the scope's served bits there, with what one sampled request costs against the phase's budget
(`sample_budgets`).  Its second stage's strata are the ones the verifier's two-stage rule registers (`STRATA`): each
block type of the cut samples its own whole count, at least one, of a sampled request's proof blocks, and `counts`
writes a point's design as that rule takes it.  A two-stage statement holds one Definition's sampled proof blocks of
a sampled request, or of the verification phase's K sampled requests (`optimize.two_stage_statements`), each
Definition taking whichever costs less, and no statement is larger than its type's on the `proof_overhead` card.  The
verifier's second stage reads one session of one table for each template of a verification phase (`Spot.SecondStage`;
a table holds at most `STATEMENT_BITS`, `table_blocks`), so a point with more `sessions` than block types prices
sessions it refuses: each point carries the count, and at β = `RATIO_BUDGET` the designs beside it, each modeled
(`session_designs`): the two with one session a block type, and those with sessions of at most 1, 16, 64 and 200
tables (`TABLES`; a session of the proof system holds at most `SESSION_TABLES`).  A design that meets a limit of the
code that runs a second stage today (`limits`: a session's tables, the challenger's release log, the second draw's
live coins) is printed as out of reach, with the first limit it meets.  `breakeven` gives each patterns'-form
line's marginal overhead and β at which its bound reaches the weights, at FP8 and FP4, each certified by Lean unless
`lean` is off (`certify`: at the β it gives, the bits Lean certifies for the planner's design are at most the weights).
Only that side is Lean's: Lean bounds a design's bits from above, so nothing certifies that a smaller β can't reach the
weights; and what a design costs, so which design fits a β, is the planner's pricing of the proof_overhead card.

    uv run python benchmarks/spot_figures/total_cost.py priced-card --model GLM-5.3 --out CARD.json
    uv run python benchmarks/spot_figures/total_cost.py render DIR/figure1.json [--scope "world year"] [--out DIR]
    uv run python benchmarks/spot_figures/total_cost.py breakeven --model GLM-5.3 --out DIR [--scope S] [--workers N] \
        [--no-lean]
    uv run python benchmarks/spot_figures/total_cost.py counts --line tokens_datacenter --beta 0.1 --out COUNTS.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import ceil, expm1, floor, fsum, inf, log, log1p, log10
from pathlib import Path
from statistics import linear_regression, median
from textwrap import fill

import cards as spot_cards
import figure1 as f1
from verity.protocols.spot import plancheck
from verity.protocols.spot.device import challenger
from verity.protocols.spot.optimize import (Cell, leak_count, two_stage_counts, two_stage_design, two_stage_exfil,
                                            two_stage_price, two_stage_statements)

SCHEMA = "spot-figure1/v2"
#: β, the share of compute for replays and proofs, from 0.01% to 1,000% at four points a decade, and the design card's:
#: past 100% so that (C), whose tax alone is several times serving, has a line
BETAS = tuple(10.0 ** (e / 4) for e in range(-16, 5))
PANELS = ("exfiltration", "infiltration", "skipped_work")
#: the throughput the certified-tick rule keeps against vLLM, with a queue-dependent tick (0.9× the step while the
#: queue stands, 1.1× otherwise): the mean run's and the worst run's, simulated on Llama-3.2-1B traces
TICKS = (0.984, 0.960)
#: a request's lifetime over the 600 s window, the share of its K/V alive at an edge: the central, least and most
LIVE = (0.15, 0.05, 0.5)
#: SHA-512 at 3.1 and 4.79 G compressions a second, 44 a 4 KiB row (`benchmarks/prove_path/window_cost.py`'s `RATES`)
HASH_RATES = (4096 / 44 * 4.79e9, 4096 / 44 * 3.1e9)
#: the pause at a replay block's edge, seconds, measured on GLM-5.3 at TP8 over 21 edges (rank 0's copy of its 37.26 GB
#: KV pool to pinned host memory, each rank's SHA-512 rows of its own, the runner rows and the scheduler state): the
#: median and the most
EDGE_PAUSE = (0.98, 1.06)
#: the same server's throughput tax against stock compiled vLLM with the step record, the token commits and an edge each
#: 30 s, measured within noise at every load: the 95% interval's top at saturation, which 3 reps can't rule out
SERVED_HIGH = 0.024
#: one engine across eight H200s serving GLM-5.3 under the chat load at 384 requests at once, stock compiled vLLM:
#: seconds an engine step, the rows it schedules and the requests it finishes a second
ENGINE_STEP, STEP_ROWS, REQUEST_RATE = 0.1467, 143, 4.897
#: the scheduler's step record on the engine core's thread, seconds a step: the serving path's writer timed off the
#: engine at `STEP_ROWS` rows, and at a full batch of 256
STEP_RECORD = (98e-6, 173e-6)
#: the two-stage boundary commitment's hooks on the engine core's thread, timed off the engine: seconds a step at
#: `STEP_ROWS` rows, and seconds a request admitted
BOUNDARY_HOOKS = (11.2e-6, 4.4e-6)
#: its commit thread's rows, salts, hashes and files for one finished request, seconds, with its class's replay block's
#: program already digested: the least and the most timed
BOUNDARY_COMMIT = (0.41e-3, 1.09e-3)
#: the bytes the card hashes for one row of a history table: the row's 576 16-bit words, its 192-byte salt and the
#: tree node above its leaf
HISTORY_ROW_BYTES = 2 * 576 + 192 + 128
#: a history table's rows a computed token: a request's table has a row for each prompt and output position of its
#: class, each count the least power of two that holds the request's own, so from one to two: the chat load's mean,
#: the least and the most
HISTORY_FILL = (1.4, 1.0, 2.0)
#: the measured serving runs of `SERVED_HIGH` and `EDGE_PAUSE`, and of `ENGINE_STEP`, `STEP_ROWS` and `REQUEST_RATE`
SERVED_RUNS = ["r20261107-155724-3f53", "r20261111-162328-f498"]
#: the deterministic DSA top-k that makes GLM-5.3's replay bit-exact, a share of serving (kernel-modeled)
TOPK_SHARE = 0.0025
#: the serve tax's part a server pays only if it snapshots at each replay block's edge
EDGE = "token commits and replay-block edge snapshot"
#: batch-invariant kernels' throughput tax against stock compiled vLLM at saturation on one H200: the MoE's
#: (Qwen3-30B-A3B), the least (Moonlight, dense MLA) and the most (Llama-3.1-8B, dense); GLM-5.3's sparse MLA has no
#: batch-invariant backend on the H200
BATCH_INVARIANT = (0.237, 0.141, 0.311)
#: the witness bits one table of the proof system holds, its limit (Flock's `M_MAX`), at which a full statement of the
#: `proof_overhead` card is proved
STATEMENT_BITS = 2 ** 35


def table_blocks(k_log: int) -> int:
    """The proof blocks of 2^`k_log` witness bits one table holds (`STATEMENT_BITS`): 32,768 at 2^20 bits, 256 at
    2^27.  The verifier's second stage reads one session of one table a template (`Spot.SecondStage`,
    `second-tables`)."""
    return max(1, STATEMENT_BITS >> k_log)


@dataclass(frozen=True)
class Configuration:
    """A line: its label, what serving commits, its replay block and design, and the panels it bounds; the `partition`
    card it cuts the circuit by (else the `workload` card's cut) and the configuration whose serve tax it pays (else
    its own).  A two-stage line charges each request in the capped sets' or patterns' `form`; its `registration` is
    its verification phase's size (`REGISTRATIONS`), of which the first stage samples a whole count and within which
    statements are shared; a variant of another line (`variant_of`) differs from it only there.
    Its `replay` is what a sampled replay block re-runs (`REPLAYS`)."""

    label: str
    commits: str
    replay_block: str
    design: str
    panels: tuple[str, ...]
    partition: str | None = None
    serve: str | None = None
    form: str | None = None
    registration: str | None = None
    variant_of: str | None = None
    replay: str = "block"


#: what replaying a sampled replay block re-runs: without batch-invariant kernels a request's tokens come out right only
#: in their served batch, so a sampled request replays one machine's whole replay block from its edge ("block") and a
#: sampled layer of one token its engine step's whole batch at that layer, the requests in flight ("step"); with them, a
#: request replays alone ("request")
REPLAYS = ("block", "step", "request")


#: a 64-machine pod: one committer gathering its machines' boundary rows into one committed trace
POD = 64
#: a verification phase's size, the machines whose census window one committed trace and its sample cover: one
#: machine, a pod or every machine of the census datacenter.  The first stage samples a whole count of its requests,
#: and they fill each Definition's statements together
REGISTRATIONS = {"machine": "one machine's", "pod": f"a {POD}-machine pod's", "datacenter": "a datacenter's"}
#: a two-stage line's second-stage strata (`optimize.two_stage_design`'s `stratify`): one a block type of the cut, the
#: strata the verifier's two-stage rule registers, one whole count a template (`Spot.Select.Counts`).  A stratum of
#: several types, the planner's width-stratified design, samples its count among them, which that rule cannot state
STRATA = "template"
#: a two-stage point's verification phase: the requests it samples (K, 0 where no design is within the budget), the
#: shares of proving in its statements and on their padding, its sessions, and what one sampled request costs over
#: the phase's budget (`sample_budgets`: under the point's design at K = 1, so the design fits a phase that many
#: times this one's size; with no design, under the least one, a single proof block of each type proved)
REG = ("registration_sample", "shared", "padding", "sessions", "sample_budgets")


def registration_gpu_seconds(name: str, c: dict[str, f1.InputCard]) -> float:
    """A verification phase's GPU-seconds: its machines' GPUs over the census window."""
    machines = {"machine": 1, "pod": POD, "datacenter": c["datacenter_machines"].value}[name]
    return machines * c["machine_gpus"].value * c["window"].value


def _spans(name: str, line: Configuration) -> dict[str, Configuration]:
    """A request-grain line at every size of verification phase in the patterns' form, and at its own in the capped
    sets' form, the comparison, for exfiltration only."""
    other = {f"{name}_{r}": replace(line, registration=r, serve=line.serve or name, variant_of=name)
             for r in REGISTRATIONS if r != line.registration}
    return {name: line} | other | {f"{name}_sets": replace(line, form="sets", panels=("exfiltration",),
                                                           serve=line.serve or name, variant_of=name)}


TOKENS = "the token ids, the schedule record and the KV cache at replay-block edges"
REQUEST = "one machine's 600 s, its whole batch re-run from its edge for each sampled request"
CONFIGURATIONS = {
    **_spans("tokens", Configuration("(A) Tokens only", TOKENS, REQUEST, "two-stage, by template",
                                     ("exfiltration", "infiltration"), form="patterns", registration="machine")),
    "tokens_bi": Configuration("(A) Tokens only, batch-invariant kernels", TOKENS, "a request, replayed alone",
                               "two-stage, by template", ("exfiltration",), form="patterns",
                               registration="machine", variant_of="tokens", replay="request"),
    "layer_of_token": Configuration("(B) Each layer's state",
                                    "the tokens and each layer's residual stream, K/V and indexer keys",
                                    "one engine step's whole batch at one layer, re-run for each sampled layer of one token",
                                    "two-stage, by template", ("exfiltration", "infiltration"), form="sets",
                                    registration="machine", replay="step"),
    **_spans("per_expert", Configuration("(A′) Tokens only, the MoE cut per expert", TOKENS, REQUEST,
                                         "two-stage, by template", ("exfiltration", "infiltration"),
                                         partition="per-expert", serve="tokens", form="patterns",
                                         registration="machine")),
    **_spans("head_units", Configuration("(A″) Tokens only, a head's key block as its scores and values units (modeled, "
                                         "key rows not held equal)", TOKENS, REQUEST, "two-stage, by template",
                                         ("exfiltration", "infiltration"), partition="head-scores-values",
                                         serve="tokens", form="patterns", registration="machine")),
    "every_block": Configuration("(C) Every proof-block boundary", "every proof block's outputs", "a proof block",
                                 "one-stage", PANELS),
}


def priced_card(model: str = "GLM-5.3", serving: str = "assumed") -> dict:
    """The model's `serve_tax` card priced from its parts: each part's GPU-seconds a token over a token's at serving's
    MFU (under `serving="measured"`, at the `serving_rate` card's rate: `figure1.token_seconds`), hashing at the
    `commitment` card's rate (and at `HASH_RATES` for the range), the tick rule's throughput loss
    (`TICKS`), the step record's and the two-stage boundary commitment's time on the engine core's thread over an
    engine step (`STEP_RECORD`, `BOUNDARY_HOOKS`, `ENGINE_STEP`), the history tables' hashing (`HISTORY_ROW_BYTES`), the
    pause at a replay block's edge (`EDGE_PAUSE`), the deterministic top-k (`TOPK_SHARE`) and, for the batch-invariant
    comparison, `BATCH_INVARIANT`.  Each part says how it got its share (`cards.PART_STATUS`), and a configuration's
    tax is the sum of its parts'.  (C)'s is a lower bound: hashing alone."""
    c = f1.cards(model, serving=serving)
    sc = f1.slot_cards(model)
    token_s = f1.token_seconds(c, c["partition"].value["served_macs_per_token"])
    rate = c["hash_rate"].value

    def hashed(part: str, nbytes: float, source: str, scale: tuple[float, float, float] = (1, 1, 1)) -> dict:
        share = [nbytes * s / r / token_s for s, r in zip(scale, (rate, *HASH_RATES))]
        return {"part": part, "status": "modeled", "share": share[0], "low": share[1], "high": share[2],
                "source": f"{nbytes:,.0f} B a token hashed at the commitment card's rate, the range at "
                          f"{HASH_RATES[0] / 1e9:.0f} and {HASH_RATES[1] / 1e9:.0f} GB/s; {source}"}

    load = (f"one engine across eight H200s under the chat load at 384 requests at once ({ENGINE_STEP * 1e3:.0f} ms a "
            f"step, {STEP_ROWS} rows a step, {REQUEST_RATE:.1f} requests finished a second, measured on stock compiled "
            "vLLM)")
    record = {"part": "step record", "status": "modeled", "share": STEP_RECORD[0] / ENGINE_STEP, "low": 0.0,
              "high": STEP_RECORD[1] / ENGINE_STEP,
              "source": "the scheduler's record of each step, written on the engine core's thread: "
                        f"{STEP_RECORD[0] * 1e6:.0f} µs a step, timed on the serving path's writer off the engine, over "
                        f"the step of {load}, as if it lengthened every step; the high at a full batch of 256 rows, "
                        f"{STEP_RECORD[1] * 1e6:.0f} µs; the low 0, vLLM's queue of two batches letting the thread "
                        f"write while the cards run; with the record on, {model}'s throughput at TP8 was measured "
                        "within noise of stock compiled vLLM's at every load"}
    hooks = BOUNDARY_HOOKS[0] / ENGINE_STEP + BOUNDARY_HOOKS[1] * REQUEST_RATE
    boundary = {"part": "two-stage boundary commitment", "status": "modeled", "share": hooks, "low": 0.0,
                "high": hooks + BOUNDARY_COMMIT[1] * REQUEST_RATE,
                "source": f"its hooks on the engine core's thread, {BOUNDARY_HOOKS[0] * 1e6:.1f} µs a step and "
                          f"{BOUNDARY_HOOKS[1] * 1e6:.1f} µs a request admitted, timed off the engine, on {load}, as if "
                          "they lengthened every step; the high adds its commit thread's rows, salts, hashes and "
                          f"files, {BOUNDARY_COMMIT[0] * 1e3:.2f} to {BOUNDARY_COMMIT[1] * 1e3:.2f} ms a finished "
                          "request, as if each second of that thread, which shares the engine core's interpreter, "
                          "were taken from serving; the low 0; priced with each size class's replay block's program "
                          "digested once: a commit thread that builds the program again for every window and class "
                          "costs more, and that is not in this part"}
    fill = HISTORY_FILL
    history = hashed("history tables", HISTORY_ROW_BYTES * fill[0],
                     f"one attention layer's table for each request, {HISTORY_ROW_BYTES:,} B hashed on the card a row "
                     f"(the row's {2 * 576:,} B, its 192 B salt and the tree node above its leaf) at {fill[0]:g} rows "
                     f"a computed token on the chat load's size classes, {fill[1]:g} to {fill[2]:g} over the range; "
                     "the hashing alone, without the rows' gather from the layer's state and their copy to the host; "
                     "no served run has built a table", (1, fill[1] / fill[0], fill[2] / fill[0]))
    served = [record, boundary, history]
    ticks = {"part": "certified-tick rule", "status": "simulated", "share": 1 / TICKS[0] - 1, "low": 1 / TICKS[0] - 1,
             "high": 1 / TICKS[1] - 1,
             "source": f"throughput {TICKS[0]} of vLLM's on the mean run and {TICKS[1]} on the worst, a queue-dependent "
                       "tick on Llama-3.2-1B traces, simulated and validated against them; on GLM-5.3 at TP8, with "
                       "certified arrivals and prefix caching off, it leaves no scheduling advice but the response "
                       "length, which stays advice until a circuit checks the stop rule; the run with the real network "
                       "certifier is pending (a client stood in for it in the measured runs, which served with prefix "
                       "caching on)"}
    tokens = hashed("token ids", c["token_bytes"].value, "the design card's token_bytes")
    window = c["window"].value
    edge = {"part": EDGE, "status": "measured",
            "share": EDGE_PAUSE[0] / window, "low": EDGE_PAUSE[0] / window,
            "high": max(EDGE_PAUSE[1] / window, SERVED_HIGH),
            "source": f"measured on {model} at TP8 against stock compiled vLLM: the serving pause at each {window:g} s "
                      f"replay block's edge to copy and hash the KV pool, {EDGE_PAUSE[0]:.2f} s at the median and "
                      f"{EDGE_PAUSE[1]:.2f} s at most; the throughput with the step record, the token commits and an "
                      f"edge each 30 s within noise at every load, up to {SERVED_HIGH:.1%} at saturation"}
    topk = {"part": "deterministic top-k", "status": "modeled", "share": TOPK_SHARE, "low": TOPK_SHARE,
            "high": TOPK_SHARE,
            "source": "the canonical DSA top-k that makes GLM-5.3's replay bit-exact at TP8 mid-traffic, "
                      "kernel-modeled"}
    invariant = {"part": "batch-invariant kernels", "status": "measured", "share": BATCH_INVARIANT[0],
                 "low": BATCH_INVARIANT[1], "high": BATCH_INVARIANT[2],
                 "source": "throughput against stock compiled vLLM at saturation on one H200, measured on these "
                           f"models: Qwen3-30B-A3B (MoE) {BATCH_INVARIANT[0]:.1%}, Moonlight (dense MLA) "
                           f"{BATCH_INVARIANT[1]:.1%}, Llama-3.1-8B (dense) {BATCH_INVARIANT[2]:.1%}; unavailable for "
                           "GLM-5.3, whose sparse MLA has no batch-invariant backend on the H200"}
    layers = hashed("each layer's state", sc["infiltration"]["value"]["boundary_bytes_per_token"],
                    "the infiltration card's boundary_bytes_per_token")
    blocks = hashed("every proof block's outputs", sc["workload"]["value"]["boundaries"]["every_block"],
                    "the workload card's every_block boundary")

    def entry(config: str, parts: list[dict], lower_bound: bool = False) -> dict:
        return {"config": config, "status": "priced", "tax": sum(p["share"] for p in parts),
                "low": sum(p["low"] for p in parts), "high": sum(p["high"] for p in parts), "lower_bound": lower_bound,
                "parts": parts} | ({"runs": SERVED_RUNS} if edge in parts else {})

    return spot_cards.check({
        "schema": spot_cards.SCHEMA, "slot": "serve_tax", "model": model, "status": "modeled", "provisional": True,
        "source": "priced from each configuration's parts against a token's GPU-seconds at "
                  + ("the serving_rate card's measured tokens a GPU-second" if "serving_rate" in c
                     else "serving's MFU on the H200's dense FP8 peak")
                  + " (total_cost.priced_card); measured values replace it configuration by configuration",
        "runs": SERVED_RUNS, "owner": "@proofs",
        "value": {"baseline": "stock vLLM, compiled, on the model's served circuit",
                  "configurations": [entry("tokens", [ticks, *served, edge, topk]),
                                     entry("tokens_bi", [ticks, *served, edge, topk, invariant]),
                                     entry("layer_of_token", [ticks, tokens, layers, topk]),
                                     entry("every_block", [ticks, blocks], lower_bound=True)]}})


def taxes(card: dict, names: tuple[str, ...] = tuple(CONFIGURATIONS)) -> dict[str, dict]:
    """Each named configuration's tax from the `serve_tax` card, every configuration that pays its own given."""
    out = {t["config"]: t for t in card["value"]["configurations"]}
    if missing := [n for n in CONFIGURATIONS if CONFIGURATIONS[n].serve is None and n not in out]:
        raise ValueError(f"the serve_tax card gives no tax for {missing}: each configuration's line starts at its own")
    return {n: out[CONFIGURATIONS[n].serve or n] for n in names}


def cut(c: dict[str, f1.InputCard], card: dict) -> dict[str, f1.InputCard]:
    """The inputs with the circuit cut by a `partition` card: its strata in place of the `workload` card's of the kinds
    it replaces."""
    v, part = card["value"], c["partition"]
    strata = [x for x in part.value["strata"] if x.get("kind") not in v["replaces"]] + v["strata"]
    art = f" ({card['art']})" if card.get("art") else ""
    return c | {"partition": replace(part, value=part.value | {"strata": strata},
                                     source=f"{part.source}; cut by the partition card {v['name']}{art}")}


def verifier_seconds(v: dict, block: str) -> tuple[float, str]:
    """The verifier's core-seconds a statement of `block`'s type from a `verifier_cost` card's value, and how they are
    known: its measured type's, its family's line through its members (or through the fixed seconds at size 0 and its
    one member; the member's own where the block gives no size), else the measured types' median."""
    by = {t["block"]: t["core_seconds"] for t in v["types"]}
    for prefix, seconds in by.items():
        if block.startswith(prefix):
            return seconds, "measured"
    for fam in v["families"]:
        if block.startswith(tuple(fam["types"])):
            size = fam["size"] and re.search(rf"[{{,]{fam['size']}=(\d+)", block)
            if not size:
                return by[fam["members"][0]], "family"
            pts = [(int(re.search(rf"[{{,]{fam['size']}=(\d+)", m)[1]), by[m]) for m in fam["members"]]
            (x0, y0), (x1, y1) = pts if len(pts) == 2 else [(0, v["fixed_seconds"]), *pts]
            return y0 + (int(size[1]) - x0) * (y1 - y0) / (x1 - x0), "family"
    return median(by.values()), "assumed"


def _verify_sentence(inputs: dict) -> str:
    """The caption's sentence on the verifier's share of β, empty where the inputs leave it out."""
    if "verifier_cost" not in inputs:
        return " β counts proving only: the verifier's CPU is not in it."
    seconds = [x["verify_core_seconds"] for x in inputs["partition"]["value"]["strata"] if x["per_token"] > 0]
    return (f" β also pays the verifier: each statement's verify, its type's core-seconds on one core "
            f"({min(seconds):,.0f} s to {max(seconds):,.0f} s), at {inputs['verifier_cost']['value']:.3g} "
            "core-seconds a GPU-second, adds to its fixed seconds, so total compute is serve tax, proving and "
            "verifying.")


def verified(c: dict[str, f1.InputCard], card: dict) -> dict[str, f1.InputCard]:
    """The priced inputs with each statement's verify, its type's core-seconds (`verifier_seconds`) at the card's
    core-seconds a GPU-second, added to the statement's fixed GPU-seconds."""
    v, part = card["value"], c["partition"]
    strata = []
    for x in part.value["strata"]:
        seconds, how = verifier_seconds(v, x["block"])
        strata.append(x | {"fixed_seconds": x["fixed_seconds"] + seconds / v["core_seconds_per_gpu_second"],
                           "verify_core_seconds": seconds, "verify_priced": how})
    art = f" ({card['art']})" if card.get("art") else ""
    return c | {"partition": replace(part, value=part.value | {"strata": strata}),
                "verifier_cost": f1.InputCard(v["core_seconds_per_gpu_second"], "core-s/GPU-s", card["status"],
                                              f"the verifier_cost card{art}: {card['source']}", card["provisional"])}


def _definition(block: str) -> tuple[str, int | None]:
    k = re.search(r"[{,]K=(\d+)", block)
    return block.split("{")[0], int(k.group(1)) if k else None


def priced(c: dict[str, f1.InputCard], card: dict) -> dict[str, f1.InputCard]:
    """The inputs with every proof block priced in GPU-seconds from the `proof_overhead` card, and the marginal
    overhead they come to.  A stratum whose type the card measures (its block, else the card's block of the same
    Definition and K) takes its seconds a block, its statement's fixed seconds and its cap.  One the card measures only
    at another size, 2^Δk its block's, takes 2^Δk its seconds and 2^-Δk its cap: Δk compares the two k_logs as the
    strata estimate them, the card's block's from the strata where they hold one of its type.  A type the card doesn't
    measure is `computed` from its staged witness bits: the measured types' median seconds a witness bit, a statement's
    fixed seconds on their least-squares line in k_log, and as many blocks as a full statement holds.  Each stratum's
    `price` is the MACs that cost its seconds at that overhead, so the planner's cells and grains prove it at them.
    `table_blocks` is the stratum's proof blocks one table holds, by the witness bits its price is of: the card's
    k_log of the block that prices it, moved by the same Δk, never the strata's own estimate, which the card's
    measured staging can differ from; a computed type's by its own k_log."""
    v, part = card["value"], c["partition"]
    if abs(c["peak_flops"].value / (2 * v["peak_mac_s"]) - 1) > 1e-3:
        raise ValueError(f"the planner's peak, {c['peak_flops'].value:.6g} FLOP/s, isn't twice the proof_overhead "
                         f"card's {v['peak_mac_s']:.6g} MAC/s, so its overheads don't price the planner's MACs")
    by_name = {b["block"]: b for b in v["blocks"]}
    types = {_definition(b["block"]): b for b in v["blocks"]}
    rate = median(b["seconds_per_block"] / 2 ** b["k_log"] for b in types.values())
    slope, intercept = linear_regression([b["k_log"] for b in types.values()],
                                         [b["fixed_seconds"] for b in types.values()])
    k_of = {x["block"]: x["k_log"] for x in part.value["strata"] if "k_log" in x}
    held = {_definition(b["block"]): (b, k_of[b["block"]]) for b in v["blocks"] if b["block"] in k_of}
    strata = []
    for x in part.value["strata"]:
        if b := by_name.get(x["block"]):
            strata.append(x | {k: b[k] for k in ("seconds_per_block", "fixed_seconds", "statement_blocks")}
                          | {"priced": "measured", "table_blocks": table_blocks(b["k_log"])})
        elif b := types.get(_definition(x["block"])):
            b, k = held.get(_definition(x["block"]), (b, b["k_log"]))
            dk = x["k_log"] - k if "k_log" in x else 0
            cap = b["statement_blocks"] << -dk if dk < 0 else max(1, b["statement_blocks"] >> dk)
            strata.append(x | {"seconds_per_block": b["seconds_per_block"] * 2.0 ** dk, "fixed_seconds": b["fixed_seconds"],
                               "statement_blocks": cap, "priced": "measured",
                               "table_blocks": table_blocks(b["k_log"] + dk)})
        elif "k_log" in x:
            strata.append(x | {"seconds_per_block": rate * 2 ** x["k_log"],
                               "fixed_seconds": max(0.0, intercept + slope * x["k_log"]),
                               "statement_blocks": table_blocks(x["k_log"]), "priced": "computed",
                               "table_blocks": table_blocks(x["k_log"])})
        else:
            raise ValueError(f"{x['block']}: the proof_overhead card doesn't measure its type, and it gives no k_log "
                             "to compute its proving from")
    seconds = sum(x["per_token"] * x["seconds_per_block"] for x in strata)
    h = seconds * v["peak_mac_s"] / sum(x["per_token"] * x["macs"] for x in strata)
    computed = sum(x["per_token"] * x["seconds_per_block"] for x in strata if x["priced"] == "computed") / seconds
    art = f" ({card['art']})" if card.get("art") else ""
    return c | {"partition": replace(part, value=part.value | {
                    "strata": [x | {"price_macs": x["seconds_per_block"] * v["peak_mac_s"] / h} for x in strata],
                    "computed_share": computed}),
                "proof_overhead": f1.InputCard(h, "x", card["status"],
                                               f"the proof_overhead card's blocks{art} by stratum, {computed:.2%} of "
                                               "the marginal computed from staged witness bits", card["provisional"])}


def _hazard(q: float, y: float) -> float:
    """h(y) = −ln(1 − q(1 − e^{−y})): a replay block's share of Λ once its incorrect blocks have rate y."""
    return -log1p(-q * -expm1(-y)) if y < inf else -log1p(-q) if q < 1 else inf


def write_form(cells: list[Cell], row: dict, replay_blocks: float, delta: float, footprint: float) -> float:
    """At least the most bits the incorrect proof blocks of an admissible set write under `row`'s two-stage design
    (`optimize.two_stage_design`'s: its strata, `figure1.split_cells`'s, and `k` by stratum, its `q`), each replay block charged at
    most `footprint` and its own blocks' bits.  An admissible set has Σ_ρ h(y_ρ) < Λ = ln(1/δ), y_ρ = Σ k_t/n_t over its
    blocks in ρ (`two_stage_exfil`'s acceptance), so the charge is at most Λ times the most a replay block charges a
    unit of h, and at most R·F.  A replay block charged c has y at least the least k_t/n_t of a block of at least c bits,
    or c times the least k_t/(n_t b) of a narrower one; c/h(y) only rises between the cells' bits, so its most is at one
    of them or at F."""
    split = f1.split_cells(cells, row["merged"])
    n: dict[int, float] = {}
    for c in split:
        n[c.stratum] = n.get(c.stratum, 0.0) + c.n
    rates = [(c.bits, inf if row["k"][c.stratum] >= n[c.stratum] else row["k"][c.stratum] / n[c.stratum])
             for c in split]
    top = min(footprint, sum(c.n * c.bits for c in split))
    best = 0.0
    for charge in {b for b, _ in rates if b < top} | {top}:
        wide = min((r for b, r in rates if b >= charge), default=inf)
        narrow = charge * min((r / b for b, r in rates if b < charge), default=inf)
        best = max(best, charge / _hazard(row["q"], min(wide, narrow)))
    return min(replay_blocks * top, log(1 / delta) * best)


def computed_proving(cells: list[Cell], row: dict, computed: tuple[bool, ...]) -> float:
    """The share of a sampled replay block's proving under `row`'s two-stage design, its statements' fixed costs
    included, that is in the cells `computed` marks: the types `priced` computes rather than measures.  A design that
    samples a few blocks of a type pays its statement's fixed cost on them, so this can be far above the type's share
    of the marginal."""
    split = f1.split_cells(cells, row["merged"])
    at = (row["k"], row["registration_sample"])
    whole = two_stage_price(split, *at)[2]
    measured = two_stage_price([replace(c, prove=0.0, fixed=0.0) if x else c
                                for c, x in zip(split, computed, strict=True)], *at)[2]
    return 1 - measured / whole


@dataclass(frozen=True)
class Grain:
    """A two-stage configuration's replay blocks on one scope: their count, cells, replay cost and footprint."""

    replay_blocks: float
    cells: tuple[Cell, ...]
    replay: float
    footprint: float


def batch(conf: Configuration, s: f1.Scope, c: dict[str, f1.InputCard]) -> float:
    """The requests a sampled replay block's replay re-runs (`REPLAYS`): one machine's replay block's, its share of the
    scope's requests; the requests in flight in an engine step, a request's lifetime over the window (`LIVE`) of
    those; or the request alone."""
    per_block = s.requests / s.gpu_seconds * registration_gpu_seconds("machine", c)
    return {"block": per_block, "step": per_block * LIVE[0], "request": 1.0}[conf.replay]


def grain(m: f1.Model, name: str, layers: float, boundary_bits: float, tokens: float, replays: float = 1.0) -> Grain:
    """A configuration's grain, its replay `replays` times its own replay block's."""
    if name != "layer_of_token":
        return Grain(m.replay.replay_blocks, m.cells, m.replay.cost * replays, m.served)
    per = tokens * layers
    return Grain(round(m.scope.requests * per), tuple(replace(c, n=c.n / per) for c in m.cells),
                 m.replay.cost * replays / per, boundary_bits)


def _plan(row: dict | None) -> dict | None:
    return None if row is None else f1._plan(row)


def _point(job: tuple) -> dict:
    """One configuration's bounds at one budget, in GPU-seconds: each panel's, with the plan and, for the capped
    exfiltration, which of the count and the served bits binds, the requests a verification phase samples and what one
    costs (`REG`), and the computed share of its design's proving.  A two-stage line's `form` and its verification
    phase's replay blocks (`span`) come with it; with `lean`, Lean's check of each line's designs
    (`figure1.lean_check`, `figure1.lean_two`).  Where the budget pays for no sampled replay block of a verification
    phase there is no design: the point is the trivial one, every replay block charged whole, with nothing for Lean to
    check."""
    name, m, g, budget, computed, form, span, lean, *rest = job
    beside = bool(rest and rest[0])
    served = f1.served_total(m)
    if name == "every_block":
        total_bits = m.scope.requests * sum(c.n * c.bits for c in m.cells)
        leak = f1.point(m, "one-stage", "leak", budget, top=max(total_bits, f1.cap(m, "leak")))
        work = f1.point(m, "one-stage", "work", budget)
        written = total_bits if leak["row"] is None else min(m.dot.out_bits * leak_count(m.dot, leak["bound"]),
                                                             total_bits)
        count, checked = leak["bound"], {"leak": {}, "work": {}}
        if lean:
            (on_leak, count), (on_work, _) = f1.lean_check([(m, "one-stage", "leak", leak),
                                                             (m, "one-stage", "work", work)])
            checked = {"leak": {"lean": on_leak}, "work": {"lean": on_work}}
        return {"exfiltration": {"y": min(count, served), "count": count,
                                 "binds": "served" if served <= count else "count", "plan": _plan(leak["row"]),
                                 **checked["leak"]},
                "infiltration": {"y": written, "plan": _plan(leak["row"]), **checked["leak"]},
                "skipped_work": {"y": work["bound"], "binds": "trivial" if work["capped"] else "count",
                                 "plan": _plan(work["row"]), **checked["work"]}}
    cells = list(g.cells)
    row = two_stage_design(cells, g.replay_blocks, m.delta, budget=budget, replay=g.replay, h=m.byte_s,
                           stratify=STRATA, served=m.served, union=form == "patterns", registration=span)
    one = g.replay + sum(x.n * x.commit_bytes for x in cells) * m.byte_s
    phase = budget / g.replay_blocks * span
    forms = {"forms": session_designs(cells, g, m, budget, form, span, served)} if beside and span else {}
    if row is None:
        count = two_stage_exfil(cells, {x.stratum: 1 for x in cells}, g.replay_blocks, 0.0, m.delta, m.served,
                                form == "patterns")
        top = min(g.footprint, sum(x.n * x.bits for x in cells))
        one += two_stage_price(f1.split_cells(cells, "template"), dict.fromkeys(range(len(cells)), 1), 1)[2]
        return {"exfiltration": {"y": min(count, served), "count": count, "binds": "served", "plan": None,
                                 **({"lean": None, "planner": None, "theta": None} if lean else {}),
                                 "registration_sample": 0, "shared": 0.0, "padding": 0.0, "sessions": 0.0,
                                 "sample_budgets": one / phase, "limit": None, "computed_proving": None, **forms},
                "infiltration": {"y": g.replay_blocks * top, "plan": None}}
    count, checked = (f1.lean_two(cells, g.replay_blocks, row, m.delta, m.served, form == "patterns") if lean
                      else (row["bound"], {}))
    one += two_stage_price(f1.split_cells(cells, row["merged"]), row["k"], 1)[2]
    return {"exfiltration": {"y": min(count, served), "count": count,
                             "binds": "served" if served <= count else "count", "plan": _plan(row), **checked,
                             **{k: row[k] for k in REG if k in row}, "sample_budgets": one / phase,
                             "limit": limits(cells, row)["limit"] if span else None,
                             "computed_proving": computed_proving(cells, row, computed), **forms},
            "infiltration": {"y": write_form(cells, row, g.replay_blocks, m.delta, g.footprint),
                             "plan": _plan(row)}}


#: the tables one session of the proof system holds at most, the scope its sessions' knowledge error is proved in
SESSION_TABLES = 200
#: the two designs with one session a template of a verification phase, printed beside a line's own, which has
#: several: whatever the session holds, and at most one table of the template's proof blocks
ONE_SESSION = {"template": "one session a block type, whatever it holds",
               "table": "one session of one table a block type"}
#: the tables a session holds at most in the designs with several sessions a template printed beside them
TABLES = (1, 16, 64, SESSION_TABLES)
#: the coins of one stream of a session's coin tree: 2^9 rounds of 512 coins (`verity_flock.rec_live.INNER_COIN_SPEC`'s
#: rounds and block, `flock-circuit`'s)
STREAM_COINS = 2 ** 9 * 512
#: the tables of one session whose coin tree the challenger samples at most: a tree holds `challenger.MAX_TREE_COINS`
#: coins, in one stream a session and two a table
TREE_TABLES = (challenger.MAX_TREE_COINS // STREAM_COINS - 1) // 2
#: the tables of one session the code that runs a second stage takes at most, least first, each by what takes it: the
#: GPU prover (one table a session), the CPU prover and the Lean verifier (their `MAX_TABLES`), the challenger's coin
#: commitment (`TREE_TABLES`) and the scope the sessions' knowledge error is proved in (`SESSION_TABLES`)
TABLE_LIMITS = ((1, "the GPU prover's"), (64, "the CPU prover's and the Lean verifier's"),
                (TREE_TABLES, "the challenger's coin commitment's"), (SESSION_TABLES, "the proved scope's"))
#: the lines of session coins one challenger session's release log holds (`challenger.MAX_TREE_LINES`), and the least
#: a session writes for each of its tables beside the 2 of its commitment: two repetitions of at least 86 rounds, a
#: line a round (`verity_flock.rec_live.rep_rounds` at the smallest table)
LOG_LINES, TABLE_LINES = challenger.MAX_TREE_LINES, 2 * 86
#: the bytes of live coins one layout of the challenger's holds (`challenger.MAX_COINS` coins of `COIN_BYTES`), which
#: the second stage's draw reads its streams from
DRAW_BYTES = challenger.MAX_COINS * challenger.COIN_BYTES


def draw_width(n: int) -> int:
    """The bytes of one uniform draw below `n` (`Flock.Draw.width`)."""
    return (n.bit_length() + 71) // 8


def limits(cells: Sequence[Cell], row: dict) -> dict:
    """What a two-stage design (`two_stage_design`'s `row` on `cells`, of a verification phase) asks of the code that
    runs a second stage, and the first limit it meets (`limit`, None when it meets none): its largest session's
    tables against `TABLE_LIMITS`; the lines of session coins its sessions write, at least 2 a session and
    `TABLE_LINES` a table, against one challenger session's `LOG_LINES` (`lines`); and the bytes of live coins its
    second draw reads, a stream of 2 k width(n) + 64 bytes for each block type in each sampled replay block
    (`Flock.Draw.Stratum.need`, one stratum a cell of `Spot.Select.secondLaw`), against one layout's `DRAW_BYTES`
    (`draw_bytes`).  A block type of fewer than one proof block a replay block counts at the chance a replay block
    has it.  A design that meets one is one no deployment runs today, whatever its bound.  The verifier's refusal of
    several sessions a template is not among them: it is the form, said beside the line."""
    count, k = row["registration_sample"], row["k"]
    split = f1.split_cells(cells, row["merged"])
    st = two_stage_statements(split, k, count, row["form"])
    lines = count * fsum(2 * r["sessions"] + TABLE_LINES * max(r["sessions"], r["taken"] / c.table)
                         for c, r in zip(split, st, strict=True))
    sizes: dict = {}
    for c in split:
        sizes[c.stratum] = sizes.get(c.stratum, 0.0) + c.n
    draw = count * fsum(min(1.0, n) * (2 * min(k[t], ceil(n)) * draw_width(ceil(n)) + 64) for t, n in sizes.items())
    tables = ceil(row["tables"] - 1e-9)
    past = [f"{what} {n:,}" for n, what in TABLE_LIMITS if tables > n]
    limit = None
    if past:
        limit = (f"its largest session in tables is {tables:,} of its type, past " +
                 (past[0] if len(past) == 1 else ", ".join(past[:-1]) + " and " + past[-1]))
    elif lines > LOG_LINES:
        limit = (f"its sessions write at least {lines:,.0f} lines of session coins, and one challenger session's "
                 f"release log holds {LOG_LINES:,}")
    elif draw > DRAW_BYTES:
        limit = (f"its second draw reads {draw:,.0f} bytes of live coins, and one layout of the challenger's holds "
                 f"{DRAW_BYTES:,}")
    return {"lines": lines, "draw_bytes": draw, "limit": limit}


def session_designs(cells: list, g: Grain, m: f1.Model, budget: float, form: str, span: float, served: float) -> dict:
    """The line's bound under each design printed beside its own, whose sessions are the cost model's statements:
    one session a block type of a verification phase (`ONE_SESSION`, `optimize.SESSIONS`), and under `tables` sessions
    of at most J tables of the type, as many as hold its sampled proof blocks, for each J of `TABLES`.  Each is the
    planner's bits, at most the scope's served bits, with the requests the verification phase samples, its sessions,
    the largest in proof blocks and in tables of its type, and the block types with a session of more than one table;
    None where no such design is within the budget, and `limits`' first limit met.  All are modeled, at the seconds
    a proof block the line's own statements are priced at.  The verifier's second stage reads one session of one
    table a template, and a session of the proof system holds at most `SESSION_TABLES` tables: `held` says whether
    the design with one session whatever it holds stays within that."""
    def design(of: list, sessions: str) -> dict | None:
        row = two_stage_design(of, g.replay_blocks, m.delta, budget=budget, replay=g.replay, h=m.byte_s,
                               stratify=STRATA, served=m.served, union=form == "patterns", registration=span,
                               sessions=sessions)
        return None if row is None else {
            "y": min(row["bound"], served), "count": row["bound"], "registration_sample": row["registration_sample"],
            "sessions": row["sessions"], "largest": row["largest"], "tables": row["tables"], "over": row["over"],
            **limits(of, row), "status": "modeled"}

    out = {sessions: design(cells, sessions) for sessions in ONE_SESSION}
    if out["template"] is not None:
        out["template"]["held"] = out["template"]["tables"] <= SESSION_TABLES
    out["tables"] = {str(j): design([replace(c, cap=j * c.table if c.table < inf else c.cap) for c in cells],
                                    "statements") for j in TABLES}
    return out


def beside_parts(got: dict, window: float) -> list[str]:
    """`session_designs`' designs as a caption and `counts` print them, one phrase each: the bound, the requests
    sampled and the largest session in proof blocks and in tables; for one session whatever it holds, no bound where
    its largest session is more tables than a session of the proof system holds; for sessions of at most J tables,
    the sessions a verification phase of `window` seconds has and the seconds between them if they run one at a
    time.  A design that meets a limit of the code that runs a second stage (`limits`) is out of reach, said first
    with the limit, its bound after it as the cost model's alone."""
    def reach(f: dict) -> str:
        return f"out of reach ({f['limit']}); modeled, " if f.get("limit") else ""

    parts = []
    for sessions, what in ONE_SESSION.items():
        f = got[sessions]
        if f is None:
            parts.append(f"{what}, no design within the budget")
            continue
        if not f.get("held", True):
            parts.append(f"{what}, out of reach: its largest session is {f['largest']:,} proof blocks and its largest "
                         f"in tables {ceil(f['tables']):,} of its type, and a session of the proof system holds at "
                         f"most {SESSION_TABLES}")
            continue
        size = f"{f['largest']:,} proof blocks" + (
            f", {f['over']:,} block types' sessions more than one table and the largest of them "
            f"{ceil(f['tables']):,} tables of its type" if f["over"] else ", within one table")
        parts.append(f"{what}, {reach(f)}{_size(f['y'])} at K = {f['registration_sample']:,}, its largest session "
                     f"{size}")
    for j, f in got["tables"].items():
        what = f"sessions of at most {j} table{'s' if j != '1' else ''} a block type"
        parts.append(f"{what}, no design within the budget" if f is None else
                     f"{what}, {reach(f)}{_size(f['y'])} at K = {f['registration_sample']:,} in "
                     f"{f['sessions']:,.0f} sessions, one each {window / f['sessions']:.3g} s if they run one at a "
                     "time")
    return parts


def _share(v: float) -> str:
    """A share of serving compute as the captions write it: a multiple from 1, a percentage from 0.01%."""
    return f"{v:.2g}×" if v >= 1 else f"{v * 100:.2g}%" if v >= 1e-4 else f"{v:.1e}"


def _size(bits: float) -> str:
    """Bits as the captions write a size: bytes in decimal units."""
    for unit, name in ((1e21, "ZB"), (1e18, "EB"), (1e15, "PB"), (1e12, "TB"), (1e9, "GB"), (1e6, "MB")):
        if bits / 8 >= unit:
            return f"{bits / 8 / unit:.3g} {name}"
    return f"{bits / 8e3:.3g} kB"


def _binds(betas: list[float], binds: list[str]) -> str:
    """Where the served bits bind on a line, as the caption says it."""
    at = [b for b, w in zip(betas, binds) if w == "served"]
    if not at:
        return "the count binds at every β plotted"
    if len(at) == len(betas):
        return "the served bits bind at every β plotted"
    return f"the served bits bind up to β = {_share(max(at))}, the count above"


def captions(out: dict, scope: str) -> dict[str, str]:
    """Each panel's caption for `scope`: each configuration's line, its tax and what its bound rests on."""
    tax, sc = out["taxes"], out["scopes"][scope]
    lines = {p: {s["config"]: s for s in out["panels"][p]["series"] if s["scope"] == scope} for p in PANELS}

    def priced(name: str) -> str:
        t = tax[name]
        kind = "priced, not measured, a lower bound: the hashing alone" if t["lower_bound"] else t["status"]
        parts = ", ".join(f"{p['part']} {_share(p['share'])}" for p in t["parts"])
        return f"serve tax {_share(t['tax'])} ({_share(t['low'])} to {_share(t['high'])}; {kind}: {parts})"

    conf = out["configurations"]
    at = out["betas"].index(f1.RATIO_BUDGET) if f1.RATIO_BUDGET in out["betas"] else None

    def at_ratio(name: str) -> str:
        """A two-stage line at β = `RATIO_BUDGET`: its bound, the requests its verification phase samples, its
        sessions beside its block types, its statements' sharing, its computed proving and the least verification
        phase its design fits; or that no design is within the budget, and what one sampled request costs against
        it."""
        if at is None:
            return ""
        s = lines["exfiltration"][name]

        def beside() -> str:
            got = (s.get("forms") or [None] * (at + 1))[at]
            if not got:
                return ""
            return " beside it, modeled at the same seconds a proof block: " + "; ".join(beside_parts(got, window)) + ";"

        if not s["registration_sample"][at]:
            return (" no design within the budget: one sampled replay block costs "
                    f"{s['sample_budgets'][at]:,.3g} budgets;{beside()}")
        shared = (f"{_share(s['shared'][at])} of its proving in the verification phase's statements, "
                  f"{_share(s['padding'][at])} padding" if s["shared"][at] else "its own statements")
        least = s["sample_budgets"][at] * regs[conf[name]["registration"]]["machines"]
        reach = f" out of reach ({s['limit'][at]}); modeled," if s.get("limit") and s["limit"][at] else ""
        return (f"{reach} {_size(s['y'][at])}; K = {s['registration_sample'][at]:,} sampled, in "
                f"{s['sessions'][at]:,.0f} sessions a verification phase for {len(s['plan'][at]['k']):,} block "
                f"types: {shared}; {_share(s['computed_proving'][at])} of its sampled request's proving computed; one "
                f"sampled request of this design costs the budget of {least:,.3g} machines' {window:g} s, the least "
                f"verification phase it fits (one machine's {least * window / 3600:,.3g} hours);{beside()}")

    head = (f"{f1.ROLES} {f1.TRUST} "
            "x is the serve tax, what each configuration's change to the production server costs on every token, plus "
            "β, the share of compute spent on replays and proofs of the sampled proof blocks; each line starts at its "
            "tax, hollow where the tax is priced. Every line proves a proof block at its type's measured seconds, a "
            "block's and its statement's fixed ones, and a type not yet measured at its staged witness bits (computed): "
            f"the registered circuit's marginal proof overhead comes to {out['overhead']:.3g}×, "
            f"{_share(conf['tokens']['computed_share'])} of it computed, measured on benchmark values, which a "
            f"replayed served step's values cost the same as.{_verify_sentence(out['inputs'])} Every β has its own "
            "plan, the one with the "
            "least bound. Replay is full-batch: the model's tokens come out bit-exact only in their served batch, so a "
            f"sampled request replays its replay block, one machine's {out['inputs']['window']['value']:g} s, whole "
            f"from its edge, {out['registrations']['machine']['requests']:,.0f} requests at {out['inputs']['replay_factor']['value']:g}× "
            "serving's busy GPU-seconds (0.97× to 1.02× measured on Moonlight on one H200), and a sampled layer of one "
            "token its engine step's whole batch at that layer.")
    regs, window = out["registrations"], out["inputs"]["window"]["value"]
    head += (f" A two-stage line's first stage is the one the verifier runs. A verification phase, one committed "
             f"trace and the sample over it, spans the {window:g} s window on one machine "
             f"({out['inputs']['machine_gpus']['value']:g} GPUs), on a {POD}-machine pod or on a datacenter's "
             f"{regs['datacenter']['machines']:,.0f} machines, and its first stage samples a whole number K of its "
             "requests (of (B), its layers of one token), never fewer than one: each point is priced and bounded at "
             "K over the phase's requests, and where β does not pay for one of them no design is within the budget "
             "and the line is at everything the scope serves. No code builds a verification phase of a datacenter's "
             "size, one committer over every machine's boundary rows: that line is the cost model's. The second "
             "stage samples each block type at its own whole count of a sampled request's proof blocks, at least "
             "one, the strata the verifier's two-stage rule registers. A two-stage "
             "statement holds one Definition's sampled proof blocks of a sampled request or of the verification "
             "phase's K, its sessions padded to one size and none larger than its type's statement on the "
             "proof_overhead card, and each Definition takes the phase's statements only "
             "where they cost less than its own. The verifier's second stage reads one session of one table for "
             "each template of a verification phase, a table holding at most 2^35 witness bits (32,768 proof blocks "
             "of 2^20 bits, 256 of 2^27), so a line with more sessions than block types prices sessions the verifier "
             "refuses. Beside each such line's bound are the designs it could be instead, all modeled at the "
             "seconds a proof block the line's own statements are priced at. Two have one session a block type: one "
             "whose session holds whatever the design samples, which the verifier refuses where it is more than one "
             f"table and no session of the proof system holds past {SESSION_TABLES} tables, so it is printed only "
             "where every session is within that; and one whose sessions are each at most one table, the form the "
             f"verifier's second stage reads. The others have sessions of at most {', '.join(map(str, TABLES[:-1]))} "
             f"and {TABLES[-1]} tables of the type, as many as hold its sampled proof blocks, each with its sessions "
             "a verification phase and the seconds between them if they run one at a time; the verifier's second "
             "stage refuses more than one session a template. A design, the line's own or one beside it, is out of "
             "reach where it meets a limit of the code that runs a second stage today, and the first it meets is "
             f"named: a session of more than {TABLE_LIMITS[0][0]} table on the GPU prover, {TABLE_LIMITS[1][0]} on "
             f"the CPU prover and in the Lean verifier, {TABLE_LIMITS[2][0]} in the challenger's coin commitment or "
             f"{SESSION_TABLES} in the proved scope; more lines of session coins than one challenger session's "
             f"release log holds ({LOG_LINES:,}, a session writing at least 2 and {TABLE_LINES} a table); or a second "
             f"draw of more bytes of live coins than one layout holds ({DRAW_BYTES:,}). Its bound is then the cost "
             "model's alone, of a deployment no code runs. "
             "The verification phase, its K and its statements' split are the "
             "protocol's, fixed before any coin, never the prover's. Each two-stage line gives at "
             f"β = {_share(f1.RATIO_BUDGET)} its bound, K and its sessions a verification phase beside its block "
             "types, the shares of its proving in the phase's statements "
             "and on padding, the share of its sampled request's proving, the statements' fixed costs included, "
             "that is computed, and the least verification phase its design fits; or that no design is within "
             "the budget, and what one sampled replay block costs against it.")
    a, b, c = CONFIGURATIONS["tokens"], CONFIGURATIONS["layer_of_token"], CONFIGURATIONS["every_block"]
    short = a.label.split(" ")[0]
    served, betas = out["served_bits_request"], out["betas"]

    def variants(name: str, form: str) -> list[str]:
        return [n for n, k in conf.items() if (n == name or k["variant_of"] == name) and k["form"] == form
                and k["replay"] != "request"]

    def tag(name: str) -> str:
        k = conf[name]
        if not k["variant_of"] and not any(v["variant_of"] == name for v in conf.values()):
            return k["label"].split(" ")[0]
        return (f"{k['label'].split(' ')[0]} at {regs[k['registration']]['label']}"
                + (" in the sets' form" if k["form"] == "sets" else "")
                + (" with batch-invariant kernels" if k["replay"] == "request" else ""))

    def binds(name: str) -> str:
        return f"{at_ratio(name)} {_binds(betas, lines['exfiltration'][name]['binds'])}."

    def span(name: str) -> str:
        r = regs[conf[name]["registration"]]
        return f" {r['label'][0].upper()}{r['label'][1:]} ({r['requests']:,.0f} requests):{binds(name)}"

    def family(name: str, style: str, what: str) -> str:
        own = [n for n in variants(name, "patterns") if n in lines["exfiltration"]]
        spans = [REGISTRATIONS[conf[n]["registration"]] for n in own]
        text = (f" {conf[name]['label']} ({style}, dark to light: {', '.join(spans[:-1])} and {spans[-1]} "
                f"verification phase): {what}" + "".join(span(n) for n in own))
        for n in variants(name, "sets"):
            r = regs[conf[n]["registration"]]["label"]
            text += (f" {conf[n]['label']} in the capped sets' form (dashed, thin; {r}), for comparison: "
                     f"{f1.CAP_STATUS['sets']}, at most the scope's served bits ({f1.SERVED_STATUS}):{binds(n)}")
        return text

    patterns = (f"the patterns' form ({f1.CAP_STATUS['patterns']}): a request's incorrect proof "
                f"blocks are charged their committed bits up to the {served:,.0f} bits it serves, so a wholly incorrect "
                f"request costs {served:,.0f} bits, and the bound is at most the {sc['served_bits']:.2g} bits the "
                f"scope serves ({f1.OUTS_STATUS}). It takes two modeling conditions: each request serves at most "
                f"S = {served:,.0f} bits, and no proof block is shared across requests, so a request's served outputs "
                "read only its own proof blocks.")
    cuts = {n: k["cut"] for n, k in conf.items() if "cut" in k and not k["variant_of"]}
    exfil_cuts = "".join(family(n, "dash-dot", f"{short}'s serving and serve tax on the partition card {k['name']}'s "
                                f"cut ({k['label']}), its marginal proof overhead {conf[n]['overhead']:.3g}×, "
                                f"{_share(conf[n]['computed_share'])} of it computed; the same forms.")
                         for n, k in cuts.items())
    infil_cuts = "".join(f" {conf[n]['label']} (dash-dot, the same verification phases): {short}'s footprint at its own "
                         "designs." for n in cuts)
    trivial = [tag(n) for n, s in lines["exfiltration"].items()
               if at is not None and s["binds"][at] == "served"]
    who = ("every line is" if len(trivial) == len(lines["exfiltration"]) else trivial[0] + " is" if len(trivial) == 1
           else f"{', '.join(trivial[:-1])} and {trivial[-1]} are" if trivial else "")
    one = len(trivial) == 1
    exfil = (f" At β = {_share(f1.RATIO_BUDGET)} {who} bounded here only by everything the scope serves "
             f"({_size(sc['served_bits'])}), the trivial bound: the audit's {'count is' if one else 'counts are'} "
             f"above it, so {'that value is not a result' if one else 'those values are not results'} of the audit."
             if trivial else "")
    exfil += (" Lean checks every two-stage point's design: its count is the bits Lean certifies there, at a θ Lean "
              "searched." if out.get("lean") else "")
    bi = conf.get("tokens_bi")
    exfil += (family("tokens", "solid", f"serving commits {a.commits}, a replay block is {a.replay_block}, "
                     f"{priced('tokens')}; {patterns}")
              + (f" {bi['label']} (dotted; {regs[bi['registration']]['label']}), for comparison: {bi['replay_block']}, "
                 f"which needs batch-invariant kernels, whose tax is measured on three other models at one H200 "
                 f"({_share(BATCH_INVARIANT[1])} to {_share(BATCH_INVARIANT[2])}) and which GLM-5.3's sparse MLA has "
                 f"none of on the H200, {priced('tokens_bi')}; the patterns' form:{binds('tokens_bi')}"
                 if bi and "tokens_bi" in lines["exfiltration"] else "")
              + f"{exfil_cuts} {b.label} ({regs[b.registration]['label']}): serving commits {b.commits}, a replay "
              f"block is {b.replay_block}, {priced('layer_of_token')}; the capped sets' form, each sampled layer of "
              f"one token charged at most its request's served bits, at least the lemma's charge:{binds('layer_of_token')}"
              f" {c.label} (dashed): serving commits {c.commits}, sampled one-stage, a statement a sampled proof "
              f"block, {priced('every_block')}; consumers.exfiltration_bound's form, which bits are incorrect and "
              "where, at most the served bits (an influence set is a set of served outputs)"
              + (", each design checked by Lean" if out.get("lean") else "")
              + f": {_binds(betas, lines['exfiltration']['every_block']['binds'])}.")
    infil = (f" The write form: the bits the incorrect proof blocks write, each sampled request or layer charged at "
             f"most its footprint, at the design its exfiltration line plans. {a.label} (solid, dark to light: its "
             f"verification phases): a footprint of a request's {served:,.0f} served bits, the cheapest explanation of what "
             f"it serves (the served form).{infil_cuts} {b.label} ({regs[b.registration]['label']}): a footprint of "
             f"its {out['footprint_bits'][1]:,.0f} boundary bits, written wherever they sit. {c.label} (dashed): the K "
             "incorrect blocks the plan detects, each the widest block's bits. The two-stage write forms need a harm "
             "bound at the footprint, which no lemma states yet; one-stage's rests on audit_profile.")
    work = (f" Only {c.label} bounds skipped work: serving commits {c.commits}, so a sampled block's inputs and "
            f"outputs were fixed at serve time; {priced('every_block')}; the work target's plan.")
    return {"exfiltration": head + exfil, "infiltration": head + infil, "skipped_work": head + work}


def run(model_name: str = "GLM-5.3", scope_names: tuple[str, ...] | None = None, request: str = "census",
        workers: int = 1, lean: bool = True, serving: str = "assumed") -> dict:
    """Figure 1's JSON in its total-cost form over the scopes named (default all), planned on `workers` processes
    (`figure1.processes`), with Lean checking every line's designs unless `lean` is off.  `serving` is where serving's
    rate comes from (`figure1.SERVING`): the measured reading's JSON says which rate it read (`serving`), and its serve
    taxes are still the `serve_tax` card's, priced at the assumed reading until `priced_card` prices one at the
    measured."""
    with f1.processes(workers, lean) as each:
        return _run(model_name, scope_names, request, each, lean, serving)


def _lines(model_name: str, request: str, verify: bool = True, serving: str = "assumed") -> tuple:
    """The model's cards and, for each configuration it has the cut of, its priced inputs (with each statement's
    verify where `verify` and the model has a `verifier_cost` card), its line (proof blocks a token, fill and marginal
    overhead) and which of its cells' types are computed."""
    sc = f1.slot_cards(model_name, serving)
    c = f1.cards(model_name, request, serving)
    cuts = {card["value"]["name"]: card for card in spot_cards.load_all("partition", model_name)}
    confs = {n: conf for n, conf in CONFIGURATIONS.items() if conf.partition is None or conf.partition in cuts}
    inputs = {n: priced(cut(c, cuts[conf.partition]) if conf.partition else c, sc["proof_overhead"])
              for n, conf in confs.items()}
    if verify and (vc := spot_cards.load("verifier_cost", model_name, required=False)):
        sc |= {"verifier_cost": vc}
        inputs = {n: verified(i, vc) for n, i in inputs.items()}

    def line(i: dict[str, f1.InputCard]) -> tuple:
        strata = i["partition"].value["strata"]
        return (f1.TokenBlocks.of(strata, i["partition"].value["served_macs_per_token"]),
                f1.Fill.of(strata, i["proof_blocks"].value), i["proof_overhead"].value)

    lines = {n: line(i) for n, i in inputs.items()}
    computed = {n: tuple(x["priced"] == "computed" for x in i["partition"].value["strata"] if x["per_token"] > 0)
                for n, i in inputs.items()}
    return sc, c, cuts, confs, inputs, lines, computed


def _run(model_name: str, scope_names: tuple[str, ...] | None, request: str, each, lean: bool,
         serving: str = "assumed") -> dict:
    sc, c, cuts, confs, inputs, lines, computed = _lines(model_name, request, serving=serving)
    tax = taxes(spot_cards.load("serve_tax", model_name), tuple(confs))
    part, fill, h = lines["tokens"]
    betas = sorted(set(BETAS) | set(c["budgets"].value))
    inf_card = sc["infiltration"]["value"]
    layers, boundary = inf_card["replay_blocks_per_token"], inf_card["cap_bits_per_block"]
    tokens = c["tokens_per_request"].value
    refs = f1.references(c, model_name, sc.get("references", []))
    notes = f1.reference_notes(sc.get("references", []), refs)
    if serving == "measured":
        for panel in PANELS:
            notes.setdefault(panel, []).append(f1.reading_note(f1.reading(sc, c, part)))
    chosen = [s for s in f1.scopes(c, part) if scope_names is None or s.name in scope_names]
    series: dict[str, list[dict]] = {p: [] for p in PANELS}
    bases, plans = {}, []
    for s in chosen:
        base = f1.model(inputs["tokens"], part, s, h, fill)
        for name, conf in confs.items():
            tb, fl, oh = lines[name]
            m = f1.model(inputs[name], tb, s, oh, fl) if conf.partition else base
            g = None if name == "every_block" else grain(m, name, layers, boundary, tokens, batch(conf, s, c))
            span = 0.0 if g is None or conf.registration is None else (
                g.replay_blocks * registration_gpu_seconds(conf.registration, c) / s.gpu_seconds)
            plans.append((s, name, conf, [(name, m, g, beta * s.gpu_seconds, computed[name], conf.form, span, lean,
                                           beta == f1.RATIO_BUDGET) for beta in betas]))
        bases[s.name] = base
    points = iter(list(each(_point, [job for *_, jobs in plans for job in jobs])))
    for s, name, conf, jobs in plans:
        pts = [next(points) for _ in jobs]
        for panel in conf.panels:
            rows = [p[panel] for p in pts]
            series[panel].append({"scope": s.name, "config": name, "beta": betas,
                                  "x": [tax[name]["tax"] + b for b in betas], "y": [r["y"] for r in rows],
                                  **{k: [r[k] for r in rows]
                                     for k in ("binds", "count", *REG, "limit", "computed_proving", "lean", "planner",
                                               "theta")
                                     if k in rows[0]},
                                  **({"forms": [r.get("forms") for r in rows]}
                                     if any("forms" in r for r in rows) else {}),
                                  "plan": [r["plan"] for r in rows]})
    out_scopes = {s.name: {"who": s.who, "period": s.period, "gpus": c[f"{s.who}_gpus"].value,
                           "gpu_seconds": s.gpu_seconds, "requests": s.requests,
                           "tokens": s.requests * tokens, "served_bits": f1.served_total(bases[s.name]),
                           "output_bits": f1.cap(bases[s.name], "leak"), "flop": f1.cap(bases[s.name], "work"),
                           "ands": f1.cap(bases[s.name], "work") * c["ands_per_mac"].value / 2} for s in chosen}
    used = {conf.partition: cuts[conf.partition] for conf in confs.values() if conf.partition}
    per_gpu_s = chosen[0].requests / chosen[0].gpu_seconds
    registrations = {r: {"label": f"{label} {c['window'].value:g} s verification phase",
                         "machines": registration_gpu_seconds(r, c) / (c["machine_gpus"].value * c["window"].value),
                         "gpu_seconds": registration_gpu_seconds(r, c),
                         "requests": per_gpu_s * registration_gpu_seconds(r, c)} for r, label in REGISTRATIONS.items()}
    out = ({"serving": f1.reading(sc, c, part)} if serving == "measured" else {}) | {
           "schema": SCHEMA, "model": model_name, "request": request, "overhead": h, "registrations": registrations,
           "lean": {"exe": plancheck.EXE, "slack": str(plancheck.SLACK), "search": plancheck.SEARCH,
                    "precision": plancheck.TWO_P} if lean else None,
           "overhead_label": sc["proof_overhead"]["value"]["label"], "betas": betas, "budgets": c["budgets"].value,
           "served_bits_request": c["served_bits"].value, "footprint_bits": [c["served_bits"].value, boundary],
           "configurations": {n: conf.__dict__ | {"panels": list(conf.panels), "overhead": lines[n][2],
                                                  "computed_share": inputs[n]["partition"].value["computed_share"]} | (
               {"cut": {k: used[conf.partition]["value"][k] for k in ("name", "label")}
                       | {"art": used[conf.partition].get("art")}} if conf.partition else {})
               for n, conf in confs.items()},
           "taxes": tax, "inputs": {k: v.__dict__ for k, v in inputs["tokens"].items()},
           "cards": sc | ({"partition": list(used.values())} if used else {}), "scopes": out_scopes,
           "panels": {"exfiltration": {"unit": "bit", "series": series["exfiltration"],
                                       "references": [r for r in refs["exfiltration"] if r["main"]]},
                      "infiltration": {"unit": "bit", "series": series["infiltration"],
                                       "references": refs["infiltration"]},
                      "skipped_work": {"unit": "FLOP", "series": series["skipped_work"],
                                       "references": refs["skipped_work"]}}}
    for name in out_scopes:
        caps = captions(out, name)
        for panel, sentences in notes.items():
            if panel in caps:
                caps[panel] += "".join(f" {x}" for x in sentences)
        out_scopes[name]["captions"] = caps
    return out


COUNTS_SCHEMA = "spot-two-stage-counts/v1"


def counts(model_name: str = "GLM-5.3", line: str = "tokens_datacenter", scope: str | None = None,
           beta: float = f1.RATIO_BUDGET, request: str = "census", serving: str = "assumed") -> dict:
    """A two-stage line's design at β on `scope` (default the first) as the verifier's two-stage rule takes it
    (`optimize.two_stage_counts`): `sampled`, the requests the first stage samples of a verification phase, and `k`,
    each block type's whole count of a sampled request's proof blocks, by the cut's name of the type; with the phase's
    size, the design's sessions a verification phase, its largest session in proof blocks and in tables of its type,
    the planner's bound in bits, and what the design asks of the code that runs a second stage with the first limit
    it meets (`limits`).  `beside` holds the designs beside it (`session_designs`), each modeled.  A β that pays for
    no sampled request of the line's verification phase has no
    design and no counts (`ValueError`)."""
    sc, c, _, confs, inputs, lines, _ = _lines(model_name, request, serving=serving)
    conf = confs[line]
    if conf.registration is None:
        raise ValueError(f"{line} is no two-stage line: the counts are a two-stage design's")
    s = next(x for x in f1.scopes(c, lines["tokens"][0]) if scope in (None, x.name))
    tb, fl, oh = lines[line]
    m = f1.model(inputs[line], tb, s, oh, fl)
    card = sc["infiltration"]["value"]
    g = grain(m, line, card["replay_blocks_per_token"], card["cap_bits_per_block"], c["tokens_per_request"].value,
              batch(conf, s, c))
    gpu_s = registration_gpu_seconds(conf.registration, c)
    row = two_stage_design(list(g.cells), g.replay_blocks, m.delta, budget=beta * s.gpu_seconds, replay=g.replay,
                           h=m.byte_s, stratify=STRATA, served=m.served, union=conf.form == "patterns",
                           registration=g.replay_blocks * gpu_s / s.gpu_seconds)
    if row is None:
        raise ValueError(f"{line} has no design within β = {beta:g} on {s.name}: the budget pays for no sampled "
                         f"replay block of {REGISTRATIONS[conf.registration]} verification phase")
    names = [x["block"] for x in inputs[line]["partition"].value["strata"] if x["per_token"] > 0]
    return {"schema": COUNTS_SCHEMA, "model": model_name, "line": line, "scope": s.name, "beta": beta,
            "verification_phase": {"machines": gpu_s / (c["machine_gpus"].value * c["window"].value),
                                   "seconds": c["window"].value, "replay_blocks": row["registration"]},
            **two_stage_counts(row, names), "sessions": row["sessions"], "largest": row["largest"],
            "tables": row["tables"], "bound_bits": row["bound"], **limits(list(g.cells), row),
            "beside": session_designs(list(g.cells), g, m, beta * s.gpu_seconds, conf.form, row["registration"],
                                      f1.served_total(m))}


def weights(c: dict[str, f1.InputCard]) -> dict[str, float]:
    """The weights a break-even reaches, in bits: the checkpoint's FP8 bytes, and FP4 at 4 bits a parameter."""
    return {"FP8": 8 * c["weights_bytes"].value, "FP4": 4 * c["params"].value}


def _bound(line: tuple, scale: float, beta: float, lean: bool = False) -> dict:
    """A request-grain line's exfiltration point at β with its marginal overhead times `scale`: `_point`'s, its `y` at
    most the scope's served bits, with Lean's check of its design if `lean`."""
    name, i, (tb, fl, oh), s, computed, conf, gpu_s, replays = line
    m = f1.model(i, tb, s, oh * scale, fl)
    g = Grain(m.replay.replay_blocks, m.cells, m.replay.cost * replays, m.served)
    job = (name, m, g, beta * s.gpu_seconds, computed, conf.form, g.replay_blocks * gpu_s / s.gpu_seconds, lean)
    return _point(job)["exfiltration"]


def _at(what: str, line: tuple, beta: float, x: float, lean: bool = False) -> dict:
    """The point a break-even searches: at the overhead's multiple `x` and β (`what` "h"), or at β = `x` and the line's
    own overhead ("beta")."""
    return _bound(line, x, beta, lean) if what == "h" else _bound(line, 1.0, x, lean)


def _solve(job: tuple) -> float | None:
    """The multiple of a line's marginal overhead (`what` "h") at which its bound at β reaches `target` bits, or the β
    ("beta") at which it does at its own overhead, by bisection in log to `rel` over its `SPANS`: the end of the last
    bracket at which the bound is at most the target, so within 1 + `rel` of the break-even on its safe side; None
    where the bound is above the target over the whole span, and the span's end where it is below it throughout.  At a
    target of None, the floor: the bound at the least overhead."""
    what, target, line, beta, rel = job
    lo, hi = SPANS[what]
    if target is None:
        return _bound(line, lo, beta)["y"]
    rising = what == "h"
    if _at(what, line, beta, lo if rising else hi)["y"] > target:
        return None
    if _at(what, line, beta, hi if rising else lo)["y"] <= target:
        return hi if rising else lo
    while hi / lo > 1 + rel:
        mid = (lo * hi) ** 0.5
        lo, hi = (mid, hi) if (_at(what, line, beta, mid)["y"] <= target) == rising else (lo, mid)
    return lo if rising else hi


#: the spans a break-even searches: the marginal overhead's multiple, and β
SPANS = {"h": (1e-12, 1e2), "beta": (1e-6, 1e6)}
#: the most steps outward from a solved break-even that Lean's certificate takes, the i-th of (1 + `rel`)^(2^i)
STEPS = 6


def certify(at, x: float, target: float, rising: bool, rel: float) -> dict:
    """Lean's certificate that a line's exfiltration bound at a break-even's point is at most `target` bits.  `at(x)` is
    the line's point at `x` with Lean's check of its design (`_point`'s exfiltration with `lean`): its count is the
    bits Lean certifies for the design the planner picks there (`figure1.lean_two`), rounded up to a float, so `y`, that
    count or the served bits if fewer, at most the target is the certificate.  Where it is above the target (Lean's
    bits above the planner's, at a point the planner puts near the target), `x` steps outward (down for a bound rising
    in it, up for one falling), the i-th step by (1 + `rel`)^(2^i), at most `STEPS` times; `at` None says no step
    reached it."""
    for step in range(STEPS + 1):
        pt = at(x)
        if pt["y"] <= target:
            return _certificate(x, step, pt)
        x = x / (1 + rel) ** 2 ** step if rising else x * (1 + rel) ** 2 ** step
    return _certificate(None, STEPS, pt)


def _certificate(x: float | None, steps: int, pt: dict) -> dict:
    return {"at": x, "steps": steps, "y": pt["y"], "binds": pt["binds"],
            **{k: pt.get(k) for k in ("count", "lean", "planner", "theta")}}


def _certify(job: tuple) -> dict | None:
    """`certify` at a break-even `_solve` found, at `x`; None where it found none.  At a target of None, the floor:
    Lean's bound at the least overhead."""
    what, target, line, beta, rel, x = job
    if x is None:
        return None
    if target is None:
        return _certificate(SPANS["h"][0], 0, _bound(line, SPANS["h"][0], beta, True))
    return certify(lambda v: _at(what, line, beta, v, True), x, target, what == "h", rel)


def _breakeven(job: tuple) -> tuple[float | None, dict | None]:
    """A break-even `_solve` finds and, with `lean`, its certificate (`_certify`), in one job, so no break-even waits on
    the others' to be certified."""
    *solve, lean = job
    x = _solve(tuple(solve))
    return x, _certify((*solve, x)) if lean else None


def breakeven(model_name: str = "GLM-5.3", scope_names: tuple[str, ...] | None = None, request: str = "census",
              workers: int = 1, rel: float = 0.01, lean: bool = True, serving: str = "assumed") -> dict:
    """Each patterns'-form line's break-even on each scope: the marginal overhead at which its bound at β =
    `RATIO_BUDGET` reaches each of the weights (`weights`), the β at which it does at the line's own overhead, and its
    floor, the bound at the least overhead `SPANS` searches.  With `lean`, each has Lean's certificate (`certify`): a
    row's `lean` holds, for its floor and each break-even, the point Lean certifies (`at`, an overhead for `h`), the
    bits it certifies there and which of them and the served bits binds, so its β is an upper bound on the β at which
    the line's bound reaches the weights, and its overhead a lower bound on the overhead at which it does.  `serving`
    is where serving's rate comes from (`figure1.SERVING`)."""
    sc, c, _, confs, inputs, lines, computed = _lines(model_name, request, serving=serving)
    targets = weights(c)
    chosen = [s for s in f1.scopes(c, lines["tokens"][0]) if scope_names is None or s.name in scope_names]
    keys = [(n, s) for s in chosen for n, conf in confs.items() if conf.form == "patterns"]
    jobs = [(what, t, (n, inputs[n], lines[n], s, computed[n], confs[n],
                       registration_gpu_seconds(confs[n].registration, c), batch(confs[n], s, c)), f1.RATIO_BUDGET, rel)
            for n, s in keys for what, t in [("h", None)] + [(w, t) for w in SPANS for t in targets.values()]]
    with f1.processes(workers, lean) as each:
        found = list(each(_breakeven, [j + (lean,) for j in jobs]))
    it, cit = iter(x for x, _ in found), iter(cert for _, cert in found)
    rows = []
    for n, s in keys:
        oh, floor_bits = lines[n][2], next(it)
        at = {what: {name: next(it) for name in targets} for what in SPANS}
        row = {"config": n, "scope": s.name, "registration": confs[n].registration, "overhead": oh,
               "floor": floor_bits, "h": {k: None if v is None else v * oh for k, v in at["h"].items()},
               "beta": at["beta"]}
        if lean:
            row["lean"] = {"floor": next(cit), **{what: {name: next(cit) for name in targets} for what in SPANS}}
            for cert in [row["lean"]["floor"], *row["lean"]["h"].values()]:
                if cert and cert["at"] is not None:
                    cert["at"] *= oh
        rows.append(row)
    return ({"serving": f1.reading(sc, c, lines["tokens"][0])} if serving == "measured" else {}) | {
            "beta": f1.RATIO_BUDGET, "targets": targets, "rel": rel, "spans": SPANS, "rows": rows,
            "lean": {"exe": plancheck.EXE, "slack": str(plancheck.SLACK), "search": plancheck.SEARCH,
                     "precision": plancheck.TWO_P, "steps": STEPS} if lean else None}


def breakeven_table(be: dict) -> str:
    """The break-even as a table: a row per line and scope, its floor, the marginal overheads at which its bound at β
    reaches each weights size and the β at which it does at its own overhead.  With Lean's certificates, each is the
    point Lean certifies, marked "(served)" where the served bits bind there, and "uncertified" where Lean's bits stay
    above the target."""
    names, lean = list(be["targets"]), be.get("lean") is not None
    head = (["line", "verification phase", "scope", "h", "floor"] + [f"h for {n} at β = {_share(be['beta'])}" for n in names]
            + [f"β for {n}" for n in names])

    def fmt(v: float | None, end: float) -> str:
        return "none" if v is None else (">" if v >= end else "") + f"{v:.3g}"

    def cell(r: dict, what: str, n: str, end: float) -> str:
        if not lean:
            return fmt(r[what][n], end)
        cert = r["lean"][what][n]
        if cert is None or cert["at"] is None:
            return "none" if cert is None else "uncertified"
        return fmt(cert["at"], end) + (" (served)" if cert["binds"] == "served" else "")

    rows = [[CONFIGURATIONS[r["config"]].label.split(" ")[0]
             + (" batch-invariant" if CONFIGURATIONS[r["config"]].replay == "request" else ""),
             r["registration"], r["scope"], f"{r['overhead']:.3g}",
             _size(r["lean"]["floor"]["y"] if lean else r["floor"])]
            + [cell(r, "h", n, be["spans"]["h"][1] * r["overhead"]) for n in names]
            + [cell(r, "beta", n, be["spans"]["beta"][1]) for n in names] for r in be["rows"]]
    note = f1.reading_note(be["serving"]) + "\n\n" if "serving" in be else ""
    return note + "\n".join(" | ".join(x) for x in [head] + rows) + "\n"


def at_budgets(out: dict) -> list[dict]:
    """Each line's point at the design card's budgets: the bound, its x, and its share of the scope's served bits
    (exfiltration, infiltration) or FLOP (skipped work).  An exfiltration row at the measured reading names the rate it
    read (`figure1.at_reading`)."""
    rows = []
    for panel, p in out["panels"].items():
        for s in p["series"]:
            sc = out["scopes"][s["scope"]]
            for beta in out["budgets"]:
                i = s["beta"].index(beta)
                whole = sc["flop"] if panel == "skipped_work" else sc["served_bits"]
                rows.append({"panel": panel, "scope": s["scope"], "config": s["config"], "beta": beta, "x": s["x"][i],
                             "bound": s["y"][i], "share": s["y"][i] / whole,
                             **({"binds": s["binds"][i]} if "binds" in s else {}),
                             **(f1.at_reading(out.get("serving"), s["y"][i]) if panel == "exfiltration" else {})})
    return rows


PRIMARY, SECONDARY = f1.PRIMARY, f1.SECONDARY
#: the line the headline scope's exfiltration panel also plots over the secondary scope, and its style there
SECONDARY_LINE, SECONDARY_STYLE = "tokens", ("#6A51A3", (0, (5, 2)), 2.0)
#: references closer than this factor stagger their labels: the lowest below its line, the rest alternating sides
CROWDED = 2.5
#: each configuration's colour, line style and width: a line's verification phases dark to light, each narrower so that one
#: plotted over another shows; the sets' form thin and dashed; (C) dashed, its tax a lower bound
CUT, SETS = (0, (7, 2, 1.5, 2)), (0, (3, 2))
STYLE = {"tokens": ("#08306B", "-", 4.2), "tokens_pod": ("#2171B5", "-", 2.8),
         "tokens_datacenter": ("#9ECAE1", "-", 1.4), "tokens_sets": ("#08306B", SETS, 1.6),
         "per_expert": ("#00441B", CUT, 4.2), "per_expert_pod": ("#238B45", CUT, 2.8),
         "per_expert_datacenter": ("#A1D99B", CUT, 1.4), "per_expert_sets": ("#00441B", SETS, 1.6),
         "head_units": ("#7A0177", CUT, 4.2), "head_units_pod": ("#C51B8A", CUT, 2.8),
         "head_units_datacenter": ("#FA9FB5", CUT, 1.4), "head_units_sets": ("#7A0177", SETS, 1.6),
         "tokens_bi": ("#6BAED6", (0, (1, 1.6)), 2.4), "layer_of_token": ("#D95F02", "-", 2.4),
         "every_block": ("#555555", (0, (6, 3)), 2.4)}
YLABELS = {"exfiltration": "Exfiltration capacity", "infiltration": "Infiltration, bits written",
           "skipped_work": "Skipped work (H100-hours)"}


def stem(scope: str, panel: str = "exfiltration") -> str:
    base = "figure1_skipped_work" if panel == "skipped_work" else "figure1"
    return base if scope == PRIMARY else f"{base}_{scope.replace(' ', '_')}"


def _percent(v: float, _pos=None) -> str:
    return f"{v * 100:g}%"


def _panel(fig, ax, out: dict, scope: str, panel: str, wrap: int) -> None:
    """One panel of `scope`'s figure: each configuration's line from its tax, hollow at its start while the tax is
    priced, with the tax's range as a bar, the references grey, and the panel's caption under it."""
    import textwrap

    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    from render import CAPTION_DROP, OFF_SCALE, REFERENCE_STYLE, _bytes, _decades, _fit, _floor, _lines, _power, _zeros

    p, tax = out["panels"][panel], out["taxes"]
    series = [s for s in p["series"] if s["scope"] == scope]
    secondary = [s for s in p["series"] if (scope, panel, s["scope"], s["config"])
               == (PRIMARY, "exfiltration", SECONDARY, SECONDARY_LINE)]
    scale = 1 / 8 if p["unit"] == "bit" else 1 / out["inputs"]["h100_hour_flop"]["value"]
    fmt = _bytes if p["unit"] == "bit" else _power
    xs = [x for s in series for x in s["x"]] + [tax[s["config"]][k] for s in series for k in ("low", "high")]
    lo_x, hi_x = 10.0 ** floor(log10(min(xs))), max(xs) * 1.3
    ax.set_xscale("log")
    ax.set_xlim(lo_x, hi_x)
    ax.xaxis.set_major_locator(FixedLocator(_decades(lo_x, hi_x, 1)))
    ax.xaxis.set_major_formatter(FuncFormatter(_percent))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Total cost, a share of serving compute: serve tax + β", labelpad=8)
    ax.set_ylabel(YLABELS[panel], labelpad=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    refs = _lines(p["references"], scope, scale)
    if panel == "exfiltration":
        refs = sorted(refs + [(out["scopes"][scope]["served_bits"] * scale, "Everything the scope serves", "dotted")])
    ys = [y * scale for s in series + secondary for y in s["y"]]
    far = [r for r in refs if r[0] > max(ys) * OFF_SCALE]
    refs = [r for r in refs if r not in far]
    lo, hi = 10.0 ** floor(log10(min(ys + [r[0] for r in refs]))), 10.0 ** (floor(log10(max(ys + [r[0] for r in refs]))) + 1)
    zeros = _zeros(p["references"], scope)
    lo /= 10 if zeros else 1
    caption = out["scopes"][scope]["captions"][panel]
    if far:
        caption += " Above the axis: " + "; ".join(f"{label} ({fmt(a)})" for a, label, _ in far) + "."
    ax.text(0, -CAPTION_DROP, textwrap.fill(caption, wrap), transform=ax.transAxes, ha="left", va="top", fontsize=14,
            color="#444444")
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(FixedLocator(_decades(lo, hi, 3 if hi / lo > 1e12 else 2)))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(which="major", color="#E2E2E2", lw=.45)
    styles = REFERENCE_STYLE | {"dotted": (0, (1.2, 2.4))}
    refs, k = sorted(refs), 0
    for i, (amount, label, style) in enumerate(refs):
        k = k + 1 if i and amount < refs[i - 1][0] * CROWDED else 0
        below = k == 0 and i + 1 < len(refs) and refs[i + 1][0] < amount * CROWDED
        ax.axhline(amount, color="#909090", ls=styles[style], lw=1.3, zorder=1)
        _fit(ax.text(lo_x * 1.3 if k % 2 else hi_x / 1.3, amount / 1.08 if below else amount * 1.08, label,
                     ha="left" if k % 2 else "right", va="top" if below else "bottom", fontsize=14, color="#555555",
                     zorder=5, bbox={"facecolor": "white", "alpha": .75, "edgecolor": "none", "pad": .5}))
    _floor(ax, zeros, styles, 14)
    for s in series:
        color, ls, lw = STYLE[s["config"]]
        t = tax[s["config"]]
        y0 = s["y"][0] * scale
        ax.plot(s["x"], [y * scale for y in s["y"]], color=color, ls=ls, lw=lw, zorder=3)
        ax.plot([t["low"], t["high"]], [y0, y0], color=color, lw=1.4, zorder=3, marker="|", markersize=12)
        ax.plot([t["tax"]], [y0], marker="o", markersize=13, color=color, zorder=4, mew=2.2,
                markerfacecolor="white" if t["status"] == "priced" else color)
    for s in secondary:
        color, ls, lw = SECONDARY_STYLE
        ax.plot(s["x"], [y * scale for y in s["y"]], color=color, ls=ls, lw=lw, zorder=3)
    configs = list(dict.fromkeys(s["config"] for s in series))
    handles = [plt_line(STYLE[n]) for n in configs] + [plt_line(SECONDARY_STYLE) for _ in secondary]
    labels = [legend(out, n) for n in configs] + [f"{legend(out, s['config'])}, over a {SECONDARY}"
                                                  for s in secondary]
    ax.legend(handles, [fill(label, 64) for label in labels], loc="lower left", bbox_to_anchor=(0, 1.01),
              ncol=2 if len(configs) > 3 else 3, frameon=False, fontsize=14, handlelength=2.6, columnspacing=1.4,
              borderaxespad=0)


def legend(out: dict, name: str) -> str:
    """A line's legend entry: its label, its verification phase and form where its line has variants, and a cut's label on
    the cut's own line."""
    k = out["configurations"][name]
    if name == "every_block":
        return k["label"] + " (priced, not measured)"
    text = k["label"] if k["variant_of"] else k["label"] + (f" ({k['cut']['label']})" if "cut" in k else "")
    if k["registration"] is None:
        return text
    return f"{text}: {out['registrations'][k['registration']]['label']}" + (
        ", the capped sets' form" if k["form"] == "sets" and k["variant_of"] else "")


def plt_line(style: tuple):
    import matplotlib.pyplot as plt

    color, ls, lw = style
    return plt.Line2D([], [], color=color, ls=ls, lw=lw)


def render(out: dict, scope: str, outdir: Path) -> list[Path]:
    """Sample `scope`'s Figure 1 (exfiltration and infiltration) and its skipped-work figure from `out` into `outdir`,
    as `stem(scope, panel)`.png and .svg; returns their paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from render import _save, _sci

    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 20, "axes.labelsize": 21, "xtick.labelsize": 18,
                         "ytick.labelsize": 18, "savefig.facecolor": "white", "figure.facecolor": "white"})
    info = out["scopes"][scope]
    where = "the world's compute" if info["who"] == "world" else "a datacenter's compute"
    paths = []
    for panels, width, height, bottom in ((("exfiltration", "infiltration"), 24, 14.5, .5),
                                          (("skipped_work",), 13, 12, .42)):
        fig, axes = plt.subplots(1, len(panels), figsize=(width, height), squeeze=False)
        left = .06 if len(panels) > 1 else .11
        fig.subplots_adjust(left=left, right=.99, bottom=bottom, top=.88, wspace=.28)
        for ax, panel in zip(axes[0], panels):
            _panel(fig, ax, out, scope, panel, 108 if len(panels) > 1 else 110)
        renderer = fig.canvas.get_renderer()
        legend_top = max(ax.get_legend().get_window_extent(renderer).transformed(fig.transFigure.inverted()).y1
                         for ax in axes[0])
        fig.text(left, max(.963, legend_top + .02), f"{out['model']} served on all of {where} for a "
                 f"{info['period']}: {_sci(info['gpus'])} H200s, {_sci(info['requests'])} requests. Hollow: a priced "
                 "serve tax; the bar, its range.", ha="left", va="bottom", fontsize=21)
        paths += _save(fig, outdir, stem(scope, panels[0]))
    return paths


#: `--serving`'s help, here and in figures.py
SERVING_HELP = ("where serving's rate comes from: rho's fleet utilization (assumed, the default), or the model's "
                "serving_rate card (measured)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("priced-card", help="write the model's priced serve_tax card")
    pc.add_argument("--model", choices=spot_cards.MODELS, default=spot_cards.MODELS[0])
    pc.add_argument("--out", type=Path, required=True)
    pc.add_argument("--serving", choices=f1.SERVING, default=f1.SERVING[0], help=SERVING_HELP)
    rd = sub.add_parser("render", help="plot Figure 1 from its JSON")
    rd.add_argument("json", type=Path)
    rd.add_argument("--scope", action="append", help="a scope's name, as the JSON's `scopes` has it; default all")
    rd.add_argument("--out", type=Path, help="default: the JSON's directory")
    be = sub.add_parser("breakeven", help="write the patterns'-form lines' break-even, breakeven.json and .txt")
    be.add_argument("--model", choices=spot_cards.MODELS, default=spot_cards.MODELS[0])
    be.add_argument("--out", type=Path, required=True)
    be.add_argument("--scope", action="append", help="a scope's name; default all")
    be.add_argument("--workers", type=int, default=1)
    be.add_argument("--no-lean", action="store_true", help="skip Lean's certificate of each break-even")
    be.add_argument("--serving", choices=f1.SERVING, default=f1.SERVING[0], help=SERVING_HELP)
    ct = sub.add_parser("counts", help="write a two-stage line's design at β as the counts the verifier's two-stage "
                                       "rule takes: the first stage's and each block type's")
    ct.add_argument("--model", choices=spot_cards.MODELS, default=spot_cards.MODELS[0])
    ct.add_argument("--line", default="tokens_datacenter", help="a two-stage configuration (default: %(default)s)")
    ct.add_argument("--scope", help="a scope's name; default the first")
    ct.add_argument("--beta", type=float, default=f1.RATIO_BUDGET, help="the audit's share of the scope's compute")
    ct.add_argument("--out", type=Path, required=True)
    ct.add_argument("--serving", choices=f1.SERVING, default=f1.SERVING[0], help=SERVING_HELP)
    args = ap.parse_args(argv)
    if args.cmd == "counts":
        out = counts(args.model, args.line, args.scope, args.beta, serving=args.serving)
        args.out.write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")
        print(f"{args.out}: {out['line']} on {out['scope']} at β = {out['beta']:g}: {out['sampled']:,} of a "
              f"verification phase's {out['verification_phase']['replay_blocks']:,.0f} replay blocks, "
              f"{len(out['k']):,} block types, {out['sessions']:,.0f} sessions, the largest "
              f"{out['largest']:,} proof blocks and none more than {out['tables']:,.3g} tables of its type; "
              f"{_size(out['bound_bits'])}" + (f"; out of reach: {out['limit']}" if out["limit"] else ""))
        for part in beside_parts(out["beside"], out["verification_phase"]["seconds"]):
            print(f"  beside it, modeled: {part}")
        return
    if args.cmd == "priced-card":
        args.out.write_text(json.dumps(priced_card(args.model, args.serving), indent=1) + "\n")
        print(args.out)
        return
    if args.cmd == "breakeven":
        out = breakeven(args.model, tuple(args.scope) if args.scope else None, workers=args.workers,
                        lean=not args.no_lean, serving=args.serving)
        args.out = f1.outdir(args.out, args.serving)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "breakeven.json").write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")
        (args.out / "breakeven.txt").write_text(breakeven_table(out))
        print(breakeven_table(out), end="")
        return
    out = json.loads(args.json.read_text())
    for scope in args.scope or out["scopes"]:
        for path in render(out, scope, args.out or args.json.parent):
            print(path)


if __name__ == "__main__":
    main()
