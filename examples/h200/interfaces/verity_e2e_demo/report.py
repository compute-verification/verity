"""Demo A's report: one or more block records (`block_record`, ``verity/block-record/v1``) and a measurements
file rendered as Markdown in three parts: 1. ran and proved, on one H200, one block;
2. measured on the demo, the inputs to the scaling claim; 3. computed from the same theorem and the measured costs, not
run; then what the demo didn't show, and every number's source.

**Numbers.** Every number is a placeholder ``⟨NAME⟩`` of ``demo_a.template`` (Markdown; not ``.md``, so the suite's
inputs, which leave Markdown out, cover it), filled from one of three places, each name from exactly one:

* `RECORD`: the first block record (the block the report is about), by the path each entry names;
* `MEASURED`: the measurements file (``verity/demo-a-measurements/v1``), each value with the run or field it came from;
* `DERIVED`: a formula over the other two, filled once all its inputs are.

A name with no value renders as ``⟨NAME⟩`` only in a draft (``render(..., draft=True)``, never ``main``), the closing
table saying what fills it; a report is refused while any number is unfilled, naming each. A value may carry a ``label`` (``PyTorch dry run``, ``H200 commit bench``), printed beside it wherever it appears,
for a number that is not the demo block's own; ``--record-label`` labels every number read from the block records,
once at the top and in the sources table.

**Demo A** composes sampled proofs, PoUW and the network certifier (`DEMO_A`): accept when all three accepted on the
block's server, reject when one rejected, pending otherwise, with their errors summed as
``Accounting.Guarantees.JointHorizonLive`` states it. PoUS, the bandwidth audit and the wipe compose at M4 (Demo B).

**The guarantee** cites ``Flock.Guarantees.JointStratified`` when ``JOINT_LOCKED`` is true (it is in the lock), else
the per-subcircuit bound, with the joint figure labelled unlocked.

**The draw** is a handful: k = `HANDFUL` units drawn uniformly over the whole partition by
the live coins, never sized to a proving budget; a block that drew another k says so in the caveats.

**Part 3's H and B** are proofs' planner's (``benchmarks/sampling_plan/demo_a.py``), read from a recorded run
of it: `sampled_proofs.optimize.for_budget` at 1% of a one-week sampling block, a unit's harm its own ANDs, its plan of
record's ``free_compute`` and ``leak`` at δ = 1%. ``H_POOLED``, ``B_POOLED`` and ``PLANNER`` (the planner, its commit and
its run) are filled together, each source naming the run.
Part 3 is circuit size, counted in AND gates, with no FLOP figure or price: each share is
stated against the world's compute in AND gates (``WORLD_GATES_YEAR``, the adversary's resources) with the
deployment's beside it.

**Stubs.** Each of `STUBS` is a caveat until the measurements file's ``fixed`` names it, with the run that shows it
fixed.

**Visible failures**: the draw stays uniform over every unit, and a drawn unit that can't be proved (one
landing in a PoUW tile lane) is a failure with no redraw. The measurements file's ``visible_failures`` names each, with
its subcircuit, unit id, step, slot, why and run; part 1 lists them beside the verdict, and Demo A's verdict on the
block counts them as failures, not skips. Left out, the list renders as ``⟨VISIBLE_FAILURES⟩``; an empty list says
every drawn unit was proved.

**Gathers** are wire references, not proof units: a measured subcircuit named for one is refused.

**Words.** Each of the six subprotocols by its name (sampled proofs, PoUW, PoUS, the bandwidth audit, the network
certifier, the wipe), "proof block", "sampling block", "server-time block" and "subcircuit": `render` refuses a report
that says "leg", "window", "template", "integrity", "work", "space" or "communication" (`BANNED`) outside a code span,
measurement text included.

**M3's report** (``m3.template``): when the measurements file has ``sections`` or ``proved``, the report is M3's,
and no block record is needed. Each section row is one number with its run, its own verdict line quoted and its target
(``{"op": "<=", "value": 0.95}``, or ``{"none": "why"}``); `render` computes its ratio and whether it met the target,
and lists every miss under visible failures. ``proved`` is the drawn-block proving run of record:
``{"run", "drawn": [the coin's indices], "lines": [its output's VERDICT and PROVEBUNDLE lines, verbatim], "text",
"rows"}``, rendered as the first section, one row per drawn proof block, then ``rows`` (section rows: the guarantee the
draw certifies). The first line is `M3_STATUS` only when all `HANDFUL`
drawn proof blocks have a VERDICT line that says accepted (Flock ``--zk``, M0 and Lean) and the final PROVEBUNDLE
line is ok with every negative refused; until then it is `FALLBACK_STATUS` (the file's ``status``, which may say
nothing else), and each block not accepted is a visible failure.

**Per-call noise's serving overhead** is `serve_noise_line`: the seconds it adds a step, its
units' and commit's parts, and two ratios, against the commit-bound step γ_step was measured on and against the plain
served step, never γ_step alone; then the units' share of the tensor peak, and the commit's own overhead as a
separate line. Demo A's part 2 renders it as ``⟨SERVE_NOISE⟩``; in M3's report a section with ``"serve_noise": true``
ends with it, and a section row about per-call noise is refused.

The measurements file::

    {"format": "verity/demo-a-measurements/v1",
     "values": {"PROVING_GPU_S": {"value": 5400.0, "source": "prove run r..., sessions' GPU-seconds", "label": null}},
     "subcircuits": [{"name": "...", "units": 1024, "drawn": 8, "ands_per_unit": 21963, "out_bits": 16,
                      "fixed_s": 30.0, "source": "..."}],
     "fixed": {"pouw-live-salt": "the run that shows it"},
     "visible_failures": [{"subcircuit": "...", "unit": 4093, "step": 12, "slot": 7, "why": "...", "source": "r..."}],
     "caveats": ["what was stubbed"]}

    python -m verity_e2e_demo.report RECORD.json [RECORD.json ...] [--measurements M.json] [--record-label TEXT] \
        [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from verity_e2e_demo import block_record

TEMPLATE = Path(__file__).parent / "demo_a.template"
MEASUREMENTS = "verity/demo-a-measurements/v1"
DELTA = Fraction(1, 100)
#: the share of compute spent on proving in part 3
BUDGET = Fraction(1, 100)
#: M3's draw: k units over the whole partition
HANDFUL = 8
#: part 3's sampling block, in H200-seconds
SAMPLING_BLOCK_S = 7 * 86400
WORLD_LABEL = "world-inference-2027"
#: the first line of M3's report while its drawn H200 proof blocks aren't all proved and accepted
FALLBACK_STATUS = "M3 fallback: live-drawn, opened, replayed (Python); not proved in C-Flock"
#: the first line once every drawn H200 proof block is proved in Flock and accepted by the Lean verifier
M3_STATUS = f"M3: {HANDFUL} live-drawn H200 proof blocks, each proved in C-Flock --zk and accepted by the Lean verifier"
#: what keeps the CPU demo's ACCEPT from being "no stubs", until the measurements file's ``fixed`` names it
STUBS = {
    "pouw-live-salt": "PoUW's salt isn't yet the live coin read in its answer, nor its rows the committed activations, "
                      "so the demo's ACCEPT isn't yet \"no stubs\".",
    "certifier-output-tie": "The network certifier's output tie isn't proved yet: the certified egress isn't yet shown "
                            "equal to the committed output rows inside the period's --zk proof, so the "
                            "demo's ACCEPT isn't yet \"no stubs\".",
    "torch-loop-wiring": "The torch loop's ReLU and per-row quantize are not proved units: each linear's input is a "
                         "committed value, not tied to the previous linear's outputs, until each linear is wired "
                         "as one ScaledMmS8_v1{128,128} unit to QuantRowS8_v1, directly or through "
                         "output-port registered reads, so the demo's ACCEPT isn't yet \"no stubs\".",
    "pouw-salt-order": "PoUW's H1 anchors (x, sx, the weight rows, sw) are committed before the draw coin but not "
                       "before the salt (coin 0), so a prover could choose x after seeing it, which breaks the H1 "
                       "anchors' binding premise, until each activation entry is committed first and the salt "
                       "released against R = SHA-512(tag ‖ W ‖ X ‖ k), so the "
                       "demo's ACCEPT isn't yet \"no stubs\".",
}
DEMO_A = ("integrity", "pouw", "certifier")
PROTOCOL_NAMES = {"integrity": "sampled proofs", "pouw": "PoUW", "certifier": "network certifier", "pous": "PoUS",
                  "bandwidth": "the bandwidth audit", "wipe": "the wipe"}
PERIODS = block_record.PERIODS
#: retired words, and the old names of the six subprotocols; a code span (a field or a quoted verdict line) is exempt
BANNED = re.compile(r"\b(legs?|windows?|templates?|integrity|work|space|communication)\b", re.IGNORECASE)
CODE_SPAN = re.compile(r"(`+)(?:(?!\1).)+?\1")
PLACEHOLDER = re.compile(r"⟨([A-Z][A-Z0-9_]*)⟩")
#: placeholders that stand for a rendered block rather than one number
BLOCKS = ("LABEL", "RECORDS", "VISIBLE_FAILURES", "SUBCIRCUITS", "PERIODS", "CAVEATS", "SOURCES", "SECTIONS", "MISSES",
          "SERVE_NOISE")
#: the numbers `serve_noise_line` states
SERVE_NOISE = ("SERVE_NOISE_S", "SERVE_NOISE_BATCH", "SERVE_NOISE_UNITS_S", "SERVE_NOISE_COMMIT_S",
               "SERVE_NOISE_GAMMA_STEP", "SERVE_NOISE_BASE_S", "SERVE_NOISE_RATIO_4562", "SERVE_STEP_4562_S",
               "SERVE_NOISE_UNITS_PEAK", "SERVE_COMMIT_OVERHEAD_S")
#: a section row that states per-call noise itself, which `serve_noise_line` states instead
NOISE_ROW = re.compile(r"per-call noise|γ_step|gamma_step", re.IGNORECASE)
#: M3's report: the measurements file's ``sections`` and ``proved``
M3_TEMPLATE = Path(__file__).parent / "m3.template"
#: what a section row names: the number, its run, its own verdict line quoted and its target
ROW_FIELDS = ("what", "value", "run", "verdict", "target")
RUN_ID = re.compile(r"\b(r\d{8}-\d{6}-[0-9a-f]{4}|art:[0-9a-f]{8,})")
TARGET_OPS = {"<=": "≤", ">=": "≥", "==": "="}
#: what names a visible failure: the drawn unit, why it wasn't proved, and the run that shows it
FAILURE_FIELDS = ("subcircuit", "unit", "step", "slot", "why", "source")

#: read from the first block record: {NAME: the record's field}
RECORD: dict[str, str] = {
    "RECORD_RUN": "run",
    "BLOCK_SERVER": "server",
    "BLOCK_STEPS": "server_time's steps [lo, hi)",
    "DEMO_A_VERDICT": "sampled proofs', PoUW's and the network certifier's answer.verdict on the block's server",
    "DRAWN_UNITS": "`integrity` › quantity.drawn",
    "UNITS_TOTAL": "`integrity` › quantity.of",
    "LEAN_VERDICT": "`integrity` › answer.verdict (the Lean verifier's audit)",
    "H_UNITS": "`integrity` › at 1/100 › value",
    "INTEGRITY_ERROR": "`integrity` › error",
    "POUW_TILES": "pouw › answer.public.check's tile Definition",
    "POUW_TILES_DRAWN": "pouw › answer.public.drawn (its count)",
    "POUW_W_CERT": "pouw › quantity.value (W_cert, at 1/100)",
    "POUW_TILES_CREDITED": "pouw › quantity.tiles",
    "POUW_ERROR": "pouw › error",
    "POUW_BACKING": "pouw › answer.public.check's tile Definition: per-call noise (v5/v6) or the v1 subset",
    "NET_K": "certifier › quantity.value (K, at 1/100)",
    "NET_ERROR": "certifier › error",
    "JOINT_HORIZON_STATUS": "composed › joint_horizon.statement (name, proved)",
    "DEMO_A_ERROR_SUM": "the three protocols' error, summed",
    "D_GPUS": "monitoring_period.deployment: machines × gpus_per_machine (`benchmarks/census/deployments.json`)",
    "DEPLOYMENT_ID": "monitoring_period.deployment.id",
}

#: filled by the measurements file: {NAME: which run or field fills it}
MEASURED: dict[str, str] = {
    "COMMITTED_ROWS": "the block run: the per-step commit calls' rows (`commit_slots`), summed "
                      "over the block's steps",
    "COMMITTED_BYTES": "the block run: the bytes those commit calls hashed",
    "BLOCK_ROOT": "the block root of the block run's drawn-unit bundle (`verity/drawn-unit-bundle`)",
    "REGISTRATION_ROOT": "the seeded weights' registration root (`e2e_h200/served_weights.py`)",
    "CHALLENGER_RUN": "the block run: the challenger's coin record for the block's draw",
    "ZK_SESSIONS": "the prove run: its C-Flock --zk sessions behind the firewall (`sessions/index.jsonl`)",
    "PROVE_RUN": "the prove run's id",
    "FLIP_VERDICT": "the flipped-bit run: the Lean verifier's verdict with one committed bit flipped",
    "FLIP_RUN": "the flipped-bit run's id",
    "JOINT_LOCKED": "true once `Flock.Guarantees.JointStratified` is in the lock (`Security/lean-audit.json`) on main",
    "EPS_KS": "the bound's knowledge-soundness term for C-Flock --zk at the block's sessions",
    "EPS_BIND": "the bound's binding term for the block's hm96-sha512 commitments",
    "H_A": "the bound (BOUND_CITATION) at δ = 1/100 on the block's draw, harm in ANDs: "
           "`sampled_proofs.optimize.certify` on the block's sizes and k",
    "H_B": "the same bound, harm in output bits",
    "BLOCK_ANDS": "the block's circuit: ANDs over all its proof blocks (circuit-check's ANDs a unit × units, per "
                  "subcircuit)",
    "BLOCK_OUTPUT_BITS": "the block's circuit: output bits over all its proof blocks",
    "POUW_EPS": "the per-call noise run: PoUW's ε at δ = 1/100 from the block's one draw, its tiles a subcircuit at the "
                "planner's k",
    "POUW_COVERAGE": "the per-call noise run: the share of the block's static INT8 MACs its tiles cover (`range`)",
    "POUW_CERTIFIED_SHARE": "the per-call noise run: PoUW's certified share at POUW_RHO's ρ",
    "POUW_RHO": "ρ, the live share in `H1J10RCheckedCreditPeriod`",
    "PROVING_GPU_S": "the prove run: GPU-seconds of proving over every drawn unit's session",
    "BLOCK_GPU_S": "the block run: H200-seconds the block served",
    "PROVE_S_PER_AND": "the prove run, fitted by `sampled_proofs.costfit`: GPU-seconds of proving per AND",
    "REPLAY_GPU_S": "the block run: H200-seconds to replay the block's drawn units (the drawn-unit bundle's builder)",
    "COMMIT_GBPS": "the block run: the commit call's throughput on the H200",
    "PLANNER_PREDICTED_S": "`sampled_proofs.optimize`'s cost for the block's draw, at the cost model fitted before the "
                           "block ran",
    "SERVE_COMMIT_S": "the block run: seconds of commits a step",
    "SERVE_STEP_S": "the block run: seconds a step",
    "SERVE_NOISE_S": "the per-call noise run (kraqwxsc): seconds per-call noise adds a step, its source naming the noise "
                     "run, then the plain run it is measured against (`r... against r...`)",
    "SERVE_NOISE_BATCH": "the batch those two runs served",
    "SERVE_NOISE_UNITS_S": "of SERVE_NOISE_S, the seconds a step of per-call noise's units",
    "SERVE_NOISE_COMMIT_S": "of SERVE_NOISE_S, the seconds a step of committing their values",
    "SERVE_NOISE_GAMMA_STEP": "γ_step, the noise run's step over the plain run's",
    "SERVE_NOISE_BASE_S": "the plain run's served step, mostly FAMILIES' slot commit (`commit_gpu`)",
    "SERVE_STEP_4562_S": "the earlier per-call build's (`percall:kraq`) served step on the H200, in seconds",
    "SERVE_NOISE_UNITS_PEAK": "the tensor-peak run: the share of the tensor peak per-call noise's units "
                              "reach, ALU-bound",
    "SERVE_COMMIT_OVERHEAD_S": "the commit overhead run: the commit's own overhead a step at 1 KiB rows, with each add's "
                               "tables dropped at their read",
    "SIMULATED_RTT_MS": "the block run: the challenger's simulated round trip",
    "STEP_S_B512": "M3's block run on the H200 at batch 512: seconds a step",
    "ROW_SLOT_NS_B512": "the same run: nanoseconds a step per row-slot",
    "STEP_S_B1024": "M3's block run on the H200 at batch 1,024: seconds a step",
    "ROW_SLOT_NS_B1024": "the same run: nanoseconds a step per row-slot",
    "STEP_S_KERNELS": "kernels' faster step at batch 512: seconds a step",
    "WORLD_GATES_YEAR": f"the world's compute, AND gates a year, labelled {WORLD_LABEL} (census entry pending)",
    "PLANNER": "proofs' planner (`benchmarks/sampling_plan/demo_a.py`) at its commit, and the recorded run of it "
               "at 1% of a one-week sampling block (`--spans week`)",
    "H_POOLED": "PLANNER's run: `for_budget` at 1% of a one-week sampling block, a unit's harm its own ANDs, the plan of "
                "record's faulty ANDs at δ = 1% (`plan_of_record-week.json` › at › 0.01 › free_compute)",
    "B_POOLED": "PLANNER's run: the same plan's leaked bits at δ = 1%, output bits plus which units are wrong "
                "(`plan_of_record-week.json` › at › 0.01 › leak)",
}
#: part 3's numbers, filled together from one recorded run of proofs' planner
PLANNED = ("PLANNER", "H_POOLED", "B_POOLED")

_PERIOD_DERIVED: dict[str, str] = {}
for _p, _ in PERIODS:
    _P = _p.upper()
    _world = f"WORLD_GATES_YEAR × a {_p} / a year"
    _PERIOD_DERIVED |= {
        f"BLOCKS_{_P}": f"D_GPUS × a {_p} / a week: the {_p}'s one-week sampling blocks",
        f"H_WORLD_SHARE_{_P}": f"H_POOLED / ({_world})",
        f"H_SHARE_{_P}": f"H_POOLED / (D_GPUS × a {_p} × BLOCK_ANDS / BLOCK_GPU_S)",
        f"B_WORLD_SHARE_{_P}": f"B_POOLED / ({_world} × BLOCK_OUTPUT_BITS / BLOCK_ANDS)",
        f"B_SHARE_{_P}": f"B_POOLED / (D_GPUS × a {_p} × BLOCK_OUTPUT_BITS / BLOCK_GPU_S)",
    }

#: formulas over the other numbers: {NAME: the formula}
DERIVED: dict[str, str] = {
    "H_A_SHARE": "H_A / BLOCK_ANDS",
    "Y": "PROVING_GPU_S / BLOCK_GPU_S",
    "PLANNER_RATIO": "PROVING_GPU_S / PLANNER_PREDICTED_S",
    "SERVE_COMMIT_SHARE": "SERVE_COMMIT_S / SERVE_STEP_S",
    "SERVE_NOISE_RATIO_4562": "(SERVE_STEP_4562_S + SERVE_NOISE_S) / SERVE_STEP_4562_S, to two figures",
    "BOUND_CITATION": "JOINT_LOCKED: `Flock.Guarantees.JointStratified` when locked, else the "
                      "per-subcircuit bound with the joint figure unlocked",
    "HORIZON_CITATION": "JOINT_LOCKED: `JointStratifiedHorizon` when locked, else "
                        "`Flock.Guarantees.IntegrityHorizon` per subcircuit with the joint figure unlocked",
    "SUBCIRCUIT_COUNT": "the subcircuits measured (part 2), else the block record's per-subcircuit draw",
    "N_HOURS": "the largest subcircuit fixed proving cost / 1%, in hours",
    "M_MINUTES": "N_HOURS × 60 / D_GPUS",
    **_PERIOD_DERIVED,
}


@dataclass(frozen=True)
class V:
    """One filled number: its value, where it came from, and the label it carries when it isn't the demo block's own."""

    value: Any
    source: str
    label: str | None = None


def _fmt(x: Any) -> str:
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, Fraction):
        x = int(x) if x.denominator == 1 else float(x)
    if isinstance(x, int):
        return f"{x:,}"
    if isinstance(x, float):
        a = abs(x)
        if a == 0:
            return "0"
        if a >= 1e12 or a < 1e-3:
            return f"{x:.3g}"
        return f"{x:,.0f}" if a >= 1000 else f"{x:.4g}"
    s = str(x)
    return f"`{s[:16]}…`" if len(s) > 40 and re.fullmatch(r"[0-9a-f]+", s) else s


def _pct(x: Fraction | float) -> str:
    return f"{float(x) * 100:.3g}%"


def _num(v: V | None) -> Fraction | None:
    if v is None or isinstance(v.value, (bool, str)):
        return None
    return Fraction(v.value) if isinstance(v.value, int) else Fraction(str(v.value))


def _tiles(public: Mapping[str, Any] | None) -> str | None:
    t = ((public or {}).get("check") or {}).get("template")
    return t.split("{", 1)[0] if t else None


def demo_a_verdict(record: Mapping[str, Any], failures: int = 0) -> str:
    """Demo A's composed verdict over `DEMO_A` on the record's block, counting its `failures` visible failures (module
    docstring)."""
    legs, server = record["legs"], record["server"]
    rejected = [p for p in DEMO_A if legs.get(p) and legs[p]["server"] == server
                and legs[p]["answer"]["verdict"] in ("reject", "error")]
    if rejected:
        return "reject (" + ", ".join(PROTOCOL_NAMES[p] for p in rejected) + ")"
    if failures:
        return f"fail ({failures:,} drawn unit{'s' * (failures != 1)} not proved: visible failures, no redraw)"
    absent = [p for p in DEMO_A if not legs.get(p) or legs[p]["server"] != server]
    if absent:
        return "pending: no " + ", no ".join(PROTOCOL_NAMES[p] for p in absent) + " on the block's server"
    waiting = [p for p in DEMO_A if legs[p]["answer"]["verdict"] != "accept"]
    if waiting:
        return "pending (" + ", ".join(f"{PROTOCOL_NAMES[p]}: {legs[p]['answer']['verdict']}" for p in waiting) + ")"
    return "accept"


def from_record(record: Mapping[str, Any], label: str | None = None) -> dict[str, V]:
    """The `RECORD` numbers of one block record."""
    if bad := block_record.check(record):
        raise ValueError(f"block record {record.get('run')}: {bad}; `python -m verity_e2e_demo.block_record check` it")
    out: dict[str, V] = {}

    def put(name: str, value: Any) -> None:
        out[name] = V(value, f"block record {record['run']} › {RECORD[name]}", label)

    legs, server = record["legs"], record["server"]
    lo, hi = record["server_time"][0]["steps"]
    put("RECORD_RUN", record["run"])
    put("BLOCK_SERVER", server)
    put("BLOCK_STEPS", f"[{lo}, {hi})")
    put("DEMO_A_VERDICT", demo_a_verdict(record))
    here = {p: legs[p] for p in DEMO_A if legs.get(p) and legs[p]["server"] == server}
    if i := here.get("integrity"):
        put("DRAWN_UNITS", i["quantity"]["drawn"])
        put("UNITS_TOTAL", i["quantity"]["of"])
        put("LEAN_VERDICT", str(i["answer"]["verdict"]).upper())
        if (at := i["at"][str(DELTA)])["holds"]:
            put("H_UNITS", at["value"])
        put("INTEGRITY_ERROR", Fraction(i["error"]))
    if w := here.get("pouw"):
        public = w["answer"].get("public") or {}
        if tiles := _tiles(public):
            per_call = bool(re.search(r"_v[56]\b", tiles))
            put("POUW_TILES", f"per-call-noise tiles ({tiles})" if per_call else f"the v1 subset's tiles ({tiles})")
            put("POUW_BACKING", f"PoUW backing: shown, by the per-call-noise tiles ({tiles})." if per_call else
                f"PoUW backing: this block's PoUW ran the v1 subset's tiles ({tiles}), not the per-call-noise tiles, "
                f"so PoUW's figure in part 1 is the v1 subset's, for what it is, not the block's backing.")
        if isinstance(public.get("drawn"), list):
            put("POUW_TILES_DRAWN", len(public["drawn"]))
        if w["at"][str(DELTA)]["holds"]:
            put("POUW_W_CERT", w["quantity"]["value"])
        put("POUW_TILES_CREDITED", w["quantity"]["tiles"])
        put("POUW_ERROR", Fraction(w["error"]))
    if c := here.get("certifier"):
        if c["at"][str(DELTA)]["holds"]:
            put("NET_K", c["quantity"]["value"])
        put("NET_ERROR", Fraction(c["error"]))
    if len(here) == len(DEMO_A):
        put("DEMO_A_ERROR_SUM", sum((Fraction(here[p]["error"]) for p in DEMO_A), Fraction(0)))
    st = record["composed"]["joint_horizon"]["statement"]
    put("JOINT_HORIZON_STATUS", f"`{st['name']}`, " + ("proved" if st.get("proved") else "stated, not yet proved"))
    dep = record["monitoring_period"]["deployment"]
    put("D_GPUS", int(dep["machines"]) * int(dep["gpus_per_machine"]))
    put("DEPLOYMENT_ID", dep["id"])
    return out


def load_measurements(doc: Mapping[str, Any]) -> tuple[dict[str, V], list[dict[str, Any]], list[str], dict[str, str],
                                                       list[dict[str, Any]] | None]:
    """The measurements file's values, subcircuits, caveats, fixed stubs and visible failures (None when it doesn't
    say); refuses a name it may not fill, a gather as a subcircuit, and a value, a fix or a failure with no source."""
    if doc.get("format") != MEASUREMENTS:
        raise ValueError(f"measurements: format {doc.get('format')!r}, not {MEASUREMENTS}")
    values: dict[str, V] = {}
    for name, x in (doc.get("values") or {}).items():
        if name not in MEASURED:
            where = ("comes from the block record" if name in RECORD else
                     f"is derived ({DERIVED[name]})" if name in DERIVED else "is no number of the report")
            raise ValueError(f"measurements: ⟨{name}⟩ {where}; fill only the names of `report.MEASURED`")
        if not isinstance(x, Mapping) or "value" not in x or x["value"] is None:
            raise ValueError(f"measurements: ⟨{name}⟩ has no value; leave it out to keep it unfilled")
        if not str(x.get("source") or "").strip():
            raise ValueError(f"measurements: ⟨{name}⟩ names no source; say which run or field it came from")
        values[name] = V(x["value"], x["source"], x.get("label"))
    if "SERVE_NOISE_S" in values and len(RUN_ID.findall(values["SERVE_NOISE_S"].source)) != 2:
        raise ValueError("measurements: ⟨SERVE_NOISE_S⟩'s source names its two runs, the noise run then the plain run "
                         "it is measured against (`r... against r...`)")
    if planned := [n for n in PLANNED if n in values]:
        runs = {n: set(RUN_ID.findall(values[n].source)) for n in planned}
        if len(planned) != len(PLANNED) or not all(runs.values()) or len(set.intersection(*runs.values())) == 0:
            raise ValueError(f"measurements: {', '.join(f'⟨{n}⟩' for n in PLANNED)} are filled together from one "
                             f"recorded run of proofs' planner, each source naming that run (`benchmarks/sampling_plan/demo_a.py "
                             f"--spans week`); got {sorted(planned)}")
    if "WORLD_GATES_YEAR" in values and values["WORLD_GATES_YEAR"].label != WORLD_LABEL:
        raise ValueError(f"measurements: ⟨WORLD_GATES_YEAR⟩ is labelled {WORLD_LABEL!r}, the census entry's")
    subcircuits = [dict(s) for s in doc.get("subcircuits") or []]
    for s in subcircuits:
        if not s.get("name") or not str(s.get("source") or "").strip():
            raise ValueError(f"measurements: subcircuit {s!r} needs a name and a source")
        if re.match(r"(All)?Gather", s["name"]):
            raise ValueError(f"measurements: subcircuit {s['name']!r}: a gather is a plain wire reference, not a proof "
                             f"unit; leave it out")
    fixed = dict(doc.get("fixed") or {})
    for k, src in fixed.items():
        if k not in STUBS or not str(src or "").strip():
            raise ValueError(f"measurements: fixed {k!r} needs to be one of {sorted(STUBS)} with the run that shows it")
    failures = doc.get("visible_failures")
    if failures is not None:
        failures = [dict(f) for f in failures]
        for f in failures:
            if any(f.get(k) is None or str(f[k]).strip() == "" for k in FAILURE_FIELDS):
                raise ValueError(f"measurements: visible failure {f!r} needs {', '.join(FAILURE_FIELDS)}")
    return values, subcircuits, [str(c) for c in doc.get("caveats") or []], fixed, failures


def derive(values: Mapping[str, V], subcircuits: Sequence[Mapping[str, Any]]) -> dict[str, V]:
    """The `DERIVED` numbers whose inputs are all filled."""
    out: dict[str, V] = {}

    def put(name: str, value: Any, *inputs: str) -> None:
        mine = sorted({values[i].label for i in inputs if i in values and values[i].label})
        out[name] = V(value, DERIVED[name], "; ".join(mine) or None)

    def ratio(name: str, a: str, b: str, pct: bool = False) -> None:
        x, y = _num(values.get(a)), _num(values.get(b))
        if x is not None and y:
            put(name, _pct(x / y) if pct else float(x / y), a, b)

    ratio("H_A_SHARE", "H_A", "BLOCK_ANDS", pct=True)
    ratio("Y", "PROVING_GPU_S", "BLOCK_GPU_S")
    ratio("PLANNER_RATIO", "PROVING_GPU_S", "PLANNER_PREDICTED_S")
    ratio("SERVE_COMMIT_SHARE", "SERVE_COMMIT_S", "SERVE_STEP_S", pct=True)
    noise, step = _num(values.get("SERVE_NOISE_S")), _num(values.get("SERVE_STEP_4562_S"))
    if noise is not None and step:
        put("SERVE_NOISE_RATIO_4562", f"{float((step + noise) / step):.2g}", "SERVE_NOISE_S", "SERVE_STEP_4562_S")
    if (lk := values.get("JOINT_LOCKED")) is not None:
        put("BOUND_CITATION", "`Flock.Guarantees.JointStratified` (locked)" if lk.value is True else
            "the per-subcircuit bound, the joint figure unlocked", "JOINT_LOCKED")
        put("HORIZON_CITATION", "`JointStratifiedHorizon` (locked)" if lk.value is True else
            "`Flock.Guarantees.IntegrityHorizon` per subcircuit, the joint figure unlocked", "JOINT_LOCKED")
    if subcircuits:
        out["SUBCIRCUIT_COUNT"] = V(len(subcircuits), DERIVED["SUBCIRCUIT_COUNT"], "; ".join(
            sorted({s["label"] for s in subcircuits if s.get("label")})) or None)
        fixed = [s.get("fixed_s") for s in subcircuits]
        if all(isinstance(f, (int, float)) and not isinstance(f, bool) for f in fixed):
            out["N_HOURS"] = V(float(Fraction(str(max(fixed))) / BUDGET / 3600), DERIVED["N_HOURS"], out[
                "SUBCIRCUIT_COUNT"].label)
    d = _num(values.get("D_GPUS"))
    n = _num(out.get("N_HOURS"))
    if n is not None and d:
        out["M_MINUTES"] = V(float(n * 60 / d), DERIVED["M_MINUTES"], out["N_HOURS"].label)
    values = {**values, **out}
    year = dict(PERIODS)["year"]
    gs, world, ands = (_num(values.get(k)) for k in ("BLOCK_GPU_S", "WORLD_GATES_YEAR", "BLOCK_ANDS"))
    for p, seconds in PERIODS:
        P = p.upper()
        if d:
            put(f"BLOCKS_{P}", float(d * seconds / SAMPLING_BLOCK_S), "D_GPUS")
        for harm, total, share in (("H_POOLED", "BLOCK_ANDS", "H"), ("B_POOLED", "BLOCK_OUTPUT_BITS", "B")):
            h, t = _num(values.get(harm)), _num(values.get(total))
            if h is not None and t and world and ands:
                put(f"{share}_WORLD_SHARE_{P}", _pct(h / (world * seconds / year * t / ands)), harm, total,
                    "WORLD_GATES_YEAR", "BLOCK_ANDS")
            if h is not None and t and gs and d:
                put(f"{share}_SHARE_{P}", _pct(h / (d * seconds * t / gs)), harm, total, "BLOCK_GPU_S", "D_GPUS")
    return out


def _cell(name: str, values: Mapping[str, V], quiet: str | None = None, fmt=_fmt) -> str:
    """A number in place, with its label unless it is ``quiet`` (the records' label, stated once at the top)."""
    v = values.get(name)
    if v is None:
        return f"⟨{name}⟩"
    return fmt(v.value) + (f" *({v.label})*" if v.label and v.label != quiet else "")


def serve_noise_line(values: Mapping[str, V], quiet: str | None = None) -> str:
    """Per-call noise's serving overhead as the seconds it adds a step, with both ratios (module docstring)."""
    c = lambda n, fmt=_fmt: _cell(n, values, quiet, fmt)

    def runs(name: str) -> str:
        v = values.get(name)
        return " against ".join(RUN_ID.findall(v.source)) if v and RUN_ID.search(v.source) else f"⟨{name}⟩'s run"

    return (f"Per-call noise adds **+{c('SERVE_NOISE_S')} s a step at batch {c('SERVE_NOISE_BATCH')}** "
            f"({c('SERVE_NOISE_UNITS_S')} s of units plus {c('SERVE_NOISE_COMMIT_S')} s of commit, "
            f"{runs('SERVE_NOISE_S')}): {c('SERVE_NOISE_GAMMA_STEP')}× against that commit-bound "
            f"{c('SERVE_NOISE_BASE_S')} s served step, which is mostly FAMILIES' slot commit (`commit_gpu`), and about "
            f"{c('SERVE_NOISE_RATIO_4562')}× against the earlier per-call build's {c('SERVE_STEP_4562_S')} s served "
            f"step. The units are ALU-bound at about {c('SERVE_NOISE_UNITS_PEAK', _pct)} of the tensor peak "
            f"({runs('SERVE_NOISE_UNITS_PEAK')}). The commit's own overhead is a separate line: "
            f"+{c('SERVE_COMMIT_OVERHEAD_S')} s at 1 KiB rows ({runs('SERVE_COMMIT_OVERHEAD_S')}).")


def _records(records: Sequence[Mapping[str, Any]], failures: int = 0) -> str:
    head = ("| block record | server | steps | " + " | ".join(PROTOCOL_NAMES[p] for p in DEMO_A)
            + " | Demo A | all six audits |")
    rows = [head, "|" + " --- |" * (len(DEMO_A) + 5)]
    def cell(r: Mapping[str, Any], p: str) -> str:
        lg = r["legs"].get(p)
        if not lg:
            return "not run"
        where = "" if lg["server"] == r["server"] else ", another server"
        return f"{lg['answer']['verdict']} ({lg['source']}{where})"

    for i, r in enumerate(records):
        lo, hi = r["server_time"][0]["steps"]
        rows.append(f"| {r['run']}{' (this report)' if i == 0 else ''} | {r['server']} | [{lo}, {hi}) | "
                    + " | ".join(cell(r, p) for p in DEMO_A)
                    + f" | {demo_a_verdict(r, failures if i == 0 else 0)} | {r['composed']['verdict']} |")
    return "\n".join(rows)


def _failures(failures: Sequence[Mapping[str, Any]] | None) -> str:
    if failures is None:
        return "⟨VISIBLE_FAILURES⟩"
    if not failures:
        return "None: every drawn unit was proved."
    rows = ["| subcircuit | unit | step | slot | why | run |", "|" + " --- |" * 6]
    rows += [f"| {f['subcircuit']} | {_fmt(f['unit'])} | {_fmt(f['step'])} | {_fmt(f['slot'])} | {f['why']} | "
             f"{f['source']} |" for f in failures]
    return "\n".join(rows)


def _subcircuits(subcircuits: Sequence[Mapping[str, Any]], quiet: str | None) -> str:
    cols = (("units", "units"), ("drawn", "drawn"), ("ands_per_unit", "ANDs a unit"),
            ("out_bits", "output bits a unit"), ("fixed_s", "fixed proving cost per sampling block (GPU-s)"))
    rows = ["| subcircuit | " + " | ".join(c for _, c in cols) + " |", "|" + " --- |" * (len(cols) + 1)]
    if not subcircuits:
        rows.append("| ⟨SUBCIRCUITS⟩ |" + " ⟨SUBCIRCUITS⟩ |" * len(cols))
    for s in subcircuits:
        lab = f" *({s['label']})*" if s.get("label") and s["label"] != quiet else ""
        rows.append(f"| {s['name']}{lab} | " + " | ".join(
            _fmt(s[k]) if s.get(k) is not None else f"⟨SUBCIRCUITS.{k}⟩" for k, _ in cols) + " |")
    return "\n".join(rows)


def _period_names(p: str) -> tuple[str, ...]:
    P = p.upper()
    return (f"BLOCKS_{P}", "H_POOLED", f"H_WORLD_SHARE_{P}", f"H_SHARE_{P}", "B_POOLED", f"B_WORLD_SHARE_{P}",
            f"B_SHARE_{P}")


def _periods(values: Mapping[str, V], quiet: str | None) -> str:
    rows = [("| period | one-week sampling blocks | H faulty AND gates | share of the world's AND gates | share of "
             "the deployment's AND gates | B leaked bits | share of the world's output bits | share of the "
             "deployment's output bits |"), "|" + " --- |" * 8]
    for p, _ in PERIODS:
        rows.append(f"| {p} | " + " | ".join(_cell(n, values, quiet) for n in _period_names(p)) + " |")
    return "\n".join(rows)


def _caveats(records: Sequence[Mapping[str, Any]], values: Mapping[str, V], extra: Sequence[str],
             quiet: str | None, fixed: Mapping[str, str]) -> str:
    lines = [STUBS[k] for k in STUBS if k not in fixed]
    if not records:
        lines += [f"Numbers marked *({lab})* are not the demo block's own." for lab in
                  sorted({v.label for v in values.values() if v.label})]
        return "\n".join(f"- {x}" for x in [*lines, *extra])
    r = records[0]
    if (k := values.get("DRAWN_UNITS")) is not None and k.value != HANDFUL:
        lines.append(f"This block drew k = {_fmt(k.value)}, not M3's handful of {HANDFUL}.")
    lines.append("The challenger and the network certifier run on the prover's node for the demo, not on a machine the "
                 "prover doesn't control: the demo assumes the prover leaves them alone.")
    for p in DEMO_A:
        lg = r["legs"].get(p)
        if lg is None or lg["server"] != r["server"]:
            lines.append(f"{PROTOCOL_NAMES[p].capitalize()} didn't run on the block's server.")
        elif lg["source"] == "fixture":
            lines.append(f"{PROTOCOL_NAMES[p].capitalize()} ran on a fixture, not on the block "
                         f"(block record {r['run']}).")
    if not r["composed"]["joint_horizon"]["statement"].get("proved"):
        lines.append("JointHorizonLive is stated, not yet proved: the error sum in part 1 is the sum it states.")
    lines.append("PRF-167: the state carried across a wipe has no cross-session binding term yet, so the bound holds "
                 "per server-time block.")
    lines.append("`pous-ring-instantiation` is M4's: PoUS's S_cert, the bandwidth audit's Q_cert and the wipe's foreign_bits "
                 "compose at M4 (Demo B), not in Demo A.")
    labels = sorted({v.label for v in values.values() if v.label})
    if quiet:
        lines.append(f"The numbers read from the block records, and the measurements labelled *({quiet})*, are the "
                     f"{quiet}'s, not the demo block's.")
    for lab in labels:
        if lab != quiet:
            lines.append(f"Numbers marked *({lab})* are not the demo block's own.")
    lines += list(extra)
    return "\n".join(f"- {x}" for x in lines)


def _sources(order: Sequence[str], values: Mapping[str, V], subcircuits: Sequence[Mapping[str, Any]],
             failures: Sequence[Mapping[str, Any]] | None) -> str:
    rows = ["| number | value | filled from |", "| --- | --- | --- |"]
    filled = 0
    for name in order:
        v = values.get(name)
        if name == "VISIBLE_FAILURES":
            filled += failures is not None
            rows.append(f"| ⟨VISIBLE_FAILURES⟩ | {'unfilled' if failures is None else f'{len(failures):,}'} | "
                        + ("; ".join(sorted({str(f['source']) for f in failures})) or "the measurements file: none"
                           if failures is not None else "the measurements file's visible_failures: each drawn unit "
                           "that couldn't be proved (one in a PoUW tile lane), with its run") + " |")
            continue
        if name == "SUBCIRCUITS":
            src = "; ".join(f"{s['name']}: {s['source']}" for s in subcircuits) or \
                  "the block run and the prove run: per subcircuit its units, drawn units, ANDs a unit, " \
                  "output bits a unit and fixed proving cost per sampling block (`sampled_proofs.costfit`'s " \
                  "statement_fixed)"
            filled += bool(subcircuits)
            rows.append(f"| ⟨SUBCIRCUITS⟩ | {'filled' if subcircuits else 'unfilled'} | {src} |")
            continue
        if v is not None:
            filled += 1
            src = v.source + (f" *({v.label})*" if v.label else "")
            rows.append(f"| ⟨{name}⟩ | {_fmt(v.value)} | {src} |")
        else:
            src = MEASURED.get(name) or DERIVED.get(name) or f"block record › {RECORD[name]}"
            rows.append(f"| ⟨{name}⟩ | unfilled | {src} |")
    head = (f"Every number filled ({filled})." if filled == len(order) else
            f"A draft: {filled} of {len(order)} numbers filled; an unfilled one shows as ⟨NAME⟩.")
    return f"{head}\n\n" + "\n".join(rows)


def render(records: Sequence[Mapping[str, Any]], measurements: Mapping[str, Any] | None = None, *,
           record_label: str | None = None, template: str | None = None, draft: bool = False) -> str:
    """The report's Markdown (module docstring); refuses one with an unfilled number unless it is a ``draft``."""
    doc = measurements or {"format": MEASUREMENTS}
    sections = list(doc.get("sections") or [])
    status = str(doc.get("status") or "").strip()
    if doc.get("proved") is not None:
        if status and status != FALLBACK_STATUS:
            raise ValueError(f"measurements: with `proved`, the status is {FALLBACK_STATUS!r} or left out; `render` "
                             f"says {M3_STATUS!r} once every drawn proof block is proved and accepted")
        section, complete = proved_section(doc["proved"])
        sections.insert(0, section)
        status = M3_STATUS if complete else FALLBACK_STATUS
    m3 = bool(sections)
    if m3 and not status:
        raise ValueError("measurements: sections need a status, the report's first line (what was and wasn't shown)")
    if not records and not m3:
        raise ValueError("render needs at least one block record: the first is the block the report is about")
    text = (M3_TEMPLATE if m3 else TEMPLATE).read_text() if template is None else template
    measured, subcircuits, caveats, fixed, failures = load_measurements(doc)
    rows = section_rows(sections)
    values = {**(from_record(records[0], record_label) if records else {}), **measured}
    if failures and records:
        v = values["DEMO_A_VERDICT"]
        values["DEMO_A_VERDICT"] = V(demo_a_verdict(records[0], len(failures)),
                                     v.source + ", with the measurements file's visible failures", v.label)
    for r in records[1:]:
        from_record(r)
    if records and "SUBCIRCUIT_COUNT" not in values and not subcircuits:
        per = (records[0]["legs"].get("integrity") or {}).get("answer", {}).get("public", {}).get("per_template")
        if per:
            values["SUBCIRCUIT_COUNT"] = V(len(per), DERIVED["SUBCIRCUIT_COUNT"], record_label)
    values |= derive(values, subcircuits)
    known = set(RECORD) | set(MEASURED) | set(DERIVED) | set(BLOCKS)
    order: list[str] = []
    for name in PLACEHOLDER.findall(text):
        if name not in known:
            raise ValueError(f"the template's ⟨{name}⟩ is no number of `report` (RECORD, MEASURED, DERIVED or BLOCKS)")
        names = [name] if name not in BLOCKS or name in ("SUBCIRCUITS", "VISIBLE_FAILURES") else []
        if name in ("BOUND_CITATION", "HORIZON_CITATION"):
            names.append("JOINT_LOCKED")
        if name == "PERIODS":
            names = [n for p, _ in PERIODS for n in _period_names(p)]
        if name == "SERVE_NOISE" or name == "SECTIONS" and any(s.get("serve_noise") for s in sections):
            names = list(SERVE_NOISE)
        order += [n for n in dict.fromkeys(names) if n not in order]
    banner = (f"> The {record_label}'s numbers are not marked in place: every number read from the block records "
              f"(block record {records[0]['run']}), and every measurement labelled *({record_label})*, is the "
              f"{record_label}'s, not the demo block's. Any other number that isn't the demo block's own is marked "
              f"in place.") if record_label and records else "> A number that isn't the demo block's own is marked in place."
    blocks = {"LABEL": banner, "RECORDS": _records(records, len(failures or ())) if records else "No block record.",
              "VISIBLE_FAILURES": _failures(failures), "SUBCIRCUITS": _subcircuits(subcircuits, record_label),
              "PERIODS": _periods(values, record_label),
              "CAVEATS": _caveats(records, values, caveats, record_label, fixed),
              "SOURCES": _sources(order, values, subcircuits, failures),
              "SECTIONS": _sections(sections, rows, serve_noise_line(values, record_label)),
              "MISSES": _misses(rows, failures), "SERVE_NOISE": serve_noise_line(values, record_label)}
    sources = "\0SOURCES\0"
    out = PLACEHOLDER.sub(lambda m: sources if m[1] == "SOURCES" else blocks[m[1]] if m[1] in blocks else
                          _cell(m[1], values, record_label), text)
    if (unfilled := list(dict.fromkeys(PLACEHOLDER.findall(out)))) and not draft:
        raise ValueError(f"the report has {len(unfilled)} unfilled number(s): "
                         + ", ".join(f"⟨{n}⟩ ({MEASURED.get(n) or DERIVED.get(n) or RECORD.get(n) or 'a block'})"
                                     for n in unfilled)
                         + "; fill each in the measurements file (a draft, `render(..., draft=True)`, is never "
                           "published)")
    out = out.replace(sources, blocks["SOURCES"])
    if status:
        out = f"{status}\n\n{out}"
    vocabulary(out)
    return out


def vocabulary(text: str) -> None:
    """Refuses a report that says "leg", "window" or "template" (retired words), or a subprotocol's old name
    ("integrity", "work", "space", "communication"), outside a code span."""
    for n, line in enumerate(text.splitlines(), 1):
        if m := BANNED.search(CODE_SPAN.sub("", line)):
            raise ValueError(f"the report says {m[0]!r} (line {n}: {line.strip()!r}): name each of the six "
                             f"subprotocols (sampled proofs, PoUW, PoUS, the bandwidth audit, the network certifier, "
                             f"the wipe), and say 'proof block', 'sampling block', 'server-time block' and 'subcircuit'")


def _code(s: str) -> str:
    fence = "``" if "`" in s else "`"
    return f"{fence}{' ' if fence == '``' else ''}{s}{' ' if fence == '``' else ''}{fence}"


def section_rows(sections: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Every section row with its ``ratio`` (value over a numeric target) and ``met``; refuses a row that names no run,
    no verdict line or no target (top's rules: each number with its run id, its own verdict line, its target and
    ratio)."""
    out = []
    for sec in sections:
        if not str(sec.get("title") or "").strip():
            raise ValueError(f"measurements: section {sec!r} needs a title")
        for row in sec.get("rows") or []:
            if missing := [k for k in ROW_FIELDS if row.get(k) is None or str(row[k]).strip() == ""]:
                raise ValueError(f"measurements: row {row.get('what')!r} names no {', '.join(missing)}; every number "
                                 f"carries its run, its own verdict line quoted, and its target")
            if NOISE_ROW.search(str(row["what"])):
                raise ValueError(f"measurements: row {row['what']!r} states per-call noise on its own: put "
                                 f"the {', '.join(n for n in SERVE_NOISE if n in MEASURED)} values in `values` and "
                                 f"\"serve_noise\": true on its section, which states the seconds it adds a step with "
                                 f"both ratios")
            if not RUN_ID.search(str(row["run"])):
                raise ValueError(f"measurements: row {row['what']!r}: run {row['run']!r} has no run id (rYYYYMMDD-"
                                 f"HHMMSS-xxxx) or art: id")
            t = row["target"]
            r = dict(row, section=sec["title"], ratio=None, met=None)
            if isinstance(t, Mapping) and "none" in t:
                if not str(t["none"]).strip():
                    raise ValueError(f"measurements: row {row['what']!r}: say why it has no target")
            elif isinstance(t, Mapping) and t.get("op") in TARGET_OPS and "value" in t:
                v, x = row["value"], t["value"]
                num = all(isinstance(y, (int, float)) and not isinstance(y, bool) for y in (v, x))
                if num and t["op"] != "==" and x:
                    r["ratio"] = v / x
                r["met"] = (v <= x if t["op"] == "<=" else v >= x) if num and t["op"] != "==" else v == x
            else:
                raise ValueError(f"measurements: row {row['what']!r}: target is {{op: <=|>=|==, value}} or "
                                 f"{{none: why}}, not {t!r}")
            out.append(r)
    return out


def _line(lines: Sequence[str], head: str) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for line in lines:
        if line.startswith(head + " {"):
            try:
                out.append((line, json.loads(line[len(head) + 1:])))
            except json.JSONDecodeError as e:
                raise ValueError(f"measurements: proved: {line[:80]!r} isn't {head} and one JSON object ({e})") from e
    return out


def proved_section(proved: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    """The drawn-block proving run as a section, one row per drawn proof block and one for the bundle's verdict, and whether every
    drawn proof block was proved and accepted (module docstring); refuses a VERDICT for a block the coin didn't draw."""
    run, drawn, lines = proved.get("run"), proved.get("drawn"), proved.get("lines")
    if not RUN_ID.search(str(run or "")) or not isinstance(lines, list) or not isinstance(drawn, list) or not drawn:
        raise ValueError("measurements: proved needs its run id, the coin's drawn indices and the run's output lines")
    if len(set(drawn)) != len(drawn) or not all(isinstance(g, int) and not isinstance(g, bool) for g in drawn):
        raise ValueError(f"measurements: proved: drawn {drawn!r} isn't distinct indices")
    verdicts: dict[int, tuple[str, dict[str, Any]]] = {}
    for line, v in _line(lines, "VERDICT"):
        if v.get("block") not in drawn:
            raise ValueError(f"measurements: proved: a VERDICT for block {v.get('block')!r}, which the coin didn't draw")
        verdicts[v["block"]] = (line, v)
    final = (_line(lines, "PROVEBUNDLE") or [(None, None)])[-1]
    rows = [{"what": "proof blocks the live coin drew", "value": len(drawn), "target": {"op": "==", "value": HANDFUL},
             "run": run, "verdict": "drawn: " + json.dumps(drawn)}]
    for g in drawn:
        line, v = verdicts.get(g, (None, None))
        value = "no VERDICT line" if v is None else "accepted" if v.get("accepted") is True else "not accepted"
        rows.append({"what": f"drawn proof block {g}: proved in C-Flock --zk behind the firewall, verified by M0 and Lean",
                     "value": value, "target": {"op": "==", "value": "accepted"}, "run": run,
                     "verdict": line or f"the run's output has no VERDICT line for block {g}"})
    line, v = final
    n = (v or {}).get("negatives") or 0
    refused = (v or {}).get("negatives_refused")
    rows.append({"what": "negatives refused by Lean (flipped bits, the tampered proof block)",
                 "value": "no PROVEBUNDLE line" if v is None else f"{refused} of {n}",
                 "target": {"op": "==", "value": f"{n} of {n}" if n else "every negative"}, "run": run,
                 "verdict": line or "the run's output has no final PROVEBUNDLE line"})
    rows.append({"what": "the bundle's verdict (PROVEBUNDLE)", "value": "no PROVEBUNDLE line" if v is None else
                 "ok" if v.get("ok") is True else "not ok", "target": {"op": "==", "value": "ok"}, "run": run,
                 "verdict": line or "the run's output has no final PROVEBUNDLE line"})
    complete = (len(drawn) == HANDFUL and all(r["value"] == "accepted" for r in rows[1:1 + len(drawn)])
                and rows[-1]["value"] == "ok" and n > 0 and refused == n)
    title = "M3 proper: the drawn H200 proof blocks proved in C-Flock"
    return {"title": title, "text": str(proved.get("text") or ""), "rows": rows + list(proved.get("rows") or [])}, complete


def _target(r: Mapping[str, Any]) -> str:
    t, unit = r["target"], f" {r['unit']}" if r.get("unit") else ""
    if "none" in t:
        return f"no target ({t['none']})"
    return f"target {TARGET_OPS[t['op']]} {_fmt(t['value'])}{unit if not isinstance(t['value'], str) else ''}"


def _row(r: Mapping[str, Any]) -> str:
    unit = f" {r['unit']}" if r.get("unit") and not isinstance(r["value"], str) else ""
    ratio = f", ratio {r['ratio']:.3g}×" if r["ratio"] is not None else ""
    met = "" if r["met"] is None else ", **met**" if r["met"] else ", **missed**"
    return (f"- **{r['what']}:** {_fmt(r['value'])}{unit} ({_target(r)}{ratio}{met}). Run {r['run']}. "
            f"Verdict: {_code(str(r['verdict']))}" + (f" {r['note']}" if r.get("note") else ""))


def _sections(sections: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]], noise: str) -> str:
    parts = []
    for n, sec in enumerate(sections, 1):
        mine = [_row(r) for r in rows if r["section"] == sec["title"]] + [f"- {noise}"] * bool(sec.get("serve_noise"))
        parts.append(f"## {n}. {sec['title']}\n\n" + (f"{sec['text']}\n\n" if sec.get("text") else "")
                     + "\n".join(mine))
    return "\n\n".join(parts)


def _misses(rows: Sequence[Mapping[str, Any]], failures: Sequence[Mapping[str, Any]] | None) -> str:
    missed = [r for r in rows if r["met"] is False]
    lines = [f"- {r['section']}: **{r['what']}**: {_fmt(r['value'])}"
             + (f" {r['unit']}" if r.get("unit") and not isinstance(r["value"], str) else "") + f" against {_target(r)}"
             + (f" (ratio {r['ratio']:.3g}×)" if r["ratio"] is not None else "") + f". Run {r['run']}." for r in missed]
    if failures:
        lines += [f"- Drawn proof block {_fmt(f['unit'])} ({f['subcircuit']}, step {_fmt(f['step'])}, slot "
                  f"{_fmt(f['slot'])}) not proved: {f['why']}. Run {f['source']}." for f in failures]
    return "\n".join(lines) or "None: every number above met its target."


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Demo A's report from block records and a measurements file.")
    ap.add_argument("records", nargs="*", type=Path,
                    help="block records (verity/block-record/v1); the first is the block. None for a report whose "
                         "measurements carry sections or proved (M3's report)")
    ap.add_argument("--measurements", type=Path)
    ap.add_argument("--record-label", help="the label every number read from the block records carries")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    md = render([json.loads(p.read_text()) for p in a.records],
                json.loads(a.measurements.read_text()) if a.measurements else None, record_label=a.record_label)
    if a.out:
        a.out.write_text(md)
    else:
        sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
