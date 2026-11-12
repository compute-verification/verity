"""The console's training panel `training/standard-workloads`: the standard covert workloads (te_core.workloads'
names: standard, diloco, rl, inference, lora and distill), each a small real run from a registered start, measured by
te_core and priced by te_core.formalism at its own session (the headline), the console's default and the registered
serving block, beside the workloads no run here measures, each priced on te_core.workloads' model of it (`MODELLED`).

    python islands/standard-workloads/standard_workloads.py --out DIR [--rows FILE] [--served FILE] [--base REV]
                                                            [--branch NAME]
                                                            [--restated PLAN.json]
    python islands/standard-workloads/standard_workloads.py --put DIR/console/standard-workloads.json --served FILE
                                                            [--dry]

The build is a research run on a node (the agent VM only edits), from benchmarks/covert_witness with `--out` under the
run's out/; the PUT is the agent VM's, which alone holds the console's token, with `--served` the panels the route
served when it launched the build, written there by te_core.console.served() and sent with the run.

`--rows` (default `ROWS`, the island's rows.json) gives the stored traces (the island's trace.py output, chosen steps
of the runs, refused unless traced by a te_core from `TRACED_SINCE` to the one it runs on, a step traced more than once
read from the newest te_core's trace) and a row per workload and measured
dial setting: its workload, model, dials (te_core.workloads' arguments), run id and record art, what the run was beside
them (`about`, and a `note`), `last_untraced` where the trace can't reach the run's last step (why, in its
`input_status`), and `rounds` for a generation run (the island's generate.py), with `learner` for an RL run read
whole, its rollouts plus its learner; or, for a workload no run measures,
`modelled` (what the model is) in place of the run, its art and its trace: its row is te_core.workloads' model of the
workload at its dials, with the model's products at each point (workloads.with_products) priced as a run's are, its
run cell `MODELLED` with what the model is, its dials cell every argument of the model, and each input the model's
(`modelled_status`). A measured row is the run's last
checkpoint as te_core.ledger.ingest writes it (into a scratch ledger) with the run's traced structures
(annotate.with_structures), each its latest traced step's, less the read-backs its trace lists past the run's last
step, the program's code after it, which no step's graph carries (`past_last`, in the row's details), measured by
te_core.formalism.measure (its op log the ledger row's macs_log), and priced by
formalism.published under `READING` at the row's points (`own`). The headline, `rho_own`, is its ρ at the row's own
session, algorithm_policy's `own_session` (te_core.policy's `session`, 4,096 tokens in and 1,024 out on the model the
row serves: its start, distillation's teacher, a modelled row's model; c `own_c_gates`), at algorithm_policy's `OWN_K`,
the console's k, one 16-bit word an input token, beside the session's packed ids (`own_k_bits`), where ρ is at least
rho_own (`OWN_K_LABEL`), as plot 2's rho_own is. Beside it, two sensitivities at `POINTS`, algorithm_policy's:
`rho_default` and `rho_console` are both its ρ at the console's default policy (`CONSOLE`, te_core.policy's: that
session of the 45M reference, attention included), with `binding_kind`, `rests_on` and `kinds` there, as plot 2's are,
and `rho_served` its ρ at the registered serving block (te_core.panels' `SERVED`); `build` refuses a row whose own
session binds on another kind or rests on another premise than the default, so `binding_kind` and `rests_on` hold for
rho_own too. A distillation row's ρ at its student's own session (the run's notes' student, `measured`) is a third
sensitivity, given in the note and the details (`student_session`), no column. ρ, its binding kind and what it rests on
are published's as it gives them (null where it has no bound, resting on "unmeasured"), a measured row's `rests_on`
opening with algorithm_policy's `TRACED` where ρ is the products term's or its floor's, as algorithm_policy's rows do
(algorithm_policy.rests_on); reading A's ρ is never infinite, so `build` refuses one. `kinds` is
algorithm_policy's cell of the terms the max takes (the products term the theorem of record's ρ_HE, given (H) and (E)),
its `partial` and `why` where published sets them, and, beside them, the sensitivities `SENSITIVITIES`: published's
(reduction, update and fresh_old, the old products figure, which rests on P-width extended, false on honest circuits)
and reading B's token and decoding forms. `capture_code` is published's, the VERSION of the code that wrote the
row's trace (null where the trace records none, one from before te_core 4.12.14, or the row has no trace).
`rho_inputs` holds `KEYS` (C_gates beside algorithm_policy's `rho_inputs`), from which algorithm_policy.page_rho gives
the dial's ρ at any (c, k), and beside them te_core.panels' `FRESH`, the old figure and the products term over c at
each k of the dial its products are classified at ([fresh_old/c, term/c]; null for a run with no product rows), which
the console's port multiplies by c; `build` refuses a
row whose rho_inputs miss formalism.at's ρ at any of its points or whose token dials miss its measured tokens
(`token_dials`).
`input_status` gives each input's source, "measured" or "assumed: " and why, from te_core's labels: a count, a trace's
quantity, E's size and the harness's decoder are measured, and a stub, an assumption or an unmeasured quantity is
assumed. An unmeasured Δ_grad names the steps te_core has none for and why (`untraced`), and the ρ columns whose Fit
verdict rests on it (`held`). Beside `KEYS`, `products` is the products kind's input (`products`): measured at the
console's default, and so at the own session, whose k is the default's, or assumed with te_core's reason or where
untraced steps that update count none, and where the serving block's k leaves it unmeasured. The words the builder
adds there and in the note name no code (`CODE`): the code is training's slowdown code, and only te_core's reasons,
which the builder relays as they are, are te_core's to word.

A `rounds` row is the run's generation rounds alone (`rounds`), its setup and training steps dropped: a round is one
counted step that codes the served weights, prefills its prompts and decodes, and the run's traces of its rounds (a
run's rounds share one structure) give its quantities as a training run's traces give its, a row with no traced round
refused. A round's forward positions are each prompt token and every generated token but the last (the run's notes',
`positions`), which its token dials count, and which its trace's tokens count where the capture puts the decode loop's
positions on the token axis the prompts' are on (`axis`, in the row's details). Where a traced round's trace counts
fewer, its products and token share are the prompts' alone, so the row is published null (`NO_AXIS`): every ρ column,
its inputs and their sources null, its binding kind unmeasured, and `NO_AXIS` its rests_on and its kinds' why; a trace
counting more is refused. A priced record is read with these substitutions (`ROUNDS`, each in the row's
`input_status`): every GEMM on the served codes (W4A4, emulated in bf16 on their decoded values) priced as NVFP4
(gemm/fp4); Δ the served update's bits from the run's notes (every served matrix's codes and scales with the gains in
bf16, or the LoRA in bf16); and Δ_grad, the partial gradient and the gradient cut 0, a round having no backward (the
trace's cut holds its decode loop's values). A `learner` row (RL's) is the whole run instead, the rollouts (its rounds,
read as a rounds row's) plus the learner (its training steps), each part's quantities its own traced step's (a
training step after the first update, whose structure the learner's steps share), a row with no traced training step
refused: its rounds' GEMMs on the served codes are priced as NVFP4 and their structure's Δ_grad, partial gradient and
gradient cut are 0 (`NO_BACKWARD`), so Fit binds on the learner's measured Δ_grad, and Δ is the run's E, as a training
row's; its `input_status` carries the rollouts' substitutions beside the learner's measured quantities (`LEARNER`), and
its run cell and details give each part's share of its gates (`shares`, of the parts te_core.formalism.of_record makes,
`part_names`).

DIR gets the uncut table, each row with its labels, parts, its own session (`own_session`) and a distillation row's
student's (`student_session`), and formalism.at, formalism.published and the one bound (reading A's, formalism.bound)
at each of its points (tables/standard-workloads.json), and the console's copy
(console/standard-workloads.json), whose note cites the tables in the research run's out/ (`cite`), which custody
stores with the run's record. With `--served` a build is refused where it would drop a served run's row or the route
serves a newer te_core (`emptied`). `--put` PUTs a build's console body (`put`), refused where the route's panel
changed since `--served` was read, where the build would drop a row or the route refuses it (`emptied`, `refused`), or
where its note cites no stored tables; then it GETs the panel and compares it with the body as numbers (the scorer's
`readback`). The panel's source names
the command at this checkout's commit and the integrator's commit it builds on (`commits`: `--base`, else their merge
base on origin, which a node's checkout hasn't), refused where te_core or the island has uncommitted changes, which
neither commit holds. Where the checkout builds on another branch than the integrator's, `--branch` names it: the
source then names that branch, which must hold this checkout's commit, and with no `--base` no commit under it.

`--restated PLAN.json` (the scorer's restate.py plan) reads each run the plan restates with its record's tokens a step
restated (`restated`): the record is restate.py's, refused unless it is the record the plan read and its restating the
plan's, and each stored trace of the run is read against the restated count of its step, its `step_tokens` and its
products block's `tokens` that count and its `token_factor` the trace's own tokens over it, the three fields in which
a trace of the restated record differs from a trace of the record as it is. The row's details say so (`restated` in
what it read).
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.append(str(Path(__file__).resolve().parents[2] / "scorer"))

import readback
from te_core import VERSION, algorithm_policy, annotate, console, formalism, panels, policy, workloads
from te_core.ledger import Ledger, ingest

HERE = Path(__file__).resolve().parent
CW = HERE.parents[1]
NAME = "standard-workloads"
PANEL = f"training/{NAME}"
COLUMNS = ["workload", "model", "dials", "run", "rho_own", "own_c_gates", "own_k_bits", "rho_inputs", "input_status",
           "rho_default", "rho_console", "reading", "te_core", "capture_code", "binding_kind", "rests_on", "kinds",
           "rho_served"]
#: the points (c, k) every row is priced at beside its own session's (`own`), algorithm_policy's: the console's
#: default policy, the one default of `rho_default` and `rho_console`, and the registered serving block, `rho_served`'s
POINTS = {"console": algorithm_policy.CONSOLE, "served": panels.SERVED}
#: the reading every row publishes (formalism.published's): A, the one bound over the kinds valid on their own, with
#: reading B's forms among the kinds cell's sensitivities
READING = "A"
#: the kinds cell's sensitivities, beside the terms the bound's max takes: published's and reading B's two forms
SENSITIVITIES = (*formalism.CELL_SENSITIVITIES, "token_form", "decoding_form")
#: the `rho_inputs` cell's keys, algorithm_policy.rho_inputs' in its order, each with its `input_status`; the cell holds
#: te_core.panels' `FRESH` beside them, whose input is `products`
KEYS = ("C_gates", *panels.PACKED, *algorithm_policy.SPLIT)
#: the first te_core whose traces te_core.formalism reads whole: their entries and partial and gradient cuts
TRACED_SINCE = "4.4"
#: the one bound's fields each row's details keep at each of its points
BOUND = ("terms", "max", "binding", "unmeasured", "sensitivities")
ROWS = HERE / "rows.json"
#: a modelled row's run cell, before what the model is: no run measures the workload, so no trace
MODELLED = "modelled: no trace"
#: how a note cites stored tables: their path in a research run's out/ (`cite`)
TABLES = re.compile(r" Tables run:r\d{8}-\d{6}-[0-9a-f]{4}/out/\S+\.")
FP4 = "float4_e2m1fn_x2"
GEMMS = ("mm", "addmm", "bmm")
#: each column's te_core quantities (algorithm_policy's `LABELLED`, and the two 4.x terms)
LABELLED = algorithm_policy.LABELLED | {"w_fwd_gates": (), "w_fit_gates": ("tokens", "token_gates"),
                                        "alpha_ratio": ("affected_gates",)}
#: a traced step's quantities each row's details keep (te_core.capture.circuit.measure's)
TRACE_FACTS = ("tokens", "sigma", "affected_share", "token_factor", "a_bits", "u1_bits", "tail_share",
               "delta_grad_bits", "state_bits")
#: te_core labels of quantities read off the run though te_core tags them bounds: E's size and the harness's decoder
READ = (formalism.STUBS["delta_bits"], formalism.DECODER["witness"])
#: why `untraced` names a step of a traced structure that updates nothing, which has no fresh products
IDLE = ("its structure updates nothing, and the slowdown code takes such a step's trace only where the trace measures "
        "the step and a traced step of the run updates something")
_PRICED = "assumed: the rounds' counted work, each GEMM on the served codes (W4A4, emulated in bf16 on decoded values) " \
          "priced as an NVFP4 GEMM"
_PER_TOKEN = _PRICED + ", times the trace's token share, per forward position"
_NO_BACKWARD = "assumed: 0, a round having no backward (the trace's gradient cut holds its decode loop's values)"
_DEFAULT = "at the console's default"
ROUNDS = {"C_gates": _PRICED, "w_gates": _PER_TOKEN, "sigma": "assumed: the trace's share of a round's gates in token ops",
          "w_fit_gates": _PER_TOKEN, "delta_grad_bits": _NO_BACKWARD}
#: a learner row's rounds' traced structure's backward quantities: none, a round having no backward
NO_BACKWARD = {"delta_grad_bits": 0.0, "partial_cut_bits": 0.0, "grad_cut_bits": 0.0}
_ROLLOUTS = ("assumed: the rollouts' counted work, each GEMM on the served codes (W4A4, emulated in bf16 on decoded "
             "values) priced as an NVFP4 GEMM")
#: a learner row's substitutions, its rollouts' (`ROUNDS`') beside its learner's measured quantities; `LEARNER_W_FIT`
#: where the rollouts' w is the least of its parts', the one Fit's compute block binds on
LEARNER = {"C_gates": f"{_ROLLOUTS}, beside the learner's counted gates",
           "w_gates": f"{_ROLLOUTS}, times their trace's token share, per forward position, beside the learner's "
                      f"measured token gates per token",
           "sigma": "assumed: the rollouts' at their trace's share of a round's gates in token ops, beside the "
                    "learner's measured share"}
LEARNER_W_FIT = f"{_ROLLOUTS}, times their trace's token share, per forward position: the rollouts' w, the least part's"
#: a rounds row's rests_on and its kinds' why where a traced round's trace counts fewer tokens than its forward
#: positions (`axis`): every ρ column, its inputs and their sources null
NO_AXIS = "unmeasured: the capture puts decode positions on no token axis"
#: a code name, which none of the builder's own words in a published string holds: they name the code only as
#: training's slowdown code
CODE = re.compile(r"te_core|io_bound|formalism|algorithm_policy|annotate|rows_of|capture_\w|gemm/|\w\.py\b|`|"
                  r"\b[A-Z][A-Z0-9]*_[A-Z0-9_]+\b")
ROUND_LABELS = {
    "gates": "counted MACs in gates (te_core.gates), the GEMMs on served codes at gemm/fp4 (standard_workloads.rounds)",
    "delta_bits": "measured: the served update's bits (generate.py's notes: every served matrix's NVFP4 codes and "
                  "scales with the gains in bf16, or the LoRA in bf16)",
}


def fetch(art: str, *paths: str) -> Path:
    """`art`'s tree (only `paths`, given any), materialised from the evidence store (`research data fetch`)."""
    cmd = ["research", "data", "fetch", art, *(x for p in paths for x in ("--path", p))]
    return Path(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip().splitlines()[-1])


def version_at(commit: str) -> str:
    """te_core's VERSION at `commit` of this repository."""
    src = subprocess.run(["git", "show", f"{commit}:benchmarks/covert_witness/te_core/__init__.py"], check=True,
                         capture_output=True, text=True, cwd=Path(__file__).parent).stdout
    return next(line.split('"')[1] for line in src.splitlines() if line.startswith("VERSION"))


def _version(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split("."))


def traces(arts: list[str]) -> dict[str, dict[int, dict]]:
    """Each run's traced steps in `arts` (the island's trace.py output: a tree of its manifest and traces, or the
    research run's record holding them in out/), each with its art and the te_core that traced it, refused unless that
    is `TRACED_SINCE` to the one it runs on; a step traced in more than one art is the newest te_core's trace (of equal
    ones, the later art's)."""
    out: dict[str, dict[int, dict]] = {}
    for art in arts:
        root = fetch(art, "*manifest.json", "*traces/*.json")
        root = root / "out" if (root / "out" / "manifest.json").exists() else root
        v = version_at(json.loads((root / "manifest.json").read_text())["te_commit"])
        if not _version(TRACED_SINCE) <= _version(v) <= _version(VERSION):
            raise SystemExit(f"{PANEL}: {art} was traced by te_core {v}, not {TRACED_SINCE} to {VERSION}: trace the "
                             f"runs again (islands/{NAME}/job.sh) and name the new traces in the rows")
        for p in sorted((root / "traces").glob("*.json")):
            t = json.loads(p.read_text())
            have = out.get(t["run"], {}).get(t["step"])
            if t.get("trace") and (have is None or _version(have["te_core"]) <= _version(v)):
                out.setdefault(t["run"], {})[t["step"]] = t | {"from": art, "te_core": v}
    return out


def commits(base: str | None = None, branch: str | None = None) -> tuple[str, str, str]:
    """This checkout's commit, the integrator's branch (pin.py's `BRANCH`) and its commit this one builds on (`base`,
    an ancestor of this one, else their merge base on origin), the commits at 10 characters, which the panel's source
    names; refused where te_core or the island has uncommitted changes, which neither commit holds. With `branch`, the
    branch is that one, refused unless it (here or on origin) holds this checkout's commit, and with no `base` the
    commit under it is this checkout's own."""
    from pin import BRANCH, git

    head = git("rev-parse", "HEAD")
    if branch:
        if all(git("merge-base", "--is-ancestor", "HEAD", ref).returncode
               for ref in (f"refs/heads/{branch}", f"refs/remotes/origin/{branch}")):
            raise SystemExit(f"{PANEL}: --branch {branch} doesn't hold this checkout's commit, here or on origin: "
                             f"push it there, or name the branch that does")
        BRANCH = branch
    on = git("rev-parse", "--verify", f"{base}^{{commit}}") if base else head if branch else \
        git("merge-base", "HEAD", f"origin/{BRANCH}")
    if head.returncode or on.returncode:
        raise SystemExit(f"{PANEL}: no commit of {BRANCH} to build on: give --base, or git fetch origin {BRANCH} and "
                         f"build again")
    if base and git("merge-base", "--is-ancestor", on.stdout.strip(), "HEAD").returncode:
        raise SystemExit(f"{PANEL}: --base {base} isn't in this checkout's history: merge it first")
    if git("status", "--porcelain", "--", str(CW / "te_core"), str(HERE)).stdout.strip():
        raise SystemExit(f"{PANEL}: te_core or islands/{NAME} has uncommitted changes, which the source's commit "
                         f"doesn't hold: commit them and build again")
    return head.stdout.strip()[:10], BRANCH, on.stdout.strip()[:10]


def coded(row: list, served: set[tuple[int, int]]) -> bool:
    """Whether a structure's log row is a GEMM on a served matrix's codes: its second operand (K, N) is a served
    (N, K) matrix transposed."""
    return row[2].split(".")[0] in GEMMS and tuple(json.loads(row[3])[-1][-2:]) in served


def rounds(record: dict, notes: dict, served: set[tuple[int, int]], learner: bool = False) -> dict:
    """`record` (generate.py's: each training step, and a round after each of `notes`' rounds) as its rounds alone,
    the setup and the tail without work and the run's last checkpoint its one, each round's tokens its prompts' (the
    counter's, which counts data-stream positions, so no sampled id) and every GEMM on served codes (`coded`) marked
    NVFP4; with `learner`, the whole run: its rounds so, and its training steps, their structures, the setup and the
    tail as they are."""
    at = round_at(notes)
    if len(record["steps"]) != notes["args"]["steps"] + len(at):
        raise ValueError(f"{len(record['steps'])} steps, not {notes['args']['steps']} training steps and "
                         f"{len(at)} rounds: not generate.py's layout")
    steps = [dict(s) for s in record["steps"] if learner or s["step"] in at]
    for s in steps:
        r = at.get(s["step"])
        if r and s["tokens"] != r["sequences"] * r["prompt"]:
            raise ValueError(f"round at step {s['step']}: {s['tokens']} tokens counted, not its prompts' ({r})")
    ours = {s["struct"] for s in steps if s["step"] in at}
    if both := ours & {s["struct"] for s in record["steps"] if s["step"] not in at}:
        raise ValueError(f"round structure {', '.join(sorted(both))} is a training step's too: not generate.py's "
                         f"layout")
    structures = dict(record["structures"]) if learner else {}
    for sid in ours:
        st = record["structures"][sid]
        log = [[*r[:3], f"{r[3]} {FP4}" if coded(r, served) else r[3], *r[4:]] for r in st["log"]]
        if any(r[0] == "bwd" for r in log):
            raise ValueError(f"round structure {sid} has a backward")
        structures[sid] = st | {"log": log}
    none = {"by": {}, "macs": 0}
    kept = {} if learner else {"setup": record["setup"] | none, "tail": record.get("tail", {}) | none}
    return record | {"steps": steps, "structures": structures, "checkpoints": record["checkpoints"][-1:]} | kept


def round_formalism(notes: dict, tensors: dict) -> dict:
    """A rounds record's own `formalism` fields: Δ the served update's bits."""
    a = notes["args"]
    if a["update"] == "lora":
        delta = notes["served_lora_bits"]
    else:
        gains = sum(math.prod(shape) for _, shape in tensors.values() if len(shape) == 1)
        delta = notes["served_code_bits"] + 16 * (gains + (math.prod(tensors["embed"][1]) if a["head"] == "bf16" else 0))
    return {"delta_bits": float(delta), "labels": ROUND_LABELS}


def round_at(notes: dict) -> dict[int, dict]:
    """Each round's step in the run's record (generate.py's: a round after each of `notes`' rounds' training step) and
    its notes."""
    return {r["after_step"] + i: r for i, r in enumerate(notes["rounds"])}


def positions(r: dict) -> int:
    """A round's forward positions (its notes'): each prompt token and every generated token but the last, which no
    forward reads."""
    return r["sequences"] * (r["prompt"] + r["generated"] - 1)


def forward_positions(notes: dict) -> int:
    """The rounds' forward positions (generate.py's notes), `positions` summed."""
    return sum(positions(r) for r in notes["rounds"])


def axis(notes: dict, traced: dict[int, dict]) -> list[dict]:
    """Each traced round's tokens (its trace's consumed positions) beside its forward positions (`positions`), which a
    trace counts where the capture puts the decode loop's positions on the token axis the prompts' are on."""
    at = round_at(notes)
    return [{"step": s, "traced": t["measured"]["tokens"], "forward": positions(at[s])}
            for s, t in sorted(traced.items()) if s in at]


def off_axis(what: str, rounds: list[dict]) -> bool:
    """Whether a rounds row's trace lacks its decode loop's positions: a traced round's tokens fewer than its forward
    positions (`axis`'s `rounds`); refused where none is a round or one counts more, which no forward reads."""
    if not rounds:
        raise SystemExit(f"{PANEL}: {what}: no traced step is a round: trace its rounds (islands/{NAME}/job.sh) and "
                         f"name the traces in the rows")
    if over := [a for a in rounds if a["traced"] > a["forward"]]:
        raise SystemExit(f"{PANEL}: {what}: the capture counts more tokens than a round's forward positions ("
                         + "; ".join(f"step {a['step']}: {a['traced']:,} traced, {a['forward']:,} forward" for a in over)
                         + "), positions no forward reads: fix the capture and trace the rounds again")
    return any(a["traced"] < a["forward"] for a in rounds)


def start(name: str) -> tuple[dict, dict]:
    """A start's shape and its tensors (name -> [dtype, shape], its start.json's), from the starts manifest."""
    s = json.loads(policy.MANIFEST.read_text())["starts"][name]
    path = f"{s.get('path', 'starts')}/{name}/start.json"
    return s["shape"], json.loads((fetch(s["art"], path) / path).read_text())["tensors"]


def untraced(record: dict, structures: dict[str, dict], last_why: str | None = None) -> list[str]:
    """The steps with tokens that te_core.annotate.with_structures gives none of the traced `structures` (by structure
    id), so no Δ_grad, each with why (and `last_why`, why the trace doesn't reach it, beside the program's last)."""
    got = annotate.with_structures(record, structures)
    last, out = record["steps"][-1]["step"], []
    for s, a in zip(record["steps"], got["steps"]):
        sid = s["struct"]
        if not s.get("tokens") or a["struct"] != sid:
            continue
        if sid in structures:
            why = IDLE
        elif s["step"] == last and sum(x["struct"] == sid for x in record["steps"]) == 1:
            why = "the program's last, its structure having no other step, not traced" + (
                f", since {last_why}" if last_why else "")
        else:
            why = "no traced step shares its structure"
        out.append(f"step {s['step']}: {why}")
    return out


def part_names(record: dict, structures: dict[str, dict], rounds: set[int]) -> list[str]:
    """What each part te_core.formalism.of_record makes of the whole run `record` with the traced `structures` holds,
    in its order: the setup's and the tail's where they have work ("setup", "tail"), then each step structure's
    (te_core.annotate.with_structures's) as its steps first come, "rollouts" where its steps are the `rounds`, else
    "learner"; refused where one holds both."""
    names = [k for k in ("setup", "tail") if (record.get(k) or {}).get("by")]
    kinds: dict[str, set[str]] = {}
    for s, a in zip(record["steps"], annotate.with_structures(record, structures)["steps"]):
        kinds.setdefault(a["struct"], set()).add("rollouts" if s["step"] in rounds else "learner")
    if both := [sid for sid, k in kinds.items() if len(k) > 1]:
        raise ValueError(f"structure {', '.join(both)} holds a round and a training step")
    return names + [k.pop() for k in kinds.values()]


def shares(m: formalism.Measured, names: list[str]) -> dict[str, float]:
    """Each named part's share of `m`'s gates (`names`, `part_names`'), the parts of a name summed."""
    return {n: sum(p.gates for p, k in zip(m.parts, names) if k == n) / m.gates for n in dict.fromkeys(names)}


def split(share: dict[str, float]) -> str:
    """A learner row's two parts' shares of its gates, as its run cell gives them."""
    return f"the rollouts {100 * share['rollouts']:.3g}% of its gates and the learner {100 * share['learner']:.3g}%"


def learner_status(m: formalism.Measured, names: list[str]) -> dict[str, str]:
    """A learner row's substitutions (`LEARNER`), w_fit's where the rollouts' w is the least of its parts'
    (`names`, `part_names`')."""
    least = min((p.token_gates / p.tokens, n) for p, n in zip(m.parts, names) if p.tokens)[1]
    return LEARNER | ({"w_fit_gates": LEARNER_W_FIT} if least == "rollouts" else {})


def past_last(structures: dict[str, dict], last: int) -> tuple[dict[str, dict], dict[str, list]]:
    """The traced `structures` without the read-backs their traces list past the run's `last` step (the program's code
    after it, which no step's graph carries, such as a harness reading the run's outputs), and those read-backs by
    structure."""
    out, after = {}, {}
    for sid, st in structures.items():
        f = st.get("formalism") or {}
        late = {k: [e for e in f.get(k) or () if min(e.get("steps") or [last]) > last] for k in formalism.READ_BACKS}
        if any(late.values()):
            after[sid] = [e for k in formalism.READ_BACKS for e in late[k]]
            f = f | {k: [e for e in f[k] if e not in late[k]] for k in formalism.READ_BACKS if f.get(k) is not None}
            st = st | {"formalism": f}
        out[sid] = st
    return out, after


def restated(raw: bytes, entry: dict, traced: dict[int, dict]) -> tuple[dict, dict[int, dict]]:
    """A run's record (`raw`, its run.json's bytes) with its tokens a step restated by the run's `entry` of a plan
    (the scorer's restate.py), and the run's stored traces read against it (module docstring). Refused where `raw`
    isn't the record the plan read, or its restating isn't the plan's."""
    sys.path.insert(0, str(CW / "scorer"))
    try:
        import restate
    finally:
        sys.path.remove(str(CW / "scorer"))
    if hashlib.sha256(raw).hexdigest() != entry["before_sha256"]:
        raise SystemExit(f"{PANEL}: --restated: the stored record isn't the one the plan read")
    record = json.loads(raw)
    restate.restate(record, entry["factor"], entry["tokens"])
    if hashlib.sha256(restate.encode(record)).hexdigest() != entry["after_sha256"]:
        raise SystemExit(f"{PANEL}: --restated: the restated record isn't the plan's")
    tokens = {s["step"]: s.get("tokens") for s in record["steps"]}
    out = {}
    for step, t in traced.items():
        t = copy.deepcopy(t)
        for f in (t["trace"].get("formalism"), (t["trace"].get("structure") or {}).get("formalism"), t.get("measured")):
            if not isinstance(f, dict) or "step_tokens" not in f:
                continue
            block = f.get("products")
            if isinstance(block, dict) and "tokens" in block:
                if block["tokens"] != f["step_tokens"]:
                    raise SystemExit(f"{PANEL}: --restated: step {step}'s products block counts {block['tokens']} "
                                     f"tokens, not the record's {f['step_tokens']}")
                block["tokens"] = tokens[step]
            f["step_tokens"] = tokens[step]
            f["token_factor"] = f["tokens"] / tokens[step] if tokens[step] and f.get("tokens") else 1.0
        out[step] = t
    return record, out


def measured(row: dict, traced: dict[int, dict], entry: dict | None = None) -> tuple[formalism.Measured, dict, dict]:
    """A row's quantities: its run's last checkpoint, or its rounds (`rounds`; a learner row's with its training
    steps), ingested with the run's `traced` steps' structures, each structure's its latest traced step's
    (`structures` in what the row read); the start's shape and what the row read, a distillation run's student's shape
    among it (`student`, from the run's notes), a rounds row's traced rounds' tokens beside their forward positions
    (`axis`) and a learner row's parts, their shares of its gates (`part_names`, `shares`) and its learner's tokens."""
    root = fetch(row["art"], "run/run.json", "meta.json")
    raw = (root / "run" / "run.json").read_bytes()
    if entry is None:
        record = json.loads(raw)
    else:
        record, traced = restated(raw, entry, traced)
    meta = json.loads((root / "meta.json").read_text())
    shape, tensors = start(record["start"]["name"])
    notes = meta.get("notes") or {}
    if row["workload"] == "distill" and "student" not in notes:
        raise SystemExit(f"{PANEL}: distill {row['run']}: its notes name no student, whose own session the note gives "
                         f"as a sensitivity: run it again with islands/{NAME}/distill.py, which writes them")
    student = {"student": {k: notes["student"][k] for k in ("d_model", "n_layer", "n_head", "d_ff", "vocab")}
                          | {"parameters": notes["student_parameters"]}} if "student" in notes else {}
    by_step = {s["step"]: s["struct"] for s in record["steps"]}
    latest = {by_step[step]: step for step in sorted(traced)}
    structures, after = past_last({sid: traced[step]["trace"]["structure"] for sid, step in latest.items()},
                                  record["steps"][-1]["step"])
    read = {"traces": [{"step": step, "struct": by_step[step], "art": t["from"], "te_core": t["te_core"],
                        "measured": {k: t["measured"].get(k) for k in TRACE_FACTS}}
                       for step, t in sorted(traced.items())], "structures": latest} | (
        {"past_last_step": after} if after else {}) | (
        {} if entry is None else {"restated": {"record_sha256": entry["after_sha256"], "factor": entry["factor"]}})
    learner, at = bool(row.get("learner")), round_at(notes) if row.get("rounds") else {}
    if row.get("rounds"):
        if not traced:
            raise SystemExit(f"{PANEL}: rounds {row['run']}: no round is traced: trace its rounds (islands/{NAME}/"
                             f"job.sh) and name the traces in the rows")
        if learner and all(step in at or step < min(at) for step in traced):
            raise SystemExit(f"{PANEL}: rl {row['run']}: no learner step is traced: trace a training step after its "
                             f"first update (islands/{NAME}/job.sh) and name the trace in the rows")
        matrices = [n for n in tensors if n.startswith("blocks.") and len(tensors[n][1]) == 2]
        names = matrices + (["embed"] if notes["args"]["head"] == "fp4" else [])
        record = rounds(record, notes, {(tensors[n][1][1], tensors[n][1][0]) for n in names}, learner)
        if learner:
            ours = {s["struct"] for s in record["steps"] if s["step"] in at}
            structures = {sid: st | {"formalism": st["formalism"] | NO_BACKWARD}
                          if sid in ours and st.get("formalism") else st for sid, st in structures.items()}
        else:
            record["formalism"] = round_formalism(notes, tensors)
        read |= {"rounds": notes["rounds"], "round_steps": [s["step"] for s in record["steps"] if s["step"] in at],
                 "forward_positions": forward_positions(notes), "axis": axis(notes, traced),
                 "fp4_macs": sum(r[5] * sum(s["struct"] == sid for s in record["steps"])
                                 for sid, st in record["structures"].items() for r in st["log"] if r[3].endswith(FP4))}
    else:
        record["checkpoints"] = record["checkpoints"][-1:]
    read["untraced"] = untraced(record, structures, row.get("last_untraced"))
    with tempfile.TemporaryDirectory(prefix="sw-ingest-") as tmp:
        d = Path(tmp) / "run"
        d.mkdir()
        (d / "run.json").write_text(json.dumps(record))
        got = ingest(d, Ledger(Path(tmp) / "ledger", NAME), direction=NAME, mechanism=row["workload"],
                     run_id=row["run"], structures=structures or None, meta=meta)[-1]
    read |= {"checkpoint": got["checkpoint_step"], "steps": len(record["steps"]),
             "tokens": sum(s.get("tokens", 0) for s in record["steps"]), "invalid": got["invalid"]} | student
    m = formalism.measure(got, shape)
    if learner:
        parts = part_names(record, structures, set(at))
        if len(parts) != len(m.parts) or {"rollouts", "learner"} - set(parts):
            raise ValueError(f"rl {row['run']}: parts {parts} beside the slowdown code's {len(m.parts)}: not the "
                             f"rollouts and the learner")
        read |= {"parts": parts, "shares": shares(m, parts),
                 "learner_tokens": sum(p.tokens for p, n in zip(m.parts, parts) if n == "learner")}
    return (no_backward(m) if row.get("rounds") and not learner else m), shape, read


def no_backward(m: formalism.Measured) -> formalism.Measured:
    """A rounds row's quantities with each part's Δ_grad, partial gradient and gradient cut 0, a round having no
    backward (`ROUNDS`)."""
    zero = {"delta_grad_bits": 0.0, "partial_bits": 0.0, "grad_bits": 0.0}
    return dataclasses.replace(m, parts=tuple(dataclasses.replace(p, **zero) for p in m.parts),
                               labels=m.labels | {"delta_grad_bits": _NO_BACKWARD})


def _measured(label: str | None) -> bool:
    return label is not None and (label.startswith(("measured", "counted")) or "trace" in label or label in READ
                                  or label in algorithm_policy.COUNTED)


def listed(xs: list[str]) -> str:
    return " and ".join(xs) if len(xs) < 3 else f"{', '.join(xs[:-1])} and {xs[-1]}"


def own(shape: dict) -> tuple[dict, dict[str, tuple[float, float]]]:
    """A row's own session, algorithm_policy's `own_session` on the model it serves (a run's start, distillation's
    teacher, or a modelled row's model), and the row's points: that session's c at algorithm_policy's `OWN_K` (`own`,
    the headline), then `POINTS`."""
    s = algorithm_policy.own_session(shape)
    return s, {"own": (s["gates"], algorithm_policy.OWN_K)} | POINTS


def held(m: formalism.Measured, points: dict[str, tuple[float, float]] = POINTS) -> list[str]:
    """The `points` (a row's, `own`'s) where Fit holds once every part with token blocks and no Δ_grad has 0 (the
    least any could have): where Fit's verdict rests on those parts' Δ_grad."""
    zero = dataclasses.replace(m, parts=tuple(dataclasses.replace(p, delta_grad_bits=0.0)
                                              if p.tokens and p.delta_grad_bits is None else p for p in m.parts))
    return [name for name, (c, k) in points.items() if formalism.at(zero, c, k)["fit"]]


def status(m: formalism.Measured, override: dict[str, str] | None = None, read: dict | None = None,
           points: dict[str, tuple[float, float]] = POINTS) -> dict[str, str]:
    """Each of `KEYS`' source (module docstring), `override`'s where it names one; an unmeasured Δ_grad's steps are
    the row's `read`'s `untraced`, where it has them, which must number te_core's, out of its `steps`, and the ρ
    columns whose Fit verdict rests on it are those at the row's `points` where it `held`."""
    out, gaps = {}, (read or {}).get("untraced")
    for key in KEYS:
        keys = LABELLED["delta_bits"] if key == "delta_t_bits" and m.delta_t_bits is None else LABELLED[key]
        bad = [m.labels.get(k) for k in keys if not _measured(m.labels.get(k))]
        out[key] = "measured" if not bad else f"assumed: {bad[0] or 'no label from the slowdown code'}"
        if key == "delta_grad_bits" and m.fit_delta_grad_bits is None:
            n = sum(p.steps for p in m.parts if p.tokens and p.delta_grad_bits is None)
            if gaps is not None and len(gaps) != n:
                raise ValueError(f"te_core has no Delta_grad for {n} steps, `untraced` names {len(gaps)}: a step "
                                 f"te_core.annotate gives a trace without a measured Delta_grad, which untraced() "
                                 f"doesn't name")
            where = (f"{n} of {read['steps']} steps ({'; '.join(gaps[:3])}{f'; and {n - 3} more' if n > 3 else ''})"
                     if gaps else f"{n} of {sum(p.steps for p in m.parts)} steps in structures with no traced "
                                  f"Δ_grad")
            on = [f"rho_{name}" for name in held(m, points)]
            then = (f"{listed(on)} rest{'s' if len(on) == 1 else ''} on it, Fit holding at "
                    f"{'its point' if len(on) == 1 else 'their points'} with it at 0" if on else
                    "with it at 0, Fit still fails at every point, so no rho column rests on it")
            have = m.delta_grad_bits
            out[key] = (f"assumed: unmeasured on {where}, so the slowdown code fails Fit at every k; {then}"
                        + (f" (the traced steps' largest {algorithm_policy.num(have)} bits)" if have is not None else ""))
    return out | (override or {})


def products(b: dict[str, dict], left: list[str] = ()) -> str:
    """The products kind's input, the run's products whose two operands both depend on the update, as the one bound
    `b` read them at each of `POINTS`: measured at the console's default, or assumed with te_core's reason (where
    unmeasured, where the dominant traced step's stand in for steps of another shape, formalism.STAND_IN, or where
    `left`, `untraced`'s steps but those that update nothing, have none, te_core counting the traced steps' alone);
    and where unmeasured at the serving block, te_core's reason."""
    con, srv = b["console"], b["served"]
    if (why := con["unmeasured"].get("fresh_products")) is not None:
        return f"assumed: {why}"
    stand = con["inputs"]["fresh_products"].get("stand_in")
    said = [f"{formalism.STAND_IN.removeprefix('approximated: ')} ({len(stand['structures'])} traced structures)"
            ] if stand else []
    if left:
        more = f"; and {len(left) - 3} more" if len(left) > 3 else ""
        said.append(f"none on {len(left)} untraced step{'s' if len(left) > 1 else ''} ({'; '.join(left[:3])}{more}), "
                    f"the slowdown code counting the traced steps' products alone")
    out = f"assumed: {'; and '.join(said)}" if said else "measured"
    if (off := srv["unmeasured"].get("fresh_products")) is not None:
        out += f"{'' if said else f' {_DEFAULT}'}; at the serving block, {off}"
    return out


def modelled_status(m: formalism.Measured, b: dict[str, dict]) -> dict[str, str]:
    """A modelled row's input sources: each of `KEYS` the workload model's, with te_core's labels for the quantities it
    reads (w_fwd, which has none, a token's forward on the model's shape), and the products the model's matrix
    products, with what `products` says of them as the one bound `b` read them."""
    out = {}
    for key in KEYS:
        keys = LABELLED["delta_bits"] if key == "delta_t_bits" and m.delta_t_bits is None else LABELLED[key]
        said = "; ".join(dict.fromkeys(m.labels[k].removeprefix("model: ") for k in keys if m.labels.get(k))) or (
            "a token's forward on the model's shape, at the slowdown code's gate prices" if key == "w_fwd_gates" else
            "no label from the slowdown code")
        out[key] = f"modelled: {said}"
    got = products(b)
    rest = got.removeprefix("measured") if got.startswith("measured") else f"; {got.removeprefix('assumed: ')}"
    return out | {"products": f"modelled: the workload model's matrix products{rest}"}


def token_dials(dials: dict, tokens: float, whose: str = "te_core measured", rollouts: float | None = None) -> list[str]:
    """Where `dials`' token counts (`tokens`, and `tokens_per_param` over `student_params` or `n_params`) miss the run's
    measured `tokens` (formalism.Measured's: each step's count scaled by its traced structure's consumed tokens over its
    count, so a step of micro-batches counts them all; a rounds row's forward positions, `whose` the notes; a learner
    row's learner's, with `samples_per_trained` its `rollouts`' forward positions over them) by more than 1%: te_core
    leaves a step before the run's first update at the counter's count, which counts one micro-batch of a step's."""
    want = {"tokens": (tokens, whose)}
    if "tokens_per_param" in dials:
        want["tokens_per_param"] = (tokens / dials.get("student_params", dials.get("n_params")), whose)
    if rollouts is not None:
        want["samples_per_trained"] = (rollouts / tokens, "its rounds' forward positions over its learner's tokens are")
    return [f"its dial {k} is {dials[k]:g}, {w} {algorithm_policy.num(v)}: the dial is the run's"
            for k, (v, w) in want.items() if k in dials and not math.isclose(dials[k], v, rel_tol=1e-2)]


def priced(m: formalism.Measured, w_fwd: float, at: dict[str, dict]) -> str:
    """`KEYS` for `m` as the panel's JSON cell, C_gates beside algorithm_policy's `rho_inputs` (numbers to 6
    significant digits), refused where page_rho on the cell misses ρ or its basis at a point of `at` (formalism.at's
    output at each of the row's points, `own`'s)."""
    v = dict(zip(algorithm_policy.INPUTS, algorithm_policy.inputs(m, w_fwd)))
    v |= {"C_gates": m.gates, "w_fit_gates": m.w_fit, "alpha_ratio": m.alpha_ratio}
    if bad := [k for k in KEYS if isinstance(v[k], float) and not math.isfinite(v[k])]:
        raise ValueError(f"{', '.join(bad)} not finite")
    text = f'{{"C_gates":{algorithm_policy.num(m.gates)},' + algorithm_policy.rho_inputs(m, w_fwd)[1:]
    cell = json.loads(text)
    for name, want in at.items():
        got, basis = algorithm_policy.page_rho(cell, want["c"], want["k"])
        if basis != ("fit" if want["fit"] else "target") or not (
                got == want["rho"] or math.isclose(got, want["rho"], rel_tol=1e-4)):
            raise ValueError(f"at {name} (c {want['c']:.6g}, k {want['k']:g}): te_core's rho {want['rho']:.6g} "
                             f"{'fit' if want['fit'] else 'target'}, its rho_inputs' {got:.6g} {basis}")
    return text


def rho_cell(v: float | None) -> float | str | None:
    return None if v is None else "inf" if math.isinf(v) else float(algorithm_policy.num(v))


def kinds(one: dict, b: dict) -> str:
    """formalism.published's `one` as the kinds cell (algorithm_policy's `kinds`), its sensitivities `SENSITIVITIES`
    from the one bound `b` at its point, less any `one`'s max takes."""
    return algorithm_policy.kinds(one | {"sensitivities": {s: b["sensitivities"][s] for s in SENSITIVITIES
                                                           if s not in one["terms"]}})


def finite(x):
    """`x` with every non-finite float a string, for JSON."""
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    if isinstance(x, dict):
        return {k: finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [finite(v) for v in x]
    return x


def compact(d: dict) -> str:
    return json.dumps(d, separators=(",", ":"), allow_nan=False)


def source(revs: tuple[str, str, str]) -> str:
    """The panel's source: the command at this checkout's commit, and the branch and commit it builds on (`revs`,
    `commits`'s); the branch alone where that commit is this checkout's own."""
    head, branch, base = revs
    on = branch if base == head else f"{branch} at {base}"
    return f"python islands/{NAME}/{Path(__file__).name} at {head} (te_core {VERSION}, on {on})"


def build(spec: dict, revs: tuple[str, str, str], stamp: str | None = None,
          plan: dict | None = None) -> tuple[dict, list[dict]]:
    """The panel and each row's details (module docstring) from the rows file's `spec`, its source at `revs`
    (`commits`'s); a run with an entry in `plan` (a restate.py plan's `restated`) is read restated."""
    tr = traces(spec.get("traces", []))
    rows, details = [], []
    for r in spec["rows"]:
        what = f"{r['workload']} {r.get('run', MODELLED)}"
        if "modelled" in r:
            model = workloads.model(r["workload"], **r["dials"])
            session, pts = own({"parameters": model.dials["n_params"]})
            ms = {name: workloads.with_products(model, c, k) for name, (c, k) in pts.items()}
            m, w_fwd, dials = ms["console"], model.w_fwd, model.dials
            read = {"modelled": r["modelled"], "derived": model.derived}
            run = f"{MODELLED} ({r['modelled']})"
        else:
            entry = (plan or {}).get(r["run"])
            m, shape, read = (measured(r, tr.get(r["run"], {})) if entry is None
                              else measured(r, tr.get(r["run"], {}), entry))
            fwd = read.get("forward_positions")
            if bad := (token_dials(r["dials"], read["learner_tokens"], "its learner's tokens are", fwd)
                       if r.get("learner") else token_dials(r["dials"], m.tokens) if fwd is None else
                       token_dials(r["dials"], fwd, "its rounds' forward positions are")):
                raise SystemExit(f"{PANEL}: {what}: {'; '.join(bad)}")
            session, pts = own(shape)
            ms, w_fwd, dials = dict.fromkeys(pts, m), policy.w_fwd(shape), r["dials"]
            said = "; ".join([*(r[x] for x in ("about", "note") if r.get(x)),
                              *([split(read["shares"])] if "shares" in read else [])])
            run = f"{r['run']} {r['art']}" + (f" ({said})" if said else "")
            if r.get("rounds") and off_axis(what, read["axis"]):
                cells = {"workload": r["workload"], "model": r["model"], "dials": compact(dials), "run": run,
                         "own_c_gates": float(algorithm_policy.num(session["gates"])), "own_k_bits": session["k_bits"],
                         "reading": READING, "te_core": VERSION, "capture_code": (m.products or {}).get("capture_code"),
                         "binding_kind": algorithm_policy.UNMEASURED, "rests_on": NO_AXIS,
                         "kinds": algorithm_policy.kinds({"terms": dict.fromkeys(formalism.KINDS),
                                                          "sensitivities": dict.fromkeys(SENSITIVITIES), "why": NO_AXIS})}
                rows.append([cells.get(c) for c in COLUMNS])
                details.append(finite({"workload": r["workload"], "run": r["run"], "art": r["art"], "modelled": None,
                                       "read": read, "w_fwd_gates": w_fwd, "own_session": session,
                                       "unmeasured": NO_AXIS}))
                continue
        at = {name: formalism.at(ms[name], c, k) for name, (c, k) in pts.items()}
        one = {name: formalism.published(ms[name], c, k, READING) for name, (c, k) in pts.items()}
        bound = {name: formalism.bound(ms[name], c, k) for name, (c, k) in pts.items()}
        student = None
        if "student" in read:
            s = algorithm_policy.own_session(read["student"])
            student = s | {"published": formalism.published(m, s["gates"], algorithm_policy.OWN_K, READING)}
        try:
            cell = priced(m, w_fwd, at)
            over = learner_status(m, read["parts"]) if r.get("learner") else ROUNDS if r.get("rounds") else None
            st = modelled_status(m, bound) if "modelled" in r else status(m, over, read, pts)
        except ValueError as e:
            raise SystemExit(f"{PANEL}: {what}: {e}") from None
        if inf := [name for name, v in (one | ({"student": student["published"]} if student else {})).items()
                   if v["reading"] == "A" and v["rho"] is not None and math.isinf(v["rho"])]:
            raise SystemExit(f"{PANEL}: {what}: reading A's rho is infinite at {listed(inf)}, which no row publishes")
        con, mine = one["console"], one["own"]
        if (mine["binding_kind"], mine["rests_on"]) != (con["binding_kind"], con["rests_on"]):
            raise SystemExit(f"{PANEL}: {what}: rho_own binds on {mine['binding_kind']} ({mine['rests_on']}) and "
                             f"rho_console on {con['binding_kind']} ({con['rests_on']}), but the row's binding_kind "
                             f"and rests_on are the console default's alone: give rho_own its own basis column, as "
                             f"plot 2's basis_own, and build again")
        if "modelled" not in r:
            left = [g for g in read.get("untraced", []) if not g.endswith(IDLE)]
            st["products"] = products(bound, left)
        cells = {"workload": r["workload"], "model": r["model"], "dials": compact(dials), "run": run,
                 "rho_own": rho_cell(mine["rho"]), "own_c_gates": float(algorithm_policy.num(session["gates"])),
                 "own_k_bits": session["k_bits"], "rho_inputs": cell, "input_status": compact(st),
                 "rho_default": rho_cell(con["rho"]), "rho_console": rho_cell(con["rho"]), "reading": con["reading"],
                 "te_core": VERSION, "capture_code": con["capture_code"], "binding_kind": con["binding_kind"],
                 "rests_on": con["rests_on"] if "modelled" in r else algorithm_policy.rests_on(con),
                 "kinds": kinds(con, bound["console"]), "rho_served": rho_cell(one["served"]["rho"])}
        rows.append([cells[c] for c in COLUMNS])
        details.append(finite({"workload": r["workload"], "run": r.get("run"), "art": r.get("art"),
                               "modelled": r.get("modelled"), "read": read, "w_fwd_gates": w_fwd, "labels": m.labels,
                               "parts": [dataclasses.asdict(p) for p in m.parts], "tail_share": m.tail_share,
                               "own_session": session, "at": at, "published": one,
                               "bound": {name: {f: v[f] for f in BOUND} for name, v in bound.items()}}
                              | ({"student_session": student} if student else {})))
    p = {"id": PANEL, "title": "Standard training under the policy: covert workloads measured on real runs",
         "kind": "table", "updated_at": stamp or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
         "source": source(revs), "note": "", "columns": COLUMNS, "rows": rows}
    return p | {"note": note(p, details=details)}, details


def figure(rho: float | None) -> str:
    """A ρ as the note gives it, to 6 significant digits, thousands separated; "unmeasured" where it has none."""
    return "unmeasured" if rho is None else f"{float(algorithm_policy.num(rho)):,.10g}"


#: a measured row's ρ's least and greatest ratio, at 32-bit attention's products over 16-bit, in each of its ρ columns:
#: the as-if re-pricing's standard-workloads/out/{control,asif}/console/standard-workloads.json
ATTENTION_32 = "0.21 to 0.52"


def note(p: dict, tables: str | None = None, details: list[dict] = ()) -> str:
    """The panel's note: its readings as its rows have them, the headline first (rho_own, each row's own session at
    algorithm_policy's `OWN_K`, and at its `V3_PACKED_K` at least that), then the sensitivities beside it (the
    console's default, the serving block, and a distillation row's ρ at its student's own session, the rows'
    `details`' `student_session`), the old figure beside them (published's `fresh_old`), the price of attention's
    products (`ATTENTION_32`), and `tables` the stored tables (`cite`'s)."""
    into, out = policy.CONSOLE_SESSION
    got = {r[COLUMNS.index("reading")] for r in p["rows"]}
    how = "Reading B, a target, not a bound (tables: reading A's)" if got <= {"B"} else "Reading A"
    model = COLUMNS.index("model")
    students = [f"{r[model]} {figure(d['student_session']['published']['rho'])}"
                for r, d in zip(p["rows"], details) if d.get("student_session")]
    return (f"{how}. rho_own: own session ({into:,} in + {out:,} out; distill's teacher), k = "
            f"{algorithm_policy.OWN_K:,.0f}; at the packed {algorithm_policy.V3_PACKED_K:,} rho is at least this. "
            f"Sensitivities: rho_default = rho_console ({policy.REFERENCE_PARAMS / 1e6:.0f}M), rho_served (serving "
            "block)" + (f"; distill at its student's: {', '.join(students)}" if students else "")
            + ". fresh_old rests on P-width extended, false on honest circuits. Attention's products are priced at 16 "
              f"bits, as the card computes them; at 32 bits a measured rho is {ATTENTION_32} of this."
            + (f" Tables {tables}." if tables else ""))


def refused(p: dict) -> list[str]:
    """What the panels route refuses in `p`: te_core.console's `refused`, and its note and body over their limits."""
    out = console.refused(p)
    if len(p["note"]) > console.NOTE_MAX:
        out.append(f"its note is {len(p['note'])} characters, over {console.NOTE_MAX}")
    if len(console.body(p)) > console.BODY_MAX:
        out.append(f"its body is {len(console.body(p))} bytes, over {console.BODY_MAX}")
    return out


def kept(r: list, columns: list[str]) -> str:
    """What a panel's row stands for, which a version switch keeps: its run's id, or a modelled row's workload and
    model."""
    run = str(r[columns.index("run")])
    if run.startswith(MODELLED):
        return f"{r[columns.index('workload')]} {r[columns.index('model')]} ({MODELLED})"
    return run.split()[0]


def emptied(p: dict, served: dict | None) -> list[str]:
    """Why `p` mustn't replace the panel the route serves (`served`): a version switch never empties a populated
    panel, so every run it serves and every workload it models stays a row (`kept`), and a te_core newer than the one
    it runs on keeps it."""
    if not served:
        return []
    te = served["columns"].index("te_core")
    mine = {kept(r, COLUMNS) for r in p["rows"]}
    out = [f"it drops the served row of {x}" for x in (kept(r, served["columns"]) for r in served["rows"])
           if x not in mine]
    newer = sorted({str(r[te]) for r in served["rows"] if _version(str(r[te])) > _version(VERSION)})
    return out + [f"the route serves te_core {v}, newer than {VERSION}" for v in newer]


def cite(out: Path) -> str | None:
    """How the note cites the tables a build writes under `out`: the research run's id and their path in its run dir
    (`run:RUN/out/.../tables`), where `out` is in the run's out/, which custody stores with the run's record; else None,
    a build no `put` takes."""
    run_dir = os.environ.get("RESEARCH_RUN_DIR")
    if not run_dir:
        return None
    try:
        rel = (out.resolve() / "tables").relative_to(Path(run_dir).resolve())
    except ValueError:
        return None
    if rel.parts[0] != "out":
        return None
    return f"run:{os.environ.get('RESEARCH_RUN_ID') or Path(run_dir).name}/{rel.as_posix()}"


def put(body: Path, sent: dict, dry: bool = False) -> int:
    """PUT a build's console body (`body`, its console/standard-workloads.json), refused where the route's panel
    isn't `sent`'s (the panels it served when the build launched: someone PUT since), where the body would drop a
    served row or the route refuses it (`emptied`, `refused`), or where its note cites no stored tables (`TABLES`);
    then GET the panel, which must be the body as numbers (`readback`). 0 where the route serves the body, else 1. `dry`
    PUTs nothing: past the same refusals, it prints the body's sha256, bytes, rows and note length, and what
    te_core.console's `put` would refuse it for."""
    q = json.loads(body.read_text())
    if q.get("id") != PANEL:
        raise SystemExit(f"{PANEL}: not PUT: {body} is {q.get('id')}'s body")
    now = console.served().get(PANEL)
    if not readback.same(now, sent.get(PANEL)):
        raise SystemExit(f"{PANEL}: not PUT: the route's panel changed after the build read it (now updated "
                         f"{(now or {}).get('updated_at')}, from {(now or {}).get('source')}): someone PUT over it, "
                         f"so tell the coordinator its time and build, and PUT nothing")
    if bad := emptied(q, now) + refused(q):
        raise SystemExit(f"{PANEL}: not PUT, since {'; '.join(bad[:5])}")
    if not TABLES.search(q["note"]):
        raise SystemExit(f"{PANEL}: not PUT: its note cites no stored tables, so build it as a research run with --out "
                         f"in the run's out/")
    if dry:
        sent_body = console.body(q)
        print(f"dry: {PANEL}: sha256 {hashlib.sha256(sent_body).hexdigest()}, {len(sent_body)} bytes, {len(q['rows'])} "
              f"rows, note {len(q['note'])} characters; console.put would refuse: {console.refusal(q) or 'nothing'}; "
              f"nothing stored, nothing PUT")
        return 0
    got = console.put(q)
    print(f"PUT {console.PANELS_URL}/{PANEL}: {got} at {datetime.now(UTC):%H:%M:%SZ}")
    if not got.startswith("2"):
        return 1
    back = console.served().get(PANEL) or {}
    if differ := readback.differ(q, back):
        print(f"GET {PANEL}: the route serves another panel than the body ({', '.join(differ)} differ)")
        return 1
    print(f"GET {PANEL}: the body, {len(back['rows'])} rows, updated_at {back['updated_at']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    do = ap.add_mutually_exclusive_group(required=True)
    do.add_argument("--out", type=Path, help="build into DIR (on a node, in the research run's out/)")
    do.add_argument("--put", type=Path, metavar="BODY", help="PUT a build's console body (the agent VM's)")
    ap.add_argument("--rows", type=Path, default=ROWS)
    ap.add_argument("--served", type=Path,
                    help="the panels the route served when the build launched (te_core.console.served()'s, as JSON)")
    ap.add_argument("--base", help="the commit of the integrator's branch this checkout builds on (default: their "
                                   "merge base on origin)")
    ap.add_argument("--branch", help="the branch this checkout builds on, where it isn't the integrator's: it must hold "
                                     "this checkout's commit, and the source names it")
    ap.add_argument("--restated", type=Path, metavar="PLAN",
                    help="read each run this plan restates (the scorer's restate.py plan) with its record's tokens a "
                         "step restated, and its stored traces against the restated count")
    ap.add_argument("--dry", action="store_true", help="with --put: PUT nothing, print what the put would send")
    a = ap.parse_args(argv)
    sent = json.loads(a.served.read_text()) if a.served else None
    if a.put:
        if sent is None:
            raise SystemExit(f"{PANEL}: --put needs --served, the panels the route served when the build launched")
        return put(a.put, sent, a.dry)
    plan = json.loads(a.restated.read_text())["restated"] if a.restated else None
    p, details = build(json.loads(a.rows.read_text()), commits(a.base, a.branch), plan=plan)
    q = p | {"note": note(p, cite(a.out), details)}
    if bad := refused(q):
        raise SystemExit(f"{PANEL}: the panels route refuses it: {'; '.join(bad[:5])}")
    if sent is not None and (bad := emptied(q, sent.get(PANEL))):
        raise SystemExit(f"{PANEL}: not built, since {'; '.join(bad[:5])}: name the served runs in the rows file")
    tables, cons = a.out / "tables", a.out / "console"
    tables.mkdir(parents=True, exist_ok=True)
    cons.mkdir(parents=True, exist_ok=True)
    (tables / f"{NAME}.json").write_text(json.dumps(p | {"details": details}, indent=1, allow_nan=False) + "\n")
    (cons / f"{NAME}.json").write_bytes(console.body(q))
    ix = {c: i for i, c in enumerate(COLUMNS)}
    for r in q["rows"]:
        print(f"{r[ix['workload']]:10} {r[ix['model']]:4} {r[ix['run']][:21]}  reading {r[ix['reading']]}  "
              f"rho_own {r[ix['rho_own']]} (c {r[ix['own_c_gates']]:g}, k_bits {r[ix['own_k_bits']]})  "
              f"rho_console {r[ix['rho_console']]}  binding {r[ix['binding_kind']]}  rho_served {r[ix['rho_served']]}")
    print(f"{PANEL}: {len(q['rows'])} rows, {len(console.body(q))} bytes, longest cell "
          f"{max(len(v) for r in q['rows'] for v in r if isinstance(v, str))} characters, note {len(q['note'])} "
          f"-> {a.out}\n{q['note']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
