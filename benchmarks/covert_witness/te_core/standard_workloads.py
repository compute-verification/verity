"""The console's training panel `training/standard-workloads`: the standard covert workloads (te_core.workloads'
names: standard, diloco, rl, inference, lora and distill), each a small real run from a registered start, measured by
this te_core and priced by te_core.formalism at two dial points.

    python -m te_core.standard_workloads --out DIR [--rows FILE] [--publish]

`--rows` (default `ROWS`, the island's rows.json) gives the stored traces (the island's trace.py output, chosen steps
of the runs, refused unless traced by a te_core from `TRACED_SINCE` to this one) and a row per workload and measured
dial setting: its workload, model, dials (te_core.workloads' arguments), run id and record art, what the run was beside
them (`about`, and a `note`), and `rounds` for a generation run (the island's generate.py). A row is the run's last
checkpoint as te_core.ledger.ingest writes it (into a scratch ledger) with the run's traced structures
(annotate.with_structures), measured by te_core.formalism.measure (its op log the ledger row's macs_log), and priced as
te_core.algorithm_policy prices a run, by formalism.published under `READING`: `rho_default` and `rho_console` are its
ρ at the dial's default (`points`: c = 4,096 tokens of the start's forward, k = 80,000 bits) and at algorithm_policy's
`CONSOLE`, the console's default session, and `binding_kind`, `rests_on` and `kinds` (algorithm_policy's `kinds`: the
terms its max takes, the sensitivities beside it) are published's at the console's (`cells`: under reading A, null,
none and "unmeasured" where Fit fails and the one bound measures no kind; a reading-A ρ is never infinite, so `build`
refuses one). `rho_inputs` holds `KEYS` (C_gates beside algorithm_policy's `rho_inputs`), from which
algorithm_policy.page_rho gives the dial's ρ at any (c, k); `build` refuses a row whose rho_inputs miss formalism.at's
ρ at either point or whose token dials miss its measured tokens (`token_dials`).
`input_status` gives each input's source, "measured" or "assumed: " and why, from te_core's labels: a count, a trace's
quantity, E's size and the harness's decoder are measured, and a stub, an assumption or an unmeasured quantity is
assumed.

A `rounds` row is the run's generation rounds alone (`rounds`), its setup and training steps dropped: a round is one
counted step that codes the served weights, prefills its prompts and decodes. The capture can't trace a round (its
decode loop's length is no token axis), so the counter's record is read with these substitutions (`ROUNDS`, each in
the row's `input_status`): a round's tokens its prompts' and every generated token but the last (the counter counts
the prompts' positions alone); every GEMM on the served codes (W4A4, emulated in bf16 on their decoded values) priced
as NVFP4 (gemm/fp4); its affected gates the forward ops that read an update tensor (te_core's untraced convention); Δ
the served update's bits from the run's notes (every served matrix's codes and scales with the gains in bf16, or the
LoRA in bf16), and Δ_grad, the partial gradient and the gradient cut 0 (no backward); τ an id per prompt token and
none per generated one; a, u and u_in te_core's stubs.

DIR gets the uncut table, each row with its labels, parts, and formalism.at, formalism.published and the one bound
(reading A's, formalism.bound) at both points (tables/standard-workloads.json), and the console's copy
(console/standard-workloads.json); `--publish` GETs what the route serves, puts tables/ in the evidence store, whose art
id the note names, and PUTs the panel, unless it would drop a served run's row or the route serves a newer te_core
(`emptied`).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from . import VERSION, algorithm_policy, console, formalism, panels, policy
from .gates import of_log
from .ledger import Ledger, ingest

NAME = "standard-workloads"
PANEL = f"training/{NAME}"
COLUMNS = ["workload", "model", "dials", "run", "rho_inputs", "input_status", "rho_default", "rho_console", "reading",
           "te_core", "binding_kind", "rests_on", "kinds"]
#: the reading every row publishes (formalism.published's): B, a target, until te_core prices the console's full
#: session, attention included, and the fresh-products kind; then A, the one bound over the kinds valid on their own
READING = "B"
#: the `rho_inputs` cell's keys, algorithm_policy.rho_inputs' in its order
KEYS = ("C_gates", *panels.PACKED, *algorithm_policy.SPLIT)
#: the first te_core whose traces this one's formalism reads whole: their entries and partial and gradient cuts
TRACED_SINCE = "4.4"
#: the one bound's fields each row's details keep at both points
BOUND = ("terms", "max", "binding", "unmeasured", "sensitivities")
ROWS = Path(__file__).resolve().parents[1] / "islands" / NAME / "rows.json"
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
_PRICED = "assumed: the rounds' counted work, each GEMM on the served codes (W4A4, emulated in bf16 on decoded values) " \
          "priced as NVFP4 (gemm/fp4)"
_PER_TOKEN = _PRICED + ", per token through the forward: each prompt token and every generated token but the last " \
                       "(the counter counts the prompts' alone)"
_UNTRACED = "; a round is not traced (its decode loop's length is no token axis of the capture)"
ROUNDS = {
    "C_gates": _PRICED, "w_gates": _PER_TOKEN, "sigma": _PRICED, "w_fit_gates": _PER_TOKEN,
    "tau_bits": "assumed: one id per prompt token and none per generated token, which the round samples" + _UNTRACED,
    "a_bits": "assumed: te_core's stub, one token's bf16 residual stream (d_model x 16)" + _UNTRACED,
    "u_points": "assumed: te_core's stub (A1), evenly spread to u(1) = Delta" + _UNTRACED,
    "u_in_points": "assumed: te_core's stub (A1'), u_in = u" + _UNTRACED,
    "alpha_ratio": "assumed: a round's affected work is its forward ops that read an update tensor (te_core's "
                   "untraced convention)" + _UNTRACED,
}
ROUND_LABELS = {
    "tokens": "measured: each round's prompt positions (the counter's) and every generated token but the last, which "
              "the counter doesn't count, a sampled id being no data-stream position (the run's notes)",
    "gates": "counted MACs in gates (te_core.gates), the GEMMs on served codes at gemm/fp4 (standard_workloads.rounds)",
    "token_gates": "counted: the rounds' forward in gates, the GEMMs on served codes at gemm/fp4",
    "affected_gates": "stub: the rounds' forward ops that read an update tensor, the GEMMs on served codes at gemm/fp4",
    "delta_bits": "measured: the served update's bits (generate.py's notes: every served matrix's NVFP4 codes and "
                  "scales with the gains in bf16, or the LoRA in bf16)",
    "delta_grad_bits": "measured: 0, a round has no backward",
    "partial_bits": "measured: 0, a round has no backward (no partial gradient, no update applied)",
    "tau_bits": "stub: ceil(log2 V) bits per prompt token, none per generated token, over the rounds' tokens",
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
    """Each run's traced steps in `arts` (the island's trace.py output), each with its art and the te_core that traced
    it, refused unless that is `TRACED_SINCE` to this one."""
    out: dict[str, dict[int, dict]] = {}
    for art in arts:
        root = fetch(art)
        v = version_at(json.loads((root / "manifest.json").read_text())["te_commit"])
        if not _version(TRACED_SINCE) <= _version(v) <= _version(VERSION):
            raise SystemExit(f"{PANEL}: {art} was traced by te_core {v}, not {TRACED_SINCE} to {VERSION}: trace the "
                             f"runs again (islands/{NAME}/job.sh) and name the new traces in the rows")
        for p in sorted((root / "traces").glob("*.json")):
            t = json.loads(p.read_text())
            if t.get("trace"):
                out.setdefault(t["run"], {})[t["step"]] = t | {"from": art, "te_core": v}
    return out


def coded(row: list, served: set[tuple[int, int]]) -> bool:
    """Whether a structure's log row is a GEMM on a served matrix's codes: its second operand (K, N) is a served
    (N, K) matrix transposed."""
    return row[2].split(".")[0] in GEMMS and tuple(json.loads(row[3])[-1][-2:]) in served


def rounds(record: dict, notes: dict, served: set[tuple[int, int]]) -> dict:
    """`record` (generate.py's: each training step, and a round after each of `notes`' rounds) as its rounds alone,
    the setup and the tail without work and the run's last checkpoint its one: each round's tokens its prompts' (the
    counter's, which counts data-stream positions, so no sampled id) and every generated token but the last, every
    GEMM on served codes (`coded`) marked NVFP4, each round structure's affected gates its forward ops that read an
    update tensor and its Δ_grad 0."""
    after = [r["after_step"] for r in notes["rounds"]]
    if len(record["steps"]) != notes["args"]["steps"] + len(after):
        raise ValueError(f"{len(record['steps'])} steps, not {notes['args']['steps']} training steps and "
                         f"{len(after)} rounds: not generate.py's layout")
    at = {t + i: r for i, (t, r) in enumerate(zip(after, notes["rounds"]))}
    steps = [dict(s) for s in record["steps"] if s["step"] in at]
    for s in steps:
        r = at[s["step"]]
        if s["tokens"] != r["sequences"] * r["prompt"]:
            raise ValueError(f"round at step {s['step']}: {s['tokens']} tokens counted, not its prompts' ({r})")
        s["tokens"] = r["sequences"] * (r["prompt"] + r["generated"] - 1)
    structures = {}
    for sid in {s["struct"] for s in steps}:
        st = record["structures"][sid]
        log = [[*r[:3], f"{r[3]} {FP4}" if coded(r, served) else r[3], *r[4:]] for r in st["log"]]
        if any(r[0] == "bwd" for r in log):
            raise ValueError(f"round structure {sid} has a backward")
        upd = {" ".join(str(x) for x in r[:4]): r[4:6] for r in log if r[:2] == ["fwd", "updated"]}
        structures[sid] = st | {"log": log, "affected_gates": of_log({"ops": upd})["gates"], "delta_grad_bits": 0.0}
    none = {"by": {}, "macs": 0}
    return record | {"steps": steps, "structures": structures, "setup": record["setup"] | none,
                     "tail": record.get("tail", {}) | none, "checkpoints": record["checkpoints"][-1:]}


def round_formalism(notes: dict, tensors: dict, vocab: int) -> dict:
    """A rounds record's own `formalism` fields: Δ the served update's bits and τ the rounds' input ids per token."""
    a = notes["args"]
    if a["update"] == "lora":
        delta = notes["served_lora_bits"]
    else:
        gains = sum(math.prod(shape) for _, shape in tensors.values() if len(shape) == 1)
        delta = notes["served_code_bits"] + 16 * (gains + (math.prod(tensors["embed"][1]) if a["head"] == "bf16" else 0))
    ids = sum(r["sequences"] * r["prompt"] for r in notes["rounds"])
    tokens = sum(r["sequences"] * (r["prompt"] + r["generated"] - 1) for r in notes["rounds"])
    return {"delta_bits": float(delta), "tau_bits": policy.id_bits(vocab) * ids / tokens, "labels": ROUND_LABELS}


def start(name: str) -> tuple[dict, dict]:
    """A start's shape and its tensors (name -> [dtype, shape], its start.json's), from the starts manifest."""
    s = json.loads(policy.MANIFEST.read_text())["starts"][name]
    path = f"{s.get('path', 'starts')}/{name}/start.json"
    return s["shape"], json.loads((fetch(s["art"], path) / path).read_text())["tensors"]


def measured(row: dict, traced: dict[int, dict]) -> tuple[formalism.Measured, dict, dict]:
    """A row's quantities: its run's last checkpoint ingested with the run's `traced` steps' structures, or its rounds
    (`rounds`); the start's shape and what the row read."""
    root = fetch(row["art"], "run/run.json", "meta.json")
    record = json.loads((root / "run" / "run.json").read_text())
    meta = json.loads((root / "meta.json").read_text())
    shape, tensors = start(record["start"]["name"])
    by_step = {s["step"]: s["struct"] for s in record["steps"]}
    structures = {by_step[step]: t["trace"]["structure"] for step, t in traced.items()}
    read = {"traces": [{"step": step, "struct": by_step[step], "art": t["from"], "te_core": t["te_core"],
                        "measured": {k: t["measured"].get(k) for k in TRACE_FACTS}}
                       for step, t in sorted(traced.items())]}
    if row.get("rounds"):
        notes = meta["notes"]
        matrices = [n for n in tensors if n.startswith("blocks.") and len(tensors[n][1]) == 2]
        names = matrices + (["embed"] if notes["args"]["head"] == "fp4" else [])
        record = rounds(record, notes, {(tensors[n][1][1], tensors[n][1][0]) for n in names})
        record["formalism"] = round_formalism(notes, tensors, shape["vocab"])
        structures = {}
        read = {"rounds": notes["rounds"], "round_steps": [s["step"] for s in record["steps"]],
                "fp4_macs": sum(r[5] * sum(s["struct"] == sid for s in record["steps"])
                                for sid, st in record["structures"].items() for r in st["log"] if r[3].endswith(FP4))}
    else:
        record["checkpoints"] = record["checkpoints"][-1:]
    with tempfile.TemporaryDirectory(prefix="sw-ingest-") as tmp:
        d = Path(tmp) / "run"
        d.mkdir()
        (d / "run.json").write_text(json.dumps(record))
        got = ingest(d, Ledger(Path(tmp) / "ledger", NAME), direction=NAME, mechanism=row["workload"],
                     run_id=row["run"], structures=structures or None, meta=meta)[-1]
    read |= {"checkpoint": got["checkpoint_step"], "steps": len(record["steps"]),
             "tokens": sum(s.get("tokens", 0) for s in record["steps"]), "invalid": got["invalid"]}
    m = formalism.measure(got, shape)
    if row.get("rounds"):
        m = dataclasses.replace(m, parts=tuple(dataclasses.replace(p, partial_bits=0.0, grad_bits=0.0)
                                               for p in m.parts))
    return m, shape, read


def _measured(label: str | None) -> bool:
    return label is not None and (label.startswith(("measured", "counted")) or "trace" in label or label in READ
                                  or label in algorithm_policy.COUNTED)


def status(m: formalism.Measured, override: dict[str, str] | None = None) -> dict[str, str]:
    """Each of `KEYS`' source (module docstring), `override`'s where it names one."""
    out = {}
    for key in KEYS:
        keys = LABELLED["delta_bits"] if key == "delta_t_bits" and m.delta_t_bits is None else LABELLED[key]
        bad = [m.labels.get(k) for k in keys if not _measured(m.labels.get(k))]
        out[key] = "measured" if not bad else f"assumed: {bad[0] or 'no label from te_core'}"
        if key == "delta_grad_bits" and m.fit_delta_grad_bits is None:
            n = sum(p.steps for p in m.parts if p.tokens and p.delta_grad_bits is None)
            have = m.delta_grad_bits
            out[key] = (f"assumed: unmeasured, {n} of {sum(p.steps for p in m.parts)} steps in structures with no "
                        f"traced Delta_grad, so te_core fails Fit at every k"
                        + (f" (the traced steps' largest {algorithm_policy.num(have)} bits)" if have is not None else ""))
    return out | (override or {})


def token_dials(dials: dict, tokens: float) -> list[str]:
    """Where `dials`' token counts (`tokens`, and `tokens_per_param` over `student_params` or `n_params`) miss the run's
    measured `tokens` (formalism.Measured's: each step's count scaled by its traced structure's consumed tokens over its
    count, so a step of micro-batches counts them all) by more than 1%: te_core leaves a step before the run's first
    update at the counter's count, which counts one micro-batch of a step's."""
    want = {"tokens": tokens}
    if "tokens_per_param" in dials:
        want["tokens_per_param"] = tokens / dials.get("student_params", dials.get("n_params"))
    return [f"its dial {k} is {dials[k]:g}, te_core measured {algorithm_policy.num(v)}: the dial is the run's"
            for k, v in want.items() if k in dials and not math.isclose(dials[k], v, rel_tol=1e-2)]


def points(w_fwd: float) -> dict[str, tuple[float, float]]:
    """The two dial points (c, k): the dial's default for a start whose forward is `w_fwd` gates a token, and the
    console's (algorithm_policy's `default` and `CONSOLE`)."""
    return {"default": algorithm_policy.default(w_fwd), "console": algorithm_policy.CONSOLE}


def priced(m: formalism.Measured, w_fwd: float, at: dict[str, dict]) -> str:
    """`KEYS` for `m` as the panel's JSON cell, C_gates beside algorithm_policy's `rho_inputs` (numbers to 6
    significant digits), refused where page_rho on the cell misses ρ or its basis at a point of `at` (formalism.at's
    output at each of `points`)."""
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


def rho_cell(v: float) -> float | str:
    return "inf" if math.isinf(v) else float(algorithm_policy.num(v))


def cells(one: dict, b: dict) -> tuple[float | str | None, str, str]:
    """formalism.published's `one` as the row's cells (ρ, its binding kind and what it rests on), beside the one bound
    `b` at its point: under reading A, where Fit fails and `b` measures no kind, null, none and "unmeasured", never
    max(1, nothing), which reads as no slowdown."""
    if one["reading"] == "A" and one["binding_kind"] != "fit" and b["max"] is None:
        return None, "none", "unmeasured"
    return rho_cell(one["rho"]), one["binding_kind"], one["rests_on"]


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


def build(spec: dict, stamp: str | None = None) -> tuple[dict, list[dict]]:
    """The panel and each row's details (module docstring) from the rows file's `spec`."""
    tr = traces(spec.get("traces", []))
    rows, details = [], []
    for r in spec["rows"]:
        m, shape, read = measured(r, tr.get(r["run"], {}))
        if bad := token_dials(r["dials"], m.tokens):
            raise SystemExit(f"{PANEL}: {r['workload']} {r['run']}: {'; '.join(bad)}")
        w_fwd = policy.w_fwd(shape)
        pts = points(w_fwd)
        at = {name: formalism.at(m, c, k) for name, (c, k) in pts.items()}
        try:
            cell = priced(m, w_fwd, at)
        except ValueError as e:
            raise SystemExit(f"{PANEL}: {r['workload']} {r['run']}: {e}") from None
        one = {name: formalism.published(m, c, k, READING) for name, (c, k) in pts.items()}
        bound = {name: formalism.bound(m, c, k) for name, (c, k) in pts.items()}
        if inf := [name for name, v in one.items() if v["reading"] == "A" and math.isinf(v["rho"])]:
            raise SystemExit(f"{PANEL}: {r['workload']} {r['run']}: reading A's rho is infinite at "
                             f"{' and '.join(inf)}, which no row publishes")
        (rho_default, _, _), (rho_console, binding, rests) = cells(one["default"], bound["default"]), \
            cells(one["console"], bound["console"])
        st = status(m, ROUNDS if r.get("rounds") else None)
        said = "; ".join(r[x] for x in ("about", "note") if r.get(x))
        run = f"{r['run']} {r['art']}" + (f" ({said})" if said else "")
        rows.append([r["workload"], r["model"], compact(r["dials"]), run, cell, compact(st), rho_default, rho_console,
                     one["console"]["reading"], VERSION, binding, rests, algorithm_policy.kinds(one["console"])])
        details.append(finite({"workload": r["workload"], "run": r["run"], "art": r["art"], "read": read,
                               "w_fwd_gates": w_fwd, "labels": m.labels,
                               "parts": [dataclasses.asdict(p) for p in m.parts], "tail_share": m.tail_share,
                               "at": at, "published": one,
                               "bound": {name: {f: v[f] for f in BOUND} for name, v in bound.items()}}))
    p = {"id": PANEL, "title": "Standard training under the policy: covert workloads measured on real runs",
         "kind": "table", "updated_at": stamp or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
         "source": f"python -m te_core.{NAME.replace('-', '_')} (te_core {VERSION}; islands/{NAME})",
         "note": "", "columns": COLUMNS, "rows": rows}
    return p | {"note": note(p)}, details


def note(p: dict, art: str | None = None) -> str:
    """The panel's note, its readings as its rows have them, and `art` the stored tables."""
    c, k = algorithm_policy.CONSOLE
    got = {r[COLUMNS.index("reading")] for r in p["rows"]}
    how = ("reading B, a target and not a bound (the tables hold reading A's bound)" if got <= {"B"} else
           "reading A, the one bound over the kinds valid on their own")
    return (f"Covert workloads, small real runs from registered starts, te_core {VERSION}: formalism.published, {how}, "
            f"at the dial's default (c = {policy.SESSION_TOKENS} w_fwd, k = {policy.REFERENCE_K_BITS:,}) and the "
            f"console's (c = {algorithm_policy.num(c, 5)} gates, the 45M reference's 5,120-token session without "
            f"attention, k = {k:,}); binding_kind, rests_on, kinds: the console's. input_status: measured, or assumed "
            "and why." + (f" Tables {art}." if art else ""))


def refused(p: dict) -> list[str]:
    """What the panels route refuses in `p`: te_core.console's `refused`, and its note and body over their limits."""
    out = console.refused(p)
    if len(p["note"]) > console.NOTE_MAX:
        out.append(f"its note is {len(p['note'])} characters, over {console.NOTE_MAX}")
    if len(console.body(p)) > console.BODY_MAX:
        out.append(f"its body is {len(console.body(p))} bytes, over {console.BODY_MAX}")
    return out


def emptied(p: dict, served: dict | None) -> list[str]:
    """Why `p` mustn't replace the panel the route serves (`served`): a version switch never empties a populated
    panel, so every run it serves stays a row, and a te_core newer than this one keeps it."""
    if not served:
        return []
    run, te = (served["columns"].index(c) for c in ("run", "te_core"))
    mine = {r[COLUMNS.index("run")].split()[0] for r in p["rows"]}
    out = [f"it drops the served row of {r[run].split()[0]}" for r in served["rows"] if r[run].split()[0] not in mine]
    newer = sorted({str(r[te]) for r in served["rows"] if _version(str(r[te])) > _version(VERSION)})
    return out + [f"the route serves te_core {v}, newer than {VERSION}" for v in newer]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", type=Path, default=ROWS)
    ap.add_argument("--publish", action="store_true",
                    help=f"put the tables in the evidence store and PUT {console.PANELS_URL}/{PANEL}")
    a = ap.parse_args(argv)
    spec = json.loads(a.rows.read_text())
    p, details = build(spec)
    if bad := refused(p | {"note": note(p, "art:" + "0" * 64)}):
        raise SystemExit(f"{PANEL}: the panels route refuses it: {'; '.join(bad[:5])}")
    if a.publish and (bad := emptied(p, console.served().get(PANEL))):
        raise SystemExit(f"{PANEL}: not PUT, since {'; '.join(bad[:5])}: name the served runs in the rows file")
    tables, cons = a.out / "tables", a.out / "console"
    tables.mkdir(parents=True, exist_ok=True)
    cons.mkdir(parents=True, exist_ok=True)
    (tables / f"{NAME}.json").write_text(json.dumps(p | {"details": details}, indent=1, allow_nan=False) + "\n")
    art = algorithm_policy.store(tables, {"what": f"{PANEL} uncut, each row's labels, parts, and formalism.at, "
                                                  f"published and the one bound at both points "
                                                  f"(te_core.standard_workloads)", "rows": len(p["rows"]),
                                          "source": p["source"]}) if a.publish else None
    q = p | {"note": note(p, art)}
    (cons / f"{NAME}.json").write_bytes(console.body(q))
    ix = {c: i for i, c in enumerate(COLUMNS)}
    for r in q["rows"]:
        print(f"{r[ix['workload']]:10} {r[ix['run']][:21]}  reading {r[ix['reading']]}  rho_default "
              f"{r[ix['rho_default']]}  rho_console {r[ix['rho_console']]}  binding {r[ix['binding_kind']]}")
    print(f"{PANEL}: {len(q['rows'])} rows, {len(console.body(q))} bytes, longest cell "
          f"{max(len(v) for r in q['rows'] for v in r if isinstance(v, str))} characters, note {len(q['note'])}"
          f"{f' ({art})' if art else ''} -> {a.out}")
    if a.publish:
        got = console.put(q)
        print(f"PUT {console.PANELS_URL}/{PANEL}: {got}")
        return 0 if got.startswith("2") else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
