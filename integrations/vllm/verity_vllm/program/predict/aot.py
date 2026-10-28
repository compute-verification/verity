"""The AOT Build: a switched row's derive dirs from the predictor, and the comparison that switches a row.

A row is switched when its entry in the circuit database's index (`manifests/records.json`) holds `aot` (`config_record.aot_of`):
the comparison run that found its predicted Programs digest-equal to its traced Build on every rank (`compare`), and that traced
Build (`trace`, a `vllm-build/v1`).  Its record path's Build (`row_tp.TpRow.build`) then writes each derive dir with `emit` in place
of the trace:

    descriptor.json.gz   the predicted Program, under the trace's annotations: the runtime correspondence (per Call, the fx occurrence
                         it was lowered from, its module, views and slots), which only the trace yields
    instances.json.gz    the predicted Program's instance sequence (`pipeline.build.write_instances`), with the Call and weight
                         names the trace gave it (`traced_names`), which no digest covers and Match and the manifest read
    the rest (CARRIED)   as the trace wrote them, with `aot` in result.json and construction_manifest.json: what predicted the
                         Program, from which trace

`emit` refuses unless the predicted Program is the trace's, digest and descriptor bytes: what it carries is about that Program only.
The trace stays the check: `pipeline.neutrality`'s heavy tier traces every row and compares with the pins, and a trace never reads
the predictor (`program.frontend.inputs_trace`).

A single-GPU row's Build of record needs no trace at all (`write_row`, `predict row`): from the row's declared inputs, each request
Program with the names its trace gives it (`naming.write_build`, `build_request_<shape>/`), GP-01's workload over them
(`write_workload`, `build_workload/`: its artifact `workload_program.json`, with the `lag` and `applicability.engine.async_scheduling`
the declared target resolves, and its `descriptor.json.gz`, which GM-01 holds the digest and the lag to), the row's required-value
manifest over them (`manifest.json`, the row's own query and policies, `row_build`) and
`predicted.json` (`records.predicted_row`), what predicted it and the manifest's sha256. That is the row's Build of record, the
ahead-of-time compiler's, which a traced Build only cross-checks. `compare --predicted DIR` holds it to
a traced Build of the same row: every Program digest, and the manifest's digest, Program digest and identities.

    verity-vllm predict emit --row ROW --rank R --config CONFIG_JSON --trace DIR --out DIR (--wrapper step --M N --pos0 N |
                             --wrapper request --LP N --T N) [--aot JSON]
    verity-vllm predict compare ROW_DIR --row ROW --config CONFIG_JSON --out comparison.json [--workers N] [--predicted DIR]
    verity-vllm predict row --row ROW --out DIR [--config CONFIG_JSON] [--jobs N]
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import multiprocessing
import os
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity_vllm.config import WORKLOADS, RowEngine, UsageError, option, row_file, workload_file
from verity_vllm.records.predicted_row import FILE, PredictedRow

from . import score as S
from .build import build_program
from .phi import Phi

SCHEMA = "verity-vllm/aot-comparison/v1"
#: what a predicted row's `build_workload/workload_program.json` names as its maker
ROW_BUILT_BY = "verity_vllm.program.predict.aot"
#: the traced derive dir's files a switched row's carries as written; `instances.json.gz` is the predicted Program's own
CARRIED = ("result.json", "artifact.json", "construction_manifest.json", "derive-report.json", "tp_links.json", "inputs.json")
REQUEST = re.compile(r"build_request_LP(\d+)_T(\d+)")
MECHANISM = "verity_vllm.program.predict: the Program from the row's declared inputs, ahead of time; the data about it from the trace"


def phi_at(row: str, rank: int, config: str) -> Phi:
    """φ of `row` at `rank` from its declared inputs (`score.phi_of`), the checkpoint's `config.json` at `config`."""
    phi = S.phi_of({"row": row, "workload": json.loads(workload_file(WORKLOADS / f"{row}.json").read_text())}, config=config)
    return Phi.from_json({**phi.to_json(), "tp": {**phi.tp, "rank": rank}})


def predict(phi: Phi, wrapper: str, *, M: int = 1, pos0: int = 8, LP: int | None = None, T: int | None = None):
    return build_program(phi, "step", M=M, pos0=pos0) if wrapper == "step" else build_program(phi, "request", LP=LP, T=T)


def traced_names(program, desc: dict, trace: str) -> tuple[dict[int, str | None], list[list[str]]]:
    """The names no digest covers, as the trace gave them: each Call's naming hint (the descriptor's `node_hints`: the Calls are the
    trace's, the descriptors being byte-equal) and each weights record field's leaf name (`derive-report.json`'s `provenance.weights`,
    in field order), refused where the fields differ in number or size."""
    from verity.ml.circuits.types import Record
    hints = {int(k): v for k, v in ((desc.get("annotations") or {}).get("node_hints") or {}).items()}
    params = []
    for n, t in program.fn.body.params:
        if isinstance(t, Record):
            leaves = sorted(json.loads(Path(trace, "derive-report.json").read_text())["provenance"]["weights"], key=lambda w: w["field"])
            if (got := [f.leaves for _, f in t.fields]) != (want := [math.prod(w["shape"]) for w in leaves]):
                raise UsageError(f"{trace}: the predicted {n} record's {len(got)} fields are not the trace's {len(want)} weight leaves in size")
            t = Record(*((w["name"], f) for w, (_, f) in zip(leaves, t.fields)))
        params.append([n, str(t)])
    return hints, params


def emit(phi: Phi, wrapper: str, trace: str, out: str, *, M: int = 1, pos0: int = 8, LP: int | None = None, T: int | None = None,
         aot: dict | None = None) -> dict:
    """One derive dir of a switched row's Build at `out`: the predicted Program, with the data `trace` (the traced derive dir at the
    same rank and shape) holds about it.  Returns the `aot` block it records."""
    from verity_vllm.correspondence.runtime import correspondence_digest
    from verity_vllm.pipeline.build import write_instances
    res = json.loads(Path(trace, "result.json").read_text())
    traced = res.get("program") or {}
    r = predict(phi, wrapper, M=M, pos0=pos0, LP=LP, T=T)
    if r.digest != traced.get("digest"):
        raise UsageError(f"{trace}: the trace's Program {str(traced.get('digest'))[:16]}… is not the predicted {r.digest[:16]}…: what it "
                         f"holds is about another Program (the row's comparison no longer holds: `verity-vllm predict compare` on a "
                         f"fresh trace, `verity-vllm neutrality --tier heavy`)")
    desc = {**r.desc, "annotations": S.load_descriptor(os.path.join(trace, "descriptor.json.gz")).get("annotations") or {}}
    raw = json.dumps(desc, sort_keys=True).encode()
    if (got := (hashlib.sha256(raw).hexdigest(), correspondence_digest(desc))) != (traced.get("descriptor_sha256"), traced.get("correspondence_digest")):
        raise UsageError(f"{trace}: the predicted descriptor under the trace's annotations is {got[0][:16]}… (correspondence {got[1][:16]}…), "
                         f"the trace's {str(traced.get('descriptor_sha256'))[:16]}… ({str(traced.get('correspondence_digest'))[:16]}…)")
    names, params = traced_names(r.program, desc, trace)
    os.makedirs(out, exist_ok=True)
    with gzip.open(os.path.join(out, "descriptor.json.gz"), "wb") as f:
        f.write(raw)
    write_instances(out, r.program, r.digest, traced.get("correspondence_digest"), {"supplied": s} if (s := traced.get("supplied")) else {},
                    names=names, params=params)
    block = {"mechanism": MECHANISM, "predicted_digest": r.digest, "code_version": S.code_version(), "timings": r.timings, **(aot or {})}
    for name in CARRIED:
        src = os.path.join(trace, name)
        if name in ("result.json", "construction_manifest.json") and os.path.exists(src):
            doc = json.loads(Path(src).read_text())
            Path(out, name).write_text(json.dumps({**doc, "aot": block}, indent=1, default=str))
        elif os.path.exists(src):
            shutil.copyfile(src, os.path.join(out, name))
    return block


def _predict_rank(task: tuple[str, int, str, str, list[tuple[str, int, int]]]) -> dict[str, Any]:
    """One rank's Programs of the traced Build: its step at the trace's (M, pos0), each request (name, LP, T), GP-01's workload over
    them.  A request's Program goes once its component is built: the composition holds the components, never every request Program."""
    row, rank, config, step_dir, requests = task
    out: dict[str, Any] = {"requests": {}, "errors": {}, "secs": {}}

    def timed(part: str, fn):
        t0 = time.time()
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 -- an unexpressible unit or a predictor bug is a compared outcome, named
            out["errors"][part] = f"{type(e).__name__}: {e}"[:800]
        finally:
            out["secs"][part] = round(out["secs"].get(part, 0) + time.time() - t0, 1)

    phi = phi_at(row, rank, config)
    st = json.loads(Path(step_dir, "result.json").read_text())
    out["step"] = getattr(timed("step", lambda: predict(phi, "step", M=int(st["M"]), pos0=int(st["pos0"]))), "digest", None)
    canon, comps = {}, {}
    for name, LP, T in requests:
        if (r := timed(name, lambda: predict(phi, "request", LP=LP, T=T))) is not None:
            out["requests"][name] = r.digest
            if (c := timed("workload", lambda: S.request_component(r, canon))) is not None:
                comps[(LP, T)] = c
    out["workload"] = (timed("workload", lambda: S.predict_workload(phi, comps, canon)) or {}).get("digest")
    return out


def _rank_dir(row_dir: str, rank: int | str) -> str:
    """The dir holding rank `rank`'s derive dirs in a traced row dir: a TP row's `build/rank<r>/`, a single-GPU row's own dir."""
    for d in (os.path.join(row_dir, "build", f"rank{rank}"), os.path.join(row_dir, "build"), row_dir):
        if os.path.isdir(os.path.join(d, "build_step")):
            return d
    raise UsageError(f"{row_dir}: no build_step of rank {rank} (under build/rank{rank}/, build/ or the row dir)")


def _traced_shapes(row_dir: str, rank: int | str, programs: dict) -> dict[str, tuple[int, int]]:
    """Each traced request Program's (LP, T) by its `build_summary.json` name (`request`: the `build_request/` envelope), as its
    derive's `result.json` declares it."""
    d, out = _rank_dir(row_dir, rank), {}
    for n in programs["requests"]:
        res = json.loads(Path(d, "build_request" if n == "request" else n, "result.json").read_text())
        out[n] = (int(res["LP"]), int(res["T"]))
    return out


def compare(row_dir: str, row: str, config: str, workers: int = 0, predicted: str | None = None) -> dict[str, Any]:
    """Every Program the traced Build under `row_dir` holds (`epoch.build_programs`), predicted at its rank and shape, a fresh
    process a rank (`workers` at once; 0 = every rank: a reused one keeps the decode caches of the ranks before), and compared
    digest by digest; with `predicted`, the predicted row's Build there too (`compare_row`)."""
    from verity_vllm.pipeline import epoch
    traced = epoch.build_programs(Path(row_dir))
    if not traced:
        raise UsageError(f"{row_dir}: no build_summary.json, so no traced Build to compare with")
    tasks = [(row, int(r), config, os.path.join(_rank_dir(row_dir, r), "build_step"),
              [(n, *s) for n, s in sorted(_traced_shapes(row_dir, r, p).items())]) for r, p in traced.items()]
    with ProcessPoolExecutor(max_workers=workers or len(tasks), mp_context=multiprocessing.get_context("spawn"), max_tasks_per_child=1) as pool:
        got = dict(zip(traced, pool.map(_predict_rank, tasks)))
    ranks = {}
    for r, p in traced.items():
        parts = {"step": (p["step"], got[r]["step"]), "workload": (p["workload"], got[r]["workload"]),
                 **{n: (d, got[r]["requests"].get(n)) for n, d in p["requests"].items()}}
        ranks[r] = {n: {"traced": t, "predicted": q, "equal": bool(t) and t == q} for n, (t, q) in parts.items()}
        ranks[r].update({k: got[r][k] for k in ("errors", "secs") if got[r][k]})
    equal = all(v["equal"] for p in ranks.values() for n, v in p.items() if n not in ("errors", "secs"))
    doc = {"schema": SCHEMA, "row": row, "build": str(Path(row_dir).resolve()), "code_version": S.code_version(), "equal": equal, "ranks": ranks}
    if predicted is not None:
        doc["predicted"] = compare_row(row_dir, predicted, traced)
        doc["equal"] = equal and doc["predicted"]["equal"]
    return doc


def _manifest_facts(path: Path) -> dict[str, Any]:
    from verity_vllm.records.required_manifest import RequiredManifest
    m = RequiredManifest.load(path)
    return {"manifest_digest": m.manifest_digest, "program_digest": m.program_digest, "identities": len(m.identities)}


def compare_row(row_dir: str, predicted: str, traced: dict | None = None) -> dict[str, Any]:
    """A predicted row's Build (`write_row`, under `predicted`) against the traced Build of the same row under `row_dir`: each
    request Program and the workload by digest, and the required-value manifest by its digest (over its identities), Program
    digest and identity count."""
    from verity_vllm.pipeline import epoch
    traced = traced if traced is not None else epoch.build_programs(Path(row_dir))
    if list(traced) != ["0"]:
        raise UsageError(f"{row_dir}: a predicted row is one rank's, and the traced Build has ranks {sorted(traced)}")
    p, rec = traced["0"], PredictedRow.of(json.loads((path := Path(predicted, FILE)).read_text()), str(path))
    by_shape = {s: p["requests"][n] for n, s in _traced_shapes(row_dir, "0", p).items()}
    parts = {"workload": (p["workload"], rec.workload_digest),
             **{n: (by_shape.get(tuple(int(x) for x in REQUEST.fullmatch(n).groups())), d) for n, d in rec.requests.items()}}
    out: dict[str, Any] = {n: {"traced": t, "predicted": q, "equal": bool(t) and t == q} for n, (t, q) in parts.items()}
    t, q = _manifest_facts(Path(row_dir, "manifest.json")), _manifest_facts(Path(predicted, "manifest.json"))
    out["manifest"] = {"traced": t, "predicted": q, "equal": bool(t["manifest_digest"]) and t == q}
    return {"dir": str(Path(predicted).resolve()), "parts": out, "equal": all(v["equal"] for v in out.values())}


def write_row(row: str, out: str, *, config: str | None = None, jobs: int = 1, log=None) -> PredictedRow:
    """A single-GPU row's Build of record at `out` from its declared inputs alone (module docstring): no trace, no vLLM.  The
    checkpoint's `config.json` is `config`, else the pin's (`pinned_hf_config`).  Returns the record it writes beside them."""
    from verity_vllm.engine.profiles.dense_generic import pinned_hf_config
    from verity_vllm.pipeline import manifest as MN
    from verity_vllm.program import prefix_plan as PP
    from verity_vllm.query.required import QUERY_OF_RECORD
    from .naming import named_program, write_build
    w = json.loads(workload_file(row_file(row)).read_text())
    if PP.prefix_caching(w):
        raise UsageError(f"{row}: the row serves with prefix caching, and its workload Program's cached components are not predicted")
    config = config or pinned_hf_config(w["model"], RowEngine.of(w).revision)
    if config is None:
        raise UsageError(f"{row}: no pinned config.json for {w['model']}@{RowEngine.of(w).revision}; pass --config")
    if os.path.exists(os.path.join(out, "build_summary.json")):
        raise UsageError(f"{out} is a traced row's dir: a predicted row's Build is written into a dir of its own")
    phi = S.phi_of({"row": row, "workload": w}, config=str(config))
    os.makedirs(out, exist_ok=True)
    t0, canon, comps, requests = time.time(), {}, {}, {}
    for LP, T in phi.request_shapes():
        r = named_program(phi, LP, T)
        name = f"build_request_{PP.shape_name(PP.shape(LP, T))}"
        write_build(os.path.join(out, name), r.program, r.named)
        requests[name], comps[(LP, T)] = r.digest, S.request_component(r, canon)
        if log:
            log(f"[predict] {name}: {r.digest[:16]}… ({r.n_root_nodes} nodes)")
    wl = write_workload(phi, comps, canon, os.path.join(out, "build_workload"))
    policy = dict(word="16/32", norm_scales=True, guarded_max=True, taps=MN.taps_of(True, True), query=QUERY_OF_RECORD,
                  jobs=jobs or MN.cpus(), log=log)
    if len(phi.requests) > 1:
        man = MN.build_global(w, out, workload_digest=wl.digest, **policy)
    else:
        man = MN.build(os.path.join(out, next(iter(requests))), workload=w, **policy)
    raw = json.dumps(man, separators=(",", ":")).encode()
    Path(out, "manifest.json").write_bytes(raw)
    if MN.incomplete_rc(man):
        raise UsageError(f"{row}: the predicted Build's manifest is incomplete (named above), so it has no {FILE}")
    rec = PredictedRow(row=row, config=str(config), code_version=S.code_version(), requests=requests, workload_digest=wl.digest,
                       lag=wl.lag, manifest_digest=man["manifest_digest"], manifest_sha256=hashlib.sha256(raw).hexdigest(),
                       program_digest=man["program_digest"], identities=man["populations"]["identities"],
                       secs=round(time.time() - t0, 1))
    Path(out, FILE).write_text(json.dumps(rec.dump(), indent=1) + "\n")
    return rec


def write_workload(phi: Phi, comps: dict, canon: dict, out: str):
    """GP-01's workload Program over the predicted request components (`score.workload_program`) at `out`, as GP-01 writes its
    own: `workload_program.json`, the Program's artifact (`WorkloadProgram.artifact`) with the `lag` block and
    `applicability.engine.async_scheduling` the declared target resolves, and `descriptor.json.gz`, from which GM-01's
    `check.match.declaration.from_gprog` recomputes the workload digest and the `lag` static the block claims."""
    from verity.ml.circuits.codec import canonical_json, descriptor_size
    wl = S.workload_program(phi, comps, canon)
    lag = phi.target_profile().async_scheduling_declaration()
    desc = wl.descriptor
    os.makedirs(out, exist_ok=True)
    with gzip.open(os.path.join(out, "descriptor.json.gz"), "wb") as f:
        f.write(cj := canonical_json(desc))
    Path(out, "workload_program.json").write_text(json.dumps(
        {**wl.artifact(), "ok": True, "function": wl.fn.id[:300], "lag": {**lag, "value": wl.lag}, "built_by": ROW_BUILT_BY,
         "applicability": {"engine": {"async_scheduling": bool(lag["async_scheduling"])}},
         "descriptor": {"bytes": len(cj) if "annotations" not in desc else descriptor_size(desc), "sha256": hashlib.sha256(cj).hexdigest(),
                        "n_definitions": len(desc["definitions"])}}, indent=1) + "\n")
    return wl


@dataclass(frozen=True, kw_only=True)
class EmitOptions:
    """One derive dir of a switched row's Build: the predicted Program with the data the trace at the same rank and shape holds."""
    row: str = option("--row")
    rank: int = option("--rank", type=int, default=0)
    config: str = option("--config", help="the pinned checkpoint's config.json")
    trace: str = option("--trace", help="the traced derive dir at the same rank and shape")
    out: str = option("--out")
    wrapper: str = option("--wrapper", choices=["step", "request"])
    M: int = option("--M", type=int, default=1)
    pos0: int = option("--pos0", type=int, default=8)
    LP: int | None = option("--LP", type=int, default=None)
    T: int | None = option("--T", type=int, default=None)
    aot: str = option("--aot", default="{}", help="the row's `aot` index entry, recorded in the dir")
    cmd: str = field(default="emit", init=False)


@dataclass(frozen=True, kw_only=True)
class CompareOptions:
    """Predict every Program of a row's traced Build and compare digests; exit 0 iff every rank's every Program is equal."""
    row_dir: str = option("row_dir", help="the traced row dir (build_summary.json, build/rank<r>/build_*)")
    row: str = option("--row")
    config: str = option("--config", help="the pinned checkpoint's config.json")
    out: str = option("--out", help="comparison.json")
    workers: int = option("--workers", type=int, default=0, help="ranks predicted at once (0: every rank)")
    predicted: str | None = option("--predicted", default=None, help="a predicted row's Build (`predict row`) to hold to the trace too")
    cmd: str = field(default="compare", init=False)


@dataclass(frozen=True, kw_only=True)
class RowOptions:
    """A single-GPU row's Build of record from its declared inputs alone: each request Program named, GP-01's workload, the manifest."""
    row: str = option("--row")
    out: str = option("--out", help="the Build's dir (build_request_<shape>/, build_workload/, manifest.json, predicted.json)")
    config: str | None = option("--config", default=None, help="the checkpoint's config.json (default: the pin's, pinned_hf_config)")
    jobs: int = option("--jobs", type=int, default=0, help="processes for the manifest's word check and components (0: the CPUs)")
    cmd: str = field(default="row", init=False)


def main(a: EmitOptions | CompareOptions | RowOptions) -> int:
    if a.cmd == "emit":
        if a.wrapper == "request" and (a.LP is None or a.T is None):
            raise UsageError("--wrapper request needs --LP and --T")
        block = emit(phi_at(a.row, a.rank, a.config), a.wrapper, a.trace, a.out, M=a.M, pos0=a.pos0, LP=a.LP, T=a.T, aot=json.loads(a.aot))
        print(json.dumps({"out": a.out, "digest": block["predicted_digest"], "timings": block["timings"]}))
        return 0
    if a.cmd == "row":
        rec = write_row(a.row, a.out, config=a.config, jobs=a.jobs, log=lambda s: print(s, file=sys.stderr, flush=True))
        print(json.dumps({k: getattr(rec, k) for k in ("row", "workload_digest", "manifest_digest", "program_digest", "identities", "secs")}))
        return 0
    doc = compare(a.row_dir, a.row, a.config, a.workers, a.predicted)
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    for r, p in doc["ranks"].items():
        print(f"rank {r}: " + "  ".join(f"{n} {'=' if v['equal'] else '≠'}" for n, v in p.items() if n not in ("errors", "secs")), flush=True)
    if "predicted" in doc:
        print(f"predicted {doc['predicted']['dir']}: " + "  ".join(f"{n} {'=' if v['equal'] else '≠'}" for n, v in doc["predicted"]["parts"].items()),
              flush=True)
    print(f"[predict] {doc['row']}: {'EQUAL on every rank' if doc['equal'] else 'NOT EQUAL'} -> {a.out}", flush=True)
    return 0 if doc["equal"] else 1
