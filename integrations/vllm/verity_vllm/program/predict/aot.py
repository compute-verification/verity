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

    verity-vllm predict emit --row ROW --rank R --config CONFIG_JSON --trace DIR --out DIR (--wrapper step --M N --pos0 N |
                             --wrapper request --LP N --T N) [--aot JSON]
    verity-vllm predict compare ROW_DIR --row ROW --config CONFIG_JSON --out comparison.json [--workers N]
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
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity_vllm.config import WORKLOADS, UsageError, option, workload_file

from . import score as S
from .build import build_program
from .phi import Phi

SCHEMA = "verity-vllm/aot-comparison/v1"
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


def _predict_rank(task: tuple[str, int, str, str, list[str]]) -> dict[str, Any]:
    """One rank's Programs of the traced Build: its step at the trace's (M, pos0), each request shape, GP-01's workload over them.
    A request's Program goes once its component is built: the composition holds the components, never every request Program."""
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
    for name in requests:
        LP, T = (int(x) for x in REQUEST.fullmatch(name).groups())
        if (r := timed(name, lambda: predict(phi, "request", LP=LP, T=T))) is not None:
            out["requests"][name] = r.digest
            if (c := timed("workload", lambda: S.request_component(r, canon))) is not None:
                comps[(LP, T)] = c
    out["workload"] = (timed("workload", lambda: S.predict_workload(phi, comps, canon)) or {}).get("digest")
    return out


def compare(row_dir: str, row: str, config: str, workers: int = 0) -> dict[str, Any]:
    """Every Program the traced Build under `row_dir` holds (`epoch.build_programs`), predicted at its rank and shape, a fresh
    process a rank (`workers` at once; 0 = every rank: a reused one keeps the decode caches of the ranks before), and compared
    digest by digest."""
    from verity_vllm.pipeline import epoch
    traced = epoch.build_programs(Path(row_dir))
    if not traced:
        raise UsageError(f"{row_dir}: no build_summary.json, so no traced Build to compare with")
    tp = len(traced) > 1 or os.path.isdir(os.path.join(row_dir, "build", "rank0"))
    step_dir = lambda r: os.path.join(row_dir, "build", *([f"rank{r}"] if tp else []), "build_step")   # noqa: E731
    tasks = [(row, int(r), config, step_dir(r), sorted(p["requests"])) for r, p in traced.items()]
    with ProcessPoolExecutor(max_workers=workers or len(tasks), mp_context=multiprocessing.get_context("spawn"), max_tasks_per_child=1) as pool:
        got = dict(zip(traced, pool.map(_predict_rank, tasks)))
    ranks = {}
    for r, p in traced.items():
        parts = {"step": (p["step"], got[r]["step"]), "workload": (p["workload"], got[r]["workload"]),
                 **{n: (d, got[r]["requests"].get(n)) for n, d in p["requests"].items()}}
        ranks[r] = {n: {"traced": t, "predicted": q, "equal": bool(t) and t == q} for n, (t, q) in parts.items()}
        ranks[r].update({k: got[r][k] for k in ("errors", "secs") if got[r][k]})
    equal = all(v["equal"] for p in ranks.values() for n, v in p.items() if n not in ("errors", "secs"))
    return {"schema": SCHEMA, "row": row, "build": str(Path(row_dir).resolve()), "code_version": S.code_version(), "equal": equal, "ranks": ranks}


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
    cmd: str = field(default="compare", init=False)


def main(a: EmitOptions | CompareOptions) -> int:
    if a.cmd == "emit":
        if a.wrapper == "request" and (a.LP is None or a.T is None):
            raise UsageError("--wrapper request needs --LP and --T")
        block = emit(phi_at(a.row, a.rank, a.config), a.wrapper, a.trace, a.out, M=a.M, pos0=a.pos0, LP=a.LP, T=a.T, aot=json.loads(a.aot))
        print(json.dumps({"out": a.out, "digest": block["predicted_digest"], "timings": block["timings"]}))
        return 0
    doc = compare(a.row_dir, a.row, a.config, a.workers)
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    for r, p in doc["ranks"].items():
        print(f"rank {r}: " + "  ".join(f"{n} {'=' if v['equal'] else '≠'}" for n, v in p.items() if n not in ("errors", "secs")), flush=True)
    print(f"[predict] {doc['row']}: {'EQUAL on every rank' if doc['equal'] else 'NOT EQUAL'} -> {a.out}", flush=True)
    return 0 if doc["equal"] else 1
