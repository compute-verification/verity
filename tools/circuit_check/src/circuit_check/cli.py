"""``circuit-check TARGET... [--out report.json]``: the standard property checks on each target, one machine-readable report.

    circuit-check F32Add_v1 'RMSNormTriton_v1{N=64,EPS=1e-05}'     Definitions
    circuit-check rope-head gemm-coordinate/k64/sm80-mma-bf16       templates, subcircuits
    circuit-check path/to/build_request_LP3_T2                      a served row's request Program
    circuit-check --all                                             every registered Definition and template (the suite's set)

Exit status 1 when a check fails that `circuit_check.known` does not list (failures are named in the summary; ``--strict`` fails
on the known ones too), and with ``--all`` when a known entry no longer fails; 0 otherwise.  Redundant and dead gates are
warnings with counts, printed and in the report (``--fail-on-warnings`` makes them fail too).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from . import cache as CC
from . import checks as CK
from . import sidecars as SC
from . import targets as TG

REPO = Path(__file__).resolve().parents[4]


def _options(a: argparse.Namespace) -> CK.Options:
    return CK.Options(vectors=a.vectors, seed=a.seed, X=a.X, W=a.W, extras=a.extras)


_TASKS: list[tuple] = []


def _info(task: tuple) -> dict:
    """A task's ``(kind, label, check[, info])``: info has ``as_call`` (in its key), ``units`` (its bindings, `targets.BINDINGS`)
    and ``roots`` (the objects it checks, built before its check)."""
    return task[3] if len(task) > 3 else {}


def _one(kind: str, label: str, fn: Callable[[], dict]) -> dict:
    t0 = time.time()
    try:
        r = fn()
    except Exception as e:  # noqa: BLE001  (a target the checks cannot run on is a failure by name, not a crash)
        r = {"failures": [{"check": "circuit-check", "code": "error", "definition": label,
                           "name": f"circuit-check/error {label}: {type(e).__name__}: {str(e)[:200]}"}], "warnings": []}
    return {**r, "kind": kind, "target": label, "seconds": round(time.time() - t0, 3)}


def _errored(r: dict) -> bool:
    """A check that raised: no verdict on the target, whatever broke (an import, the machine)."""
    return any(f.get("code") == "error" for f in r.get("failures", []))


def _checked(task: tuple) -> tuple[dict, dict | None]:
    """A task's result, and with tracing what its check read or resolved (`cache.attributed`), recorded from its start."""
    kind, label, fn = task[:3]
    if not CC.tracing():
        return _one(kind, label, fn), None
    info = _info(task)
    with CC.recording() as fr:
        r = _one(kind, label, fn)
    return r, CC.attributed(fr, info.get("units", ()), info.get("roots", ()))


def _observe(r: dict, got: dict | None) -> tuple[dict | None, list[str], dict | None]:
    """What a result read, what the tracer missed, and the reads as this process sees them right after the check."""
    untraced = CC.untraced()
    return got, untraced, (CC.seen(REPO, got) if got is not None and not untraced and not _errored(r) else None)


def _work(i: int) -> tuple[int, dict, dict | None, list[str], dict | None]:
    r, got = _checked(_TASKS[i])
    return i, r, *_observe(r, got)


def target_key(kind: str, label: str, opts: CK.Options, as_call: bool | None, roots: tuple = ()) -> dict:
    """A target's key: its request, and for a Definition the digest of what it reaches as descriptors (`cache.descriptor`)."""
    request = {"target": [kind, label], "as_call": as_call, "options": {k: getattr(opts, k) for k in opts.__dataclass_fields__}}
    if roots:
        request["descriptor"] = CC.descriptor(roots[0])
    return CC.key(request)


def _tasks(specs: list[str], opts: CK.Options, everything: bool, as_call: bool | None) -> list[tuple]:
    """The targets' checks in target order (deterministic: a worker rebuilds the same list and runs entries by index).  Traced
    (`cache.tracing`), each check gets a memo of its own, so its result never rests on what an earlier target's check computed,
    and the pins it looks up are recorded (`cache.recorded_pins`); a suite Definition's ``as_call`` is resolved here (the catalog's
    `targets.call_families`), so it is in the target's key rather than a read of its check."""
    from verity.primitives.circuits.evaluation.reference import resolve

    TG.load_registries()
    fresh = CC.tracing()
    shared: dict = {}
    memo = (lambda: {}) if fresh else (lambda: shared)
    pins = CC.recorded_pins(CK.load_pins()) if fresh else CK.load_pins()
    tasks: list[tuple] = []
    if everything:
        for fid, fn in TG.suite().items():
            call = TG.family(resolve(fn)) in TG.call_families()
            tasks.append(("definition", fid, lambda fn=fn, call=call: CK.check_definition(fn, opts, memo=memo(), pins=pins, as_call=call),
                          {"as_call": call, "units": sorted(TG.BINDINGS.get(fid, ())), "roots": [fn]}))
        for label, sub, sts in TG.templates():
            tasks.append(("template", label, lambda sub=sub, sts=sts: CK.check_subcircuit(sub, opts, instance_statics=sts, memo=memo(), pins=pins),
                          {"units": sorted(TG.BINDINGS.get(label, ()))}))
    for s in specs:
        kind = TG.kind_of(s)
        if kind == "row":
            tasks.append(("row", s, lambda s=s: CK.check_row(s, opts, memo=memo(), pins=pins)))
        elif kind == "template":
            def tpl(s=s):
                sub, sts = TG.subcircuit(s)
                return CK.check_subcircuit(sub, opts, instance_statics=sts, memo=memo(), pins=pins)
            tasks.append(("template", s, tpl))
        else:
            tasks.append(("definition", s, lambda s=s: CK.check_definition(TG.definition(s), opts, memo=memo(), pins=pins, as_call=as_call)))
    return tasks


def _init(specs: list[str], opts: CK.Options, everything: bool, as_call: bool | None, tracing: bool) -> None:
    if tracing:
        CC.trace(REPO)
    _TASKS[:] = _tasks(specs, opts, everything, as_call)


def parallel(fn: Callable, items: list, jobs: int, initializer: Callable | None = None, initargs: tuple = ()):
    """``fn`` of each item in up to ``jobs`` forkserver workers, yielded as each finishes. A worker that dies (the OOM killer)
    raises ``BrokenProcessPool``: ``multiprocessing.Pool`` would wait for its task forever."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(min(jobs, len(items)), mp_context=mp.get_context("forkserver"), initializer=initializer,
                             initargs=initargs) as pool:
        for f in as_completed([pool.submit(fn, x) for x in items]):
            yield f.result()


def run(specs: list[str], opts: CK.Options, *, everything: bool = False, as_call: bool | None = None, log=print, jobs: int = 1,
        cache: Path | None = None) -> dict:
    """Every target's checks, in target order; ``jobs`` > 1 runs them in that many worker processes, each started from a fresh
    forkserver (never a fork of this possibly multi-threaded process: under pytest-xdist that deadlocked) and rebuilding the
    task list with its own memo, so a worker may repeat a Definition another checked: the results are the same.

    With ``cache`` each target's result is reused while everything its check read or resolved is unchanged
    (`circuit_check.cache`): what was read and run from the start of its check, the builds of the Definitions it reached, its
    own bindings and pins, and those modules' import closures; not what the task list's build or another target read.  A result
    is not stored once its process did something the tracer cannot see, nor when the check raised (no verdict), nor when the
    reads as that process sees them right after its check differ from what would be recorded (the tree changed during the run)."""
    tracing = cache is not None and CC.tracing()
    tasks = _tasks(specs, opts, everything, as_call)
    out: list[dict | None] = [None] * len(tasks)
    keys = [target_key(t[0], t[1], opts, _info(t).get("as_call", as_call), tuple(_info(t).get("roots", ()))) for t in tasks] \
        if tracing else []
    stats = {"hits": 0, "stored": 0, "untraced": [], "errors": [], "changed": []}

    def done(i: int, r: dict, got: dict | None = None, untraced: list[str] | None = None, worker: dict | None = None) -> None:
        out[i] = r
        if tracing and got is not None:
            if _errored(r):
                stats["errors"].append(r["target"])
            elif untraced:
                stats["untraced"].append({"target": r["target"], "because": untraced[:3]})
            elif CC.store(cache, keys[i], REPO, got, r, worker) is None:
                stats["changed"].append(r["target"])
            else:
                stats["stored"] += 1
        if log:
            across = (r.get("partition") or {}).get("recomputed_across")
            qc = (r.get("partition") or {}).get("q_call") or {}
            call = (f" q_call={qc['units']}u/{qc['committed_bits']}b" if "units" in qc else
                    f" q_call=refused:{qc['inapplicable'].get('why')}" if "inapplicable" in qc else "")
            log(f"{'FAIL' if r['failures'] else 'ok  '} {r['kind']:10s} {r['target'][:90]}  failures={len(r['failures'])} "
                f"warnings={len(r['warnings'])}" + (f" recomputed_across={across}" if across else "") + call + f" ({r['seconds']:.1f}s)")

    todo = list(range(len(tasks)))
    if tracing:
        todo = []
        for i, k in enumerate(keys):
            if (hit := CC.lookup(cache, k, REPO)) is not None and not _errored(hit[0]):
                out[i] = {**hit[0], "cache": {"hit": hit[1]}}
                stats["hits"] += 1
            else:
                todo.append(i)
    if jobs > 1 and len(todo) > 1:
        init = (specs, opts, everything, as_call, tracing)
        for i, r, got, untraced, worker in parallel(_work, todo, jobs, _init, init):
            done(i, r, got, untraced, worker)
    else:
        for i in todo:
            r, got = _checked(tasks[i]) if tracing else (_one(*tasks[i][:3]), None)
            done(i, r, *(_observe(r, got) if tracing else (None, None, None)))
    rep = CK.report([r for r in out if r is not None], opts)
    if tracing:
        rep["cache"] = stats
    return rep


def update_pins(rep: dict) -> dict:
    """Every pin, with each unpinned or changed count the report saw set to the count measured now, in the file that holds the pin
    or, for a new one, the sidecar of the module that defines its Definition (`sidecars.write_pins`)."""
    updates = {}
    for t in rep["targets"]:
        for f in t.get("failures", []) + t.get("warnings", []):
            if f["check"] == "lowering" and f["code"] in ("unpinned", "pin-mismatch"):
                updates[f["pins"], f["pin"]] = f["counts"]
    CK.load_pins()
    SC.write_pins(updates)
    return CK.load_pins()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="circuit-check", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="*", help="Definition ids, template or subcircuit ids, request Program paths")
    ap.add_argument("--all", action="store_true", help="every registered Definition and template at the catalog's bindings")
    ap.add_argument("--out", type=Path, help="write the JSON report here")
    ap.add_argument("-j", "--jobs", type=int, default=1, help="targets checked in parallel by this many workers (0: every core)")
    ap.add_argument("--vectors", type=int, default=CK.Options.vectors)
    ap.add_argument("--seed", type=int, default=CK.Options.seed)
    ap.add_argument("-X", type=int, default=CK.Options.X)
    ap.add_argument("-W", type=int, default=CK.Options.W)
    ap.add_argument("--extras", type=int, default=CK.Options.extras)
    ap.add_argument("--as-call", action="store_true", help="hold a Definition target to the partition invariant as a Call "
                    "(default: when its family is a Call of a served Program)")
    ap.add_argument("--strict", action="store_true", help="fail on the known failures too (circuit_check.known)")
    ap.add_argument("--fail-on-warnings", action="store_true")
    ap.add_argument("--update-pins", action="store_true", help="pin the lowering counts measured now (review the diff)")
    ap.add_argument("--cache", type=Path, default=None, metavar="DIR",
                    help="reuse a report whose every input is unchanged (circuit_check.cache), else run and record one")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)
    if a.cache and not a.update_pins:
        # the registries build their C++ twins on first use (g++, untraceable): once, in a process of its own, before tracing
        prebuild = ("from circuit_check import targets; targets.load_registries()\n"
                    "try:\n    from verity_vllm.program.kernels import twins\nexcept ImportError:\n    twins = None\n"
                    "if twins:\n    twins.preload()        # the three C++ models, which checks otherwise compile on first use\n")
        subprocess.run([sys.executable, "-c", prebuild], cwd=REPO, capture_output=True)
        CC.trace(REPO)
    if not a.targets and not a.all:
        ap.error("give a target or --all")
    jobs = a.jobs or len(os.sched_getaffinity(0))
    opts = _options(a)
    rep = run(a.targets, opts, everything=a.all, as_call=True if a.as_call else None,
              log=None if a.quiet else lambda s: print(s, flush=True), jobs=jobs, cache=a.cache if a.cache and not a.update_pins else None)
    if c := rep.get("cache"):
        print(f"circuit-check: {c['hits']} of {len(rep['targets'])} targets reused (every file, listing, probe, catalog unit and pin their checks read or resolved is "
              f"unchanged), {c['stored']} stored, {len(c['untraced'])} not cacheable (the tracer could not see what they read), "
              f"{len(c['errors'])} not stored because the check raised, {len(c['changed'])} because the tree changed while they ran")
    if a.update_pins:
        update_pins(rep)
    s = rep["summary"]
    stale = []
    if a.all:
        from .known import stale as stale_known

        stale = stale_known(f for t in rep["targets"] if t["kind"] == "definition" for f in t.get("failures", []))
        s["stale_known"] = stale
    for name in s["new_failures"]:
        print(f"FAIL {name}")
    for name in s["known_failures"]:
        print(f"known {name}")
    for k in stale:
        print(f"FAIL circuit_check.known lists {k}, which no longer fails: drop it")
    for k, w in s["warnings"].items():
        print(f"warn {k}: {w['total']} over {w['definitions']} target(s)")
    if a.out:
        a.out.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    across = s.get("recomputed_across") or {}
    refused = s.get("q_call_refused") or {}
    print(f"circuit-check: {len(rep['targets'])} target(s), {len(s['new_failures'])} new failure(s), {len(s['known_failures'])} known"
          + (f", {len(across)} Boolean Call(s) recomputing {sum(across.values())} gate(s) across units (Q_word v2)" if across else "")
          + (f", {len(refused)} Definition(s) Q_call v1 refuses" if refused else "")
          + (f", report {a.out}" if a.out else ""))
    failed = bool(s["new_failures"] or stale or (a.strict and s["known_failures"]) or (a.fail_on_warnings and s["warnings"]))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
