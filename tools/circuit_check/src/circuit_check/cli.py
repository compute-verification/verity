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
from . import targets as TG

REPO = Path(__file__).resolve().parents[4]


def _options(a: argparse.Namespace) -> CK.Options:
    return CK.Options(vectors=a.vectors, seed=a.seed, X=a.X, W=a.W, extras=a.extras)


_TASKS: list[tuple[str, str, Callable[[], dict]]] = []


def _one(kind: str, label: str, fn: Callable[[], dict]) -> dict:
    t0 = time.time()
    try:
        r = fn()
    except Exception as e:  # noqa: BLE001  (a target the checks cannot run on is a failure by name, not a crash)
        r = {"failures": [{"check": "circuit-check", "code": "error", "definition": label,
                           "name": f"circuit-check/error {label}: {type(e).__name__}: {str(e)[:200]}"}], "warnings": []}
    return {**r, "kind": kind, "target": label, "seconds": round(time.time() - t0, 3)}


def _work(i: int) -> tuple[int, dict, dict, list[str]]:
    kind, label, fn = _TASKS[i]
    r = _one(kind, label, fn)
    return i, r, CC.reads(), CC.untraced()


def target_key(kind: str, label: str, opts: CK.Options, as_call: bool | None) -> dict:
    return CC.key({"target": [kind, label], "as_call": as_call, "options": {k: getattr(opts, k) for k in opts.__dataclass_fields__}})


def _tasks(specs: list[str], opts: CK.Options, everything: bool, as_call: bool | None) -> list[tuple[str, str, Callable[[], dict]]]:
    """The targets' checks in target order (deterministic: a worker rebuilds the same list and runs entries by index)."""
    TG.load_registries()
    memo: dict = {}
    pins = CK.load_pins()
    tasks: list[tuple[str, str, Callable[[], dict]]] = []
    if everything:
        for fid, fn in TG.suite().items():
            tasks.append(("definition", fid, lambda fn=fn: CK.check_definition(fn, opts, memo=memo, pins=pins)))
        for label, sub, sts in TG.templates():
            tasks.append(("template", label, lambda sub=sub, sts=sts: CK.check_subcircuit(sub, opts, instance_statics=sts, memo=memo, pins=pins)))
    for s in specs:
        kind = TG.kind_of(s)
        if kind == "row":
            tasks.append(("row", s, lambda s=s: CK.check_row(s, opts, memo=memo, pins=pins)))
        elif kind == "template":
            def tpl(s=s):
                sub, sts = TG.subcircuit(s)
                return CK.check_subcircuit(sub, opts, instance_statics=sts, memo=memo, pins=pins)
            tasks.append(("template", s, tpl))
        else:
            tasks.append(("definition", s, lambda s=s: CK.check_definition(TG.definition(s), opts, memo=memo, pins=pins, as_call=as_call)))
    return tasks


def _init(specs: list[str], opts: CK.Options, everything: bool, as_call: bool | None, tracing: bool) -> None:
    if tracing:
        CC.trace(REPO)
    _TASKS[:] = _tasks(specs, opts, everything, as_call)


def run(specs: list[str], opts: CK.Options, *, everything: bool = False, as_call: bool | None = None, log=print, jobs: int = 1,
        cache: Path | None = None) -> dict:
    """Every target's checks, in target order; ``jobs`` > 1 runs them in that many worker processes, each started from a fresh
    forkserver (never a fork of this possibly multi-threaded process: under pytest-xdist that deadlocked) and rebuilding the
    task list with its own memo, so a worker may repeat a Definition another checked: the results are the same.

    With ``cache`` each target's result is reused while everything its check read is unchanged (`circuit_check.cache`). A
    result is stored with every read its process made up to it (the task list, the modules it imported, the earlier targets
    whose memoized results it may use), and not at all once that process did something the tracer cannot see."""
    tracing = cache is not None and CC.tracing()
    tasks = _tasks(specs, opts, everything, as_call)
    out: list[dict | None] = [None] * len(tasks)
    keys = [target_key(kind, label, opts, as_call) for kind, label, _ in tasks] if tracing else []
    stats = {"hits": 0, "stored": 0, "untraced": []}

    def done(i: int, r: dict, got: dict | None = None, untraced: list[str] | None = None) -> None:
        out[i] = r
        if tracing and got is not None:
            if untraced:
                stats["untraced"].append({"target": r["target"], "because": untraced[:3]})
            else:
                CC.store(cache, keys[i], REPO, got, r)
                stats["stored"] += 1
        if log:
            log(f"{'FAIL' if r['failures'] else 'ok  '} {r['kind']:10s} {r['target'][:90]}  failures={len(r['failures'])} "
                f"warnings={len(r['warnings'])} ({r['seconds']:.1f}s)")

    todo = list(range(len(tasks)))
    if tracing:
        todo = []
        for i, k in enumerate(keys):
            if (hit := CC.lookup(cache, k, REPO)) is not None:
                out[i] = {**hit[0], "cache": {"hit": hit[1]}}
                stats["hits"] += 1
            else:
                todo.append(i)
    if jobs > 1 and len(todo) > 1:
        import multiprocessing as mp

        init = (specs, opts, everything, as_call, tracing)
        with mp.get_context("forkserver").Pool(min(jobs, len(todo)), initializer=_init, initargs=init) as pool:
            for i, r, got, untraced in pool.imap_unordered(_work, todo, chunksize=1):
                done(i, r, got, untraced)
    else:
        for i in todo:
            kind, label, fn = tasks[i]
            r = _one(kind, label, fn)
            done(i, r, CC.reads() if tracing else None, CC.untraced() if tracing else None)
    rep = CK.report([r for r in out if r is not None], opts)
    if tracing:
        rep["cache"] = stats
    return rep


def update_pins(rep: dict) -> dict:
    """The pins file with every unpinned or changed count the report saw set to the count measured now."""
    pins = CK.load_pins()
    for t in rep["targets"]:
        for f in t.get("failures", []) + t.get("warnings", []):
            if f["check"] == "lowering" and f["code"] in ("unpinned", "pin-mismatch"):
                pins.setdefault(f["pins"], {})[f["pin"]] = f["counts"]
    for kind in pins:
        pins[kind] = dict(sorted(pins[kind].items()))
    CK.PINS.write_text(json.dumps(pins, indent=1, sort_keys=True) + "\n")
    return pins


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
        subprocess.run([sys.executable, "-c", "from circuit_check import targets; targets.load_registries()"], cwd=REPO,
                       capture_output=True)
        CC.trace(REPO)
    if not a.targets and not a.all:
        ap.error("give a target or --all")
    jobs = a.jobs or len(os.sched_getaffinity(0))
    opts = _options(a)
    rep = run(a.targets, opts, everything=a.all, as_call=True if a.as_call else None,
              log=None if a.quiet else lambda s: print(s, flush=True), jobs=jobs, cache=a.cache if a.cache and not a.update_pins else None)
    if c := rep.get("cache"):
        print(f"circuit-check: {c['hits']} of {len(rep['targets'])} targets reused (every file, listing and probe their checks read is "
              f"unchanged), {c['stored']} stored, {len(c['untraced'])} not cacheable (the tracer could not see what they read)")
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
    print(f"circuit-check: {len(rep['targets'])} target(s), {len(s['new_failures'])} new failure(s), {len(s['known_failures'])} known"
          + (f", report {a.out}" if a.out else ""))
    failed = bool(s["new_failures"] or stale or (a.strict and s["known_failures"]) or (a.fail_on_warnings and s["warnings"]))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
