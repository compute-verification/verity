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


def _work(i: int) -> tuple[int, dict, dict]:
    kind, label, fn = _TASKS[i]
    return i, _one(kind, label, fn), CC.reads()


def run(specs: list[str], opts: CK.Options, *, everything: bool = False, as_call: bool | None = None, log=print, jobs: int = 1,
        reads: dict | None = None) -> dict:
    """Every target's checks, in target order; ``jobs`` > 1 runs them in that many forked workers (each with its own memo, so a
    worker may repeat a Definition another checked: the results are the same).  ``reads`` collects what the workers read."""
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
    out: list[dict | None] = [None] * len(tasks)

    def done(i: int, r: dict) -> None:
        out[i] = r
        if log:
            log(f"{'FAIL' if r['failures'] else 'ok  '} {r['kind']:10s} {r['target'][:90]}  failures={len(r['failures'])} "
                f"warnings={len(r['warnings'])} ({r['seconds']:.1f}s)")

    if jobs > 1 and len(tasks) > 1:
        import multiprocessing as mp

        _TASKS[:] = tasks
        with mp.get_context("fork").Pool(min(jobs, len(tasks))) as pool:
            for i, r, got in pool.imap_unordered(_work, range(len(tasks))):
                done(i, r)
                if reads is not None:
                    for k2, v in got.items():
                        reads.setdefault(k2, set()).update(v)
    else:
        for i, (kind, label, fn) in enumerate(tasks):
            done(i, _one(kind, label, fn))
    return CK.report([r for r in out if r is not None], opts)


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
        CC.trace(REPO)
    if not a.targets and not a.all:
        ap.error("give a target or --all")
    jobs = a.jobs or len(os.sched_getaffinity(0))
    opts = _options(a)
    hit = None
    if a.cache and not a.update_pins:
        key = CC.key({"targets": a.targets, "all": a.all, "as_call": a.as_call,
                      "options": {k: getattr(opts, k) for k in opts.__dataclass_fields__}})
        hit = CC.lookup(a.cache, key, REPO)
    if hit is not None:
        rep, eid = hit
        rep["cache"] = {"hit": eid}
        print(f"circuit-check: cache hit {eid[:16]}: every file, listing and probe the recorded run read is unchanged")
    else:
        got: dict = {}
        rep = run(a.targets, opts, everything=a.all, as_call=True if a.as_call else None,
                  log=None if a.quiet else lambda s: print(s, flush=True), jobs=jobs, reads=got)
        if a.cache and not a.update_pins:
            for k2, v in CC.reads().items():
                got.setdefault(k2, set()).update(v)
            eid = CC.store(a.cache, key, REPO, {k2: sorted(v) for k2, v in got.items()}, rep)
            rep = {**rep, "cache": {"stored": eid}}
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
