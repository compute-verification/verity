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
import sys
import time
from pathlib import Path

from . import checks as CK
from . import targets as TG


def _options(a: argparse.Namespace) -> CK.Options:
    return CK.Options(vectors=a.vectors, seed=a.seed, X=a.X, W=a.W, extras=a.extras)


def run(specs: list[str], opts: CK.Options, *, everything: bool = False, as_call: bool | None = None, log=print) -> dict:
    TG.load_registries()
    memo: dict = {}
    pins = CK.load_pins()
    out = []

    def one(kind: str, label: str, fn) -> None:
        t0 = time.time()
        try:
            r = fn()
        except Exception as e:  # noqa: BLE001  (a target the checks cannot run on is a failure by name, not a crash)
            r = {"failures": [{"check": "circuit-check", "code": "error", "definition": label,
                               "name": f"circuit-check/error {label}: {type(e).__name__}: {str(e)[:200]}"}], "warnings": []}
        r = {**r, "kind": kind, "target": label, "seconds": round(time.time() - t0, 3)}
        out.append(r)
        if log:
            log(f"{'FAIL' if r['failures'] else 'ok  '} {kind:10s} {label[:90]}  failures={len(r['failures'])} warnings={len(r['warnings'])}"
                f" ({r['seconds']:.1f}s)")

    if everything:
        for fid, fn in TG.suite().items():
            one("definition", fid, lambda fn=fn: CK.check_definition(fn, opts, memo=memo, pins=pins))
        for label, sub, sts in TG.templates():
            one("template", label, lambda sub=sub, sts=sts: CK.check_subcircuit(sub, opts, instance_statics=sts, memo=memo, pins=pins))
    for s in specs:
        kind = TG.kind_of(s)
        if kind == "row":
            one("row", s, lambda s=s: CK.check_row(s, opts, memo=memo, pins=pins))
        elif kind == "template":
            def tpl(s=s):
                sub, sts = TG.subcircuit(s)
                return CK.check_subcircuit(sub, opts, instance_statics=sts, memo=memo, pins=pins)
            one("template", s, tpl)
        else:
            one("definition", s, lambda s=s: CK.check_definition(TG.definition(s), opts, memo=memo, pins=pins, as_call=as_call))
    return CK.report(out, opts)


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
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)
    if not a.targets and not a.all:
        ap.error("give a target or --all")
    rep = run(a.targets, _options(a), everything=a.all, as_call=True if a.as_call else None,
              log=None if a.quiet else lambda s: print(s, flush=True))
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
