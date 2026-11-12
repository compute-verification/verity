"""Where the served position merge's catalog build and check spend their seconds, for CIR-255 Part C.

python cir255_prof.py v1|v2 [traced]

v1: the catalog binding `dsa_topk_reads.circuit-check#_model_roots:DsaPositionsMerge_v1-served`; v2: the binding
`dsa_topk_reads.circuit-check#_merge_calls_roots` (which holds `DsaPositionsMerge_v2{M=8,K=2048}`).

1. apart: the binding built in a forked child as circuit-check's records run builds it (`units.built_apart`, described=True):
   its CPU seconds (the figure a record keeps), wall seconds and peak.  With ``traced``, under `cache.trace` (the records run
   traces its reads), and nothing else is done.
2. build: the same binding built in this process under cProfile: the body (the trace), `_reach` without descriptors, and the
   descriptors (`_reach` described).
3. check: `checks.check_definition` of the served target under cProfile.
Writes $OUT/<which>[-traced].json and, for 2 and 3, <which>-build.prof / <which>-check.prof with their top functions as text.
"""
from __future__ import annotations

import cProfile
import io
import json
import os
import pstats
import resource
import sys
import time
from pathlib import Path

from circuit_check import cache as CC
from circuit_check import checks as CK
from circuit_check import cli
from circuit_check import targets as TG
from circuit_check import units as UN

OUT = Path(os.environ["OUT"])
which = sys.argv[1]
traced = "traced" in sys.argv[2:]
noprof = "noprof" in sys.argv[2:]
skip_apart = "skip-apart" in sys.argv[2:]
tag = which + ("-traced" if traced else "")
rec: dict = {"which": which, "traced": traced}


def cpu() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def dump() -> None:
    (OUT / f"{tag}.json").write_text(json.dumps(rec, indent=1, default=str))


def top(pr: cProfile.Profile, name: str) -> None:
    pr.dump_stats(str(OUT / f"{name}.prof"))
    for key in ("cumulative", "tottime"):
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats(key).print_stats(45)
        (OUT / f"{name}.{key}.txt").write_text(s.getvalue())


t = time.time()
if traced:
    CC.trace(cli.REPO)
TG.load_registries()
want = "_model_roots:DsaPositionsMerge_v1-served" if which == "v1" else "_merge_calls_roots"
pick = None
for units, make, kind, label in TG.catalog_units():
    if "dsa_topk_reads" in units[0] and units[0].endswith(want):
        pick = (units, make, kind, label)
        break
assert pick is not None, want
units, make, kind, label = pick
roots = TG._produce(units, make)
rec["binding"] = units[0]
rec["roots"] = [getattr(r, "id", repr(r)) for r in roots]
rec["setup_s"] = round(time.time() - t, 1)
print(tag, "setup", rec["setup_s"], "s", flush=True)

e = UN.Entry(units, kind, label, UN.key(units, 0), roots)
t, c = time.time(), cpu()
if skip_apart:
    rec["apart"] = "skipped"
else:
  if traced:
    CC.warm()
  kept, dropped, peaks, _ = UN.built_apart([e], 1, described=True, pinned=TG.pinned(),
                                           side=[] if traced else None, strict=True)
  rec["apart"] = {"wall_s": round(time.time() - t, 1), "child_cpu_s": e.seconds, "peak_gb_past_start": peaks,
                  "parent_cpu_s": round(cpu() - c, 1), "nodes": len(e.nodes)}
  print(tag, "apart", rec["apart"], flush=True)
dump()
if traced:
    sys.exit(0)

pr = cProfile.Profile()
if noprof:
    pr.enable = pr.disable = (lambda: None)
phases = {}
t, c = time.time(), cpu()
pr.enable()
with CC.producing(units):
    for r in roots:
        if UN._kind(r) == "S":
            _ = r.body
pr.disable()
phases["body"] = (round(time.time() - t, 1), round(cpu() - c, 1))
t, c = time.time(), cpu()
pr.enable()
nodes, objs = UN._reach(roots, described=False)
pr.disable()
phases["reach"] = (round(time.time() - t, 1), round(cpu() - c, 1))
t, c = time.time(), cpu()
pr.enable()
nodes, objs = UN._reach(roots, described=True, pinned=TG.pinned())
pr.disable()
phases["reach_described"] = (round(time.time() - t, 1), round(cpu() - c, 1))
rec["build_profiled"] = {"phases_wall_cpu_s": phases, "maxrss_gb": cli._maxrss_gb(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                         "gates": {fid: n.get("g") for fid, n in nodes.items() if "DsaPositionsMerge" in fid}}
print(tag, "build", rec["build_profiled"], flush=True)
noprof or top(pr, f"{which}-build")
dump()

from verity.circuits.defs import bind
from verity.definitions.registry import dsa_topk_reads as R

fn = bind(R.DsaPositionsMerge if which == "v1" else R.DsaPositionsMergeV2, M=8, K=2048)
rec["check_target"] = fn.id
rec["check_target_is_root_object"] = any(fn is r for r in roots) or any(fn is o for o in objs.values())
pr = cProfile.Profile()
if noprof:
    pr.enable = pr.disable = (lambda: None)
t, c = time.time(), cpu()
pr.enable()
res = CK.check_definition(fn, CK.Options(), memo={}, pins=CK.load_pins())
pr.disable()
rec["check_profiled"] = {"wall_s": round(time.time() - t, 1), "cpu_s": round(cpu() - c, 1),
                         "failures": [f.get("name") if isinstance(f, dict) else str(f) for f in res.get("failures", [])], "maxrss_gb": cli._maxrss_gb(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                         "seconds_by_check": res.get("seconds_by_check"),
                         "keys": sorted(res)[:60]}
print(tag, "check", rec["check_profiled"], flush=True)
noprof or top(pr, f"{which}-check")
rec["check_result"] = {k: (v if not isinstance(v, (list, dict)) else json.dumps(v, default=str)[:3000]) for k, v in res.items()}
dump()
