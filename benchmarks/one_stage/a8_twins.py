"""Run A8's twin over every Call of some kinds: each Call of the chosen kinds (families) in the slice's steps and layers evaluated
whole on the capture's operands and held to the capture's output snapshots, return component by return component. `a8.py` holds
a kind's first ``--per-kind`` Calls and moves on at a disagreement; this names every Call and component that departs. It also
digests the fold's wiring (each node's Definition and argument refs), so that two folds of one log can be compared.

    PYTHONPATH=<tree>:<tree>/verity/ml/flock/python:<tree>/integrations/vllm \\
      python benchmarks/one_stage/a8_twins.py --values DIR --roles derive-report.json --kinds DsaIndexerKRow_v1,DsaMqaLogits_v3 \\
      --out twins.json [--steps 0,1] [--layers 0,3]

Operands as `a8.kind_values` takes them: each operand view's snapshot first, else the producer's output snapshot (provenance),
and provenance again for a Call that disagrees on the first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.setrecursionlimit(100000)

import a8  # noqa: E402
import a8_kinds as K  # noqa: E402


def components(store, m: dict, got: list) -> dict[int, dict]:
    """Each return component an output snapshot holds: its leaves, how many of them differ, the first two that do."""
    from verity_vllm.check.fold_compare import _component_ranges
    comps, views = _component_ranges(m), m["output_views"]
    out = {}
    first = len(comps) - len(views)
    for k, ((lo, hi), view) in enumerate(zip(comps[first:], views), start=first):
        s = None if view is None else store.find(view, at=m["event"])
        if s is None:
            continue
        want = store.extract(s, view)
        d = [i for i in range(lo, hi) if got[i] is not None and int(got[i]) != int(want[i - lo])]
        out[k] = {"leaves": hi - lo, "differ": len(d), "first": d[:2]}
    return out


def wiring(res) -> str:
    """sha256 over the fold's nodes in order: each one's Definition id and its arguments' ref descriptors (`Refs.to_json`)."""
    h = hashlib.sha256()
    for node in res.body.nodes:
        h.update(node.fn.id.encode() + b"\0")
        for arg in node.args:
            h.update(json.dumps(arg.refs.to_json(), separators=(",", ":")).encode() + b"\0")
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", required=True, help="a8_slice.py's tar (and the checkpoint subset's), extracted")
    ap.add_argument("--roles", default=None, help="the Build's derive-report.json")
    ap.add_argument("--profile", default="derived_GLM53_tp8")
    ap.add_argument("--kinds", required=True, help="comma-separated families (DsaIndexerKRow_v1,...)")
    ap.add_argument("--steps", default="0,1")
    ap.add_argument("--layers", default="0,3")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kinds = a.kinds.split(",")
    steps, layers = [int(s) for s in a.steps.split(",")], {int(x) for x in a.layers.split(",")}
    res, store, _weights, vals, fold_s = a8.fold_values(Path(a.values), a.profile, Path(a.roles) if a.roles else None,
                                                        last_step=max(steps))
    print(f"A8 twins fold: {len(res.instances)} Calls to step {max(steps)}, {fold_s} s", flush=True)
    doc: dict = {"values": a.values, "steps": steps, "layers": sorted(layers), "fold_s": fold_s,
                 "unsupported": res.report.unsupported, "kinds": {}}
    for kind in kinds:
        t0, calls = time.time(), []
        for m in res.instances:
            if K.family(m["spec"]) != kind or m["step"] not in steps:
                continue
            lm = a8.LAYER.search(m["name"])
            if lm and int(lm[1]) not in layers:
                continue
            spec = res.body.nodes[m["node"]].fn
            entry = {"name": m["name"], "spec": m["spec"], "step": m["step"], "event": m["event"], "row": m.get("row"),
                     "out_offsets": [None if v is None else v.offset for v in m["output_views"]]}
            try:
                args = a8.operands(res, store, vals, m)
            except LookupError as e:
                calls.append({**entry, "error": str(e)[:300]})
                continue
            comp = components(store, m, a8.evaluate_units(spec, spec, args, 1, 0)["call_out"])
            entry["operands"] = "snapshot"
            if any(c["differ"] for c in comp.values()):
                try:
                    alt = a8.operands(res, store, vals, m, first="provenance")
                except LookupError:
                    alt = args
                if any(not np.array_equal(x, y) for x, y in zip(alt, args)):
                    comp2 = components(store, m, a8.evaluate_units(spec, spec, alt, 1, 0)["call_out"])
                    if not any(c["differ"] for c in comp2.values()):
                        comp, entry["operands"] = comp2, "provenance"
            calls.append({**entry, "components": comp})
        pattern = Counter("error" if "error" in c else ",".join(str(k) for k, v in c["components"].items() if v["differ"]) or "agree"
                          for c in calls)
        doc["kinds"][kind] = {"calls": len(calls), "agree": pattern.get("agree", 0),
                              "differing_components": dict(sorted(pattern.items())), "seconds": round(time.time() - t0, 1),
                              "each": calls}
        print(f"A8 twins {kind}: {len(calls)} Calls, " + json.dumps(dict(sorted(pattern.items()))), flush=True)
    doc["wiring_sha256"] = wiring(res)
    print(f"A8 twins wiring: {doc['wiring_sha256']} over {len(res.body.nodes)} nodes", flush=True)
    Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))
    return 0 if all(k["calls"] and k["agree"] == k["calls"] for k in doc["kinds"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
