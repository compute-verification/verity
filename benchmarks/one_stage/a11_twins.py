"""Run A11's twin over every Call of some kinds: each Call of the chosen kinds (families) in the slice's steps and layers evaluated
whole on the capture's operands and held to the capture's output snapshots, return component by return component. `a11.py` holds
a kind's first ``--per-kind`` Calls and moves on at a disagreement; this names every Call and component that departs. It also
digests the fold's wiring (each node's Definition and argument refs), so that two folds of one log can be compared.

    PYTHONPATH=<tree>:<tree>/verity/ml/flock/python:<tree>/integrations/vllm \\
      python benchmarks/one_stage/a11_twins.py --values DIR --roles derive-report.json --kinds DsaIndexerKRow_v1,DsaMqaLogits_v3 \\
      --out twins.json [--steps 0,1] [--layers 0,3]

Operands as `a11.kind_values` takes them: each operand view's snapshot first, else the producer's output snapshot (provenance),
and provenance again for a Call that disagrees on the first. A component that disagrees with the snapshot at its Call's event
is held to its allocation's last snapshot before the next Call writes that allocation (``late``, its event): what a reader saw.
A kernel the capture doesn't see writes after its Call's event: DeepGEMM's MQA logits, folded at their `torch.empty`
(`dsa.mqa_logits.invisible_deepgemm`), are first held at the top-k's read.
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

import a11  # noqa: E402
import a11_kinds as K  # noqa: E402


def write_events(res) -> dict[int, list[int]]:
    """Each allocation's writing Calls' events (an output view's), ascending."""
    w: dict[int, set[int]] = {}
    for m in res.instances:
        for v in m["output_views"]:
            if v is not None:
                w.setdefault(v.alloc, set()).add(m["event"])
    return {a: sorted(e) for a, e in w.items()}


def components(store, m: dict, got: list, writes: dict[int, list[int]]) -> dict[int, dict]:
    """Each return component an output snapshot holds: its leaves, how many of them differ, the first two that do; one that
    differs at the Call's event is held to its allocation's last snapshot before the next write (the module docstring)."""
    from verity_vllm.check.fold_compare import _component_ranges
    comps, views = _component_ranges(m), m["output_views"]
    out = {}
    first = len(comps) - len(views)
    for k, ((lo, hi), view) in enumerate(zip(comps[first:], views), start=first):
        s = None if view is None else store.find(view, at=m["event"])
        if s is None:
            continue

        def differ(snap, lo=lo, hi=hi, view=view):
            want = store.extract(snap, view)
            return [i for i in range(lo, hi) if got[i] is not None and int(got[i]) != int(want[i - lo])]
        d = differ(s)
        out[k] = {"leaves": hi - lo, "differ": len(d), "first": d[:2]}
        nxt = next((e for e in writes.get(view.alloc, []) if e > m["event"]), None)
        later = [x for x in store.by_alloc.get(view.alloc, []) if x.event > m["event"] and (nxt is None or x.event < nxt)
                 and store.covers(x, view)]
        if d and later:
            d2 = differ(later[-1])
            out[k].update({"differ": len(d2), "first": d2[:2], "late": later[-1].event, "differ_at_event": len(d)})
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
    ap.add_argument("--values", required=True, help="a11_slice.py's tar (and the checkpoint subset's), extracted")
    ap.add_argument("--roles", default=None, help="the Build's derive-report.json")
    ap.add_argument("--profile", default="derived_GLM53_tp8")
    ap.add_argument("--kinds", required=True, help="comma-separated families (DsaIndexerKRow_v1,...)")
    ap.add_argument("--steps", default="0,1")
    ap.add_argument("--layers", default="0,3")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kinds = a.kinds.split(",")
    steps, layers = [int(s) for s in a.steps.split(",")], {int(x) for x in a.layers.split(",")}
    res, store, _weights, vals, fold_s = a11.fold_values(Path(a.values), a.profile, Path(a.roles) if a.roles else None,
                                                        last_step=max(steps))
    print(f"A11 twins fold: {len(res.instances)} Calls to step {max(steps)}, {fold_s} s", flush=True)
    doc: dict = {"values": a.values, "steps": steps, "layers": sorted(layers), "fold_s": fold_s,
                 "unsupported": res.report.unsupported, "kinds": {}}
    writes = write_events(res)
    for kind in kinds:
        t0, calls = time.time(), []
        for m in res.instances:
            if K.family(m["spec"]) != kind or m["step"] not in steps:
                continue
            lm = a11.LAYER.search(m["name"])
            if lm and int(lm[1]) not in layers:
                continue
            spec = res.body.nodes[m["node"]].fn
            entry = {"name": m["name"], "spec": m["spec"], "step": m["step"], "event": m["event"], "row": m.get("row"),
                     "out_offsets": [None if v is None else v.offset for v in m["output_views"]]}
            try:
                args = a11.operands(res, store, vals, m)
            except LookupError as e:
                calls.append({**entry, "error": str(e)[:300]})
                continue
            comp = components(store, m, a11.evaluate_units(spec, spec, args, 1, 0)["call_out"], writes)
            entry["operands"] = "snapshot"
            if any(c["differ"] for c in comp.values()):
                try:
                    alt = a11.operands(res, store, vals, m, first="provenance")
                except LookupError:
                    alt = args
                if any(not np.array_equal(x, y) for x, y in zip(alt, args)):
                    comp2 = components(store, m, a11.evaluate_units(spec, spec, alt, 1, 0)["call_out"], writes)
                    if not any(c["differ"] for c in comp2.values()):
                        comp, entry["operands"] = comp2, "provenance"
            calls.append({**entry, "components": comp})
        pattern = Counter("error" if "error" in c else ",".join(str(k) for k, v in c["components"].items() if v["differ"]) or "agree"
                          for c in calls)
        late = sum(any("late" in v for v in c.get("components", {}).values()) for c in calls)
        doc["kinds"][kind] = {"calls": len(calls), "agree": pattern.get("agree", 0), "late": late,
                              "differing_components": dict(sorted(pattern.items())), "seconds": round(time.time() - t0, 1),
                              "each": calls}
        print(f"A11 twins {kind}: {len(calls)} Calls, " + json.dumps(dict(sorted(pattern.items()))) + f", {late} held late",
              flush=True)
    doc["wiring_sha256"] = wiring(res)
    print(f"A11 twins wiring: {doc['wiring_sha256']} over {len(res.body.nodes)} nodes", flush=True)
    Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))
    return 0 if all(k["calls"] and k["agree"] == k["calls"] for k in doc["kinds"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
