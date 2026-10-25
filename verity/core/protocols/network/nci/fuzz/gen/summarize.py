"""What `nci-desc-fuzz`'s output says, per statement: how many Programs passed the preconditions and were checked, the
shapes they covered, how many fail each statement in each family (`by_family_fails`), and every failure (the first few of
each kind in full).

    python gen/summarize.py OUT.jsonl[.gz] [OUT.jsonl[.gz] ...] [--json FILE]
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json

SHAPE_MAX = ("depth", "defs", "activations", "walkBound", "gates", "root_nodes", "scans", "scanN", "batches",
             "composite_batches", "calls", "prims", "strided", "explicit", "pass_through_defs")
SHAPE_ANY = ("scans", "scan0", "scan1", "scan_xs", "batches", "composite_batches", "batch0", "mapped", "inner_axis",
             "empty_bodies", "pass_through_defs", "strided", "explicit")
S3_MAX = ("fan_in", "fan_out", "node_fan_out", "segs", "copies", "outside_reads")
KEEP = 5


def summarize(paths: list[str]) -> dict:
    pre = collections.Counter()
    by_family = collections.defaultdict(collections.Counter)
    refusals = collections.Counter()
    mx = collections.Counter()
    anyc = collections.Counter()
    s = {k: collections.Counter() for k in ("s1", "s1prim", "s2", "s3", "s3_unwalkable", "agree")}
    fails = collections.defaultdict(list)
    family_fails = collections.defaultdict(collections.Counter)
    most_vs_bound = 0.0
    for p in paths:
        for line in (gzip.open(p, "rt") if p.endswith(".gz") else open(p)):
            r = json.loads(line)
            pre[r["pre"]] += 1
            by_family[r["family"]][r["pre"]] += 1
            if r["pre"] in ("decode", "analyze"):
                refusals[f"{r['pre']}: {r['why'][:90]}"] += 1
            if r["pre"] == "walkable" and "s3" in r:
                s3 = r["s3"]
                key = ("hold" if "segs" in s3 and s3.get("fail") is None else "fail" if s3.get("fail")
                       else f"{s3['pre']} refused" if "pre" in s3 else f"skipped: {s3.get('skipped')}")
                s["s3_unwalkable"][key] += 1
                if key == "fail":
                    fails["s3_unwalkable"].append({"id": r["id"], "fail": s3["fail"]})
            if r["pre"] != "ok":
                continue
            for k in ("s1", "s2", "s3", "agree"):
                family_fails[r["family"]][k] += r[k].get("fail") is not None
            sh = r["shape"]
            for k in SHAPE_MAX:
                mx[k] = max(mx[k], sh[k])
            for k in SHAPE_ANY:
                anyc[k] += sh[k] > 0
            s1 = r["s1"]
            complete = not s1["cut"] and s1["capped"] is None
            for k in ("disagree", "anomaly"):
                if s1[k] is not None:
                    fails[f"harness: s1 {k}"].append({"id": r["id"], k: s1[k]})
            if s1["fail"] is not None:
                s["s1"]["fail"] += 1
                if len(fails["s1"]) < KEEP:
                    fails["s1"].append({"id": r["id"], "fail": s1["fail"]})
            elif complete:
                s["s1"]["hold"] += 1
            else:
                s["s1"]["incomplete"] += 1
            if s1["fail_prim"] is not None:
                s["s1prim"]["fail"] += 1
                fails["s1prim"].append({"id": r["id"], "fail": s1["fail_prim"]})
            elif complete:
                s["s1prim"]["hold"] += 1
            else:
                s["s1prim"]["incomplete"] += 1
            mx["most_steps"] = max(mx["most_steps"], s1["most_steps"])
            mx["s1.activations"] = max(mx["s1.activations"], s1["activations"])
            mx["s1.members"] = max(mx["s1.members"], s1["members"])
            if sh["walkBound"]:
                most_vs_bound = max(most_vs_bound, s1["most_steps"] / sh["walkBound"])
            s2 = r["s2"]
            if s2.get("fail") is not None:
                s["s2"]["fail"] += 1
                fails["s2"].append({"id": r["id"], "fail": s2["fail"]})
            else:
                s["s2"]["hold"] += 1
                s["s2"]["sampled" if s2["cut"] else "enumerated"] += 1
                mx["s2.defs"] = max(mx["s2.defs"], s2["defs"])
            s3 = r["s3"]
            if s3.get("fail") is not None:
                s["s3"]["fail"] += 1
                fails["s3"].append({"id": r["id"], "fail": s3["fail"]})
            elif "segs" in s3:
                s["s3"]["hold"] += 1
                for k in S3_MAX:
                    mx[f"s3.{k}"] = max(mx[f"s3.{k}"], s3[k])
                anyc["s3.copies"] += s3["copies"] > 0
            elif "pre" in s3:
                s["s3"]["circuitOf refused"] += 1
            else:
                s["s3"]["skipped: " + s3["skipped"]] += 1
            ag = r["agree"]
            if ag.get("fail") is not None:
                s["agree"]["fail"] += 1
                fails["agree"].append({"id": r["id"], "fail": ag["fail"]})
            elif "verdicts" in ag:
                s["agree"]["programs"] += 1
                s["agree"]["verdicts"] += ag["verdicts"]
                s["agree"]["given"] += ag.get("given", 0)
            else:
                s["agree"]["skipped"] += 1
    return {"programs": sum(pre.values()), "preconditions": dict(pre),
            "by_family": {f: dict(c) for f, c in sorted(by_family.items())},
            "by_family_fails": {f: dict(c) for f, c in sorted(family_fails.items())},
            "statements": {k: dict(v) for k, v in s.items()}, "max": dict(mx),
            "programs_with": dict(anyc), "most_steps_over_walkBound": most_vs_bound,
            "refusals": dict(refusals.most_common(25)), "failures": {k: v[:50] for k, v in fails.items()},
            "failure_counts": {k: len(v) for k, v in fails.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--json")
    a = ap.parse_args()
    out = summarize(a.paths)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    brief = {k: v for k, v in out.items() if k not in ("failures", "refusals")}
    print(json.dumps(brief, indent=1))
    for k, v in out["failures"].items():
        print(f"{k}: {out['failure_counts'][k]} (first: {json.dumps(v[0])[:400]})")


if __name__ == "__main__":
    main()
