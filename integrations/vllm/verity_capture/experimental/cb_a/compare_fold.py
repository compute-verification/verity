"""CB-A comparator against a record that ships only the eager fold's SUMMARY (`fold_summary.json`: digest, n_instances, per-step
spec histograms) and not `program.json`/`instances.jsonl` (the R8 family `.small.tgz` bundles). Verdict granularity is therefore
per-step operator/spec multisets and totals — operand wiring cannot be checked here (compare.py does that when instances exist).

    python -m verity_capture.experimental.cb_a.compare_fold --derived <dir> --fold <fold_summary.json> --out <json>
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys

from .compare import load_derived


_F64 = re.compile(r'\{"f64":"([^"]+)"\}')


def norm_spec(spec: str) -> str:
    """The fold summary prints float statics as Python floats (`EPS=1e-05`); the codec's spec id prints them as exact hex
    (`EPS={"f64":"0x1.4f8b588e368f1p-17"}`). Same value, one spelling."""
    if spec.startswith("batch:"):
        spec = spec[len("batch:"):]
    return _F64.sub(lambda m: repr(float.fromhex(m.group(1))), spec)


def step_hist(P, idxs) -> collections.Counter:
    c: collections.Counter = collections.Counter()
    for i in idxs:
        c[norm_spec(P.inst[i]["spec"])] += 1
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--derived", required=True)
    ap.add_argument("--fold", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="derived")
    a = ap.parse_args()
    fs = json.load(open(a.fold))
    P = load_derived(a.derived, a.label)
    rec_steps = fs["per_step"]
    out = {"derived": {"dir": a.derived, "digest": P.digest, "instances": len(P.inst), "steps": len(P.steps)},
           "record": {"fold_summary": a.fold, "profile_id": fs.get("profile_id"), "digest": fs.get("digest"), "n_instances": fs.get("n_instances"),
                      "n_steps": fs.get("n_steps"), "unsupported": fs.get("unsupported"), "errors": fs.get("errors"), "spec": fs.get("spec")},
           "steps": [], "totals": {"steps_equal": 0, "steps_different": 0, "instances_equal": 0, "instances_only_derived": 0, "instances_only_record": 0}}
    n = max(len(P.steps), len(rec_steps))
    first_mismatch = None
    for s in range(n):
        d = step_hist(P, P.steps[s]) if s < len(P.steps) else collections.Counter()
        r = collections.Counter(rec_steps[s]["spec_histogram"]) if s < len(rec_steps) else collections.Counter()
        only_d = {k: v for k, v in (d - r).items()}
        only_r = {k: v for k, v in (r - d).items()}
        eq = sum((d & r).values())
        row = {"step": s, "phase": rec_steps[s].get("phase") if s < len(rec_steps) else None, "rows": rec_steps[s].get("rows") if s < len(rec_steps) else None,
               "derived_instances": sum(d.values()), "record_instances": sum(r.values()), "equal": not only_d and not only_r,
               "only_derived": only_d, "only_record": only_r}
        out["steps"].append(row)
        out["totals"]["instances_equal"] += eq
        out["totals"]["instances_only_derived"] += sum(only_d.values())
        out["totals"]["instances_only_record"] += sum(only_r.values())
        if row["equal"]:
            out["totals"]["steps_equal"] += 1
        else:
            out["totals"]["steps_different"] += 1
            if first_mismatch is None:
                first_mismatch = {"step": s, "only_derived": dict(list(only_d.items())[:8]), "only_record": dict(list(only_r.items())[:8])}
    out["first_mismatch"] = first_mismatch
    out["spec_multiset_equal_all_steps"] = out["totals"]["steps_different"] == 0 and len(P.steps) == len(rec_steps)
    out["digests_equal"] = P.digest == fs.get("digest")
    out["note"] = ("per-step spec multisets and instance counts only: the record bundle has no instances.jsonl, so operand wiring is not "
                   "compared here; digests are expected to differ (input layout / root naming) even when the structure matches")
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("spec_multiset_equal_all_steps", "digests_equal", "totals", "first_mismatch")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
