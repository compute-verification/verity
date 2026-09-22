"""[R14 moe] `commit/coverage.json` for the finalizer (COORD 17:08Z contract) FROM the BIND-01 checker's own records -- a translation, not a
verdict: every field is copied or recomputed (`commit_verdict.binding_check`) from the `commit.binding` record that `commit_delta` wrote for
each instrumented run (`commit/runs.jsonl`) and from `commit/verdict.json`.  Nothing here decides PASS; the checker did.

    python -m verity_capture.sweep.coverage_from_binding <evidence-dir> [--negatives "<suite id + result>"] [--program-digest HEX]

Labels honestly what the population is: BIND-01's collector-independent population = model structure x workload (requests, steps, ranks),
NOT yet lane manifest's Program/query-derived manifest -- `manifest_source` says so; the finalizer / report decide what that is worth.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("evidence_dir")
    ap.add_argument("--negatives", default=None, help="negatives suite id + result as run on this tree (e.g. 'test_binding_negatives.py 9/9 passed <sha>')")
    ap.add_argument("--program-digest", default=None, help="Program digest the binding map was built against (from binding_map_p*.json)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    ev = a.evidence_dir
    runs_p = os.path.join(ev, "commit", "runs.jsonl")
    verdict_p = os.path.join(ev, "commit", "verdict.json")
    if not (os.path.exists(runs_p) and os.path.exists(verdict_p)):
        print("need commit/runs.jsonl and commit/verdict.json", file=sys.stderr)
        return 2
    from verity_capture.bench.commit_verdict import binding_check

    V = json.load(open(verdict_p))
    recs = [json.loads(l) for l in open(runs_p) if l.strip()]
    inst = [r for r in recs if r.get("arm") in ("instrumented", "treatment")]
    bnds = [(r.get("commit") or {}).get("binding") for r in inst]
    if not bnds or any(b is None for b in bnds):
        out = {"checker": "verity/coverage BIND-01 (verity_capture.commit.binding)", "verdict": "ABSENT",
               "reason": "no commit.binding record in the instrumented runs -- commit_delta ran without BIND-01 (counts + classes only)"}
    else:
        checks = [binding_check(b) for b in bnds]
        covs = [b.get("coverage") or {} for b in bnds]
        pops = [b.get("population") or {} for b in bnds]
        ops = [b.get("openings_population") or {} for b in bnds]
        all_ok = all(ok for ok, _ in checks) and bool(V.get("commit_pass"))
        pd = a.program_digest
        if pd is None:
            for f in sorted(os.listdir(os.path.join(ev, "commit"))):
                if f.startswith("binding_map_p") and f.endswith(".json"):
                    try:
                        pd = json.load(open(os.path.join(ev, "commit", f))).get("program_digest")
                    except Exception:
                        pd = None
                    break
        out = {
            "schema": "verity/coverage/v1 (translated from BIND-01 records)",
            "checker": f"verity/coverage BIND-01 verity_capture.commit.binding {bnds[0].get('rules_version')} + commit_verdict.binding_check",
            "verdict": "PASS" if all_ok else "FAIL",
            "reason": None if all_ok else "; ".join(why for ok, why in checks if not ok) or V.get("first_fail_reason"),
            "runs_instrumented": len(bnds),
            "program_digest": pd,
            "manifest_digest": pops[0].get("digest"),
            "manifest_source": "BIND-01 expected population = " + str(pops[0].get("source")) + " -- collector-independent, NOT lane manifest's Program/query-derived manifest (interim)",
            "manifest_n": pops[0].get("n"),
            "manifest_digest_all_runs_equal": len({p.get("digest") for p in pops}) == 1,
            "binding_map_digest": bnds[0].get("map_digest"),
            "binding_map_digests": [b.get("map_digest") for b in bnds],
            "binding_map_entries": [b.get("map_entries_n") for b in bnds],
            "identity_coverage": all(c.get("ok") for c in covs),
            "coverage": [{k: c.get(k) for k in ("expected_n", "bound_n", "missing_n", "extra_n", "physical_copies_n")} for c in covs],
            "coverage_missing_first": next((c.get("missing") or [None])[0] for c in covs if c.get("missing")) if any(c.get("missing") for c in covs) else None,
            "openings_checked": sum(int(o.get("opened") or 0) for o in ops),
            "openings_ok": sum(int(o.get("verified") or 0) for o in ops),
            # the finalizer reads the label ("expected population" = drawn from the expected population, not from the committed layout);
            # the per-run numbers travel in `openings_population_detail`
            "openings_population": ("expected population" if ops and all(int(o.get("n") or 0) > 0 and int(o.get("unbound_n") or 0) == 0 for o in ops)
                                    else "not from the expected population (see openings_population_detail)"),
            "openings_population_detail": {"drawn_per_run": [o.get("n") for o in ops], "unbound": [o.get("unbound_n") for o in ops],
                                           "late_step_in_sample": [o.get("late_step_in_sample") for o in ops], "families": sorted({f for o in ops for f in (o.get("families") or [])})},
            "value_correctness": all(int(o.get("verified") or 0) == int(o.get("opened") or 0) and int(o.get("opened") or 0) > 0 for o in ops) and all(bool(r.get("tokens_eq", r.get("tokens_equal", True))) for r in inst),
            "context_ok": all((b.get("context") or {}).get("ok") for b in bnds),
            "collector": bnds[0].get("collector"),
            "negatives_suite": a.negatives,
            "binding_check": [{"ok": ok, "why": why} for ok, why in checks],
            "run_roots": V.get("run_roots"),
            "source_files": ["commit/runs.jsonl (commit.binding per instrumented run)", "commit/verdict.json"],
        }
    p = a.out or os.path.join(ev, "commit", "coverage.json")
    json.dump(out, open(p, "w"), indent=1, default=str)
    print(f"{p}: verdict {out['verdict']}" + (f" coverage {out.get('coverage')} openings {out.get('openings_ok')}/{out.get('openings_checked')}" if out["verdict"] != "ABSENT" else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
