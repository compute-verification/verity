"""(e') [T1] The sampled-replay PARTITION of record: `commit/sampled_replay_p<k>.json`'s population (the VUs by family, the
not-evaluable rows by name, the strata and their sizes, the window-lifetime split), its seeded stratified sample and the per-request phase
anchors, as the record states them, beside a candidate's or a record kept on the node.  The record's own `result` / `grade` are carried
as facts beside the partition.

Inputs: `commit/sampled_replay_p0.json` (records).  `sampled_replay_p1/p2.json` are `reused_from` p0 on the B=1 rows: the partition is p0's.
"""
from __future__ import annotations

import argparse
import json

from .. import resolver

NAME = "replay_partition"
TIER = "T1"
STRENGTH = "strong"
INPUTS = ("commit/sampled_replay_p0.json",)


def applies(row: resolver.Row) -> str | None:
    return None if row.has("commit/sampled_replay_p0.json") else "no sampled_replay record on this row (the row's Commit never reached C2)"


def _body(doc: dict) -> dict:
    body = doc.get("sampled_replay")
    return body if isinstance(body, dict) else doc


def lift(sr: dict) -> dict:
    """The partition contract from a `sampled_replay` document (the record's, a candidate's, or the recomputation's)."""
    pop = sr.get("population") or {}
    smp = sr.get("sample") or {}
    sk = sr.get("strata_keys") or {}
    life = sk.get("lifetime") or {}
    nex = pop.get("not_executed_by_rule") or {}
    rule = (pop.get("not_executed_rule") or {}).get("requests") or {}
    return {
        "population": {"vus": pop.get("vus"), "requests": pop.get("requests"), "by_family": pop.get("by_family"),
                       "not_evaluable_by_family": pop.get("not_evaluable_by_family"), "not_evaluable": pop.get("not_evaluable"),
                       "strata_n": pop.get("strata_n"),
                       "not_executed": {"vus": nex.get("vus"), "vus_by_request": nex.get("vus_by_request"),
                                        "strata_excluded": {k: v.get("population") for k, v in (nex.get("strata_excluded") or {}).items()}}},
        "executed_prefix": {rid: {k: v.get(k) for k in ("a_r", "lag", "cap", "n_exec", "faults")} for rid, v in rule.items() if isinstance(v, dict)},
        "strata": {k: {"population": v.get("population"), "sampled": v.get("sampled")} for k, v in (sr.get("strata") or {}).items()},
        "sample": {**{k: smp.get(k) for k in ("per_stratum", "seed", "picked", "stratified_picked", "cap_effective", "max_evaluate", "targeted_n")},
                   "targets": sorted((smp.get("targeted") or {}).keys()),
                   "picks_by_request_step": {r: {str(s): (v[0] if isinstance(v, list) else v) for s, v in d.items()} for r, d in (smp.get("picks_by_request_step") or {}).items()},
                   "picks_by_request_arrival": smp.get("picks_by_request_arrival")},
        "lifetime": {"values": life.get("values"), "sampled": life.get("sampled"), "ring_slots": life.get("ring_slots")},
        "phase_anchor": {r: {k: a.get(k) for k in ("last_step", "anchor", "cap_last_step", "n_exec", "lag_step")} for r, a in (sk.get("phase_anchor") or {}).items()},
    }


def expected(row: resolver.Row) -> dict:
    sr = _body(row.json("commit/sampled_replay_p0.json"))
    out = lift(sr)
    out["record"] = {"result": sr.get("result"), "grade": sr.get("grade")}          # carried, not recomputed (the evaluations need the binding map)
    return out


def actual(row: resolver.Row) -> dict:
    if kept := resolver.kept_on_the_node(row, "commit/sampled_replay_p0.json"):
        return kept
    cand = row.candidate("commit/sampled_replay_p0.json")
    if cand is not None:
        sr = _body(resolver.read_json(cand))
        out = lift(sr)
        out["record"] = {"result": sr.get("result"), "grade": sr.get("grade")}
        out["source"] = "candidate"
        return out
    raise resolver.NotResolvable(f"{row.key}: no sampled_replay_p0.json kept or in a candidate; the partition is not recomputed")

def main(argv: list[str] | None = None) -> int:
    """`python -m tests.regression.checks.replay_partition <row-key|row-number> [--out FILE]`: expected vs actual, differences named."""
    ap = argparse.ArgumentParser()
    ap.add_argument("row")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    row = next(r for r in resolver.rows() if r.key == a.row or str(r.number) == a.row)
    exp = expected(row)
    act = actual(row)
    diff = resolver.diff(exp, act)
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"row": row.key, "equal": not diff, "differences": diff, "actual": act, "expected": exp}, f, indent=1, default=str)
    print(json.dumps({"row": row.key, "equal": not diff, "n_differences": len(diff), "differences": diff[:30], "source": act.get("source"),
                      "population": (act.get("population") or {}).get("vus"), "strata_n": (act.get("population") or {}).get("strata_n"),
                      "picked": (act.get("sample") or {}).get("picked")}, default=str))
    return 0 if not diff else 1


if __name__ == "__main__":
    raise SystemExit(main())
