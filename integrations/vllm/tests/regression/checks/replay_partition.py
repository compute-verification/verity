"""(e') [T1] The sampled-replay PARTITION reproduces: `commit/sampled_replay_p<k>.json`'s population (every Program row classified: the VUs
by family, the not-evaluable rows by name, the strata and their sizes, the window-lifetime split), its `not_executed_by_rule` split,
the seeded stratified sample (the per-stratum sample counts, the round-robin cap, the picks per (request, request-relative step)) and the
per-request phase anchors -- recomputed with `verity_vllm.check.replay`'s own pure functions (`ProgramIndex`, `population`,
`executed_prefix_of_record`, `phase_anchor`, `split_not_executed`, `sample`) over the row's request Programs under the v2 query's address
rule + the facts the record states (seed, per-stratum count, cap, targets, ring depth, arrivals, the executed-prefix facts of record).
STRONG: a row classified into another family / stratum, a moved phase boundary, a different draw -- each moves a count.

What this does NOT recompute: the VU evaluations themselves (the `evaluated` / `equal` / `mismatch` counts and the Q(P) reconciliation
`query_total` / `sampled_strata` / `uncovered_by_name` need the binding map, `commit/binding_map_p*.json`, 0.5-1.5 GB, vault core tars only).
The record's own `result` / `grade` are carried as facts beside the partition, not recomputed.

Inputs: `commit/sampled_replay_p0.json` (records), `build_request*/instances.json.gz` (programs; the B=1 Programs are
0.9-1.7 GB compressed and need 120-250 GB of RAM as Python objects: a big pod; the B>=2 rows' request Programs are small).  Every GREEN
row and #57 carry a sampled-replay record; the FAIL rows #4, #70, #75 never reached C2.  `sampled_replay_p1/p2.json` are `reused_from`
p0 on the B=1 rows (the same run root, seed and code): the partition is p0's; the B>=2 rows' further pairs are not compared here.
"""
from __future__ import annotations

import gzip
import json
import os
from collections import Counter
from typing import Any

from .. import resolver

NAME = "replay_partition"
TIER = "T1"
STRENGTH = "strong"
INPUTS = ("commit/sampled_replay_p0.json", "build_request*/instances.json.gz")

FACT_KEYS = ("a_r", "a_r_source", "lag", "cap", "arrive", "eos_declared", "served_len_commit", "linked", "lag_attested", "rows_past_stop",
             "served", "eos_ids", "account_missing")


def applies(row: resolver.Row) -> str | None:
    return None if row.has("commit/sampled_replay_p0.json") else "no sampled_replay record on this row (the row's Commit never reached C2)"


def _body(doc: dict) -> dict:
    return doc.get("sampled_replay") if isinstance(doc.get("sampled_replay"), dict) else doc


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


def _load_program(p: str) -> dict:
    with gzip.open(p, "rt") as f:
        return json.load(f)


def recompute(sr: dict, programs: dict[str, dict], program_dirs: dict[str, str] | None = None) -> dict:
    """The partition over `programs` (request label -> Program instances doc) with the record's stated parameters -- `sampled_replay`'s
    population / split / draw, without the evaluations, under the v2 query's address rule (`program_dirs` = the Build dirs whose
    correspondence the manifest is built with).  A record that predates the v2 rule (no `addresses.rule`: every reference record) was
    addressed by the retired v1 rule; its partition is recomputed under v2 here (decision `retire-v1` where that moves a count).  Returns a
    `sampled_replay`-shaped document (`lift` reads it)."""
    from verity_vllm.check.replay import index as IX, population as POP, sample as SM
    smp, sk = sr.get("sample") or {}, sr.get("strata_keys") or {}
    seed, per_stratum = int(smp["seed"]), int(smp.get("per_stratum") or 2)
    max_evaluate = int(smp.get("max_evaluate") or 2000)
    targets = sorted((smp.get("targeted") or {}).keys()) or None
    ring_slots = (sk.get("lifetime") or {}).get("ring_slots")
    arrivals = {str(r): int(a) for r, a in (smp.get("picks_by_request_arrival") or {}).items()}
    ranks = [int(r) for r in (sr.get("applicability") or {}).get("ranks") or [0]]
    facts = ((sr.get("executed_prefix_of_record") or {}).get("facts_of_record") or {})
    executed_prefix = {rid: POP.executed_prefix_of_record(rid, **{k: v for k, v in f.items() if k in FACT_KEYS}) for rid, f in facts.items() if isinstance(f, dict)} or None
    all_vus, not_ev, not_ev_fam, indices, anchors = [], Counter(), Counter(), {}, {}
    for rid, inst in programs.items():
        pi = indices.get(id(inst))
        if pi is None:
            pi = IX.ProgramIndex(inst, program_dir=(program_dirs or {}).get(rid))
            indices[id(inst)] = pi
        anchors[rid] = POP.phase_anchor(pi.n_steps, (executed_prefix or {}).get(str(rid)))
        vus, ne, nef = POP.population(pi, rid, rank=(ranks[0] if len(ranks) == 1 else 0), arrive=arrivals.get(rid, 0), ring_slots=ring_slots,
                                     addresses=None, last_step=int(anchors[rid]["last_step"]))
        all_vus.extend(vus)
        not_ev.update(ne)
        not_ev_fam.update(nef)
    pool, not_exec = POP.split_not_executed(all_vus, executed_prefix)
    picked_all, strata, tgt = SM.sample(pool, per_stratum, seed, targets=targets)
    n_strat = int(tgt["stratified"])
    cap = max(max_evaluate, len(strata))
    picked = picked_all[:min(cap, n_strat)] + picked_all[n_strat:]
    by_step: dict[str, dict[str, int]] = {}
    for v in picked:
        d = by_step.setdefault(str(v["request_id"]), {})
        d[str(int(v["step"]))] = d.get(str(int(v["step"])), 0) + 1
    return {"population": {"vus": len(all_vus), "requests": sorted(programs), "by_family": dict(Counter(v["family"] for v in all_vus)),
                           "not_evaluable_by_family": dict(not_ev_fam), "not_evaluable": dict(not_ev), "strata_n": len(strata),
                           "not_executed_by_rule": {"vus": len(not_exec), "vus_by_request": dict(Counter(v["request_id"] for v in not_exec)),
                                                    "strata_excluded": {k: {"population": int(n)} for k, n in sorted(Counter(v["stratum"] for v in not_exec).items())}},
                           "not_executed_rule": {"requests": {rid: {k: ep.get(k) for k in ("a_r", "lag", "cap", "n_exec", "faults")} for rid, ep in (executed_prefix or {}).items()}}},
            "sample": {"per_stratum": per_stratum, "seed": seed, "picked": len(picked), "stratified_picked": min(cap, n_strat), "cap_effective": cap,
                       "max_evaluate": max_evaluate, "targeted_n": int(tgt["added"]), "targeted": tgt["hits"],
                       "picks_by_request_step": {r: {s: [n] for s, n in sorted(d.items(), key=lambda kv: int(kv[0]))} for r, d in sorted(by_step.items())},
                       "picks_by_request_arrival": {r: arrivals.get(r, 0) for r in by_step}},
            "strata": strata,
            "strata_keys": {"lifetime": {"values": dict(Counter(v["lifetime"] for v in all_vus)), "sampled": dict(Counter(v["lifetime"] for v in picked)),
                                         "ring_slots": ring_slots},
                            "phase_anchor": anchors}}


def actual(row: resolver.Row) -> dict:
    cand = row.candidate("commit/sampled_replay_p0.json")
    if cand is not None:
        sr = _body(json.load(open(cand)))
        out = lift(sr)
        out["record"] = {"result": sr.get("result"), "grade": sr.get("grade")}
        out["source"] = "candidate"
        return out
    sr = _body(row.json("commit/sampled_replay_p0.json"))
    progs_root = row.programs_root()
    if progs_root is None:
        raise resolver.NotResolvable(f"{row.key}: no request Programs (build_request*/instances.json.gz) reachable here")
    programs: dict[str, dict] = {}
    dirs: dict[str, str] = {}
    loaded: dict[str, dict] = {}
    for rid, ent in (sr.get("programs") or {}).items():
        base = os.path.basename(str((ent or {}).get("dir") or "build_request"))
        p = str(progs_root / base / "instances.json.gz")
        if not os.path.isfile(p):
            raise resolver.NotResolvable(f"{row.key}: request {rid}'s Program {base}/instances.json.gz not under {progs_root}")
        if p not in loaded:
            loaded[p] = _load_program(p)
        programs[str(rid)] = loaded[p]
        dirs[str(rid)] = str(progs_root / base)
    doc = recompute(sr, programs, dirs)
    out = lift(doc)
    out["record"] = {"result": sr.get("result"), "grade": sr.get("grade")}
    out["source"] = "recomputed: sampled_replay.{ProgramIndex, population, executed_prefix_of_record, phase_anchor, split_not_executed, sample} over the Programs"
    return out


def main(argv: list[str] | None = None) -> int:
    """`python -m tests.regression.checks.replay_partition <row-key|row-number> [--out FILE]`: expected vs actual, differences named."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("row")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    row = next(r for r in resolver.rows() if r.key == a.row or str(r.number) == a.row)
    exp = expected(row)
    act = actual(row)
    diff = resolver.diff(exp, act)
    if a.out:
        json.dump({"row": row.key, "equal": not diff, "differences": diff, "actual": act, "expected": exp}, open(a.out, "w"), indent=1, default=str)
    print(json.dumps({"row": row.key, "equal": not diff, "n_differences": len(diff), "differences": diff[:30], "source": act.get("source"),
                      "population": (act.get("population") or {}).get("vus"), "strata_n": (act.get("population") or {}).get("strata_n"),
                      "picked": (act.get("sample") or {}).get("picked")}, default=str))
    return 0 if not diff else 1


if __name__ == "__main__":
    raise SystemExit(main())
