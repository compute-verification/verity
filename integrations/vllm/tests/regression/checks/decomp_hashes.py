"""(b) [T1] Fold / decomposition hashes reproduce: per request `canonical_sha256{a,b}` (a = the derived component, b = the record's
projection), `derived_digest`, `n_instances`, the shape facts (LP, T, sampling events, steps, prefill chunks), the cross-request reads and the
attribution table (`how`, unattributed, row-rule disagreements) of `match_decomp.json`, recomputed over the Match record with the v2 path --
`verity_vllm.observe.contract.iter_observations` (the stream) -> `verity_vllm.check.match.resolve` (occurrence resolution: the request channel wraps
batch_decomp's meta / dataflow / weight-only / rows rules) -> `verity_vllm.check.match.resolve_decomp.decompose` (the per-request projection and the
canonical hash are v1's own `batch_decomp.Projection` / `compare.compare_steps`, wrapped).  STRONG: a value attributed to another request,
a missed weight-only value, a request-relative step cut elsewhere -- each moves a hash or a count.

Inputs: `match/program.json` (0.3-1.4 GB) + `match/instances.jsonl` (0.6-5.8 GB) -- pinned by sha256 in `global_match.json.impl.fold_load`
and checked here as read (`fold_record_pins`) -- the row's request Programs (`build_request*/instances.json.gz`), and `match/capture/log.jsonl.gz`
when it is beside the record (the reference rows archived it to the vault: without it the rows channel and the meta-vs-rows cross-check
are unavailable and the result says so; every row of a reference record is stamped, so attribution itself does not need it).  100-500 s, 10-40 GB
RAM: a pod.  No B=1 row has a decomposition (`match_decomp.json` absent on #11, #39).
"""
from __future__ import annotations

import glob
import json
import os
from typing import Any

from .. import resolver

NAME = "decomp_hashes"
TIER = "T1"
STRENGTH = "strong"
INPUTS = ("match_decomp.json", "match/program.json", "match/instances.jsonl", "match/capture/log.jsonl.gz", "build_request*/instances.json.gz")

REQUEST_FIELDS = ("canonical_sha256", "derived_digest", "xreq", "n_instances", "LP", "T", "token_selects", "steps", "prefill_chunks")
ATTRIBUTION_FIELDS = ("how", "unattributed", "attributed", "row_rule_disagreements", "weight_only_shared")


def applies(row: resolver.Row) -> str | None:
    return None if row.has("match_decomp.json") else "no match_decomp.json (B=1 rows have no decomposition; FAIL rows never reached it)"


def lift(d: dict, *, pins: dict | None = None) -> dict:
    """The contract, from a `match_decomp.json`-shaped document (the record's, a candidate's, or v2's recomputation)."""
    reqs = d.get("requests") if isinstance(d.get("requests"), (dict, list)) else None
    per: dict = {}
    items = reqs.items() if isinstance(reqs, dict) else enumerate(reqs or [])
    for k, v in items:
        if not isinstance(v, dict):
            continue
        rid = v.get("request_id", k)
        ent = {kk: v.get(kk) for kk in REQUEST_FIELDS if kk in v}
        if isinstance(ent.get("xreq"), dict):
            ent["xreq"] = {"counts": ent["xreq"].get("counts") or {}}          # the counts are the contract; the examples are prose
        per[str(rid)] = ent
    att = d.get("attribution") or {}
    return {"verdict": d.get("verdict"), "requests": per, "xreq_total": d.get("xreq_total"),
            "attribution": {k: att.get(k) for k in ATTRIBUTION_FIELDS if k in att},
            "oracle": {k: (d.get("oracle") or {}).get(k) for k in ("digest", "instances", "n_input_nodes")},
            "fold_record_pins": pins}


def _pins_of_record(row: resolver.Row) -> dict | None:
    if not row.has("global_match.json"):
        return None
    fl = ((row.json("global_match.json").get("impl") or {}).get("fold_load") or {})
    for _dir, notes in fl.items():
        if isinstance(notes, dict):
            return {k: v for k, v in notes.items() if k.endswith("sha256")}
    return None


def expected(row: resolver.Row) -> dict:
    return lift(row.json("match_decomp.json"), pins=_pins_of_record(row))


def _derived_dirs(root) -> list[str]:
    out = sorted(d for d in glob.glob(os.path.join(str(root), "build_request*")) if os.path.isfile(os.path.join(d, "instances.json.gz")))
    if not out:
        out = sorted(d for d in glob.glob(os.path.join(str(root), "build", "rank*", "build_request*")) if os.path.isfile(os.path.join(d, "instances.json.gz")))
    return out


def actual(row: resolver.Row) -> dict:
    cand = row.candidate("match_decomp.json")
    if cand is not None:                                   # the candidate's decomposition, lifted field for field like the record's
        d = json.load(open(cand))
        pins = None
        gm = row.candidate("global_match.json")
        if gm is not None:
            fl = ((json.load(open(gm)).get("impl") or {}).get("fold_load") or {})
            pins = next(({k: v for k, v in n.items() if k.endswith("sha256")} for n in fl.values() if isinstance(n, dict)), None)
        return dict(lift(d, pins=pins), source="candidate")
    # the fold record of record: program.json + instances.jsonl beside each other under match/ (not frozen in files.toml: multi-GB;
    # tied to the record by the sha256 pins of global_match.json, compared below)
    pj = row.path("match/program.json", verify=False)
    inst = row.path("match/instances.jsonl", verify=False)
    match_dir = os.path.dirname(str(inst))
    if os.path.dirname(str(pj)) != match_dir:
        raise resolver.NotResolvable(f"{row.key}: match/program.json and match/instances.jsonl resolve to different directories ({pj} / {inst})")
    progs = row.programs_root()
    if progs is None:
        raise resolver.NotResolvable(f"{row.key}: no request Programs (build_request*/instances.json.gz) reachable here")
    dirs = _derived_dirs(progs)
    from verity_vllm.check.match import resolve_decomp as RD
    doc = RD.decompose(match_dir, dirs)
    notes = doc.get("fold_load") or {}
    pins = {k: v for k, v in notes.items() if k.endswith("sha256")} or None
    out = dict(lift(doc, pins=pins), source="recomputed: observe.iter_observations -> resolve -> resolve_decomp (batch_decomp.Projection / compare.compare_steps wrapped)")
    out["log_state"] = doc.get("log_state")
    out["faults"] = (doc.get("attribution") or {}).get("faults")
    out["seconds"] = doc.get("seconds")
    return out


def main(argv: list[str] | None = None) -> int:
    """`python -m tests.regression.checks.decomp_hashes <row-key> [--out FILE]`: expected vs actual through the resolver, differences named."""
    import argparse
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("row")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    row = next(r for r in resolver.rows() if r.key == a.row or str(r.number) == a.row)
    exp = expected(row)
    act = actual(row)
    diff = resolver.diff(exp, act)
    res = {"row": row.key, "equal": not diff, "differences": diff, "actual": act, "expected": exp}
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, default=str)
    print(json.dumps({"row": row.key, "equal": not diff, "n_differences": len(diff), "differences": diff[:30],
                      "source": act.get("source"), "log_state": act.get("log_state"), "faults": act.get("faults"), "seconds": act.get("seconds")}, default=str))
    return 0 if not diff else 1


if __name__ == "__main__":
    raise SystemExit(main())
