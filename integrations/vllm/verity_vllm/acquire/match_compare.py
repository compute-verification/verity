"""The GPU gate's comparison: a Match record produced with the plan's hooks installed (ACQUIRE_ENGINE=v2) against the v1 Match record
of the same row -- field for field what the regression harness lifts (`tests/regression/checks/{global_match_checks,decomp_hashes}`),
plus the fold's canonical digest and the instances themselves.

    python -m verity_vllm.acquire.match_compare --staged <row dir> --v2 <row dir> --out compare.json

Compared: tokens.json (capture / control: generated ids per request); match/instances.jsonl (header digest + n_instances, line count, a
streamed sha256 of every instance line -- the decomposition hashes at the finest grain); match_decomp.json (verdict, oracle digest and
count, xreq_total, per-request canonical shas / counts); global_match.json (GM-01's lifted verdict fields when BOTH records have it);
match/gates.json (verdict + per-gate status).  The v2 record's own acquisition documents (acquisition_plan.json, site_compare.json,
acquisition_observed.json) are summarised beside: occurrences per site kind against the engine steps of the run.  Exit 0 iff every
comparable field is equal and the v2 plan was complete with no unexplained site difference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def _json(p: Path) -> Any | None:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def tokens_of(doc: Any) -> dict[str, list[int]] | None:
    if not isinstance(doc, dict):
        return None
    return {str(r["request_id"]): list(r.get("generated_token_ids") or []) for r in (doc.get("requests") or [])}


def instances_summary(p: Path) -> dict[str, Any] | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    n = 0
    header = None
    with p.open("rb") as f:
        for line in f:
            if header is None:
                try:
                    header = json.loads(line)
                except ValueError:
                    header = {}
                continue
            h.update(line.rstrip(b"\n"))
            h.update(b"\n")
            n += 1
    return {"digest": (header or {}).get("digest"), "n_instances": (header or {}).get("n_instances"), "schema": (header or {}).get("schema"),
            "lines": n, "body_sha256": h.hexdigest(), "bytes": p.stat().st_size}


def decomp_lift(d: Any) -> dict[str, Any] | None:
    if not isinstance(d, dict):
        return None
    reqs = d.get("requests")
    per: dict[str, Any] = {}
    items = reqs.items() if isinstance(reqs, dict) else enumerate(reqs or [])
    for k, v in items:
        if isinstance(v, dict):
            per[str(v.get("request_id", k))] = {kk: v.get(kk) for kk in ("canonical_sha256", "canonical_sha256_a", "canonical_sha256_b", "derived_digest", "xreq", "attributed",
                                                                          "unattributed", "n_instances", "LP", "T", "steps", "token_selects") if kk in v}
    return {"verdict": d.get("verdict"), "xreq_total": d.get("xreq_total"), "oracle": d.get("oracle"), "attribution": d.get("attribution"),
            "within_step_criterion": d.get("within_step_criterion"), "engine_steps": d.get("engine_steps"), "requests": per, "reasons": d.get("reasons")}


def gm_lift(gm: Any) -> dict[str, Any] | None:
    if not isinstance(gm, dict):
        return None
    out = {k: gm.get(k) for k in ("verdict", "failed_checks", "eos_truncated_requests", "rules_version", "within_step_criterion", "chronology_unit")}
    out["checks"] = {c["id"]: {"status": c.get("status"), "n_problems": c.get("n_problems", len(c.get("problems") or []))}
                     for c in (gm.get("checks") or []) if isinstance(c, dict) and c.get("id")}
    x = gm.get("x09") or {}
    out["x09"] = {"verdict": x.get("verdict"), "xreq_total": x.get("xreq_total"), "within_step_criterion": x.get("within_step_criterion"),
                  "unattributed": ((x.get("attribution") or {}).get("unattributed"))}
    alt = gm.get("alternate_verdict") or {}
    out["alternate_verdict"] = {"criterion": alt.get("criterion"), "verdict": alt.get("verdict"), "failed_checks": alt.get("failed_checks")}
    out["requests_n"] = len(gm.get("requests") or [])
    return out


def gates_lift(g: Any) -> dict[str, Any] | None:
    if not isinstance(g, dict):
        return None
    return {"verdict": g.get("verdict"), "gates": {k: (v or {}).get("status") for k, v in (g.get("gates") or {}).items()}}


def diff(a: Any, b: Any, path: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out: list[str] = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: {'only staged' if k in a else 'only v2'}")
            else:
                out += diff(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if a != b:
            return [f"{path}: lists differ (len {len(a)} vs {len(b)})"]
        return []
    return [] if a == b else [f"{path}: {str(a)[:80]!r} != {str(b)[:80]!r}"]


def compare(staged: Path, v2: Path) -> dict[str, Any]:
    parts: dict[str, Any] = {}
    problems: list[str] = []
    for arm in ("capture", "control"):
        ta, tb = tokens_of(_json(staged / "match" / arm / "tokens.json")), tokens_of(_json(v2 / "match" / arm / "tokens.json"))
        if ta is None or tb is None:
            parts[f"tokens_{arm}"] = {"comparable": False, "staged": ta is not None, "v2": tb is not None}
            continue
        neq = sorted(r for r in set(ta) | set(tb) if ta.get(r) != tb.get(r))
        parts[f"tokens_{arm}"] = {"comparable": True, "requests": len(ta), "equal": not neq, "unequal_requests": neq[:8]}
        if neq:
            problems.append(f"tokens ({arm}): {len(neq)} request(s) differ")
    # the cross-arm equality that the v1 record itself required (control tokens == capture tokens): the staged control against the v2 capture
    tc, tv = tokens_of(_json(staged / "match" / "control" / "tokens.json")), tokens_of(_json(v2 / "match" / "capture" / "tokens.json"))
    if tc is not None and tv is not None:
        neq = sorted(r for r in set(tc) | set(tv) if tc.get(r) != tv.get(r))
        parts["tokens_staged_control_vs_v2_capture"] = {"equal": not neq, "unequal_requests": neq[:8]}
        if neq:
            problems.append(f"staged control tokens != v2 capture tokens for {len(neq)} request(s)")
    ia, ib = instances_summary(staged / "match" / "instances.jsonl"), instances_summary(v2 / "match" / "instances.jsonl")
    parts["instances"] = {"staged": ia, "v2": ib}
    if ia and ib:
        d = diff({k: ia[k] for k in ("digest", "n_instances", "lines", "body_sha256")}, {k: ib[k] for k in ("digest", "n_instances", "lines", "body_sha256")})
        parts["instances"]["equal"] = not d
        parts["instances"]["differences"] = d
        problems += [f"instances.jsonl {x}" for x in d]
    else:
        parts["instances"]["comparable"] = False
    da, db = decomp_lift(_json(staged / "match_decomp.json")), decomp_lift(_json(v2 / "match_decomp.json"))
    parts["match_decomp"] = {"staged": da is not None, "v2": db is not None}
    if da and db:
        # `seconds` / paths never compared; the lifted fields only
        d = diff(da, db)
        parts["match_decomp"].update({"equal": not d, "differences": d[:40], "verdicts": [da["verdict"], db["verdict"]], "oracle_digests": [(da.get("oracle") or {}).get("digest"), (db.get("oracle") or {}).get("digest")]})
        problems += [f"match_decomp {x}" for x in d]
    ga, gb = gm_lift(_json(staged / "global_match.json")), gm_lift(_json(v2 / "global_match.json"))
    parts["global_match"] = {"staged": ga is not None, "v2": gb is not None, "v2_lifted": gb}
    if ga and gb:
        d = diff(ga, gb)
        parts["global_match"].update({"equal": not d, "differences": d[:40]})
        problems += [f"global_match {x}" for x in d]
    xa, xb = gates_lift(_json(staged / "match" / "gates.json")), gates_lift(_json(v2 / "match" / "gates.json"))
    parts["gates"] = {"staged": xa, "v2": xb}
    if xa and xb:
        d = diff(xa, xb)
        parts["gates"].update({"equal": not d, "differences": d})
        problems += [f"gates {x}" for x in d]
    # the v2 record's acquisition documents
    acq: dict[str, Any] = {}
    cap = v2 / "match" / "capture"
    for name in ("acquisition_plan.json", "site_compare.json", "acquisition_observed.json", "runtime_tree.json"):
        acq[name] = (cap / name).exists()
    sc = _json(cap / "site_compare.json")
    if sc:
        acq["site_compare"] = {"identical": sc.get("identical"), "counts": sc.get("counts"), "unexplained": sc.get("unexplained"), "plan_complete": sc.get("plan_complete"),
                               "differences": sc.get("differences")}
        if not sc.get("plan_complete"):
            problems.append("v2 plan incomplete")
        if sc.get("unexplained"):
            problems.append(f"v2 site compare: {len(sc['unexplained'])} unexplained difference(s)")
    ob = _json(cap / "acquisition_observed.json")
    if ob:
        by_kind: dict[str, dict[str, Any]] = {}
        for s in ob.get("sites") or []:
            k = by_kind.setdefault(s["kind"], {"sites": 0, "occurrences": 0, "min": None, "max": None, "non_tensor": 0})
            k["sites"] += 1
            k["occurrences"] += s["occurrences"]
            k["min"] = s["occurrences"] if k["min"] is None else min(k["min"], s["occurrences"])
            k["max"] = s["occurrences"] if k["max"] is None else max(k["max"], s["occurrences"])
            k["non_tensor"] += s.get("non_tensor", 0)
        tok = _json(cap / "tokens.json") or {}
        acq["observed"] = {"by_kind": by_kind, "engine_steps": tok.get("engine_steps"), "plan_digest": ob.get("plan_digest"),
                           "zero_occurrence_sites": [[s["module"], s["kind"], s["slot"]] for s in ob.get("sites") or [] if s["occurrences"] == 0][:20]}
    parts["acquisition"] = acq
    return {"schema": "verity-vllm/acquisition-match-compare/v1", "staged": str(staged), "v2": str(v2), "equal": not problems, "problems": problems, "parts": parts}


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staged", required=True)
    ap.add_argument("--v2", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    r = compare(Path(a.staged), Path(a.v2))
    Path(a.out).write_text(json.dumps(r, indent=1, default=str) + "\n")
    print(f"[match_compare] equal={r['equal']} problems={len(r['problems'])} -> {a.out}", flush=True)
    for p in r["problems"][:30]:
        print(f"[match_compare]   {p}", flush=True)
    for k, v in r["parts"].items():
        s = json.dumps(v, default=str)
        print(f"[match_compare] {k}: {s[:400]}", flush=True)
    return 0 if r["equal"] else 1


if __name__ == "__main__":
    sys.exit(_main())
