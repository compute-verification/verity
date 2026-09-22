"""(b) [T1, STUB] Fold / decomposition hashes reproduce: per request `canonical_sha256{a,b}`, attribution and XREQ counts of
`match_decomp.json` recomputed by `cb_a.batch_decomp` / `global_match` over the Match record (`match/program.json` 0.3-1.4 GB +
`match/instances.jsonl` 0.6-5.8 GB, pinned by sha256 in `global_match.json.impl.fold_load`; 100-500 s, 10-30 GB RAM).  STRONG; no B=1
row has a decomposition (`match_decomp.json` absent on #11, #39).

Not implemented tonight: the fold records live in the vault core tars (not extracted) and on the live pods; the expected side
(`match_decomp.json` per-request shas and counts) IS lifted so a later `actual` has its contract ready.
"""
from __future__ import annotations

from .. import resolver

NAME = "decomp_hashes"
TIER = "T1"
STRENGTH = "strong"
INPUTS = ("match_decomp.json", "match/program.json", "match/instances.jsonl", "build_request*/instances.json.gz")


def applies(row: resolver.Row) -> str | None:
    return None if row.has("match_decomp.json") else "no match_decomp.json (B=1 rows have no decomposition; FAIL rows never reached it)"


def expected(row: resolver.Row) -> dict:
    d = row.json("match_decomp.json")
    reqs = d.get("requests") if isinstance(d.get("requests"), (dict, list)) else None
    per: dict = {}
    items = reqs.items() if isinstance(reqs, dict) else enumerate(reqs or [])
    for k, v in items:
        if not isinstance(v, dict):
            continue
        rid = v.get("request_id", k)
        per[str(rid)] = {kk: v.get(kk) for kk in ("canonical_sha256", "canonical_sha256_a", "canonical_sha256_b", "derived_digest", "xreq", "attributed", "unattributed") if kk in v}
    return {"verdict": d.get("verdict"), "requests": per, "xreq_total": d.get("xreq_total"),
            "fold_record_pins": {k: v for k, v in ((row.json("global_match.json").get("impl") or {}).get("fold_load") or {}).items() if k.endswith("sha256")} if row.has("global_match.json") else None}


def actual(row: resolver.Row) -> dict:
    cand = row.candidate("match_decomp.json")
    if cand is not None:                                   # the candidate's decomposition, lifted field for field like the record's
        import json
        from types import SimpleNamespace
        d = json.load(open(cand))
        fake = SimpleNamespace(json=lambda rel: d, has=lambda rel: rel == "match_decomp.json", key=row.key)
        return dict(expected(fake), source="candidate")  # type: ignore[arg-type]
    raise NotImplementedError("decomp_hashes: recomputation over the multi-GB fold record is T1 (pod) and not implemented yet")
