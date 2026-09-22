"""Independent recomputation of the manifest v2 headline from the JSON records alone (what CMT-3's coverage checker does).

    python -m verity_capture.bench.interface_manifest_check out/gen/r9/cmt-ref/interface_manifest.json

Reads only `rows[*].{correspondence.evidence_label, candidates[*].{stage, binding_test.status}, measurement.{status, samples}, required}`
and `classes[*].{row, logical_bytes, logical_count}`; re-derives the four statuses with the published rules and the three weightings,
and compares with the embedded `headline.counts`.  Exit code 1 on any disagreement.
"""
from __future__ import annotations

import json
import sys


def derive(row: dict) -> dict:
    corr = row["correspondence"]["evidence_label"] in ("measured-correspondence", "source-inspected")
    impl = any(c["stage"] in ("production", "built", "inherited") and c["binding_test"]["status"] in ("pass", "production", "inherited")
               for c in row["candidates"])
    bench = row["measurement"].get("status") == "measured" and int(row["measurement"].get("samples") or 0) > 0
    return {"correspondence_established": corr, "secure_path_implemented": corr and impl,
            "secure_path_benchmarked": corr and impl and bench, "unresolved": (not corr) or (not impl)}


def recompute(doc: dict) -> dict:
    rows = {r["id"]: r for r in doc["rows"]}
    st_rows = {rid: derive(r) for rid, r in rows.items()}
    keys = ("correspondence_established", "secure_path_implemented", "secure_path_benchmarked", "unresolved")
    acc = {k: {"classes": 0, "bytes": 0.0, "identities": 0} for k in keys}
    for c in doc["classes"]:
        rids = c["row"]
        if not rids or all(not rows[r]["required"] for r in rids):
            continue
        for k in keys:
            hit = any(st_rows[r][k] for r in rids) if k == "unresolved" else all(st_rows[r][k] for r in rids)
            if hit:
                acc[k]["classes"] += 1
                acc[k]["bytes"] += float(c["logical_bytes"])
                acc[k]["identities"] += int(c["logical_count"])
    return {"rows": st_rows, "counts": acc}


def main(path: str) -> int:
    doc = json.load(open(path))
    rc = recompute(doc)
    bad = 0
    for r in doc["rows"]:
        mine = rc["rows"][r["id"]]
        theirs = {k: r["derived_status"][k] for k in mine}
        if mine != theirs:
            bad += 1
            print(f"ROW {r['id']}: recomputed {mine} != embedded {theirs}")
    emb = doc["headline"]["counts"]
    for k, v in rc["counts"].items():
        e = {kk: emb[k][kk] for kk in v}
        if {kk: round(float(x), 3) for kk, x in v.items()} != {kk: round(float(x), 3) for kk, x in e.items()}:
            bad += 1
            print(f"COUNT {k}: recomputed {v} != embedded {e}")
    print(("OK" if not bad else "MISMATCH") + f": {path} — rows {len(doc['rows'])}, classes {len(doc['classes'])}; recomputed headline:")
    for k, v in rc["counts"].items():
        print(f"  {k:28s} classes {v['classes']:3d}  MB {v['bytes'] / 1e6:9.3f}  identities {v['identities']:>13,}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "out/gen/r9/cmt-ref/interface_manifest.json"))
