"""[E10 NIGHT 2026-09-12] Compare the committed BOUNDARY rows of two records for one request, leaf by leaf.

The canonical ids carry the record's profile_id (`p=<profile_id>/r=<req>/...`), which changes with the pod / wheel path, so the comparison keys
on the id with the `p=…/` prefix stripped; equality of the per-position `leaf` (the store's leaf hash of the row bytes) is bit-identity of the row.
Weights are compared through `commitments.json#trees` (root of the weights tree).  Prints one line per RU step and a verdict; writes a JSON.

    python -m record_v5.compare_boundary --a <record A> --b <record B> --request <req> --out <json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def leaves(rec: Path, req: str) -> dict[str, dict[str, str]]:
    """Per RU step: stripped canonical id -> sha256(dtype, shape, raw row bytes) when values.bin is present (the leaf hash itself covers the
    canonical id INCLUDING the record's profile_id, so two records of the same bits have different leaves); falls back to the leaf when a record
    has no values.bin (slim archive) -- then only records with the same profile_id can compare identical."""
    out: dict[str, dict[str, str]] = {}
    for idx in sorted((rec / "adapter" / "selective" / "ru" / req).glob("*/index.json")):
        doc = json.loads(idx.read_text())
        vb = idx.parent / "values.bin"
        values = vb.read_bytes() if vb.exists() else None
        rows = {}
        for e in doc["entries"]:
            k = re.sub(r"^p=[0-9a-f]+/", "", e["canonical_id"])
            if values is not None:
                raw = values[e["offset"]: e["offset"] + e["nbytes"]]
                rows[k] = "bytes:" + hashlib.sha256(f"{e['dtype']}|{e['shape']}|".encode() + raw).hexdigest()
            else:
                rows[k] = "leaf:" + e["leaf"]
        out[idx.parent.name] = rows
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--request", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    A, B = Path(args.a), Path(args.b)
    la, lb = leaves(A, args.request), leaves(B, args.request)
    steps = sorted(set(la) | set(lb))
    report = {"schema": "veritor/e10/boundary-compare/v1", "a": str(A), "b": str(B), "request": args.request, "steps": {}, "totals": {"rows_compared": 0, "identical": 0, "different": 0, "only_a": 0, "only_b": 0}}
    for s in steps:
        ra, rb = la.get(s, {}), lb.get(s, {})
        common = set(ra) & set(rb)
        ident = sum(1 for k in common if ra[k] == rb[k])
        diff = sorted(k for k in common if ra[k] != rb[k])
        row = {"rows_a": len(ra), "rows_b": len(rb), "compared": len(common), "identical": ident, "different": len(diff), "only_a": len(set(ra) - set(rb)), "only_b": len(set(rb) - set(ra)), "first_different": diff[:3]}
        report["steps"][s] = row
        for k in ("compared", "identical", "different", "only_a", "only_b"):
            report["totals"]["rows_compared" if k == "compared" else k] += row[k]
        print(f"[compare_boundary] {s}: compared {len(common)} identical {ident} different {len(diff)} only_a {row['only_a']} only_b {row['only_b']}")
    ta = json.loads((A / "adapter" / "selective" / "commitments.json").read_text())["trees"]
    tb = json.loads((B / "adapter" / "selective" / "commitments.json").read_text())["trees"]
    wa, wb = ta.get("weights", {}).get("root"), tb.get("weights", {}).get("root")
    report["weights_root_identical"] = bool(wa and wa == wb)
    t = report["totals"]
    report["verdict"] = "IDENTICAL" if t["different"] == 0 and t["only_a"] == 0 and t["only_b"] == 0 and t["rows_compared"] > 0 and report["weights_root_identical"] else "DIFFERENT"
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print(f"[compare_boundary] {report['verdict']}: {t}; weights root identical {report['weights_root_identical']} -> {args.out}")
    return 0 if report["verdict"] == "IDENTICAL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
