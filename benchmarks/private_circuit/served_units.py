"""Every proof unit class of a served row, ranked by gates, from the Boolean export of the served rows
(`verity_flock.boolean_export`, kind ``flock-boolean-export/v1``: ``commitments.json`` is each Call's cut under the no-recompute
`Q_word` v1 {16, 32}, each unit class with its AND / XOR / NOT from C-Flock's pieces).

    python benchmarks/private_circuit/served_units.py EXPORT_DIR ROW_KEY --out DIR

A unit's gates are its AND + XOR + NOT under the export's rule (each distinct linear form a chain of |wires| - 1 XORs and a NOT
when its constant is 1, no sharing across a type's calls): the 2-input gates of its netlist, an upper bound on the universal
unit's G. Its input bits are its reads' (other units' committed values and its Call's inputs) where the record has a part graph,
else its Call's parameters' bits. ``DIR/units.json``: every class, largest first, and the largest class's record.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("export")
    ap.add_argument("row")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ex = Path(a.export)
    recs = json.loads((ex / "commitments.json").read_text())
    C = recs["records"]
    row = json.loads((ex / f"{a.row}.boolean.json").read_text())
    call_bits: dict[str, int] = {}
    ids = set()
    for n in row["nodes"]:
        rids = [n["commitment"]] if n.get("commitment") else list((n.get("commitment_by_T") or {}).values())
        params = n.get("params") or []
        bits = sum(p["bits"] if isinstance(p["bits"], int) else max(p["bits"]) for p in params) if isinstance(params, list) else None
        for r in rids:
            ids.add(r)
            if bits is not None:
                call_bits[r] = bits

    def reads(u: dict):
        if "reads" in u:
            return u["reads"]
        g = C.get(u.get("graph"))
        for v in (g or {}).get("units", []):
            if v["head"] == u["head"] and v.get("word_gates") == u.get("word_gates") and "reads" in v:
                return v["reads"]
        return None

    rows = []
    for rid in sorted(ids):
        for u in C[rid]["units"]:
            rd = reads(u)
            rows.append({"gates": u["and"] + u["xor"] + u["not"], "and": u["and"], "xor": u["xor"], "not": u["not"], "call": rid,
                         "head": u["head"], "kind": u["kind"], "scope": C[rid]["scopes"][u["scope"]] if C[rid].get("scopes") else None,
                         "word_gates": u["word_gates"], "out_bits": u["out_bits"], "per_call": u["count"],
                         "in_bits": sum(r[2] for r in rd) if rd else None,
                         "in_bits_source": "reads" if rd else "call parameters (the unit is the whole Call)" if C[rid]["units_per_call"] == 1 else None,
                         "not_yet_lowered": u.get("not_yet_lowered") or {}})
            if rows[-1]["in_bits"] is None and C[rid]["units_per_call"] == 1:
                rows[-1]["in_bits"] = call_bits.get(rid)
    rows.sort(key=lambda r: -r["gates"])
    distinct, seen = [], set()
    for r in rows:
        fam = (re.sub(r"\{.*", "", r["call"]), r["head"], r["kind"])
        if fam not in seen:
            seen.add(fam)
            distinct.append(r)
    top = rows[0]
    rec = {k: v for k, v in C[top["call"]].items() if k not in ("committed", "structure")}
    doc = {"row": a.row, "query": recs.get("query"), "code": recs.get("code"), "unit_classes": len(rows),
           "largest": top, "largest_record": rec, "top_distinct": distinct[:10], "all": rows}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "units.json").write_text(json.dumps(doc, indent=1))
    print(json.dumps({"largest": top, "top_distinct": distinct[:6]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
