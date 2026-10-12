"""The protocol-kernel gate's L2 half for the filler: each `fill_*` launch of `bench.py --l2-probe`, measured by ncu, writes no
more L2 sectors than its outputs declare (`lts__t_sectors_op_write.sum`) and makes no more atomic sectors than its fold and
queue declare (`lts__t_sectors_op_atom.sum` + `lts__t_sectors_op_red.sum`); there are no partials to declare.

  python3 l2_gate.py DECLARED_JSON NCU_CSV OUT_JSON      exit 0 passed, 7 refused, 3 unmeasured (no ncu rows)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

WRITE, ATOM, RED, DRAM = ("lts__t_sectors_op_write.sum", "lts__t_sectors_op_atom.sum", "lts__t_sectors_op_red.sum",
                          "dram__bytes_write.sum")


def measured(text: str) -> list[dict]:
    """ncu's CSV (details page): one dict of metric values per profiled kernel, in launch order."""
    lines = text.splitlines()
    start = next((i for i, x in enumerate(lines) if x.startswith('"ID"')), None)
    if start is None:
        return []
    by_id: dict[str, dict] = {}
    for r in csv.DictReader(lines[start:]):
        k = by_id.setdefault(r["ID"], {"kernel": r["Kernel Name"].split("(")[0]})
        try:
            k[r["Metric Name"]] = float(r["Metric Value"].replace(",", ""))
        except (KeyError, ValueError):
            continue
    return [by_id[i] for i in sorted(by_id, key=int)]


def judge(declared: dict, rows: list[dict]) -> dict:
    launches = declared["launches"]
    if len(rows) != len(launches):
        return {"status": "unmeasured", "why": f"{len(rows)} profiled launches, {len(launches)} declared"}
    out, bad = [], []
    for d, m in zip(launches, rows):
        w, a = m.get(WRITE), (m.get(ATOM) or 0.0) + (m.get(RED) or 0.0)
        if m["kernel"] != d["kernel"] or w is None:
            return {"status": "unmeasured", "why": f"profiled {m['kernel']} with {sorted(m)}, declared {d['kernel']}"}
        row = {**d, "write_sectors": w, "atom_sectors": a, "dram_write_bytes": m.get(DRAM),
               "write_ratio": w / d["declared_write_sectors"],
               "atom_ratio": a / d["declared_atom_sectors"] if d["declared_atom_sectors"] else (0.0 if a == 0 else float("inf"))}
        out.append(row)
        if w > d["declared_write_sectors"] or a > d["declared_atom_sectors"]:
            bad.append(f"{d['kernel']}: {w:.0f} write / {d['declared_write_sectors']} declared, "
                       f"{a:.0f} atomic / {d['declared_atom_sectors']} declared")
    return {"status": "REFUSED" if bad else "passed", "refusals": bad, "launches": out,
            "max_write_ratio": max(r["write_ratio"] for r in out), "max_atom_ratio": max(r["atom_ratio"] for r in out)}


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    dec, csv_path = Path(sys.argv[1]), Path(sys.argv[2])
    if not dec.exists() or not csv_path.exists():
        res = {"status": "unmeasured", "why": f"missing {dec if not dec.exists() else csv_path}"}
    else:
        res = judge(json.loads(dec.read_text()), measured(csv_path.read_text()))
    Path(sys.argv[3]).write_text(json.dumps(res, indent=1) + "\n")
    print(f"L2 write gate: {res['status']}", *res.get("refusals", []), res.get("why", ""))
    sys.exit({"passed": 0, "REFUSED": 7}.get(res["status"], 3))
