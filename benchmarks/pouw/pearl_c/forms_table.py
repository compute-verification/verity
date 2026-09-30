#!/usr/bin/env python3
"""Which hashing forms each panel shape ships, from GPU 1's forms fill (the RTX PRO coordinator's 18:26Z item 3): run.py's
timing rows, each carrying `call_e2e_ms_by_form` (every A-form/forming pair on the same die, in the same process) and, on the
fused 128 x 128 path, `call_e2e_ms_by_epilogue`.

Per (shape, variant), over every row of every chunk and die:
- each pair's median call; the pick is the fastest, except that a pair within TIE (0.5%, above the rows' interquartile spread)
  of the fastest ties it, and among ties the earlier in PREFER wins (forming split before fold, then the A forms in order);
- the epilogue: `w` if its median is under `b`'s by more than TIE, else `b`;
- a pick stands only on rows from at least two dies and only if no row of the shape failed its gate or a consistency count;
  otherwise it is provisional.

  python3 forms_table.py [DIR ...] [--json OUT]   (default DIR: node 2's /workspace/pouw/gpu1-pearlc/forms/*)
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys

TIE = 0.005
PREFER = [f"{a}/{f}" for f in ("split", "fold") for a in ("w,nodes", "s,one", "p,one", "p,fold")]
DEFAULT = "w,nodes/split"


def rows(paths):
    for path in paths:
        check_ok, uuid = None, None
        for line in open(path):
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue
            if j.get("kind") == "env":
                uuid = j.get("uuid")
            elif j.get("kind") == "check":
                check_ok = all(v["pearl_c_ok"] for v in j["variants"].values())
            elif j.get("kind") == "timing":
                yield j, uuid, check_ok


def table(paths) -> dict:
    by: dict = {}
    for j, uuid, check_ok in rows(paths):
        shape = "x".join(map(str, [j.get("m_real", j["shape"][0]), *j["shape"][1:]]))
        e = by.setdefault((shape, j["variant"]), {"forms": {}, "epilogue": {}, "dies": set(), "rows": 0, "bad": 0})
        e["rows"] += 1
        e["dies"].add(uuid)
        e["bad"] += check_ok is False or any(v for v in j.get("consistency", {}).values()) or not j.get("gates", {}).get("pearl_c")
        for f, ms in (j.get("call_e2e_ms_by_form") or {}).items():
            e["forms"].setdefault(f, []).append(ms)
        for f, ms in (j.get("call_e2e_ms_by_epilogue") or {}).items():
            e["epilogue"].setdefault(f, []).append(ms)
    out: dict = {}
    for (shape, variant), e in sorted(by.items()):
        med = {f: statistics.median(xs) for f, xs in e["forms"].items()}
        if not med:
            continue
        best = min(med.values())
        ties = [f for f in PREFER if f in med and med[f] <= best * (1 + TIE)]
        pick = ties[0] if ties else min(med, key=med.get)
        epi = {f: statistics.median(xs) for f, xs in e["epilogue"].items()}
        epi_pick = "w" if "w" in epi and "b" in epi and epi["w"] < epi["b"] * (1 - TIE) else ("b" if epi else None)
        stands = len(e["dies"]) >= 2 and not e["bad"]
        out.setdefault(shape, {})[variant] = {
            "pick": pick, "epilogue": epi_pick, "stands": stands, "rows": e["rows"], "dies": len(e["dies"]), "bad_rows": e["bad"],
            "ms": {f: round(v, 4) for f, v in sorted(med.items())}, "epilogue_ms": {f: round(v, 4) for f, v in sorted(epi.items())},
            "vs_default": round(med[pick] / med[DEFAULT] - 1, 4) if DEFAULT in med else None}
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dirs", nargs="*", default=sorted(glob.glob("/workspace/pouw/gpu1-pearlc/forms/*")))
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    paths = sorted(p for d in a.dirs for p in glob.glob(f"{d}/*.jsonl"))
    t = table(paths)
    if a.json:
        open(a.json, "w").write(json.dumps({"files": len(paths), "shapes": t}, indent=1) + "\n")
    print("| shape | variant | ships | epilogue | vs default | rows (dies) | stands |")
    print("|---|---|---|---|---|---|---|")
    for shape, vs in t.items():
        for v, r in vs.items():
            d = f"{100 * r['vs_default']:+.1f}%" if r["vs_default"] is not None else "—"
            print(f"| {shape} | {v} | `{r['pick']}` | {r['epilogue'] or '—'} | {d} | {r['rows']} ({r['dies']}) | "
                  f"{'yes' if r['stands'] else 'provisional'} |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
