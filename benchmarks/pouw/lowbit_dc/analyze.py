"""fpx's records into the exactness tables (note:20261017T0643Z-draft-lowbit-dc).

  python analyze.py OUT [--json FILE]     # OUT holds fpx's records.jsonl, windows.jsonl, summary.json; markdown to stdout

Per chain config: the first K whose partials an FP32 can't hold (first_nonrep), the first K with a wrong output
(first_inexact), bad_rep (wrong while every partial was representable: must be 0 for "exact iff representable"), and
how many outputs a chain of one FP32 add per MMA, rounded toward zero, reproduces at every K (rz_all). Per window test:
the largest gap at which every output is exact, and what happens past it. Per emu8 route: wrong accumulators and words.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def per_config(records: list[dict]) -> list[dict]:
    by = defaultdict(list)
    for r in records:
        by[r["cfg"]].append(r)
    rows = []
    for cfg, rs in sorted(by.items()):
        rs.sort(key=lambda r: r["K"])
        head = {k: rs[0][k] for k in ("cfg", "fmt", "grid", "s", "dist", "c0", "c0bits", "sa", "sb", "unit_num", "unit_e",
                                      "control")}
        nonrep = [r["K"] for r in rs if r["rep"] < r["n"]]
        inexact = [r["K"] for r in rs if r["inexact"] > 0]
        rows.append(head | {
            "K_max": rs[-1]["K"],
            "first_nonrep": nonrep[0] if nonrep else None,
            "first_inexact": inexact[0] if inexact else None,
            "bad_rep": sum(r["bad_rep"] for r in rs),
            "rz_all": all(r["rz"] == r["n"] for r in rs),
            "rn_all": all(r["rn"] == r["n"] for r in rs),
            "rz_frac": sum(r["rz"] for r in rs) / max(1, sum(r["n"] for r in rs)),
            "negzero": sum(r["negzero"] for r in rs),
            "bad_word": rs[-1]["bad_word"],
            "bad_acc": rs[-1]["bad_acc"],
            "maxerr": max(r["maxerr"] for r in rs),
        })
    return rows


def group_chains(rows: list[dict]) -> list[dict]:
    """One line per (format, grid, scales, C0) over its distributions."""
    by = defaultdict(list)
    for r in rows:
        if r["fmt"] in ("e4m3", "mxf4", "nvf4") and not r["control"]:
            by[(r["fmt"], r["grid"], r["s"], r["sa"], r["sb"], r["c0"])].append(r)
    out = []
    for (fmt, grid, s, sa, sb, c0), rs in sorted(by.items()):
        fi = [r["first_inexact"] for r in rs if r["first_inexact"] is not None]
        fn = [r["first_nonrep"] for r in rs if r["first_nonrep"] is not None]
        out.append({
            "fmt": fmt, "grid": grid, "s": s, "sa": sa, "sb": sb, "c0": c0, "c0bits": rs[0]["c0bits"],
            "unit": f"{rs[0]['unit_num']}*2^{rs[0]['unit_e']}", "configs": len(rs), "K_max": max(r["K_max"] for r in rs),
            "first_nonrep": min(fn) if fn else None, "first_inexact": min(fi) if fi else None,
            "bad_rep": sum(r["bad_rep"] for r in rs), "rz_all": all(r["rz_all"] for r in rs),
            "rn_all": all(r["rn_all"] for r in rs), "rz_frac": min(r["rz_frac"] for r in rs),
            "negzero": sum(r["negzero"] for r in rs),
        })
    return out


def windows(win: list[dict]) -> list[dict]:
    by = defaultdict(list)
    for w in win:
        by[(w["fmt"], w["test"])].append(w)
    out = []
    for (fmt, test), ws in sorted(by.items()):
        ws.sort(key=lambda w: w["gap"])
        exact_gaps = [w["gap"] for w in ws if w["exact"] == w["n"]]
        bad = [w for w in ws if w["exact"] < w["n"]]
        out.append({
            "fmt": fmt, "test": test, "gaps": [ws[0]["gap"], ws[-1]["gap"]],
            "max_exact_gap": max((g for g in exact_gaps if all(x["exact"] == x["n"] for x in ws if x["gap"] <= g)),
                                 default=None),
            "first_bad_gap": bad[0]["gap"] if bad else None,
            "past": {k: sum(w[k] for w in bad) for k in ("n", "exact", "rz", "rn", "dropped", "other")},
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    summary = json.loads((a.out / "summary.json").read_text()) if (a.out / "summary.json").exists() else {}
    rows = per_config(load(a.out / "records.jsonl"))
    chains = group_chains(rows)
    wins = windows(load(a.out / "windows.jsonl"))
    emu = [r for r in rows if r["fmt"] in ("k3b", "k4b", "k3z", "k4z")]
    controls = [r for r in rows if r["control"]]
    print(f"# fpx on {summary.get('device', '?')} (cc {summary.get('cc', '?')})\n")
    print(f"summary: {json.dumps(summary)}\n")
    print("## Chains (one line per format, grid, scales, C0; over its distributions)\n")
    print("| fmt | grid | s | sa | sb | C0 | unit | configs | K max | first non-rep K | first inexact K | bad_rep | "
          "RZ chain all | min RZ frac | -0 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in chains:
        print(f"| {c['fmt']} | {c['grid']} | {c['s']} | {c['sa']} | {c['sb']} | {c['c0']} {c['c0bits']} | {c['unit']} | "
              f"{c['configs']} | {c['K_max']} | {c['first_nonrep']} | {c['first_inexact']} | {c['bad_rep']} | "
              f"{c['rz_all']} | {c['rz_frac']:.4f} | {c['negzero']} |")
    print("\n## Windows\n")
    print("| fmt | test | gaps | largest all-exact gap | first bad gap | past it: n, exact, RZ, RN, dropped, other |")
    print("|---|---|---|---|---|---|")
    for w in wins:
        p = w["past"]
        print(f"| {w['fmt']} | {w['test']} | {w['gaps']} | {w['max_exact_gap']} | {w['first_bad_gap']} | "
              f"{p['n']}, {p['exact']}, {p['rz']}, {p['rn']}, {p['dropped']}, {p['other']} |")
    print("\n## emu8 routes (every k32 prefix word)\n")
    print("| route | dist | K | bad accumulators | bad words |")
    print("|---|---|---|---|---|")
    for r in emu:
        if not r["control"]:
            print(f"| {r['fmt']} | {r['dist']} | {r['K_max']} | {r['bad_acc']} | {r['bad_word']} |")
    print("\n## Controls (no MMA issued; each must count wrong outputs)\n")
    for r in controls:
        caught = r["bad_rep"] + r["bad_word"] + r["bad_acc"] + (0 if r["fmt"].startswith("k") else
                                                                  int(r["first_inexact"] is not None))
        print(f"- {r['fmt']} {r['dist']} C0 {r['c0']}: {'caught' if caught else 'MISSED'}")
    if a.json:
        a.json.write_text(json.dumps({"summary": summary, "chains": chains, "windows": wins, "emu": emu,
                                      "controls": controls, "configs": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
