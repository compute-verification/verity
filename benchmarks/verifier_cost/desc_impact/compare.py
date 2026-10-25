"""The impact study's verdict table: each recorded and rerun results.json (run.py measure's or nci_glm.py measure's) read
as one verdict per case, and whether it is the same in every one.

    compare.py LABEL=RESULTS.json ... [--out TABLE.json]

A case's verdict is its compliance, failure (or error) and its IUs: the whole list for run.py's rows (one IU per Call),
the IU count with the worst IU's (computed, novel) for nci_glm.py's.  Prints one line per case: each label's verdict,
`same` when all agree, else `DIFFERS`; exit 1 if any case differs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def verdicts(path: Path) -> dict[str, dict]:
    doc = json.loads(path.read_text())
    out = {}
    for r in doc["rows"]:
        if "case" in r:
            out[r["case"]] = {"complies": r.get("complies"), "failure": r.get("failure") or r.get("error"),
                              "ius": r.get("ius"), "worst_iu": r.get("worst_iu"), "novel": r.get("novel")}
        else:
            v = r.get("verdict") or {}
            u = r.get("unrolled") or {}
            out[f"{r['config']}/{r['ir']}"] = {"complies": v.get("complies"), "failure": v.get("failure") or v.get("error"),
                                               "ius": v.get("ius"), "unrolled_agrees": u.get("agrees")}
    return out


def short(v: dict | None) -> str:
    if v is None:
        return "-"
    s = "complies" if v["complies"] else f"fails {json.dumps(v['failure'])}"
    if "novel" in v:
        s += f" ius={v['ius']} worst={v['worst_iu']}"
    else:
        s += f" ius={len(v['ius'] or [])}"
    return s


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results", nargs="+", metavar="LABEL=RESULTS.json")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    runs = {}
    for spec in a.results:
        label, _, p = spec.partition("=")
        runs[label] = verdicts(Path(p))
    cases = sorted({c for v in runs.values() for c in v})
    table, bad = [], 0
    for c in cases:
        got = {label: v.get(c) for label, v in runs.items()}
        present = [{k: x[k] for k in ("complies", "failure", "ius", "worst_iu", "novel") if k in x} for x in got.values() if x]
        same = all(x == present[0] for x in present)
        bad += not same
        table.append({"case": c, "same": same, **got})
        print(f"{c}: {'same' if same else 'DIFFERS'}  " + "  ".join(f"{label}: {short(x)}" for label, x in got.items()))
    summary = {"cases": len(cases), "differ": bad, "labels": list(runs)}
    print(json.dumps(summary))
    if a.out:
        a.out.write_text(json.dumps({"summary": summary, "rows": table}, indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
