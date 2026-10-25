"""The `shared` vectors of #4066's description-sized difftest data, one row each, under every build's `nci-check`.

    python shared.py DATA --exe NAME PATH [--exe NAME PATH ...] --out FILE

DATA is the `desc_difftest` directory `scripts/desc_difftest_vectors.py` writes. A row is a vector whose declaration is
one of `shared_decl`'s (`NciSharedDecl…`): the Python reference's verdict (`check_desc`, gate by gate), and per build
`nci-check --policy` on the instance (`NciCheck.Desc`) and `nci-check --policy --unrolled` on its expansion
(`NciCheck.check`). `desc_ok` is the build's Desc verdict equal to the reference's, `unrolled_ok` its unrolled one's.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path


FIELDS = ("error", "gates", "ius", "complies", "failure")


def fields(v: object) -> dict:
    """`v` as `NciCheck.Difftest.fields` compares it: those keys, a missing one null."""
    return {k: v.get(k) for k in FIELDS} if isinstance(v, dict) else {"error": v}


def verdict(exe: str, *args: str) -> dict:
    r = subprocess.run([exe, "--policy", *args], capture_output=True, text=True)
    if r.returncode != 0:
        return {"error": f"rc {r.returncode}: {r.stderr[-500:]}"}
    return fields(json.loads(r.stdout))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("--exe", nargs=2, action="append", metavar=("NAME", "PATH"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        lines = zip(*(open(a.data / f"{n}.jsonl") for n in ("instances", "expanded", "python")), strict=True)
        for i, (li, le, lp) in enumerate(lines):
            inst = json.loads(li)
            m = re.search(r'"NciSharedDecl(\w+)"', inst.get("program") or "")
            if not m:
                continue
            ip, ep = Path(tmp) / f"{i}.json", Path(tmp) / f"{i}.expanded.json"
            ip.write_text(li)
            ep.write_text(le)
            python = fields(json.loads(lp))
            row = {"line": i, "decl": m.group(1), "owner": inst["owner"], "theta": inst["theta"], "python": python}
            for name, exe in a.exe:
                desc = verdict(exe, str(ip))
                unrolled = None if json.loads(le) is None else verdict(exe, "--unrolled", str(ep))
                row[name] = {"desc": desc, "unrolled": unrolled, "desc_ok": desc == python,
                             "unrolled_ok": None if unrolled is None else unrolled == python}
            rows.append(row)
    names = [n for n, _ in a.exe]
    summary = {n: {"rows": len(rows), "desc_differs": sum(not r[n]["desc_ok"] for r in rows),
                   "unrolled_differs": sum(r[n]["unrolled_ok"] is False for r in rows),
                   "desc_accepts_where_reference_refuses": sum(
                       r[n]["desc"]["complies"] is True and r["python"]["complies"] is False for r in rows)}
               for n in names}
    a.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n")
    for r in rows:
        cells = "  ".join(f"{n}: {json.dumps(r[n]['desc']['ius'])} complies={r[n]['desc']['complies']} "
                          f"{'ok' if r[n]['desc_ok'] else 'DIFFERS'}" for n in names)
        print(f"line {r['line']} {r['decl']} owner {r['owner']}: reference {json.dumps(r['python']['ius'])} "
              f"complies={r['python']['complies']}  {cells}")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
