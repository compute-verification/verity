"""Per family, the Programs whose `ReadsExact` (s3) or `agree` failed in one fuzz-TREE directory of cpu.sh fuzz, with the
first few ids and the clause s3 failed on.

    fuzz_tally.py OUT/fuzz-TREE ... [--out TALLY.json]
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path


def failed(x) -> dict | None:
    if isinstance(x, str):
        x = json.loads(x)
    return x.get("fail") if isinstance(x, dict) else None


def tally(d: Path) -> dict:
    out: dict = {}
    for f in sorted(d.glob("*.failures.jsonl.gz")):
        fams: dict = {}
        with gzip.open(f, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                t = fams.setdefault(r.get("family", f.name.split(".")[0]), {"s3": 0, "agree": 0, "both": 0, "ids": [],
                                                                               "clauses": Counter()})
                s3, ag = failed(r.get("s3")), failed(r.get("agree"))
                t["s3"] += bool(s3)
                t["agree"] += bool(ag)
                t["both"] += bool(s3 and ag)
                if s3:
                    t["clauses"][s3.get("clause")] += 1
                if (s3 or ag) and len(t["ids"]) < 8:
                    t["ids"].append(r["id"])
        for fam, t in fams.items():
            if t["s3"] or t["agree"]:
                out[fam] = {**t, "clauses": dict(t["clauses"])}
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    res = {d.name: tally(d) for d in a.dirs}
    for name, fams in res.items():
        tot = {k: sum(t[k] for t in fams.values()) for k in ("s3", "agree", "both")}
        print(f"{name}: {tot}")
        for fam, t in sorted(fams.items()):
            print(f"  {fam}: s3 {t['s3']}, agree {t['agree']}, both {t['both']}; {t['clauses']}; e.g. {', '.join(t['ids'][:4])}")
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
