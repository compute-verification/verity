"""The staged statements lincheck_modes.sh checks, and the summary of its selftest outputs.

    python3 lincheck_modes.py list STAGE_CACHE            # "<dir> <n> <K>" per staged statement, smallest first
    python3 lincheck_modes.py summary OUT/*.out > summary.jsonl

A summary line per statement and FC_LINCHECK mode: the verdict and reason on each proof set (honest three times, z_partial,
round), the median verify_s over the honest verifications, and the median over them of each phase summed over the reps.
"""
import json
import re
import statistics
import sys
from pathlib import Path


def statements(cache: Path) -> list[tuple[Path, int, int]]:
    out = []
    for d in cache.iterdir():
        if (d / "rec.json").exists():
            r = json.loads((d / "rec.json").read_text())
            out.append((d, int(r["stage"]["n"]), int(re.search(r"K=(\d+)", r["definition"]).group(1))))
    return sorted(out, key=lambda x: (x[1], x[2]))


def summary(path: Path) -> list[dict]:
    neg = [json.loads(ln.split("\t", 1)[1]) for ln in path.read_text().splitlines() if ln.startswith("NEG\t")]
    case = next((c for c in neg if c.get("case") == "lincheck_modes_agree"), None)
    if case is None:
        return [{"file": path.name, "pass": False, "why": "no lincheck_modes_agree record"}]
    rows = {}
    for s in case.get("runs") or []:
        for m in s["modes"]:
            r = rows.setdefault(m["mode"], {"file": path.name, "mode": m["mode"], "verdicts": [], "honest": []})
            r["verdicts"].append([s["proofs"], m["accepted"], m["why"]])
            if s["proofs"] == "honest":
                ph = {}
                for p in m.get("phases") or []:
                    for k, v in p.items():
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            ph[k] = ph.get(k, 0) + v
                    for k, v in (p.get("lincheck_comb") or {}).items():
                        ph["comb." + k] = ph.get("comb." + k, 0) + v
                r["honest"].append({"verify_s": m.get("verify_s"), **ph})
    out = []
    for r in rows.values():
        h = r.pop("honest")
        keys = sorted({k for x in h for k in x})
        r["median"] = {k: round(statistics.median([x.get(k) or 0 for x in h]), 4) for k in keys}
        r["n_honest"] = len(h)
        r["case_pass"] = case.get("pass")
        out.append(r)
    return out


if __name__ == "__main__":
    if sys.argv[1] == "list":
        for d, n, k in statements(Path(sys.argv[2])):
            print(d, n, k)
    else:
        for f in sys.argv[2:]:
            for r in summary(Path(f)):
                print(json.dumps(r))
