"""One row of the layers table (`layers.sh`) from a6.py's records: prints the row and writes it through the harness
kit's result. Every number is summed over the row's sessions from their own records: the prover's `prove_total_s` and
`proof_bytes` (flock-circuit's), the session verifier's `verify_s`, and the Lean verifier's (`verity-verify-proof verify`)
seconds and verdict.

    python3 benchmarks/one_stage/layers_row.py ROW A6_OUT --rc RC
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from verity_bench import kit

WHAT = {"flock": "Flock, plain, every unit", "zk": "Flock with zero knowledge, every unit",
        "sampled": "SPOT's one-stage audit (stratified draw, --zk)"}


def sessions(row: str, s: dict) -> list[dict]:
    """Each session's units and numbers, as a6.py's summary holds them for `row`."""
    if row == "flock":
        return [{"name": w["name"], "units": w["units"], "prove_s": w["live"].get("prove_total_s"),
                 "proof_bytes": w["live"].get("proof_bytes"), "verify_s": w.get("verify_s"),
                 "lean_s": w["lean_verify"].get("verify_s"), "lean": w["lean_verify"].get("accepted") is True,
                 "why": w["lean_verify"].get("why")} for w in s.get("windows", [])]
    verdicts = (s.get("audit") or {}).get("verdicts") or {}
    why = (s.get("audit") or {}).get("why") or {}
    return [{"name": m, "units": (s.get("draw_counts") or {}).get(m), "prove_s": t["timing"].get("prove_total_s"),
             "proof_bytes": t["timing"].get("proof_bytes"), "verify_s": t.get("verify_s"), "lean_s": t.get("lean_verify_s"),
             "lean": verdicts.get(f"lean-verify/{m}") is True, "why": why.get(f"lean-verify/{m}")}
            for m, t in sorted((s.get("timing") or {}).items())]


def total(rows: list[dict], key: str) -> float | None:
    got = [r[key] for r in rows]
    return None if not got or any(x is None for x in got) else round(sum(got), 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("row", choices=sorted(WHAT))
    ap.add_argument("a6", type=Path)
    ap.add_argument("--rc", type=int, required=True)
    a = ap.parse_args()
    path = a.a6 / "summary.json"
    s = json.loads(path.read_text()) if path.is_file() else {}
    rows = sessions(a.row, s)
    lean = sum(r["lean"] for r in rows)
    units = total(rows, "units")
    population = s.get("population") or (units if a.row == "flock" else None)
    line = {"row": a.row, "what": WHAT[a.row], "units_proved": units, "population": population, "sessions": len(rows),
            "prove_s": total(rows, "prove_s"), "verify_s": total(rows, "verify_s"), "lean_verify_s": total(rows, "lean_s"),
            "proof_bytes": total(rows, "proof_bytes"), "lean_accepted": f"{lean} of {len(rows)} sessions", "a6_rc": a.rc,
            "audit": {x: (s.get("audit") or {}).get(x) for x in ("accepted", "profile")} if a.row != "flock" else None,
            "not_accepted": {r["name"]: r["why"] for r in rows if not r["lean"]}}
    print("LAYERS ROW " + json.dumps(line, sort_keys=True, default=str))

    w = kit.writer(a.a6, "bench-result/v1")
    w.fingerprint(script="benchmarks/one_stage/layers.sh", row=a.row, members=s.get("members_spec"), k=s.get("k"))
    part = "measured" if rows and len(rows) == lean and a.rc == 0 else "partial"
    for name, unit in (("prove_s", "s"), ("verify_s", "s"), ("lean_verify_s", "s"), ("proof_bytes", "bytes"),
                       ("units_proved", "units")):
        value = line[name]
        w.measurement(f"layers/{a.row}/{name}", value, unit, mark=part, why=None if value is not None else "no session record",
                      inputs=["a6 summary.json"], extra={"sessions": len(rows)})
    ok = bool(rows) and lean == len(rows) and a.rc == 0
    w.validation("passed" if ok else "failed", "benchmarks/one_stage/layers_row.py",
                 f"Lean verifier accepted {lean} of {len(rows)} sessions; a6.py exit {a.rc}", evidence=[])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
