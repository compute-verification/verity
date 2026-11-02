"""AUDIT check (measurement auditor; seconds, no run, nothing published): which te_core ledger records the scorer's
`submissions` scores. Each case is one te-core/submission/v1 record as `te_core.ledger` writes it, built here, passed
through the scorer's own `te_rec_row` and `submissions` beside the launch.py rows it may name.

    TE_CORE_PATH=benchmarks/covert_witness python islands/audit/scorer/ledger_check.py SCORER.py OUT.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def load(path: str):
    spec = importlib.util.spec_from_file_location("scorer_under_audit", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def record(i: int, run: str, direction: str, macs: float, bpb: float, rho: dict) -> dict:
    return {"kind": "submission", "id": f"audit-{i}", "run_id": run, "start": "m100", "mechanism": "audit",
            "agent": direction, "direction": direction, "commit": "", "written_at": "2026-11-02T10:20:00Z",
            "invalid": [], "macs": macs, "val_bpb": bpb, "phi": 0.0, "depth": 0, "rho": rho, "beta": 16.0,
            "reference": False, "tier": 0, "fraction": 1.0, "output": {"e_bytes": 0}}


def main() -> int:
    sc = load(sys.argv[1])
    one = {k: 1.0 for k in sc.KEYS}
    launch_invalid = {"id": "art:launch-invalid", "source": "launch.py", "run": "r-invalid", "valid": False,
                      "direction": "lowrank", "note": "x", "start": "m100"}
    cases = {
        "no launch.py record behind it": record(0, "r-none", "lowrank", 1e15, 1.0, one),
        "names a run that doesn't exist, as a ref- direction": record(1, "r-none-2", "ref-anything", 1e14, 1.0, one),
        "names an invalid launch.py run": record(2, "r-invalid", "lowrank", 1e15, 1.0, one),
        "rho below 1": record(3, "r-none-3", "lowrank", 1e15, 1.0, {k: 0.01 for k in sc.KEYS}),
    }
    out = {}
    for name, rec in cases.items():
        row = sc.te_rec_row(rec, f"art:{rec['id']}", rec["written_at"])
        subs = sc.submissions([row, launch_invalid], {"m100": {}})
        got = [s for s in subs if s.get("te_id") == rec["id"]]
        out[name] = {"scored": bool(got), "reference": got[0]["reference"] if got else None,
                     "rho_10kB_r1": got[0]["rho"].get("10kB/r1") if got else None,
                     "w_star_term": got[0]["macs"] * got[0]["rho"]["10kB/r1"] if got else None}
    Path(sys.argv[2]).write_text(json.dumps({"scorer": sys.argv[1], "cases": out}, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
