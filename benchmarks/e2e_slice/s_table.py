"""The checkpoint-spacing table from sweep.sh's outputs: for each window (OUT/<fold>-s<S>/summary.json, from any number of
sweep run dirs) the statement's ANDs and lowered size, its C-Flock --zk prove and verify, and the rounds it passed; the
receipt bytes a call at o_proj (512 MiB / S). Markdown on stdout.

    python3 benchmarks/e2e_slice/s_table.py RUN_OUT...
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def rows(outs: list[str]) -> list[dict]:
    got = []
    for o in outs:
        for d in sorted(Path(o).glob("*-s*/")):
            m = re.fullmatch(r"(duo|xin)-s(\d+)", d.name)
            if not m or not (d / "summary.json").exists():
                continue
            s = json.loads((d / "summary.json").read_text())
            p, c, w = s["prover_per_round_s"], s["challenger_per_round_s"], s["window"]
            got.append({"fold": m[1], "S": int(m[2]), "ands": s["circuit"]["ands"], "nnz": s["circuit"]["unit_nnz"],
                        "circuit_gb": s["circuit"]["circuit_bytes"] / 1e9, "prove_s": p.get("prove_s"),
                        "prove_cpu_s": p.get("prove_cpu_s"), "prove_gb": (p.get("prove_maxrss_mb") or 0) / 1024,
                        "stage_s": p.get("stage_s"), "proof_mb": (c.get("proof_bytes") or 0) / 1e6,
                        "verify_s": c.get("session_verify_s"), "verify_cpu_s": c.get("verify_cpu_s"),
                        "passed": f"{w['passed']}/{w['rounds']}", "accepted": s["accepted"], "dir": str(d)})
    return sorted(got, key=lambda r: (r["fold"], r["S"]))


def main(outs: list[str]) -> int:
    rs = rows(outs)
    print("| fold | S | receipts MiB a call | ANDs | unit nnz | circuit GB | prove s | prove CPU s | prove GB | stage s |"
          " proof MB | verify s | verify CPU s | rounds passed |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    f = lambda v, n=2: "-" if v is None else f"{v:.{n}f}"  # noqa: E731
    for r in rs:
        print(f"| {r['fold']} | {r['S']} | {512 / r['S']:.0f} | {r['ands']:,} | {r['nnz']:,} | {f(r['circuit_gb'])} | "
              f"{f(r['prove_s'], 1)} | {f(r['prove_cpu_s'], 1)} | {f(r['prove_gb'], 1)} | {f(r['stage_s'], 1)} | "
              f"{f(r['proof_mb'])} | {f(r['verify_s'])} | {f(r['verify_cpu_s'], 1)} | {r['passed']} |")
    return 0 if rs and all(r["accepted"] for r in rs) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
