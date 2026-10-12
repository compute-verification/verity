"""The filler library's SASS gate: refuses the build unless ptxas reports no spill or stack frame for any kernel, and each
filler kernel (`fill_*`) and PoUW's own `gemm_pearlc_d` carries exactly the tile's tensor-core instructions and nothing
lowered: 16 `QMMA.16832.F32.E4M3.E4M3` (the 64-deep stage's two k32 steps over the warp's 1 x 4 atoms, unrolled), 16
`HMMA.16816.F32.BF16` (the peel's four k16 steps), no other MMA form, no local memory, and no `.FTZ` but the integer
division idiom `F2I.FTZ.U32.TRUNC.NTZ` (the tile index's divisions, as in `gemm_pearlc_d`).

  python3 sass_gate.py PTXAS_LOG SASS OUT_JSON      exit 7 on a refusal
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter

ATOM, PEEL = "QMMA.16832.F32.E4M3.E4M3", "HMMA.16816.F32.BF16"
WANT = {ATOM: 16, PEEL: 16}


def kernels(sass: str) -> dict[str, str]:
    parts = re.split(r"\n\s*Function : (\S+)\n", sass)
    return dict(zip(parts[1::2], parts[2::2]))


def judge(ptxas: str, sass: str) -> tuple[dict, list[str]]:
    bad = [line.strip() for line in ptxas.splitlines() if re.search(r"[1-9]\d* bytes (spill|stack frame)", line)]
    rows = {}
    for name, body in kernels(sass).items():
        if not (name.startswith("fill_") or name == "gemm_pearlc_d"):
            continue
        mma = Counter(re.findall(r"\b([QHIO]MMA[.\w]*)", body))
        ftz = [x for x in re.findall(r"\b([A-Z0-9]+(?:\.[A-Z0-9]+)*\.FTZ[.\w]*)", body) if not x.startswith("F2I.FTZ.U32.TRUNC")]
        local = len(re.findall(r"\b(LDL|STL)\b", body))
        rows[name] = {"mma": dict(mma), "ftz": ftz, "local": local}
        if dict(mma) != WANT:
            bad.append(f"{name}: MMA {dict(mma)}, want {WANT}")
        if ftz:
            bad.append(f"{name}: FTZ instructions {sorted(set(ftz))}")
        if local:
            bad.append(f"{name}: {local} local-memory instructions")
    if not any(n.startswith("fill_") for n in rows):
        bad.append("no fill_ kernel in the SASS")
    return rows, bad


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    with open(sys.argv[1]) as p, open(sys.argv[2]) as q:
        rows, bad = judge(p.read(), q.read())
    with open(sys.argv[3], "w") as f:
        json.dump({"kernels": rows, "refusals": bad}, f, indent=1)
    if bad:
        print("SASS gate REFUSED:\n  " + "\n  ".join(bad), file=sys.stderr)
        sys.exit(7)
    print(f"SASS gate passed: {len(rows)} kernels")
