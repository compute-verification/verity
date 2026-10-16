"""The filler library's SASS gate: refuses the build unless ptxas reports no spill or stack frame for any kernel, each filler
kernel (`fill_*`) and PoUW's own `gemm_pearlc_d` carries exactly the tile's tensor-core instructions and nothing lowered: 16
`QMMA.16832.F32.E4M3.E4M3` (the 64-deep stage's two k32 steps over the warp's 1 x 4 atoms, unrolled), 16
`HMMA.16816.F32.BF16` (the peel's four k16 steps); at the 128 x 128 tile (`fill_*128_*`, a warp's 4 x 4 atoms) 32 and 64;
and in the receipt kernels (`fill_*_3`) 2 x 16 x ceil(RB / 8) `IMMA.16832.S8.S8` more (the sketch's two matrices, its
16 k32 steps a warp, one n8 group per 8 blocks: 32 at 64 x 64, 64 at 128 x 128); no other MMA form, no local memory, and no `.FTZ` but the integer
division idiom `F2I.FTZ.U32.TRUNC.NTZ` (the tile index's divisions, as in `gemm_pearlc_d`); and the protocol-kernel gate's
resource half: `cuobjdump -res-usage` reports `STACK:0` and `LOCAL:0` for every one of them.

  python3 sass_gate.py PTXAS_LOG SASS RES_USAGE OUT_JSON      exit 7 on a refusal
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter

ATOM, PEEL = "QMMA.16832.F32.E4M3.E4M3", "HMMA.16816.F32.BF16"
WANT = {ATOM: 16, PEEL: 16}
WANT128 = {ATOM: 32, PEEL: 64}
SKETCH = "IMMA.16832.S8.S8"


def gated(name: str) -> bool:
    return name.startswith("fill_") or name == "gemm_pearlc_d"


def kernels(sass: str) -> dict[str, str]:
    parts = re.split(r"\n\s*Function : (\S+)\n", sass)
    return dict(zip(parts[1::2], parts[2::2]))


def resources(res: str) -> dict[str, dict[str, int]]:
    """`cuobjdump -res-usage`: each function's REG, STACK, SHARED and LOCAL."""
    out = {}
    for name, line in re.findall(r"Function (\S+):\s*\n\s*(REG:.*)", res):
        out[name] = {k: int(v) for k, v in re.findall(r"\b(REG|STACK|SHARED|LOCAL):(\d+)", line)}
    return out


def judge(ptxas: str, sass: str, res: str) -> tuple[dict, list[str]]:
    bad = [line.strip() for line in ptxas.splitlines() if re.search(r"[1-9]\d* bytes (spill|stack frame)", line)]
    usage = resources(res)
    rows = {}
    for name, body in kernels(sass).items():
        if not gated(name):
            continue
        mma = Counter(SKETCH if m.startswith(SKETCH) else m for m in re.findall(r"\b([QHIO]MMA[.\w]*)", body))
        ftz = [x for x in re.findall(r"\b([A-Z0-9]+(?:\.[A-Z0-9]+)*\.FTZ[.\w]*)", body) if not x.startswith("F2I.FTZ.U32.TRUNC")]
        local = len(re.findall(r"\b(LDL|STL)\b", body))
        u = usage.get(name)
        rows[name] = {"mma": dict(mma), "ftz": ftz, "local": local, "res_usage": u}
        want = dict(WANT128 if "128_" in name else WANT)
        if name.endswith("_3"):
            want[SKETCH] = 64 if "128_" in name else 32
        if dict(mma) != want:
            bad.append(f"{name}: MMA {dict(mma)}, want {want}")
        if ftz:
            bad.append(f"{name}: FTZ instructions {sorted(set(ftz))}")
        if local:
            bad.append(f"{name}: {local} local-memory instructions")
        if u is None:
            bad.append(f"{name}: not in cuobjdump -res-usage")
        elif u.get("STACK") != 0 or u.get("LOCAL") != 0:
            bad.append(f"{name}: res-usage STACK:{u.get('STACK')} LOCAL:{u.get('LOCAL')}, want 0 and 0")
    if not any(n.startswith("fill_") for n in rows):
        bad.append("no fill_ kernel in the SASS")
    return rows, bad


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    with open(sys.argv[1]) as p, open(sys.argv[2]) as q, open(sys.argv[3]) as r:
        rows, bad = judge(p.read(), q.read(), r.read())
    with open(sys.argv[4], "w") as f:
        json.dump({"kernels": rows, "refusals": bad}, f, indent=1)
    if bad:
        print("SASS gate REFUSED:\n  " + "\n  ".join(bad), file=sys.stderr)
        sys.exit(7)
    print(f"SASS gate passed: {len(rows)} kernels, STACK:0 LOCAL:0 in each")
