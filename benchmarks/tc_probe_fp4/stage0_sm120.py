"""Stage 0 on sm_120: which FP4-path ``mma`` mnemonics ptxas accepts for ``sm_120a`` and what each lowers to, before any semantics.

    python benchmarks/tc_probe_fp4/stage0_sm120.py --out DIR [--arch sm_120a]

For every row below, one ``.entry`` holding that single instruction is assembled with ``ptxas -arch=<arch>``; the return code and
diagnostic are recorded and, when it assembles, the SASS opcodes (``cuobjdump -sass``) and their count.  Record-only: nothing
executes, nothing is gated except that the dense rows ``probe.py`` issues must assemble.  The sparse rows (2:4 ``mma.sp``) are
for the FP4 attacker (GPU 7): whether a 2:4 sparse FP4 instruction exists on this part, and at what shape.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: name -> (mnemonic, A regs, B regs, sparse, scaled); the dense rows marked (probe) are the ones probe.py issues
ROWS = {
    "nvf4 (probe)": ("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3", 4, 2, False, True),
    "mxf4 (probe)": ("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0", 4, 2, False, True),
    "f8f6f4 e2m1 (probe)": ("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32", 4, 2, False, False),
    "f8f6f4 e2m1.e4m3 (probe)": ("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e4m3.f32", 4, 2, False, False),
    "mxf8f6f4 e2m1 1X (probe)": ("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e2m1.e2m1.f32.ue8m0", 4, 2, False, True),
    "mxf8f6f4 e2m1.e4m3 1X (probe)": ("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e2m1.e4m3.f32.ue8m0", 4, 2, False, True),
    "mxf4nvf4 4X ue8m0": ("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue8m0", 4, 2, False, True),
    "mxf4nvf4 2X ue4m3": ("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue4m3", 4, 2, False, True),
    "mxf4 2X ue8m0": ("mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0", 4, 2, False, True),
    "mxf4 (PTX order)": ("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0", 4, 2, False, True),
    "sparse f8f6f4 e2m1 k64": ("mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32", 4, 4, True, False),
    "sparse mxf8f6f4 e2m1 1X k64": ("mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e2m1.e2m1.f32.ue8m0", 4, 4, True, True),
    "sparse mxf4nvf4 4X ue4m3 k128": ("mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3", 4, 4, True, True),
    "sparse mxf4nvf4 2X ue8m0 k128": ("mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0", 4, 4, True, True),
    "sparse mxf4 2X ue8m0 k128": ("mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0", 4, 4, True, True),
}

PTX = """\
.version {version}
.target {target}
.address_size 64
.visible .entry k(.param .u64 p) {{
    .reg .b32 %a<4>; .reg .b32 %b<4>; .reg .f32 %c<4>; .reg .f32 %d<4>; .reg .b32 %s<2>; .reg .b32 %e; .reg .u64 %rd; .reg .u64 %ra;
    mov.b32 %a0, 0; mov.b32 %a1, 0; mov.b32 %a2, 0; mov.b32 %a3, 0; mov.b32 %b0, 0; mov.b32 %b1, 0; mov.b32 %b2, 0; mov.b32 %b3, 0;
    mov.f32 %c0, 0f00000000; mov.f32 %c1, 0f00000000; mov.f32 %c2, 0f00000000; mov.f32 %c3, 0f00000000;
    mov.b32 %s0, 0; mov.b32 %s1, 0; mov.b32 %e, 0;
    {instr};
    ld.param.u64 %rd, [p];
    cvta.to.global.u64 %ra, %rd;
    st.global.v4.f32 [%ra], {{%d0,%d1,%d2,%d3}};
    ret;
}}
"""


def instruction(mnemonic: str, na: int, nb: int, sparse: bool, scaled: bool) -> str:
    a = ",".join(f"%a{i}" for i in range(na))
    b = ",".join(f"%b{i}" for i in range(nb))
    ops = f"{{%d0,%d1,%d2,%d3}}, {{{a}}}, {{{b}}}, {{%c0,%c1,%c2,%c3}}"
    if sparse:
        ops += ", %e, 0x0"
    if scaled:
        ops += ", {%s0}, {0, 0}, {%s1}, {0, 0}"
    return f"{mnemonic} {ops}"


def tool(name: str) -> str:
    return shutil.which(name) or f"/usr/local/cuda/bin/{name}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="sm_120a")
    ap.add_argument("--ptx-version", default="8.7")
    args = ap.parse_args()
    out = Path(os.path.expandvars(args.out))
    out.mkdir(parents=True, exist_ok=True)
    ptxas_version = subprocess.run([tool("ptxas"), "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1:]
    rows = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, (mn, na, nb, sparse, scaled) in ROWS.items():
            src = Path(tmp) / "k.ptx"
            cubin = Path(tmp) / "k.cubin"
            src.write_text(PTX.format(version=args.ptx_version, target=args.arch, instr=instruction(mn, na, nb, sparse, scaled)))
            p = subprocess.run([tool("ptxas"), f"-arch={args.arch}", "-o", str(cubin), str(src)], capture_output=True, text=True)
            rec = {"mnemonic": mn, "sparse": sparse, "scaled": scaled, "a_regs": na, "b_regs": nb, "ptxas_rc": p.returncode,
                   "diagnostic": [ln.split(";", 1)[-1].strip() for ln in p.stderr.splitlines() if ln.strip()][:4]}
            if p.returncode == 0:
                sass = subprocess.run([tool("cuobjdump"), "-sass", str(cubin)], capture_output=True, text=True).stdout
                ops = [ln.split("*/")[1].split(";")[0].strip() for ln in sass.splitlines()
                       if "*/" in ln and re.search(r"\b[A-Z]MMA\b|[A-Z]MMA\.", ln)]
                rec["sass_mma"] = ops
                rec["sass_mma_count"] = len(ops)
            rows[name] = rec
            print(f"{name:34s} rc={p.returncode} " + (" | ".join(rec.get("sass_mma", [])) or "; ".join(rec["diagnostic"]))[:150], flush=True)
    failures = [f"{n}: ptxas refused a row probe.py issues" for n, r in rows.items() if "(probe)" in n and r["ptxas_rc"] != 0]
    doc = {"schema": "pouw-rtxpro/stage0-sm120/v1", "arch": args.arch, "ptx_version": args.ptx_version, "ptxas": ptxas_version,
           "rows": rows, "failures": failures,
           "note": "offline: ptxas acceptance and SASS only; a sparse row's register counts follow the PTX ISA's compressed-A layout, "
                   "and a refusal is recorded with ptxas's diagnostic (an operand-count error means the row exists with other counts)"}
    (out / "stage0_sm120.json").write_text(json.dumps(doc, indent=1))
    print("FAILURES" if failures else "stage 0 recorded", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
