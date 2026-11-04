"""build.sh's gate on s8_wgmma.cubin: ptxas's report and cuobjdump's SASS, through the harness's `mma_gate`
(benchmarks/pouw/harness/sass_gate.py).  Refuses (exit 1, the reasons on stderr) on a ptxas warning or serialized wgmma, a
stack frame, a spill or local memory, and on any GEMM kernel whose tensor-core instructions aren't its s8 atom alone,
IGMMA.64xBNx32 on s8 operands at the kernel's BN, in a multiple of 4 (a 128-deep stage), or whose epilogue isn't the
Definition's instructions: an s32-to-f32 conversion, FMUL and an F2FP to BF16 present, and no FFMA (a fused multiply-add
would round once) and no .FTZ anywhere but integer division's (`INT_DIV`).  A reduction kernel (s8mr*_128xBN_rR) may also run
the reduction's atom, IGMMA.64xRx32 on s8 operands, BN / 8 of them a tile.  Each kernel's record holds its highest register
(`max_reg`): a consumer warpgroup's budget after setmaxnreg, which ptxas's count doesn't show.  j10r_sm90's job, which ships
this file without the tree, puts the harness's gate beside it.

  python3 sass_gate.py PTXAS_TXT SASS_TXT OUT_JSON
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path


def _harness():
    name = "pouw_harness_sass_gate"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[2] / "pouw" / "harness" / "sass_gate.py")
        sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[name])
    return sys.modules[name]


H = _harness()
KERNEL = re.compile(r"s8mm2?_(64|128)x(64|128|256)$")
REDUCE = re.compile(r"s8mrb?_(128)x(128|256)_r(8|16|32|64|128)$")
#: the reduction kernels s8_wgmma.cu doesn't build (wgmma.py's UNBUILT)
UNBUILT = {"s8mr_128x256_r8", "s8mr_128x256_r16"}
#: the GEMM kernels and the reduction kernels the cubin must hold
GEMMS = {f"s8mm{c}_{tm}x{bn}" for c in ("", "2") for tm in (64, 128) for bn in (64, 128, 256)}
REDUCTIONS = {f"{f}_128x{bn}_r{r}" for f in ("s8mr", "s8mrb") for bn in (128, 256) for r in (8, 16, 32, 64, 128)} - UNBUILT
REG = re.compile(r"\bR(\d+)\b")
#: the compiler's unsigned integer division (the raster's, by a runtime group and tile count) truncates a reciprocal it
#: computed from the divisor with this F2I; it never sees a value, so it is counted apart from every other .FTZ
INT_DIV = r"\bF2I\.FTZ\.U32\.TRUNC\.NTZ\b"
COUNTS = {"LDL": r"\bLDL\b", "STL": r"\bSTL\b", "FFMA": r"\bFFMA\b", "INT_DIV": INT_DIV, "FMUL": r"\bFMUL\b",
          "I2F": r"\bI2FP?\.F32\.S32\b", "F2FP_BF16": r"\bF2FP\.BF16\S*", "UTMALDG": r"\bUTMALDG\b"}


def atom_ok(form: str, bn: int) -> bool:
    """IGMMA.64x<bn>x32 with both operands s8 (cuobjdump spells the types after the shape; an unsigned operand reads U8)."""
    return form.startswith(f"IGMMA.64x{bn}x32") and "U8" not in form


def _atom(name: str, form: str, r: dict) -> bool:
    km, rm = KERNEL.match(name), REDUCE.match(name)
    widths = (int(km.group(2)),) if km else (int(rm.group(2)), int(rm.group(3))) if rm else ()
    if not any(atom_ok(form, n) for n in widths):
        return False
    r["IGMMA"] += 1
    if form not in r["forms"]:
        r["forms"].append(form)
    return True


def _line(r: dict, text: str) -> None:
    r["FTZ"] += len(re.findall(r"\.FTZ\b", text)) - len(re.findall(INT_DIV, text))
    r["max_reg"] = max([r["max_reg"]] + [int(v) + 1 for v in REG.findall(text)])


def _check(name: str, r: dict) -> list[str]:
    out = []
    if r.get("FFMA") or r.get("FTZ"):
        out.append(f"{name}: {r.get('FFMA')} FFMA, {r.get('FTZ')} .FTZ")
    if name in GEMMS | REDUCTIONS:
        least = 4 + (int(REDUCE.match(name).group(2)) // 8 if name in REDUCTIONS else 0)
        if r.get("IGMMA", 0) < least or r["IGMMA"] % 4:
            out.append(f"{name}: {r.get('IGMMA', 0)} s8 atoms (want a multiple of 4, at least {least})")
        if not (r.get("I2F") and r.get("FMUL") and r.get("F2FP_BF16") and r.get("UTMALDG")):
            out.append(f"{name}: epilogue or loads missing: {({k: r.get(k) for k in ('I2F', 'FMUL', 'F2FP_BF16', 'UTMALDG')})}")
    return out


def gate(ptxas: str, sass: str) -> tuple[dict, list[str]]:
    return H.mma_gate(ptxas, sass, GEMMS | REDUCTIONS, arch="sm_90a", counts=COUNTS, atom=_atom, line=_line, check=_check,
                      record={**dict.fromkeys(COUNTS, 0), "FTZ": 0, "IGMMA": 0, "forms": [], "other_mma": [], "max_reg": 0},
                      allowed=lambda name: "its s8 atom")


def main(argv: list[str]) -> int:
    ptxas, sass, out = argv[1:4]
    funcs, problems = gate(Path(ptxas).read_text(), Path(sass).read_text())
    return H.write_verdict(out, {"kernels": funcs, "problems": problems}, problems, lambda: "s8_wgmma.cubin: " + ", ".join(
        f"{n} {funcs[n]['regs']} regs (max R{funcs[n]['max_reg'] - 1}) {funcs[n]['IGMMA']} {funcs[n]['forms']}"
        for n in sorted(funcs) if KERNEL.match(n) or REDUCE.match(n)))


if __name__ == "__main__":
    sys.exit(main(sys.argv))
