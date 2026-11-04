"""build.sh's gate on one attn_bf16 cubin (sm_90a or sm_120a): ptxas's report and cuobjdump's SASS, through the harness's
`mma_gate` (benchmarks/pouw/harness/sass_gate.py).  Refuses (exit 1, the reasons on stderr) on a ptxas warning, a kernel
compiled for another arch, a stack frame, a spill or local memory, and on any kernel whose floating-point instructions aren't
the Definition's: every tensor-core instruction an HMMA.16816.F32.BF16 (D/16 of them in QK^T's unrolled k loop, 4 D/16 in
_s16's, at least one in PV's and in the _ref kernels'), the cvt an F2FP.BF16(.F32).PACK_AB, and no other floating-point
instruction (an HFMA2.MMA of -RZ and RZ, the compiler's way of loading a constant, adds nothing, and an FSEL whose operands
carry no negation or absolute value moves a word unchanged).

  python3 sass_gate.py ARCH PTXAS_TXT SASS_TXT OUT_JSON
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
HMMA = "HMMA.16816.F32.BF16"
CVT = re.compile(r"^F2FP\.BF16(\.F32)?\.PACK_AB$")
FLOAT = re.compile(r"\b(FADD|FMUL|FFMA|FMNMX|FSETP|FSET|FSEL|FCHK|FRND|FSWZADD|MUFU|F2F|F2I|I2F|DADD|DMUL|DFMA|DSETP|DMNMX|"
                   r"HADD2|HMUL2|HFMA2|HMNMX2|HSETP2|HSET2|F2FP)(\.[\w.]+)?\b")
CONSTANT = re.compile(r"\bHFMA2\.MMA R\d+, -RZ, RZ\b")
SELECT = re.compile(r"\bFSEL R\d+, [^;|\-]*;")
#: kernel: (exact HMMA count, or None for at least one); _s16's QK^T unrolls its 4 key tiles, _ref's loops
WANT = {f"attn_{op}_d{d}{tile}": ((4 if tile == "_s16" else 1) * d // 16 if op == "qk" and tile != "_ref" else None)
        for tile in ("", "_h8", "_h16", "_s16", "_ref") for op in ("qk", "pv") for d in (64, 128)}


def _atom(name: str, form: str, r: dict) -> bool:
    if form != HMMA:
        return False
    r["HMMA"] += 1
    return True


def _line(r: dict, text: str) -> None:
    for fl in FLOAT.finditer(text):
        if CVT.match(fl.group(0)):
            r["F2FP"] += 1
        elif not CONSTANT.search(text) and not (fl.group(1) == "FSEL" and SELECT.search(text)):
            r["float"].append(fl.group(0))


def _check(name: str, r: dict) -> list[str]:
    out = []
    if r.get("float"):
        out.append(f"{name}: floating-point instructions other than {HMMA} and {CVT.pattern}: {sorted(set(r['float']))}")
    want = WANT.get(name, 0)
    if (r.get("HMMA", 0) != want) if want is not None else not r.get("HMMA"):
        out.append(f"{name}: {r.get('HMMA')} {HMMA}, want {want if want is not None else 'at least one'}")
    if name in WANT and not r.get("F2FP"):
        out.append(f"{name}: no {CVT.pattern}")
    return out


def gate(arch: str, ptxas: str, sass: str) -> tuple[dict, list[str]]:
    return H.mma_gate(ptxas, sass, WANT, arch=arch, counts={"LDL": r"\bLDL\b", "STL": r"\bSTL\b"}, atom=_atom, line=_line,
                      check=_check, record={"HMMA": 0, "F2FP": 0, "LDL": 0, "STL": 0, "other_mma": [], "float": []},
                      allowed=lambda name: HMMA, unreported=True)


def main(argv: list[str]) -> int:
    arch, ptxas, sass, out = argv[1:5]
    funcs, problems = gate(arch, Path(ptxas).read_text(), Path(sass).read_text())
    return H.write_verdict(out, {"arch": arch, "want_hmma": WANT, "kernels": funcs, "problems": problems}, problems,
                           lambda: f"attn_bf16.{arch}.cubin: " + ", ".join(
                               f"{n} {funcs[n]['regs']} regs {funcs[n]['HMMA']} HMMA {funcs[n]['F2FP']} F2FP" for n in WANT)
                           + "; no other floating-point instruction, no local memory", refused=f"REFUSED ({arch})")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
