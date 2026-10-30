"""build.sh's gate on pearl_c_sm120.cubin: ptxas's report and cuobjdump's SASS, through the harness's `mma_gate`
(benchmarks/pouw/harness/sass_gate.py).  Refuses (exit 1, the reasons on stderr) on a ptxas warning, a stack frame, a spill or
local memory, and on any kernel whose tensor-core instructions aren't the intended ones: the FP8 atom as
QMMA.16832.F32.E4M3.E4M3 in the counts `WANT` states, the peel as HMMA.16816.F32.BF16, no other MMA.

  python3 sass_gate.py PTXAS_TXT SASS_TXT OUT_JSON
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _harness():
    name = "pouw_harness_sass_gate"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "harness" / "sass_gate.py")
        sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[name])
    return sys.modules[name]


H = _harness()
QMMA, HMMA = "QMMA.16832.F32.E4M3.E4M3", "HMMA.16816.F32.BF16"
#: kernel: (QMMA multiple, HMMA count); a multiple of one stage's atoms: the 128 x 128 tiles (mainloop_sm120.cuh: a warp of
#: 64 x 32) 16 atoms per k32, 64 per 128-deep stage, and 64 of the peel;
#: 64 x 64 (_d) and 128 x 32 (gemm_words) tiles 4 per k32, 8 per 64-deep stage (16 per 128-deep one, decode's served chain
#: gemm_pearlc_d), and 16 of the peel; the words kernels one atom per k32 per warp
WANT = {**{g: (64, 0) for g in ("gemm_plain", "gemm_fast", "gemm_ct", "gemm_ct_u")},
        **{g: (8, 0) for g in ("gemm_ct_d", "gemm_ct_d_u", "gemm_words", "gemm_words_u")},
        **{g: (64, 64) for g in ("gemm_pearlc", "gemm_pearlc_u", "gemm_pearlc_s", "gemm_pearlc_s_u")},
        **{g: (16, 16) for g in ("gemm_pearlc_d", "gemm_pearlc_d_u")}, **{g: (8, 16) for g in ("gemm_pearlc4_d", "gemm_pearlc4_d_u")},
        "words": (1, 0), "words_u": (1, 0), "words_k": (1, 0), "form_s5": (1, 0), "form_s5_r144": (1, 0), "form_s5_scalar": (1, 0),
        **{f: (1, 0) for f in ("form_rate", "form_rate_r144", "form_rate_scalar", "form_rate_nocast", "form_rate_nocast_r144",
                               "form_rate_atoms")}}


COUNTS = {op: rf"\b{op}\b" for op in ("LDL", "STL", "LDSM", "LDGSTS")}


def _atom(name: str, form: str, r: dict) -> bool:
    key = {QMMA: "QMMA", HMMA: "HMMA"}.get(form)
    if key:
        r[key] += 1
    return key is not None


def _check(name: str, r: dict) -> list[str]:
    q, h = WANT.get(name, (0, 0))
    if (r.get("QMMA", 0) < q or (q and r["QMMA"] % q) or (not q and r.get("QMMA"))) or r.get("HMMA", 0) != h:
        return [f"{name}: {r.get('QMMA')} {QMMA}, {r.get('HMMA')} {HMMA} (want a multiple of {q}, and {h})"]
    return []


def gate(ptxas: str, sass: str) -> tuple[dict, list[str]]:
    return H.mma_gate(ptxas, sass, WANT, arch="sm_120a", smem=True, record={**dict.fromkeys(COUNTS, 0), "QMMA": 0, "HMMA": 0,
                      "other_mma": []}, counts=COUNTS, atom=_atom, allowed=lambda name: f"{QMMA} and {HMMA}", check=_check)


def main(argv: list[str]) -> int:
    ptxas, sass, out = argv[1:4]
    funcs, problems = gate(Path(ptxas).read_text(), Path(sass).read_text())
    return H.write_verdict(out, {"want": {k: list(v) for k, v in WANT.items()}, "kernels": funcs, "problems": problems}, problems,
                           lambda: f"pearl_c_sm120.cubin: {len(funcs)} kernels; " + ", ".join(
        f"{n} {funcs[n]['regs']} regs {funcs[n]['QMMA']} QMMA" + (f" {funcs[n]['HMMA']} HMMA" if funcs[n]["HMMA"] else "")
        for n in ("gemm_plain", "gemm_ct", "gemm_pearlc", "gemm_pearlc_u", "gemm_pearlc_d", "words", "words_k", "form_s5"))
        + "; no local memory")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
