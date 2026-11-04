"""build.sh's gate on pearl_c_sm90.cubin: ptxas's report and cuobjdump's SASS, through the harness's `mma_gate`
(benchmarks/pouw/harness/sass_gate.py, which also holds the .FTZ and FP32 MUFU gate).  Refuses (exit 1, the reasons on
stderr) on a ptxas warning or serialized wgmma, a stack frame, a spill or local memory (but `LOCAL_OK`'s), and on any kernel
whose tensor-core instructions aren't the intended ones: the FP8 atom as QGMMA.64xNx32.F32.E4M3.E4M3 at the kernel's N in
the counts `WANT` states (and `WINDOW_ATOMS`' A'F_B atoms at N = 32), the peel as HGMMA.64xNx16.F32.BF16, no other MMA.

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
COUNTS = {op: rf"\b{op}\b" for op in ("LDL", "STL", "UTMALDG", "SYNCS", "F2FP")}


def qgmma(n: int) -> str:
    return f"QGMMA.64x{n}x32.F32.E4M3.E4M3"


def hgmma(n: int) -> str:
    return f"HGMMA.64x{n}x16.F32.BF16"


#: kernel: (N, QGMMA multiple, HGMMA count).  Each GEMM's stage is four k32 atoms (BK = 128) on its warpgroup's 64 x N tile,
#: so its QGMMA count is a multiple of 4 (the compiler may unroll the stage loop); the peel is four k16 steps; form_s5 and
#: form_s5_scalar one atom.  N is the tile's columns: 128, 64 (_n64, _d) or 32 (gemm_words).  The grouped calls' kernels
#: (_g) are their ungrouped ones' at the same tile
WANT = {**{f"{g}{t}": (n, 4, 0) for g in ("gemm_plain", "gemm_fast", "gemm_ct") for t, n in (("", 128), ("_n64", 64), ("_d", 64))},
        **{f"{g}{t}": (n, 4, 4) for g in ("gemm_pearlc", "gemm_pearlc_s") for t, n in (("", 128), ("_n64", 64), ("_d", 64))},
        "gemm_words": (32, 4, 0), "gemm_words_d": (32, 4, 0), "gemm_words_win_d": (32, 4, 0), "form_s5": (128, 1, 0), "form_s5_scalar": (128, 1, 0),
        "gemm_pearlc_g_d": (64, 4, 4), "gemm_words_g_d": (32, 4, 0), "form_s5_g": (128, 1, 0), "form_s5_ep": (128, 1, 0),
        "form_s5_epw": (128, 1, 0)}
#: kernels that also make one of A'F_B's windows (gemm_words_win_d's tile): exactly this many QGMMA.64x32x32 atoms
WINDOW_ATOMS = {"form_s5_epw": 4}
#: kernels that may use a little local memory, each with at most these bytes of stack and of spills, and why: none of
#: them is on the served path or the GEMMs'
LOCAL_OK = {"s512_rows": (16, "hash_h3.cuh's reference row digest (PEARLC_SHA=ref, or a caller without segdig_<tree>; the served "
                               "path gives every segdig buffer and runs hash_fast.cuh's): 255 registers and 4 bytes spilled at "
                               "sm_90a, r20261023-065436-805b")}


def _atom(name: str, form: str, r: dict) -> bool:
    n = WANT.get(name, (0,))[0]
    key = ("QGMMA" if n and form == qgmma(n) else "HGMMA" if n and form == hgmma(n)
           else "QGMMA_window" if name in WINDOW_ATOMS and form == qgmma(32) else None)
    if key:
        r[key] += 1
    return key is not None


def _check(name: str, r: dict) -> list[str]:
    n, q, h = WANT.get(name, (0, 0, 0))
    got, out = r.get("QGMMA", 0), []
    if (got < q or (q and got % q) or (not q and got)) or r.get("HGMMA", 0) != h:
        out.append(f"{name}: {got} {qgmma(n)}, {r.get('HGMMA')} {hgmma(n)} (want a multiple of {q}, and {h})")
    if r.get("QGMMA_window", 0) != WINDOW_ATOMS.get(name, 0):
        out.append(f"{name}: {r.get('QGMMA_window', 0)} {qgmma(32)} for A'F_B's window (want {WINDOW_ATOMS.get(name, 0)})")
    return out


def gate(ptxas: str, sass: str) -> tuple[dict, list[str]]:
    n = lambda name: WANT.get(name, (0,))[0]  # noqa: E731
    return H.mma_gate(ptxas, sass, WANT, arch="sm_90a", smem=True, local_ok={k: v[0] for k, v in LOCAL_OK.items()},
                      record={**dict.fromkeys(COUNTS, 0), "QGMMA": 0, "HGMMA": 0, "QGMMA_window": 0, "other_mma": []},
                      counts=COUNTS, atom=_atom, allowed=lambda name: f"{qgmma(n(name))} and {hgmma(n(name))}", check=_check)


def main(argv: list[str]) -> int:
    ptxas, sass, out = argv[1:4]
    funcs, problems = gate(Path(ptxas).read_text(), Path(sass).read_text())
    return H.write_verdict(out, {"want": {k: list(v) for k, v in WANT.items()}, "local_ok": LOCAL_OK, "kernels": funcs,
                                 "problems": problems}, problems, lambda: f"pearl_c_sm90.cubin: {len(funcs)} kernels; " + ", ".join(
        f"{n} {funcs[n]['regs']} regs {funcs[n]['QGMMA']} QGMMA" + (f" {funcs[n]['HGMMA']} HGMMA" if funcs[n]["HGMMA"] else "")
        for n in ("gemm_plain", "gemm_ct", "gemm_pearlc", "gemm_pearlc_n64", "gemm_pearlc_d", "gemm_words", "form_s5"))
        + f"; no local memory but {sorted(LOCAL_OK)}'s")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
