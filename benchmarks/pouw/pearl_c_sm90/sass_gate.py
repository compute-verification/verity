"""build.sh's gate on pearl_c_sm90.cubin: ptxas's report and cuobjdump's SASS.  Refuses (exit 1, the reasons on stderr) on a
ptxas warning or serialized wgmma, a stack frame, a spill or local memory, and on any kernel whose tensor-core instructions
aren't the intended ones: the FP8 atom as QGMMA.64xNx32.F32.E4M3.E4M3 at the kernel's N in the counts `WANT` states (and
`WINDOW_ATOMS`' A'F_B atoms at N = 32), the peel as HGMMA.64xNx16.F32.BF16, no other MMA.  (The .FTZ and FP32 MUFU gate is the harness's, benchmarks/pouw/harness/sass_gate.py.)

  python3 sass_gate.py PTXAS_TXT SASS_TXT OUT_JSON
"""

from __future__ import annotations

import json
import re
import sys

MMA = re.compile(r"\b([QHOIU]MMA|[QH]GMMA|UTC\w*MMA)(\.[\w.]+)?")


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


def gate(ptxas: str, sass: str) -> tuple[dict, list[str]]:
    problems = [line for line in ptxas.splitlines() if re.search(r"warning|error|C751\d|serializ", line, re.I)]
    funcs: dict[str, dict] = {}
    for m in re.finditer(r"Compiling entry function '(\w+)' for 'sm_90a'(.*?)(?=Compiling entry function|\Z)", ptxas, re.S):
        body = m.group(2)
        frame = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", body)
        smem = re.search(r"(\d+) bytes smem", body)
        funcs[m.group(1)] = {"regs": int(re.search(r"Used (\d+) registers", body).group(1)), "stack": int(frame.group(1)),
                             "spill_stores": int(frame.group(2)), "spill_loads": int(frame.group(3)),
                             "static_smem": int(smem.group(1)) if smem else 0}
    cur, ops = None, ("LDL", "STL", "UTMALDG", "SYNCS", "F2FP")
    for line in sass.splitlines():
        fm = re.search(r"Function : (\w+)", line)
        if fm:
            cur = fm.group(1)
            funcs.setdefault(cur, {}).update({op: 0 for op in ops} | {"QGMMA": 0, "HGMMA": 0, "QGMMA_window": 0, "other_mma": []})
            continue
        if not cur:
            continue
        for op in ops:
            if re.search(r"\b" + op + r"\b", line):
                funcs[cur][op] += 1
        n = WANT.get(cur, (0,))[0]
        for mm in MMA.finditer(line):
            form = mm.group(0)
            if n and form == qgmma(n):
                funcs[cur]["QGMMA"] += 1
            elif n and form == hgmma(n):
                funcs[cur]["HGMMA"] += 1
            elif cur in WINDOW_ATOMS and form == qgmma(32):
                funcs[cur]["QGMMA_window"] += 1
            else:
                funcs[cur]["other_mma"].append(form)
    for name, r in funcs.items():
        bound = LOCAL_OK.get(name, (0,))[0]
        if max(r.get("stack", 0), r.get("spill_stores", 0), r.get("spill_loads", 0)) > bound or \
                (not bound and (r.get("LDL") or r.get("STL"))):
            problems.append(f"{name}: local memory {r}")
        n, q, h = WANT.get(name, (0, 0, 0))
        if r.get("other_mma"):
            problems.append(f"{name}: MMA forms other than {qgmma(n)} and {hgmma(n)}: {sorted(set(r['other_mma']))}")
        got = r.get("QGMMA", 0)
        if (got < q or (q and got % q) or (not q and got)) or r.get("HGMMA", 0) != h:
            problems.append(f"{name}: {got} {qgmma(n)}, {r.get('HGMMA')} {hgmma(n)} (want a multiple of {q}, and {h})")
        if r.get("QGMMA_window", 0) != WINDOW_ATOMS.get(name, 0):
            problems.append(f"{name}: {r.get('QGMMA_window', 0)} {qgmma(32)} for A'F_B's window (want {WINDOW_ATOMS.get(name, 0)})")
    missing = sorted(set(WANT) - set(funcs))
    if missing:
        problems.append(f"kernels missing from the cubin: {missing}")
    return funcs, problems


def main(argv: list[str]) -> int:
    ptxas, sass, out = argv[1:4]
    funcs, problems = gate(open(ptxas).read(), open(sass).read())
    json.dump({"want": {k: list(v) for k, v in WANT.items()}, "local_ok": LOCAL_OK, "kernels": funcs, "problems": problems},
              open(out, "w"), indent=1)
    if problems:
        print("REFUSED: " + "; ".join(problems), file=sys.stderr)
        return 1
    print(f"pearl_c_sm90.cubin: {len(funcs)} kernels; " + ", ".join(
        f"{n} {funcs[n]['regs']} regs {funcs[n]['QGMMA']} QGMMA" + (f" {funcs[n]['HGMMA']} HGMMA" if funcs[n]["HGMMA"] else "")
        for n in ("gemm_plain", "gemm_ct", "gemm_pearlc", "gemm_pearlc_n64", "gemm_pearlc_d", "gemm_words", "form_s5"))
        + f"; no local memory but {sorted(LOCAL_OK)}'s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
