"""build.sh's gate on pearl_c_sm120.cubin: ptxas's report and cuobjdump's SASS.  Refuses (exit 1, the reasons on stderr) on a
ptxas warning, a stack frame, a spill or local memory, and on any kernel whose tensor-core instructions aren't the intended
ones: the FP8 atom as QMMA.16832.F32.E4M3.E4M3 in the counts `WANT` states, the peel as HMMA.16816.F32.BF16, no other MMA.

  python3 sass_gate.py PTXAS_TXT SASS_TXT OUT_JSON
"""

from __future__ import annotations

import json
import re
import sys

QMMA, HMMA = "QMMA.16832.F32.E4M3.E4M3", "HMMA.16816.F32.BF16"
MMA = re.compile(r"\b([QHOIU]MMA|[QH]GMMA|UTC\w*MMA)(\.[\w.]+)?")
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


def gate(ptxas: str, sass: str) -> tuple[dict, list[str]]:
    problems = [line for line in ptxas.splitlines() if re.search(r"warning|error|C7514|serializ", line, re.I)]
    funcs: dict[str, dict] = {}
    for m in re.finditer(r"Compiling entry function '(\w+)' for 'sm_120a'(.*?)(?=Compiling entry function|\Z)", ptxas, re.S):
        body = m.group(2)
        frame = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", body)
        smem = re.search(r"(\d+) bytes smem", body)
        funcs[m.group(1)] = {"regs": int(re.search(r"Used (\d+) registers", body).group(1)), "stack": int(frame.group(1)),
                             "spill_stores": int(frame.group(2)), "spill_loads": int(frame.group(3)),
                             "static_smem": int(smem.group(1)) if smem else 0}
    cur, ops = None, ("LDL", "STL", "LDSM", "LDGSTS")
    for line in sass.splitlines():
        fm = re.search(r"Function : (\w+)", line)
        if fm:
            cur = fm.group(1)
            funcs.setdefault(cur, {}).update({op: 0 for op in ops} | {"QMMA": 0, "HMMA": 0, "other_mma": []})
            continue
        if not cur:
            continue
        for op in ops:
            if re.search(r"\b" + op + r"\b", line):
                funcs[cur][op] += 1
        for mm in MMA.finditer(line):
            form = mm.group(0)
            if form == QMMA:
                funcs[cur]["QMMA"] += 1
            elif form == HMMA:
                funcs[cur]["HMMA"] += 1
            else:
                funcs[cur]["other_mma"].append(form)
    for name, r in funcs.items():
        if r.get("stack") or r.get("spill_stores") or r.get("spill_loads") or r.get("LDL") or r.get("STL"):
            problems.append(f"{name}: local memory {r}")
        if r.get("other_mma"):
            problems.append(f"{name}: MMA forms other than {QMMA} and {HMMA}: {sorted(set(r['other_mma']))}")
        q, h = WANT.get(name, (0, 0))
        if (r.get("QMMA", 0) < q or (q and r["QMMA"] % q) or (not q and r.get("QMMA"))) or r.get("HMMA", 0) != h:
            problems.append(f"{name}: {r.get('QMMA')} {QMMA}, {r.get('HMMA')} {HMMA} (want a multiple of {q}, and {h})")
    missing = sorted(set(WANT) - set(funcs))
    if missing:
        problems.append(f"kernels missing from the cubin: {missing}")
    return funcs, problems


def main(argv: list[str]) -> int:
    ptxas, sass, out = argv[1:4]
    funcs, problems = gate(open(ptxas).read(), open(sass).read())
    json.dump({"want": {k: list(v) for k, v in WANT.items()}, "kernels": funcs, "problems": problems}, open(out, "w"), indent=1)
    if problems:
        print("REFUSED: " + "; ".join(problems), file=sys.stderr)
        return 1
    print(f"pearl_c_sm120.cubin: {len(funcs)} kernels; " + ", ".join(
        f"{n} {funcs[n]['regs']} regs {funcs[n]['QMMA']} QMMA" + (f" {funcs[n]['HMMA']} HMMA" if funcs[n]["HMMA"] else "")
        for n in ("gemm_plain", "gemm_ct", "gemm_pearlc", "gemm_pearlc_u", "gemm_pearlc_d", "words", "words_k", "form_s5"))
        + "; no local memory")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
