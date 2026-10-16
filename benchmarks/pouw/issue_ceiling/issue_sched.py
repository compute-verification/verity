#!/usr/bin/env python3
"""Combine the round-2 scheduling run's JSONL with its SASS: MMA MAC fraction of alone, total warp-instructions a
SMSP-cycle, and the SASS confirmation of the opcode that issued, for the warp-specialized (`k_spec`) and fully-unrolled
(`k_mixu`) schedules. Report the best (lowest-displacement) schedule at 8 and 16 ops a MMA.

$OUT/issue_sched.jsonl        specmma_alone / spec / mixu rows
$OUT/sass/issue_sched.sass    every kernel's SASS; this reads each loop and counts opcodes
-> $OUT/sched.md, $OUT/sched.json
"""
import json
import re
import sys
from pathlib import Path

ADDR = re.compile(r"/\*([0-9a-fA-F]+)\*/\s+(?:@!?U?P[T0-9]+\s+)?([A-Z@][A-Z0-9_.]*)")
BRA = re.compile(r"\bBRA\b.*?(0x[0-9a-fA-F]+)")
FUNC = re.compile(r"Function\s*:\s*(\S+)")
TMPL_INT = re.compile(r"Li(\d+)E")

# enum orders must match issue_sched.cu
M_IMMA, M_OMMA = 0, 1
M_NAME = ["imma_s8_16832", "omma_nvf4_16864"]
M_SASS = ["IMMA", "OMMA"]
X_NAME = ["iadd3", "idp4a", "hfma2", "mov", "sel", "ffma", "ldg", "mix3",
          "vimnmx", "f2fp", "popc", "mufu", "flo", "brev", "brx", "s2r", "vote", "ldg128"]
X_SASS = [["IADD3"], ["IDP"], ["HFMA2"], ["MOV"], ["SEL"], ["FFMA"], ["LDG"], ["IADD3", "IDP", "HFMA2"],
          ["VIMNMX", "IMNMX"], ["F2FP"], ["POPC"], ["MUFU"], ["FLO"], ["BREV"], ["BRX"], ["S2R", "CS2R"],
          ["VOTE"], ["LDG"]]
m_idx = {n: i for i, n in enumerate(M_NAME)}
x_idx = {n: i for i, n in enumerate(X_NAME)}

A_MMA = 9.781  # JI2: 9.781 SMSP-cycles a MMA + 1 for every other instruction (note:20261015T1814Z §12)


def tmpl_ints(mangled):
    return [int(n) for n in TMPL_INT.findall(mangled)]


def loops_from_text(lines):
    """Return every backward-BRA loop body as (len, hist) using full lines (target is on the BRA line)."""
    instrs = []
    for line in lines:
        m = ADDR.search(line)
        if m:
            tgt = None
            b = BRA.search(line)
            if b:
                tgt = int(b.group(1), 16)
            instrs.append((int(m.group(1), 16), m.group(2).lstrip("@"), tgt))
    res = []
    for addr, op, tgt in instrs:
        if op.split(".")[0] != "BRA" or tgt is None or tgt >= addr:
            continue
        body = [(a, o) for a, o, _ in instrs if tgt <= a <= addr]
        hist = {}
        for _, o in body:
            k = o.split(".")[0]
            hist[k] = hist.get(k, 0) + 1
        res.append((len(body), hist))
    res.sort(reverse=True)
    return res


def funcs_with_lines(path):
    funcs = {}
    cur = None
    for line in path.read_text(errors="replace").splitlines():
        m = FUNC.search(line)
        if m:
            cur = m.group(1)
            funcs[cur] = []
            continue
        if cur is not None:
            funcs[cur].append(line)
    return funcs


def by_tmpl(funcs, stem, tmpl):
    pre = f"_Z{len(stem)}{stem}I"
    for name, lines in funcs.items():
        if name.startswith(pre) and tmpl_ints(name) == list(tmpl):
            return name, lines
    return None, None


def loop_containing(ls, needles):
    """The largest loop whose histogram contains any of `needles` (base opcodes); else the largest loop."""
    for n, hist in ls:
        if any(hist.get(x, 0) > 0 for x in needles):
            return n, hist
    return ls[0] if ls else (0, {})


def main():
    out = Path(sys.argv[1])
    rows = []
    jl = out / "issue_sched.jsonl"
    for line in jl.read_text().splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    funcs = funcs_with_lines(out / "sass" / "issue_sched.sass")

    alone = {r["mma"]: r["mma_macs_per_sm_per_clk"] for r in rows if r.get("mode") == "specmma_alone"}
    alone_hi = {r["mma"]: r["mma_macs_per_sm_per_clk"] for r in rows if r.get("mode") == "specmma_hi_alone"}

    result = {"alone": alone, "alone_hi": alone_hi, "spec": [], "spec_hi": [], "mixu": []}
    md = ["# Round-2 schedules: beating round-1's MMA+lane overlap (sm_120, RTX PRO 6000)\n"]
    md.append("Frac = MMA MAC rate / spec-geometry MMA-alone. Displacement = 1 − frac. Perfect pipe overlap keeps")
    md.append("frac ≈ 1.0 when tensor-bound (ratios 4, 8) and ≈ tensor/lane demand when lane-bound (ratio 16 ≈ 0.50).\n")
    md.append("## MMA-alone baselines (spec geometry, 2 MMA warps a SMSP)\n")
    for m, v in alone.items():
        md.append(f"- `{m}`: {v:.1f} MAC/SM/clk (16-warp geometry)")
    for m, v in alone_hi.items():
        md.append(f"- `{m}`: {v:.1f} MAC/SM/clk (32-warp high-occupancy geometry)")
    md.append("")

    def sass_for(mode, M, X, R):
        stem = {"spec": "k_spec", "spec_hi": "k_spec_hi", "mixu": "k_mixu"}[mode]
        name, lines = by_tmpl(funcs, stem, [M, X, R])
        if not lines:
            return None
        ls = loops_from_text(lines)
        return ls, name

    md.append("## Schedules\n")
    md.append("| sched | MMA | op | ops/MMA | MMA frac | displ. | total warp-instr/SMSP (min) | op SASS confirmed |")
    md.append("|---|---|---|---|---|---|---|---|")
    best = {}  # ratio -> (frac, descr, row)
    for r in rows:
        mode = r.get("mode")
        if mode not in ("spec", "spec_hi", "mixu"):
            continue
        M, X = m_idx[r["mma"]], x_idx[r["op"]]
        # R = ops a loop body: spec lane warp does R (ratio R/8); spec_hi does R (ratio R/4); mixu body has 8*R.
        R = int(round(r["ops_per_mma"] * (4 if mode == "spec_hi" else 8)))
        frac = r["mma_macs_frac_of_alone"]
        total_min_smsp = (r["mma_calls_per_sm_per_clk"] + r["op_calls_per_sm_per_clk"]) / 4.0
        # SASS confirmation. intended op count in the loop body: R (spec/spec_hi lane warp) or 8*R (mixu body = 8 sub x R).
        intended = 8 * R if mode == "mixu" else R
        confirmed = "—"
        got = 0
        total_sass_smsp = None
        # JI2 coverage: per-SMSP window counts from SASS. #MMA = IMMAs on the SMSP; #other = every other instruction
        # issued on the SMSP (lane ops + all loop overhead). Satisfies JI2 iff 9.781·#MMA + #other ≤ cycles.
        n_mma = n_oth = ji2_ratio = None
        iters, cyc = r["iters"], r["cyc"]
        s = sass_for(mode, M, X, R)
        if s:
            ls, name = s
            if mode in ("spec", "spec_hi"):
                mma_len, mma_h = loop_containing(ls, [M_SASS[M]])
                alu_len, alu_h = loop_containing(ls, X_SASS[X])
                got = sum(alu_h.get(x, 0) for x in X_SASS[X])
                mw, aw = r["mma_warps"], r["alu_warps"]
                total_sass_smsp = (mw * mma_len + aw * alu_len) * iters / cyc / 4.0
                imma = mma_h.get(M_SASS[M], 0)
                mw_s, aw_s = mw / 4.0, aw / 4.0  # warps a SMSP
                n_mma = mw_s * imma * iters
                n_oth = (mw_s * (mma_len - imma) + aw_s * alu_len) * iters
            else:
                body_len, body_h = ls[0] if ls else (0, {})
                got = sum(body_h.get(x, 0) for x in X_SASS[X])
                total_sass_smsp = body_len * r["warps"] * iters / cyc / 4.0
                imma = body_h.get(M_SASS[M], 0)
                w_s = r["warps"] / 4.0
                n_mma = w_s * imma * iters
                n_oth = w_s * (body_len - imma) * iters
            if n_mma is not None and cyc > 0:
                ji2_ratio = (A_MMA * n_mma + n_oth) / cyc
        eliminated = got < 0.5 * intended  # ptxas removed most of the op (e.g. MOV copy-propagated)
        confirmed = (f"{got}/{intended}×{'/'.join(X_SASS[X])}"
                     + (" **eliminated**" if eliminated else ""))
        row = dict(r)
        row["total_warp_instr_smsp_min"] = total_min_smsp
        if total_sass_smsp is not None:
            row["total_warp_instr_smsp_sass"] = total_sass_smsp
        row["op_sass_confirmed"] = confirmed
        row["op_sass_got"] = got
        row["op_sass_intended"] = intended
        row["op_eliminated"] = eliminated
        if ji2_ratio is not None:
            row["ji2_n_mma_per_iter_smsp"] = n_mma / iters
            row["ji2_n_other_per_iter_smsp"] = n_oth / iters
            row["ji2_cyc_per_iter_smsp"] = cyc / iters
            row["ji2_ratio"] = ji2_ratio  # (9.781·#MMA + #other) / cycles; ≤ 1 satisfies JI2
            row["ji2_satisfies"] = ji2_ratio <= 1.0 + 1e-6
        result[mode].append(row)
        md.append(f"| {mode} | {r['mma']} | {r['op']} | {r['ops_per_mma']:.0f} | {frac:.3f} | {1-frac:.3f} | "
                  f"{total_min_smsp:.4f} | {confirmed} |")
        # best (lowest displacement = highest frac) per ratio, IMMA only, over ops that genuinely issued and aren't
        # memory-latency-bound (LDG)
        if (r["mma"] == M_NAME[M_IMMA] and r["ops_per_mma"] in (8.0, 16.0)
                and not eliminated and r["op"] != "ldg"):
            key = int(r["ops_per_mma"])
            if key not in best or frac > best[key][0]:
                best[key] = (frac, f"{mode} IMMA+{r['op']}", row)

    md.append("")
    md.append("## Best (lowest-displacement) schedule\n")
    md.append("| ops/MMA | best schedule | MMA frac | displacement | round-1 frac |")
    md.append("|---|---|---|---|---|")
    r1 = {8: 0.756, 16: 0.444}
    for ratio in (8, 16):
        if ratio in best:
            frac, descr, _ = best[ratio]
            md.append(f"| {ratio} | {descr} | {frac:.3f} | {1-frac:.3f} | {r1[ratio]:.3f} |")
    result["best"] = {str(k): {"frac": v[0], "schedule": v[1]} for k, v in best.items()}

    (out / "sched.md").write_text("\n".join(md) + "\n")
    (out / "sched.json").write_text(json.dumps(result, indent=2))

    # ---- JI2 coverage (round 3) ------------------------------------------------------------------------------------
    ji_rows = [r for r in result["spec"] + result["mixu"] if "ji2_ratio" in r]
    ji_rows.sort(key=lambda r: (x_idx[r["op"]], r["mode"], r["ops_per_mma"]))
    jm = ["# Round-3 JI2 coverage: 9.781·#MMA + #other ≤ cycles a SMSP (sm_120, RTX PRO 6000)\n"]
    jm.append("JI2 = 9.781 SMSP-cycles a MMA + 1 for every other instruction (note:20261015T1814Z-draft-rs-machine-model")
    jm.append("§12). #MMA and #other are per-SMSP window counts from SASS (#other = every non-MMA instruction issued on")
    jm.append("the SMSP: lane ops + all loop overhead). JI2 = (9.781·#MMA + #other) / cycles; **≤ 1 satisfies** (JI2 is a")
    jm.append("valid lower bound on cycles), **> 1 violates** (the mix runs in fewer cycles than JI2 charges).\n")
    jm.append("| op | mode | ops/MMA | MMA frac | instr/SMSP-cyc | #MMA/it·smsp | #other/it·smsp | JI2 ratio | SASS |")
    jm.append("|---|---|---|---|---|---|---|---|---|")
    viol = []
    per_op = {}
    for r in ji_rows:
        ok = r["ji2_satisfies"]
        tag = "✓" if ok else "✗ VIOLATES"
        instr = r.get("total_warp_instr_smsp_sass", r["total_warp_instr_smsp_min"])
        jm.append(f"| {r['op']} | {r['mode']} | {r['ops_per_mma']:.0f} | {r['mma_macs_frac_of_alone']:.3f} | "
                  f"{instr:.3f} | {r['ji2_n_mma_per_iter_smsp']:.3f} | {r['ji2_n_other_per_iter_smsp']:.3f} | "
                  f"{r['ji2_ratio']:.3f} {tag} | {r['op_sass_confirmed']} |")
        per_op.setdefault(r["op"], []).append(r["ji2_ratio"])
        if not ok:
            viol.append(r)
    jm.append("")
    jm.append("## Per-kind worst (max) JI2 ratio\n")
    jm.append("| op | worst JI2 ratio | covered by JI2? |")
    jm.append("|---|---|---|")
    for op in X_NAME:
        if op in per_op:
            worst = max(per_op[op])
            jm.append(f"| {op} | {worst:.3f} | {'yes' if worst <= 1.0 + 1e-6 else '**NO**'} |")
    jm.append("")
    if viol:
        jm.append("## Kinds that VIOLATE JI2\n")
        for r in viol:
            jm.append(f"- **{r['op']}** ({r['mode']}, {r['ops_per_mma']:.0f} ops/MMA): JI2 ratio "
                      f"{r['ji2_ratio']:.3f} > 1 — runs in fewer cycles than 9.781·#MMA + #other.")
    else:
        jm.append("## No kind violates JI2\n")
        jm.append("Every tested kind satisfies 9.781·#MMA + #other ≤ cycles at every (mode, ratio) point, so JI2 covers "
                  "them: the frozen τ = 0.2447 holds.")
    result["ji2"] = {"a_mma": A_MMA, "violations": [
        {k: r[k] for k in ("mode", "op", "ops_per_mma", "mma_macs_frac_of_alone", "ji2_ratio",
                           "ji2_n_mma_per_iter_smsp", "ji2_n_other_per_iter_smsp", "op_sass_confirmed")} for r in viol]}
    result["ji2"]["per_op_worst"] = {op: max(v) for op, v in per_op.items()}

    (out / "ji3.md").write_text("\n".join(jm) + "\n")
    (out / "ji3.json").write_text(json.dumps(result, indent=2))
    print("\n".join(jm))


if __name__ == "__main__":
    main()
