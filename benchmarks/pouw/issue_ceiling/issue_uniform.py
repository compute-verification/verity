#!/usr/bin/env python3
"""Does a uniform-datapath instruction take the dispatch slot? Combine issue_uniform.jsonl with its SASS.

Saturate the SMSP dispatch with a vector kind (FFMA/IADD3) at ~1 instr a SMSP-cycle, add a live UIADD3 chain at
densities D (uniform a SMSP-iter). slots a uniform = Δ(cycles a outer-iter) / Δ(#uniform a outer-iter), i.e. the slope of
cycles-per-iter vs D, / warps-a-SMSP. ~1 => uniform takes a full slot (VF2's 0.594 is a valid lower bound); ~0 =>
uniform takes no slot (0.594 exceeds real cost => VF2 unsound). The LOW-density points (D < NACC, the uniform pipe not
yet saturated) are the discriminator: high density saturates any port, so it can't separate co-issue from slot-sharing.

$OUT/issue_uniform.jsonl, $OUT/sass/issue_uniform.sass -> $OUT/uniform.md, $OUT/uniform.json
"""
import json
import re
import sys
from pathlib import Path

ADDR = re.compile(r"/\*([0-9a-fA-F]+)\*/\s+(?:@!?U?P[T0-9]+\s+)?([A-Z@][A-Z0-9_.]*)")
BRA = re.compile(r"\bBRA\b.*?(0x[0-9a-fA-F]+)")
FUNC = re.compile(r"Function\s*:\s*(\S+)")
TMPL_INT = re.compile(r"Li(-?\d+)E")

VK_NAME = ["ffma", "iadd3"]
VK_SASS = ["FFMA", "IADD3"]
vk_idx = {n: i for i, n in enumerate(VK_NAME)}
A_UNI = 0.594  # VF2 §13 price of a uniform-datapath instruction (SMSP-cycles)


def tmpl_ints(mangled):
    return [int(n) for n in TMPL_INT.findall(mangled)]


def lin_slope(pts):
    """Least-squares slope of y vs x over pts=[(x,y),...]."""
    n = len(pts)
    if n < 2:
        return 0.0
    sx = sum(x for x, _ in pts); sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts); sxy = sum(x * y for x, y in pts)
    den = n * sxx - sx * sx
    return (n * sxy - sx * sy) / den if den else 0.0


def funcs_with_lines(path):
    funcs, cur = {}, None
    for line in path.read_text(errors="replace").splitlines():
        m = FUNC.search(line)
        if m:
            cur = m.group(1)
            funcs[cur] = []
        elif cur is not None:
            funcs[cur].append(line)
    return funcs


def loops_from_text(lines):
    instrs = []
    for line in lines:
        m = ADDR.search(line)
        if m:
            b = BRA.search(line)
            tgt = int(b.group(1), 16) if b else None
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


def by_tmpl(funcs, stem, tmpl):
    pre = f"_Z{len(stem)}{stem}I"
    for name, lines in funcs.items():
        if name.startswith(pre) and tmpl_ints(name) == list(tmpl):
            return loops_from_text(lines)
    return None


def main():
    out = Path(sys.argv[1])
    rows = [json.loads(l) for l in (out / "issue_uniform.jsonl").read_text().splitlines()
            if l.strip().startswith("{")]
    rows = [r for r in rows if r.get("probe") == "uniform" and "density" in r]
    funcs = funcs_with_lines(out / "sass" / "issue_uniform.sass")

    md = ["# Uniform-datapath probe: does a uniform instruction take the dispatch slot? (sm_120, RTX PRO 6000)\n"]
    md.append("Saturate the SMSP dispatch with a vector kind, add a live UIADD3 chain at density D (uniform a SMSP-iter).")
    md.append(f"VF2 §13 prices a uniform instruction at **{A_UNI} SMSP-cycles**. `slots/uni` = Δ(cycles a outer-iter)/Δ")
    md.append("(#uniform a outer-iter) — ~1 means it takes a full dispatch slot, ~0 means it takes none.\n")
    result = {"a_uni": A_UNI, "by_vk": {}}

    nacc = next((r["nacc"] for r in rows if "nacc" in r), 32)
    for vk in (0, 1):
        vr = sorted([r for r in rows if r["vk"] == VK_NAME[vk]], key=lambda r: r["density"])
        if not vr:
            continue
        base = next((r for r in vr if r["density"] == 0), None)
        warps_smsp = (base["warps"] if base else vr[0]["warps"]) / 4.0  # warps a SMSP (4 SMSPs a SM)

        def uget_of(dd):  # UIADD3 in the largest loop body of the k_vec<vk,dd> SASS
            lls = by_tmpl(funcs, "k_vec", [vk, dd])
            if not lls:
                return None
            return max(lls, key=lambda x: x[0])[1].get("UIADD3", 0)
        base_u0 = uget_of(0)  # loop-control UIADD3 at D=0; the chain adds D more on top
        md.append(f"## Vector = {VK_NAME[vk].upper()}\n")
        md.append("| density D (uni/SMSP-it) | vec/SMSP-cyc | uni/SMSP-cyc | cyc/outer-it | slots/uni (vs D=0) | SASS (vec / uni) | promoted? |")
        md.append("|---|---|---|---|---|---|---|")
        pts = []  # (D, cyc_per_iter)
        rowsout = []
        for r in vr:
            D = r["density"]
            cyc_it = r["cyc"] / r["iters"]
            n_uni_it = r["n_uni_smsp"] / r["iters"]  # = warps_smsp*D, uniform instrs a SMSP a outer-iter
            pts.append((D, cyc_it))
            ls = by_tmpl(funcs, "k_vec", [vk, D])
            vget = uget = 0
            uni_other = 0
            if ls:
                body = max(ls, key=lambda x: x[0])[1] if ls else {}
                vget = body.get(VK_SASS[vk], 0)
                uget = body.get("UIADD3", 0)
                uni_other = sum(c for k, c in body.items() if (k.startswith("U") and k != "UIADD3") or k in ("R2UR", "CS2UR"))
            slots = ((cyc_it - base["cyc"] / base["iters"]) / n_uni_it) if (base and D > 0) else None
            # the chain is D fully-unrolled UIADD3 on top of the loop-control UIADD3 (base_u0); promotion succeeded iff
            # the body has base_u0 + D of them.
            chain_u = (uget - base_u0) if base_u0 is not None else None
            promoted = (chain_u == D) if chain_u is not None else False
            prom = "ok" if promoted else f"NO (chain {chain_u}≠{D})"
            sass = f"{vget}×{VK_SASS[vk]} / {uget}×UIADD3" + (f" (+{uni_other} U-other)" if uni_other else "")
            md.append(f"| {D} ({n_uni_it:.0f}) | {r['vec_per_smsp_per_cyc']:.4f} | {r['uni_per_smsp_per_cyc']:.4f} | "
                      f"{cyc_it:.1f} | {('%.3f' % slots) if slots is not None else '—'} | {sass} | {prom} |")
            rowsout.append({"density": D, "cyc_per_iter": cyc_it, "vec_per_smsp_per_cyc": r["vec_per_smsp_per_cyc"],
                            "uni_per_smsp_per_cyc": r["uni_per_smsp_per_cyc"], "slots_per_uni_vs0": slots,
                            "sass_vec": vget, "sass_uiadd3": uget, "sass_uni_other": uni_other,
                            "promoted": promoted, "iters": r["iters"]})
        # slots a uniform = slope of cyc/outer-it vs D, / warps_smsp (each unit D is warps_smsp uni a SMSP a iter).
        # LOW density (D < nacc, uniform pipe unsaturated) is the discriminator: with the vector pipe the bottleneck,
        #   co-issue => flat (slope~0, slots~0); a shared slot => cycles rise at once (slots~1). HIGH density saturates
        #   either way, so it can't tell them apart.
        lo = [(d, c) for d, c in pts if d <= nacc // 2]
        slope_lo = lin_slope(lo)
        slope_all = lin_slope(pts)
        slots_lo = slope_lo / warps_smsp
        slots_all = slope_all / warps_smsp
        sat = base["vec_per_smsp_per_cyc"] if base else 0.0
        saturated = sat > 0.9
        all_promoted = all(ro["promoted"] for ro in rowsout if ro["density"] > 0)
        md.append("")
        md.append(f"- Uniform chain promoted to UIADD3: **{'yes' if all_promoted else 'NO — chain stayed vector, result INVALID'}** "
                  f"(loop-control UIADD3 at D=0 = {base_u0}).")
        md.append(f"- Dispatch saturation (D=0): **{sat:.4f}** {VK_NAME[vk].upper()} a SMSP-cycle"
                  f" ({'saturated ~1/cyc' if saturated else 'below 1/cyc — NOT a full saturator, low-density test confounded'}).")
        md.append(f"- **Low-density slots/uni = {slots_lo:.3f}** (D≤{nacc//2}, the discriminating regime; slope {slope_lo:.3f} cyc/iter a unit D, /{warps_smsp:.0f} warps a SMSP).")
        md.append(f"- Full-range slots/uni = {slots_all:.3f} (slope {slope_all:.3f}); high D saturates any port, so this over-states co-issue cost.")
        if not all_promoted:
            verdict = "INVALID — the chain did not promote to UIADD3 (it ran on the vector datapath), so this measures vector cost, not uniform cost"
        elif not saturated:
            verdict = f"inconclusive — the vector saturator only reached {sat:.3f}/cyc, so spare dispatch could hide uniforms for free"
        elif slots_lo > 0.9:
            verdict = f"takes a full dispatch slot (~{slots_lo:.2f}) — VF2's {A_UNI} is a valid lower bound (SOUND)"
        elif slots_lo < A_UNI - 0.05:
            verdict = f"takes ~{slots_lo:.2f} slot, below VF2's {A_UNI} — VF2 OVER-prices a uniform instruction (UNSOUND)"
        else:
            verdict = f"takes ~{slots_lo:.2f} slot — consistent with VF2's {A_UNI}"
        md.append(f"- Verdict ({VK_NAME[vk].upper()} saturator): a uniform UIADD3 {verdict}.")
        md.append("")
        result["by_vk"][VK_NAME[vk]] = {"saturation_d0": sat, "saturated": saturated, "warps_smsp": warps_smsp,
                                        "all_promoted": all_promoted, "loop_control_uiadd3": base_u0,
                                        "slots_per_uni_low_density": slots_lo, "slots_per_uni_full": slots_all,
                                        "slope_low": slope_lo, "slope_full": slope_all, "rows": rowsout}

    (out / "uniform.md").write_text("\n".join(md) + "\n")
    (out / "uniform.json").write_text(json.dumps(result, indent=2))
    print("\n".join(md))


if __name__ == "__main__":
    main()
