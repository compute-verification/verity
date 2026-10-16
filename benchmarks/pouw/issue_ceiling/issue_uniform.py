#!/usr/bin/env python3
"""Does a uniform-datapath instruction take the dispatch slot? Combine issue_uniform.jsonl with its SASS.

Saturate the SMSP dispatch with a vector kind (FFMA/IADD3) at ~1 instr a SMSP-cycle, add a live UIADD3 chain at
densities D (uniform a SMSP-iter). slots a uniform = Δ(cycles a outer-iter) / Δ(#uniform a outer-iter), i.e. the slope of
cycles-per-iter vs D, /4 (4 warps a SMSP). ~1 => uniform takes a full slot (VF2's 0.594 is a valid lower bound); ~0 =>
uniform takes no slot (0.594 exceeds real cost => VF2 unsound).

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

    for vk in (0, 1):
        vr = sorted([r for r in rows if r["vk"] == VK_NAME[vk]], key=lambda r: r["density"])
        if not vr:
            continue
        base = next((r for r in vr if r["density"] == 0), None)
        md.append(f"## Vector = {VK_NAME[vk].upper()}\n")
        md.append("| density (uni/SMSP-it) | vec/SMSP-cyc | uni/SMSP-cyc | cyc/outer-it | slots/uni (vs D=0) | SASS (vec / uni) |")
        md.append("|---|---|---|---|---|---|")
        pts = []  # (D, cyc_per_iter, n_uni_per_iter)
        rowsout = []
        for r in vr:
            D = r["density"]
            cyc_it = r["cyc"] / r["iters"]
            n_uni_it = r["n_uni_smsp"] / r["iters"]  # = 4*D
            pts.append((D, cyc_it, n_uni_it))
            ls = by_tmpl(funcs, "k_vec", [vk, D])
            vget = uget = 0
            uni_other = 0
            if ls:
                body = max(ls, key=lambda x: x[0])[1] if ls else {}
                vget = body.get(VK_SASS[vk], 0)
                uget = body.get("UIADD3", 0)
                uni_other = sum(c for k, c in body.items() if (k.startswith("U") and k != "UIADD3") or k in ("R2UR", "CS2UR"))
            slots = ((cyc_it - base["cyc"] / base["iters"]) / n_uni_it) if (base and D > 0) else None
            sass = f"{vget}×{VK_SASS[vk]} / {uget}×UIADD3" + (f" (+{uni_other} U-other)" if uni_other else "")
            md.append(f"| {D} ({n_uni_it:.0f}) | {r['vec_per_smsp_per_cyc']:.4f} | {r['uni_per_smsp_per_cyc']:.4f} | "
                      f"{cyc_it:.1f} | {('%.3f' % slots) if slots is not None else '—'} | {sass} |")
            rowsout.append({"density": D, "cyc_per_iter": cyc_it, "vec_per_smsp_per_cyc": r["vec_per_smsp_per_cyc"],
                            "uni_per_smsp_per_cyc": r["uni_per_smsp_per_cyc"], "slots_per_uni_vs0": slots,
                            "sass_vec": vget, "sass_uiadd3": uget, "sass_uni_other": uni_other, "iters": r["iters"]})
        # linear fit cyc_per_iter = a + slope*D  => slots/uni = slope/4 (4 warps a SMSP)
        n = len(pts)
        sx = sum(d for d, _, _ in pts); sy = sum(c for _, c, _ in pts)
        sxx = sum(d * d for d, _, _ in pts); sxy = sum(d * c for d, c, _ in pts)
        slope = (n * sxy - sx * sy) / (n * sxx - sx * sx) if (n * sxx - sx * sx) else 0.0
        slots_fit = slope / 4.0
        sat = base["vec_per_smsp_per_cyc"] if base else 0.0
        md.append("")
        md.append(f"- Dispatch saturation (D=0): **{sat:.4f}** {VK_NAME[vk].upper()} a SMSP-cycle"
                  f" ({'saturated ~1/cyc' if sat > 0.9 else 'below 1/cyc — not a full saturator'}).")
        md.append(f"- **Linear fit: {slots_fit:.3f} dispatch slots a uniform instruction** (slope {slope:.3f} cyc/iter a"
                  f" unit density, /4 warps a SMSP).")
        verdict = ("takes a full slot — VF2's 0.594 is a valid lower bound (sound)" if slots_fit > 0.9 else
                   f"takes ~{slots_fit:.2f} slot — compare to VF2's {A_UNI}: "
                   + ("below the price, VF2 OVER-prices (unsound)" if slots_fit < A_UNI - 0.05 else "consistent with VF2"))
        md.append(f"- Verdict: a uniform UIADD3 {verdict}.")
        md.append("")
        result["by_vk"][VK_NAME[vk]] = {"saturation_d0": sat, "slots_per_uni_fit": slots_fit,
                                        "slope_cyc_per_iter_per_density": slope, "rows": rowsout}

    (out / "uniform.md").write_text("\n".join(md) + "\n")
    (out / "uniform.json").write_text(json.dumps(result, indent=2))
    print("\n".join(md))


if __name__ == "__main__":
    main()
