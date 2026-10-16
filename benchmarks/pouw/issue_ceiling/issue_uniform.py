#!/usr/bin/env python3
"""Does a uniform-datapath instruction take the dispatch slot? Combine issue_uniform.jsonl with its SASS.

Saturate the SMSP dispatch with a vector kind (FFMA/IADD3) at ~1 instr a SMSP-cycle, then run a uniform-datapath INNER
LOOP at runtime trip counts `ucount`. The SASS gives the uniform-datapath op count a inner-iteration (U-ops); the timing
gives the cycles added a inner-iteration. slots a inner-iteration vs (U-ops + 1 BRA) says whether uniform instructions
consume dispatch slots: ~ (U-ops+1) => each uniform op takes a full slot (VF2's 0.594 is a valid lower bound); ~1 (only
the BRA) => uniform ops co-issue on their own port (0.594 exceeds real cost => VF2 unsound). LOW density is the
discriminator; high density saturates any port.

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
    """Every backward-branch loop as (body_len, histogram-of-opcodes)."""
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


def is_uop(op):
    k = op.split(".")[0]
    return k.startswith("U") or k in ("CS2UR", "S2UR", "R2UR")


def uop_count(hist):
    return sum(c for k, c in hist.items() if is_uop(k))


def main():
    out = Path(sys.argv[1])
    rows = [json.loads(l) for l in (out / "issue_uniform.jsonl").read_text().splitlines()
            if l.strip().startswith("{")]
    rows = [r for r in rows if r.get("probe") == "uniform" and "density" in r]
    funcs = funcs_with_lines(out / "sass" / "issue_uniform.sass")

    md = ["# Uniform-datapath probe: does a uniform instruction take the dispatch slot? (sm_120, RTX PRO 6000)\n"]
    md.append("Saturate the SMSP dispatch with a vector kind; run a uniform-datapath inner loop at runtime trip `ucount`.")
    md.append(f"VF2 §13 prices a uniform instruction at **{A_UNI} SMSP-cycles**. The inner loop body is `W` uniform ops +")
    md.append("1 BRA; `slots/inner-it` ≈ W+1 means each uniform op takes a full dispatch slot, ≈1 (just the BRA) means they")
    md.append("co-issue for free. LOW density is the discriminator; high density saturates any port.\n")
    result = {"a_uni": A_UNI, "by_vk": {}}

    for vk in (0, 1):
        vr = sorted([r for r in rows if r["vk"] == VK_NAME[vk]], key=lambda r: r["density"])
        if not vr:
            continue
        base = next((r for r in vr if r["density"] == 0), None)
        warps_smsp = (base["warps"] if base else vr[0]["warps"]) / 4.0

        # the SASS has one k_vec<vk> kernel; its outer loop holds the vector saturator, a nested inner loop holds the
        # uniform-datapath work. Inner loop = the backward-branch loop with uniform ops and no vector saturator.
        ls = by_tmpl(funcs, "k_vec", [vk]) or []
        outer = max(ls, key=lambda x: x[0])[1] if ls else {}
        vec_sat = outer.get(VK_SASS[vk], 0)
        inner_cands = [h for _, h in ls if uop_count(h) > 0 and h.get(VK_SASS[vk], 0) == 0]
        inner = max(inner_cands, key=uop_count) if inner_cands else {}
        w_uops = uop_count(inner)          # uniform-datapath ops a inner-iteration
        bra_inner = inner.get("BRA", 0)    # branch a inner-iteration (control, not uniform datapath)
        uop_mix = ", ".join(f"{c}×{k}" for k, c in sorted(inner.items()) if is_uop(k)) or "—"

        md.append(f"## Vector = {VK_NAME[vk].upper()}  (inner loop: {w_uops} uniform ops + {bra_inner} BRA a iteration: {uop_mix})\n")
        md.append("| ucount | vec/SMSP-cyc | inner-it/SMSP-cyc | cyc/outer-it | slots/inner-it (vs 0) | slots/uniform-op |")
        md.append("|---|---|---|---|---|---|")
        pts = []  # (ucount, cyc_per_outer_iter)
        rowsout = []
        base_cyc_it = base["cyc"] / base["iters"] if base else None
        for r in vr:
            uc = r["density"]
            cyc_it = r["cyc"] / r["iters"]
            n_inner_it = r["n_inner_smsp"] / r["iters"]  # = warps_smsp*ucount, inner-iters a SMSP a outer-iter
            pts.append((uc, cyc_it))
            slots_it = ((cyc_it - base_cyc_it) / n_inner_it) if (base_cyc_it is not None and uc > 0) else None
            slots_uop = (slots_it / w_uops) if (slots_it is not None and w_uops) else None
            md.append(f"| {uc} | {r['vec_per_smsp_per_cyc']:.4f} | {r['inner_per_smsp_per_cyc']:.4f} | {cyc_it:.1f} | "
                      f"{('%.3f' % slots_it) if slots_it is not None else '—'} | "
                      f"{('%.3f' % slots_uop) if slots_uop is not None else '—'} |")
            rowsout.append({"ucount": uc, "cyc_per_iter": cyc_it, "vec_per_smsp_per_cyc": r["vec_per_smsp_per_cyc"],
                            "inner_per_smsp_per_cyc": r["inner_per_smsp_per_cyc"], "slots_per_inner_it": slots_it,
                            "slots_per_uop": slots_uop, "iters": r["iters"]})

        lo = [(u, c) for u, c in pts if u <= 16]
        slope_lo = lin_slope(lo) / warps_smsp          # cyc added a inner-iteration (low density)
        slope_all = lin_slope(pts) / warps_smsp
        slots_uop_lo = slope_lo / w_uops if w_uops else None
        sat = base["vec_per_smsp_per_cyc"] if base else 0.0
        promoted = w_uops > 0 and bra_inner > 0  # a real uniform-datapath inner loop exists
        expected_if_slots = w_uops + bra_inner     # every op (uniform + branch) takes one dispatch slot
        md.append("")
        md.append(f"- Uniform inner loop found: **{'yes' if promoted else 'NO — no uniform-datapath inner loop in SASS, INVALID'}** "
                  f"({w_uops} uniform ops + {bra_inner} BRA a iteration).")
        md.append(f"- Dispatch saturation (ucount=0): **{sat:.4f}** {VK_NAME[vk].upper()} a SMSP-cycle"
                  f" ({'saturated ~1/cyc' if sat > 0.9 else 'below 1/cyc — spare dispatch can hide some uniforms'}).")
        md.append(f"- **Low-density slots a inner-iteration = {slope_lo:.3f}** (ucount≤16); if every op took a slot this would be ~{expected_if_slots} (={w_uops} uniform + {bra_inner} BRA).")
        md.append(f"- Low-density slots a uniform op = **{slots_uop_lo:.3f}**" if slots_uop_lo is not None else "- (no uniform ops counted)")
        md.append(f"- Full-range slots a inner-iteration = {slope_all:.3f} (high density saturates any port).")
        if not promoted:
            verdict = "INVALID — no uniform-datapath inner loop was emitted"
        elif slope_lo >= expected_if_slots - 1.0:
            verdict = (f"each uniform op takes ~{slots_uop_lo:.2f} dispatch slot — VF2's {A_UNI} is a valid lower bound (SOUND)")
        elif slope_lo <= bra_inner + 0.5:
            verdict = (f"uniform ops add ~0 slots beyond the BRA (slots a inner-it {slope_lo:.2f} ≈ {bra_inner} BRA) — they "
                       f"co-issue on their own port; VF2's {A_UNI}/op OVER-prices (UNSOUND)")
        else:
            verdict = (f"each uniform op takes ~{slots_uop_lo:.2f} slot — compare to VF2's {A_UNI}: "
                       + ("below the price, VF2 over-prices (unsound)" if (slots_uop_lo or 0) < A_UNI - 0.05 else "consistent with VF2"))
        md.append(f"- Verdict ({VK_NAME[vk].upper()} saturator): {verdict}.")
        md.append("")
        result["by_vk"][VK_NAME[vk]] = {"saturation_d0": sat, "warps_smsp": warps_smsp, "vec_saturator_count": vec_sat,
                                        "uniform_ops_per_inner_it": w_uops, "bra_per_inner_it": bra_inner,
                                        "inner_uop_mix": uop_mix, "promoted": promoted,
                                        "slots_per_inner_it_low": slope_lo, "slots_per_inner_it_full": slope_all,
                                        "slots_per_uop_low": slots_uop_lo, "rows": rowsout}

    (out / "uniform.md").write_text("\n".join(md) + "\n")
    (out / "uniform.json").write_text(json.dumps(result, indent=2))
    print("\n".join(md))


if __name__ == "__main__":
    main()
