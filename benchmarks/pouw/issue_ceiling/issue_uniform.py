#!/usr/bin/env python3
"""Does a uniform instruction take a vector dispatch slot? Warp-specialized test, issue_uniform.jsonl + its SASS.

On each SMSP: nf warps run a pure vector saturator (FFMA/IADD3), nu warps run pure uniform-datapath work (a runtime-trip
inner loop). We run vector-alone (nf,0), uniform-alone (0,nu) and both (nf,nu) and compare cycles a OUTER iteration (each
outer iteration both warp groups do one unit of work):
  - both ≈ vector-alone + uniform-alone  => one shared dispatch port: a uniform instruction consumes a vector slot
    (VF2's 0.594/op is a valid lower bound).
  - both ≈ max(vector-alone, uniform-alone) => a separate uniform port: a uniform instruction takes NO vector slot
    (0.594/op over-prices => VF2 unsound).
The SASS gives the uniform ops a inner-iteration, so we can turn the added dispatch time into slots a uniform instruction.

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
        if len(body) < 2:
            continue
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
    rows = [r for r in rows if r.get("probe") == "spec" and "ucount" in r]
    funcs = funcs_with_lines(out / "sass" / "issue_uniform.sass")

    md = ["# Uniform-datapath probe: does a uniform instruction take a vector dispatch slot? (sm_120, RTX PRO 6000)\n"]
    md.append("Warp-specialized: nf vector warps + nu uniform warps on the same SMSPs. Compare cycles a OUTER iteration of")
    md.append("vector-alone (a), uniform-alone (b) and both (c). c≈a+b => shared dispatch (uniform takes a slot); c≈max(a,b)")
    md.append(f"=> separate uniform port (uniform takes no slot => VF2's {A_UNI}/op over-prices).\n")
    result = {"a_uni": A_UNI, "by_vk": {}}

    for vk in (0, 1):
        vr = [r for r in rows if r["vk"] == VK_NAME[vk]]
        if not vr:
            continue
        ls = by_tmpl(funcs, "k_spec", [vk]) or []
        cand = [(L, h) for L, h in ls if uop_count(h) >= 2]
        n_inner, inner = min(cand, key=lambda x: x[0]) if cand else (0, {})
        w_uops = uop_count(inner)
        mix = ", ".join(f"{c}×{k}" for k, c in sorted(inner.items())) or "—"

        md.append(f"## Vector = {VK_NAME[vk].upper()}  (uniform inner loop: N={n_inner} instrs, {w_uops} uniform a iteration: {mix})\n")
        md.append("| ucount | a: vec-alone cyc/it | b: uni-alone cyc/it | c: both cyc/it | max(a,b) | a+b | verdict | vec rate alone→both | slots/uniform-op |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        vkout = {"n_inner_instrs": n_inner, "uniform_ops_per_inner_it": w_uops, "inner_mix": mix, "rows": []}
        for uc in sorted(set(r["ucount"] for r in vr)):
            g = {(r["nf"], r["nu"]): r for r in vr if r["ucount"] == uc}
            va, ua, both = g.get((4, 0)), g.get((0, 4)), g.get((4, 4))
            if not (va and ua and both):
                continue
            a = va["cyc"] / va["iters"]
            b = ua["cyc"] / ua["iters"]
            c = both["cyc"] / both["iters"]
            maxab, sumab = max(a, b), a + b
            # uniform instructions added per SMSP a outer-iteration in "both": nu * ucount * w_uops
            uni_instrs_it = both["nu"] * uc * w_uops
            # dispatch time the uniform work added on top of the vector saturator; slots a uniform op = added / uni instrs
            added = c - a
            slots_uop = added / uni_instrs_it if uni_instrs_it else None
            # classify by whether both is closer to max(a,b) [parallel] or a+b [serial]
            if abs(c - maxab) <= abs(c - sumab) and (sumab - maxab) > 0.05 * sumab:
                cls = "co-issue (separate port)"
            elif abs(c - sumab) < abs(c - maxab):
                cls = "shared dispatch"
            else:
                cls = "ambiguous"
            vrate_alone = va["vec_per_smsp_per_cyc"]
            vrate_both = both["vec_per_smsp_per_cyc"]
            md.append(f"| {uc} | {a:.1f} | {b:.1f} | {c:.1f} | {maxab:.1f} | {sumab:.1f} | {cls} | "
                      f"{vrate_alone:.3f}→{vrate_both:.3f} | {('%.3f' % slots_uop) if slots_uop is not None else '—'} |")
            vkout["rows"].append({"ucount": uc, "cyc_vec_alone_it": a, "cyc_uni_alone_it": b, "cyc_both_it": c,
                                  "max_ab": maxab, "sum_ab": sumab, "class": cls, "vec_rate_alone": vrate_alone,
                                  "vec_rate_both": vrate_both, "uni_instrs_per_it": uni_instrs_it,
                                  "slots_per_uniform_op": slots_uop})
        md.append("")
        # headline: average over ucount of slots/uniform-op and the dominant class
        srows = vkout["rows"]
        if srows:
            classes = [r["class"] for r in srows]
            dom = max(set(classes), key=classes.count)
            su = [r["slots_per_uniform_op"] for r in srows if r["slots_per_uniform_op"] is not None]
            su_avg = sum(su) / len(su) if su else None
            md.append(f"- Dominant behaviour: **{dom}**.")
            if su_avg is not None:
                md.append(f"- Slots a uniform instruction (added dispatch time ÷ uniform instrs, both case): **{su_avg:.3f}**"
                          f" — vs VF2's {A_UNI}: "
                          + ("at/above the price (VF2 is a valid lower bound, SOUND)" if su_avg >= A_UNI - 0.05
                             else "BELOW the price (VF2 over-prices a uniform instruction, UNSOUND)") + ".")
            vkout["dominant_class"] = dom
            vkout["slots_per_uniform_op_avg"] = su_avg
        md.append("")
        result["by_vk"][VK_NAME[vk]] = vkout

    (out / "uniform.md").write_text("\n".join(md) + "\n")
    (out / "uniform.json").write_text(json.dumps(result, indent=2))
    print("\n".join(md))


if __name__ == "__main__":
    main()
