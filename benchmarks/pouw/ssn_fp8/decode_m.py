"""ssn-1l-tag at decode on the H200, m = 1..64 rows a call: T1's shape term and T2, by tensor instruction.

    python decode_m.py THEORY_DIR INPUTS.json OUT.json > table.md

Drives the whole-step model (`step_cover.py` in THEORY_DIR, with its `noise_sweep.py`) at decode points dec<m>_<ctx>.
Prices are its per-register H200 prices (`wgmma_price`, `TENSOR`; wgmma at N 24, 40, 48 and 56 is its 16 + 1024/N
between the widths it prices). T2 is anchored on a 64-row forward of the 64 linear calls, measured and read from
INPUTS.json, over the step model's decode stock step.
"""
from __future__ import annotations

import dataclasses
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import step_cover as sc

MS = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64]
CTX = [2048, 8192]
READING = "paired"
INP = json.loads(Path(sys.argv[2]).read_text())


def ceil(x: float, q: int) -> int:
    return q * math.ceil(x / q)


def options(m: int) -> list[tuple[str, float, float]]:
    """(instruction, units a tile MAC, tile MACs a real MAC) for s8 at m rows."""
    kreg = sc.TENSOR["s8"]["kreg"]
    mma = sc.TENSOR["s8"]["mma"] / kreg
    return [("wgmma m64 (rows padded to 64)", sc.wgmma_price(256, READING) / kreg, ceil(m, 64) / m),
            (f"wgmma swapped (N {min(256, ceil(m, 8))})", sc.wgmma_price(min(256, ceil(m, 8)), READING) / kreg,
             ceil(m, 8) / m),
            ("mma.sync m16n8k32 swapped (N 8)", mma, ceil(m, 8) / m),
            ("mma.sync m16n8k32 (M 16)", mma, ceil(m, 16) / m),
            ("mma.sync m16n8k16 swapped (N 8)", 32.01 / 16, ceil(m, 8) / m)]


def instr_rows(m: int) -> list[dict]:
    """Each instruction's shape term over stock (stock at its cheapest instruction, its padding not in its favour):
    committed padding forced at min(price, 1) a tile MAC (H32's floor), TensorSemantics at its price, uncommitted
    padding forced at 1 a real MAC."""
    opts = options(m)
    stock = min(p * f for _, p, f in opts)
    out = []
    for name, p, f in opts:
        out.append({"instr": name, "units_tile_mac": p, "pad": f, "units_real_mac": p * f, "stock_units_real_mac": stock,
                    "shape_committed": (p - min(p, 1.0)) * f / stock,
                    "shape_ts": 0.0,
                    "shape_uncommitted": (p * f - min(p, 1.0)) / stock,
                    "honest_over_stock": p * f / stock})
    return out


def lin_terms(m: int, ctx: int, plan) -> dict:
    """The linears alone under `plan` at dec<m>: their stock, honest, forced and the shape part of the gap."""
    pt = f"dec{m}_{ctx // 1024}k"
    comps = [c for c in sc.components("llama", pt) if c.family in ("linear", "lm_head")]
    acc = {"linear": dict(stock=0.0, honest=0.0, forced=0.0, forming=0.0, words=0.0, shape=0.0),
           "lm_head": dict(stock=0.0, honest=0.0, forced=0.0, forming=0.0, words=0.0, shape=0.0)}
    instr = {}
    for c in comps:
        s = sc.tensor_comp(c, plan, READING, stock=True)
        r = sc.tensor_comp(c, plan, READING, stock=False)
        a = acc[c.family]
        a["stock"] += s["stock"]
        a["honest"] += r["honest"]
        a["forced"] += r["forced"]
        a["forming"] += r["forming"]
        a["words"] += r["words"]
        a["shape"] += r["honest"] - r["forming"] - r["forced"]
        instr[c.family] = r["instr"]
    return {"acc": acc, "instr": instr}


def stock_ms(m: int, ctx: int) -> float:
    return sc.stock_ms("llama", f"dec{m}_{ctx // 1024}k")[0]


def forward_ms(m: int, scheme: str, pad: str) -> float:
    """The linears' forward (16 layers x 4 calls) at m rows: the measured 64-row forward under `pad` = "64" (the wgmma
    kernel tiles M by 64, so every m <= 64 runs it); under "8" (mma.sync swapped) the 64-row forward's parts, the
    words' bytes scaled by ceil(m, 8)/64 and the per-call latencies (root(A_j), the trees, coins) held."""
    f = INP["forward_64"][scheme]
    if pad == "64":
        return f["forward_ms"]
    share = INP["parts_64_serial"]
    s = ceil(m, 8) / 64
    if scheme.startswith("serial"):
        scaled = share["leaves"] + share["export"] + share["gemm"]
        held = share["tree"] + share["root_a"] + share["coins_forming"]
        return f["forward_ms"] * (scaled * s + held)
    gemm_path = f["root_a_ms"] + f["coins_forming_ms"] + f["gemm_export_ms"] * s
    return max(gemm_path, f["forward_ms"] * s)


def main() -> int:
    for m in MS:
        for ctx in CTX:
            sc.POINTS[f"dec{m}_{ctx // 1024}k"] = {"prompts": m, "tokens": 1, "context": ctx, "decode": True,
                                                   "label": f"decode, batch {m}, context {ctx // 1024}K"}
    base = sc.PLAN["best_p4_nkd"]
    no_ts = frozenset(base.premises - {"TensorSemantics"})
    plans = {
        "tcbest_ts": base,
        "tcbest_nots": dataclasses.replace(base, name="best_p4_nkd-noTS", premises=no_ts),
        "pad64_nots": dataclasses.replace(base, name="best_p4_nkd-pad64-noTS", premises=no_ts, small_m="wgmma64"),
        "uncommitted_nots": dataclasses.replace(base, name="best_p4_nkd-uncommitted-noTS", premises=no_ts,
                                                commit_pad=False),
    }
    rep = {"reading": READING, "ms": MS, "ctx": CTX, "inputs": INP, "instr": {}, "linears": {}, "step": {}, "t2": {}}
    for m in MS:
        rep["instr"][m] = instr_rows(m)
    for ctx in CTX:
        for m in MS:
            key = f"{m}_{ctx}"
            pt = f"dec{m}_{ctx // 1024}k"
            rep["linears"][key] = {}
            rep["step"][key] = {}
            for pk, plan in plans.items():
                lt = lin_terms(m, ctx, plan)
                ev = sc.evaluate("llama", pt, plan, READING)
                lin = lt["acc"]["linear"]
                head = lt["acc"]["lm_head"]
                rep["linears"][key][pk] = {
                    "instr": lt["instr"], "lin_shape_over_lin_stock": lin["shape"] / lin["stock"],
                    "lin_shape_over_step": lin["shape"] / ev["stock_units"],
                    "head_shape_over_step": head["shape"] / ev["stock_units"],
                    "lin_words_per_row": lin["words"] / m, "head_words_per_row": head["words"] / m,
                    "lin_stock_share": lin["stock"] / ev["stock_units"]}
                rep["step"][key][pk] = {k: ev[k] for k in ("t1", "t1_matmul", "coverage", "t1_coins_uncredited",
                                                           "linears_gap", "linears_stock_share", "w1_per_mac",
                                                           "stock_units", "hash_bytes")}
                rep["step"][key][pk]["t2_model"] = {k: v for k, v in ev["t2"].items() if k.startswith("t2_")}
                rep["step"][key][pk]["groups_gap"] = {g: v["gap_share"] for g, v in ev["groups"].items()}
            st = stock_ms(m, ctx)
            t2 = {"stock_ms": st}
            for scheme in INP["forward_64"]:
                for pad in ("64", "8"):
                    fw = forward_ms(m, scheme, pad)
                    plain = INP["plain_forward_64_ms"] * (1.0 if pad == "64" else ceil(m, 8) / 64)
                    t2[f"{scheme}_pad{pad}"] = {"forward_ms": fw, "t2": (st + fw - plain) / st}
            for pk, pad in (("pad64_nots", "64"), ("tcbest_nots", "8")):
                hb = 4 * rep["linears"][key][pk]["head_words_per_row"] * m
                t2[f"lm_head_pad{pad}"] = {"bytes": hb, **{
                    f"added_t2_{s}": hb * INP["units_per_byte"][s] / INP["units_per_s"] * 1e3 / st
                    for s in INP["units_per_byte"]}}
            rep["t2"][key] = t2
    rep["periods"] = {}
    for m in (1, 8, 64):
        for pk in ("tcbest_nots", "pad64_nots"):
            key = f"{m}_2048"
            li, st = rep["linears"][key][pk], rep["step"][key][pk]
            rep["periods"][f"{m}_2048_{pk}"] = {"share_of_step": li["lin_shape_over_step"],
                                                "gates": sc.periods(li["lin_shape_over_step"], st["w1_per_mac"])}
    Path(sys.argv[3]).write_text(json.dumps(rep, indent=1, default=str))
    print(table(rep))
    return 0


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def table(rep: dict) -> str:
    out = ["## Instructions at m rows (s8, paired; price a real MAC, shape term over the linears' stock)", "",
           "| m | stock's best (units a real MAC) | wgmma m64 pad 64 | wgmma swapped | mma.sync m16n8k32 swapped | "
           "mma.sync m16n8k32 M16 | m16n8k16 swapped |", "|---|---|---|---|---|---|---|"]
    for m in rep["ms"]:
        rows = rep["instr"][m]
        best = min(rows, key=lambda r: r["units_real_mac"])
        cells = [f"{r['units_real_mac']:.2f} / {pct(r['shape_committed'])} / {pct(r['shape_uncommitted'])}"
                 for r in rows]
        out.append(f"| {m} | {best['units_real_mac']:.2f} ({best['instr']}) | " + " | ".join(cells) + " |")
    out += ["", "Cells: units a real MAC / shape term with padding committed (no TensorSemantics) / uncommitted.", ""]
    for ctx in rep["ctx"]:
        out += [f"## Step at decode {ctx // 1024}K (Llama-3.2-1B, best_p4_nkd and variants; T1 paired)", "",
                "| m | linears' share | tc_best + TS: T1 | tc_best no TS: linears' shape (of linears / of step) / step T1 | "
                "pad64 no TS: linears' shape / step T1 | uncommitted no TS: step T1 | T2 model tc_best "
                "(serial / overlapped) | T2 model pad64 |",
                "|---|---|---|---|---|---|---|---|"]
        ks = ("t2_serial", "t2_overlapped")
        for m in rep["ms"]:
            s = rep["step"][f"{m}_{ctx}"]
            li = rep["linears"][f"{m}_{ctx}"]
            t2a, t2b = s["tcbest_ts"]["t2_model"], s["pad64_nots"]["t2_model"]
            out.append(f"| {m} | {pct(li['tcbest_ts']['lin_stock_share'])} | {pct(s['tcbest_ts']['t1'])} | "
                       f"{pct(li['tcbest_nots']['lin_shape_over_lin_stock'])} / "
                       f"{pct(li['tcbest_nots']['lin_shape_over_step'])} / {pct(s['tcbest_nots']['t1'])} | "
                       f"{pct(li['pad64_nots']['lin_shape_over_step'])} / {pct(s['pad64_nots']['t1'])} | "
                       f"{pct(s['uncommitted_nots']['t1'])} | "
                       + " / ".join(f"{t2a[k]:.1f}×" for k in ks) + " | "
                       + " / ".join(f"{t2b[k]:.1f}×" for k in ks) + " |")
        out.append("")
        out += [f"## The linears' T2 at decode {ctx // 1024}K, anchored on the measured 64-row forward", "",
                "| m | stock ms (est) | " + " | ".join(f"{k} pad64 / pad8" for k in rep["inputs"]["forward_64"])
                + " | the lm_head under ssn_1l adds, pad64 (beside / serial) |",
                "|---|---|" + "---|" * (len(rep["inputs"]["forward_64"]) + 1)]
        for m in rep["ms"]:
            t = rep["t2"][f"{m}_{ctx}"]
            h = t["lm_head_pad64"]
            out.append(f"| {m} | {t['stock_ms']:.2f} | " + " | ".join(
                f"{t[k + '_pad64']['t2']:.1f}× / {t[k + '_pad8']['t2']:.1f}×" for k in rep["inputs"]["forward_64"])
                + f" | +{h['added_t2_beside']:.1f}× / +{h['added_t2_serial']:.1f}× |")
        out.append("")
    out += ["## The linears' shape term per monitoring period (AND gates, decode 2K, all compute at the point)", "",
            "| m, plan | share of the step | scope | window | week | month | year |", "|---|---|---|---|---|---|---|"]
    for k, v in rep["periods"].items():
        for scope, g in v["gates"].items():
            out.append(f"| {k} | {pct(v['share_of_step'])} | {scope} | " + " | ".join(
                f"{g[p]:.3g}" for p in ("window", "week", "month", "year")) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
