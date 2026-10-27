"""beside.sh's tables and VERDICT lines, from its output directory.

lowreg: per variant, register cap and warpgroups an SM, the median over rounds of compressions a second and an SM-clock,
against BAR (sx::compress<0, 1>'s full-occupancy rate on the H200, 4.355 G/s at 1830 MHz: 0.01803 an SM-clock), the
registers ptxas gave and the local memory (spills). The verdict: for one warpgroup an SM, the fewest registers at which a
variant reaches 90% of BAR an SM-clock, and the best rate at each cap.

beside: per kernel, the median over rounds of each arm. The co-issue: the GEMM's span (%globaltimer, first MMA to last
epilogue) beside hash warpgroups that hash throughout (free) against the GEMM alone (mma), and the hash's rate beside the
GEMM against alone; the handoff (the pipeline) against the larger of its GEMM and its hash alone, and against wgmma.linear's
own shape (wgl_n256_s4, three warpgroups, n256, no O tile) alone. Nsight Compute's pipe shares for NCU_RUNS beside them,
and its warp-stall counters: the hash warps' stalls beside the GEMM are the free arm's less the mma arm's.
The served model: the served commit hashes each output's u32 word in 16 KiB tile-local rows, about 133 compressions a row
(its row_v2 digest's 130, its commit string and tree leaf), so the GEMM at M x N owes 4 M N / 16384 x 133 compressions. A
kernel's modeled call is its GEMM beside the hash (free) and then the rest at its hash-alone rate, against the GEMM alone in
wgmma.linear's shape and then the hash at the bar on every SM (serial).

    python3 beside_summary.py OUT_DIR    # markdown tables, then VERDICT lines; summary.json beside them
"""
from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

BAR_GS, BAR_CLK = 4.355, 0.01803
CAPS = (64, 72, 80, 96, 128, 255)


def jsonl(path: Path) -> list[dict]:
    out, bad = [], []
    if not path.exists():
        return out
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.startswith("{"):
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            bad.append((n, str(e)))
    if bad:
        print(f"beside_summary: {path}: {len(bad)} line(s) left out, unparsed (first, line {bad[0][0]}: {bad[0][1]})",
              file=sys.stderr)
    return out


def lowreg(o: Path, summary: dict) -> list[str]:
    rows = [r for r in jsonl(o / "lowreg-rates.jsonl") if "var" in r]
    if not rows:
        return ["VERDICT lowreg: not run (no lowreg-rates.jsonl)"]
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        cells[(r["var"], r["cap"], r["wgs"])].append(r)
    med = {}
    for k, rs in cells.items():
        med[k] = {"gcomp_per_s": statistics.median(r["gcomp_per_s"] for r in rs),
                  "per_clk": statistics.median(r["comp_per_sm_clk"] for r in rs),
                  "mhz": statistics.median(r["mhz"] for r in rs), "regs": rs[0]["regs"], "local": rs[0]["local_bytes"],
                  "rounds": len(rs)}
    summary["lowreg"] = {f"{v}/{c}/{w}": m for (v, c, w), m in med.items()}
    variants = list(dict.fromkeys(k[0] for k in med))
    lines = []
    for w in sorted({k[2] for k in med}):
        lines += [f"\n### {w} warpgroup(s) an SM: G compressions/s (an SM-clock as % of {BAR_CLK}), regs, spill B", "",
                  "| variant | " + " | ".join("none" if c == 255 else str(c) for c in CAPS) + " |",
                  "|---" * (len(CAPS) + 1) + "|"]
        for v in variants:
            cols = []
            for c in CAPS:
                m = med.get((v, c, w))
                cols.append("" if m is None else
                            f"{m['gcomp_per_s']:.3f} ({100 * m['per_clk'] / BAR_CLK:.0f}%) {m['regs']}r"
                            + (f" {m['local']}B" if m["local"] else ""))
            lines.append(f"| {v} | " + " | ".join(cols) + " |")
    lines.append("")
    for c in CAPS:
        best = max(((v, med[(v, c, 1)]) for v in variants if (v, c, 1) in med), key=lambda x: x[1]["per_clk"], default=None)
        if best:
            v, m = best
            lines.append(f"VERDICT lowreg cap {'none' if c == 255 else c}, 1 warpgroup an SM: best {v} {m['gcomp_per_s']:.3f} "
                         f"G/s at {m['mhz']:.0f} MHz, {m['per_clk']:.5f} an SM-clock = {m['per_clk'] / BAR_CLK:.3f}x the bar "
                         f"{BAR_CLK} ({BAR_GS} G/s at 1830 MHz); {m['regs']} registers, {m['local']} B local")
    hit = [(c, v) for c in CAPS for v in variants if (v, c, 1) in med and med[(v, c, 1)]["per_clk"] >= 0.9 * BAR_CLK]
    if hit:
        c, v = min(hit)
        lines.append(f"VERDICT lowreg: the fewest registers at which one warpgroup an SM reaches 90% of the bar: cap "
                     f"{'none' if c == 255 else c} ({v}, {med[(v, c, 1)]['per_clk'] / BAR_CLK:.3f}x)")
    else:
        top = max(((v, c) for v in variants for c in CAPS if (v, c, 1) in med), key=lambda k: med[(k[0], k[1], 1)]["per_clk"])
        lines.append(f"VERDICT lowreg: no cap reaches 90% of the bar with one warpgroup an SM; the best is {top[0]} at cap "
                     f"{top[1]}, {med[(top[0], top[1], 1)]['per_clk'] / BAR_CLK:.3f}x")
    return lines


REF = "wgl_n256_s4"
O_BYTES, LEAF_BYTES, LEAF_COMP = 4, 16384, 133
NCU_COLS = (("tensor", ("sm__pipe_tensor_op_gmma_cycles_active", "sm__pipe_tensor_cycles_active")),
            ("alu", ("sm__pipe_alu_cycles_active",)), ("fma", ("sm__pipe_fma_cycles_active",)),
            ("fmaheavy", ("sm__pipe_fmaheavy_cycles_active",)), ("alu inst", ("sm__inst_executed_pipe_alu",)),
            ("fma inst", ("sm__inst_executed_pipe_fma",)), ("lsu inst", ("sm__inst_executed_pipe_lsu",)),
            ("shared", ("sm__pipe_shared_cycles_active",)), ("issue", ("smsp__issue_active",)))


NUMBER = re.compile(r"-?\d+(\.\d+)?([eE][-+]?\d+)?")


def ncu_value(cell: str) -> float | None:
    """A raw-page cell as a number ("1,234.5"), or None for what ncu prints when it has none ("", "n/a")."""
    s = cell.replace(",", "").strip()
    return float(s) if NUMBER.fullmatch(s) else None


def ncu_row(path: Path) -> dict[str, float]:
    """The last launch's pct_of_peak_sustained_elapsed for NCU_COLS' metrics, from ncu's raw page (header, units, rows)."""
    rows = list(csv.reader(path.open()))
    if len(rows) < 3:
        return {}
    head, last = rows[0], rows[-1]
    out = {}
    for label, names in NCU_COLS:
        for n in names:
            col = f"{n}.avg.pct_of_peak_sustained_elapsed"
            v = ncu_value(last[head.index(col)]) if col in head else None
            if v is not None:
                out[label] = v
                break
    return out


STALL = "smsp__warps_issue_stalled_"


def ncu_stalls(path: Path) -> dict[str, float]:
    """The last launch's warps a scheduler in each stall reason an active cycle (`smsp__warps_issue_stalled_*` and
    `smsp__warps_active`, .avg.per_cycle_active), keyed by the reason ("active" for the warps resident)."""
    rows = list(csv.reader(path.open()))
    if len(rows) < 3:
        return {}
    head, last = rows[0], rows[-1]
    out = {}
    for i, col in enumerate(head):
        name = col[len(STALL):] if col.startswith(STALL) else "active" if col.startswith("smsp__warps_active.") else None
        v = ncu_value(last[i]) if name is not None and col.endswith(".avg.per_cycle_active") else None
        if v is not None:
            out[name.split(".")[0]] = v
    return out


def stall_verdicts(stalls: dict[str, dict[str, float]]) -> list[str]:
    """For each kernel with mma, hash and free runs: the hash warps' stalls beside the GEMM, free less mma (the MMA
    warpgroups' own stalls, taken as unchanged beside the hash), against the hash alone, the largest reasons first."""
    out = []
    for k in sorted({n.rsplit("-", 1)[0] for n in stalls}):
        mma, alone, free = (stalls.get(f"{k}-{a}") for a in ("mma", "hash", "free"))
        if not (mma and alone and free):
            continue
        beside = {r: free[r] - mma.get(r, 0.0) for r in free if r != "active"}
        top = sorted(beside, key=lambda r: -max(beside[r], alone.get(r, 0.0)))[:5]
        parts = ", ".join(f"{r} {beside[r]:.3f} (alone {alone.get(r, 0.0):.3f})" for r in top)
        out.append(f"VERDICT beside stalls {k}: the hash warps beside the GEMM (free - mma), warps a scheduler an active "
                   f"cycle: {parts}")
    return out


def _n(n: int, name: str, v: int) -> str:
    return f"{name} {v}" if n == 1 else f"{n} x {name} {v}"


def roles(m: dict) -> str:
    return f"{m['P']} / {m['nm']} x {m['Y']} / " + (f"{m['nh']} x {m['H']}" if m["nh"] else "-")


def budget(m: dict) -> str:
    """P + NM Y + NH H against the launch's registers x the warpgroups (each warpgroup holds a warp on every SM sub-partition,
    so 128 x the sum <= 64K an SM)."""
    terms = [f"P {m['P']}", _n(m["nm"], "Y", m["Y"])] + ([_n(m["nh"], "H", m["H"])] if m["nh"] else [])
    total = m["P"] + m["nm"] * m["Y"] + m["nh"] * m["H"]
    return (f"{' + '.join(terms)} = {total} <= {m['regs'] * m['nwg']} registers a thread across the warpgroups ({m['regs']} "
            f"at launch x {m['nwg']} warpgroups; 128 x that <= 64K an SM)")


def served_call(head: dict, med: dict, k: str, ref: dict | None) -> dict | None:
    a, h, f = (med.get((k, x)) for x in ("mma", "hash", "free"))
    if not (a and h and f and ref and h["gcomp"] and f["mma_us"] and h["mhz"]):
        return None
    comp = O_BYTES * head["M"] * head["N"] / LEAF_BYTES * LEAF_COMP
    done = min(comp, f["gcomp"] * 1e3 * f["mma_us"])
    us = f["mma_us"] + (comp - done) / (h["gcomp"] * 1e3)
    serial = ref["mma_us"] + comp / (BAR_CLK * h["mhz"] * head["sms"])
    return {"comp": comp, "beside": done, "gemm_us": f["mma_us"], "alone_gs": h["gcomp"], "us": us, "serial_us": serial}


def beside(o: Path, summary: dict) -> list[str]:
    lines_in = jsonl(o / "beside-time.jsonl")
    head = next((r for r in lines_in if "sms" in r), None)
    rows = [r for r in lines_in if "kernel" in r]
    if not rows or head is None:
        return ["VERDICT beside: not run (no beside-time.jsonl)"]
    grid = head["grid"]
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        cells[(r["kernel"], r["arm"])].append(r)
    med = {}
    for k, rs in cells.items():
        m = {f: statistics.median(r[f] for r in rs) for f in ("ms", "mma_us", "hash_us", "compressions", "tops", "mhz",
                                                               "watts")}
        m.update({f: rs[0][f] for f in ("P", "Y", "H", "nwg", "nform", "stages", "var", "regs", "local_bytes")})
        m["nm"] = rs[0].get("nm", 2)
        m["ot"] = rs[0].get("ot", 0 if m["stages"] == 4 else 1)
        m["nh"] = rs[0].get("nh", m["nwg"] - 1 - m["nm"])
        m["grid"] = rs[0].get("grid", grid)
        m["gcomp"] = m["compressions"] / (m["hash_us"] * 1e3) if m["hash_us"] else 0.0
        m["per_clk"] = m["compressions"] / (m["hash_us"] * m["mhz"] * m["grid"]) if m["hash_us"] and m["mhz"] else 0.0
        m["rounds"] = len(rs)
        med[k] = m
    summary["beside"] = {f"{k}/{a}": m for (k, a), m in med.items()}
    summary["beside_head"] = head
    kernels = list(dict.fromkeys(k for k, _ in med))
    ref = med.get((REF, "mma"))
    lines = [f"\n### beside, M {head['M']} N {head['N']} K {head['K']}, grid {grid} at 128-row tiles on {head['sms']} SMs "
             f"({head['device']})",
             "", "| kernel | P / NM x Y / NH x H | hash | regs, spill B | GEMM alone us (TOPS) | GEMM beside the hash us (+%) | "
             "hash alone G/s (an SM-clock, x bar) | hash beside the GEMM G/s (x alone) | handoff ms (TOPS) | handoff / "
             "max(alone) | handoff / wgmma.linear's |", "|---" * 11 + "|"]
    verdicts = []
    for k in kernels:
        a, h, f, ho = (med.get((k, x)) for x in ("mma", "hash", "free", "handoff"))
        if a is None:
            continue
        regs = f"{a['regs']}" + (f", {a['local_bytes']}" if a["local_bytes"] else "")
        cols = [k, roles(a), a["var"] if a["nh"] else "-", regs, f"{a['mma_us']:.1f} ({a['tops']:.0f})"]
        cols.append(f"{f['mma_us']:.1f} ({100 * (f['mma_us'] / a['mma_us'] - 1):+.1f}%)" if f else "")
        cols.append(f"{h['gcomp']:.3f} ({h['per_clk']:.5f}, {h['per_clk'] / BAR_CLK:.2f}x)" if h else "")
        cols.append(f"{f['gcomp']:.3f} ({f['gcomp'] / h['gcomp']:.2f}x)" if f and h and h["gcomp"] else "")
        if ho:
            alone = max(a["ms"], h["ms"]) if h else a["ms"]
            cols += [f"{ho['ms']:.4f} ({ho['tops']:.0f})", f"{ho['ms'] / alone:.3f}",
                     f"{ho['ms'] / ref['ms']:.3f}" if ref else ""]
        else:
            cols += ["", "", ""]
        lines.append("| " + " | ".join(cols) + " |")
        if f and h and h["gcomp"]:
            verdicts.append(f"VERDICT beside co-issue {k} ({budget(a)}, {a['var']}): the GEMM beside {a['nh']} hash "
                            f"warpgroup(s) that hash throughout takes {f['mma_us']:.1f} us against {a['mma_us']:.1f} us "
                            f"alone ({100 * (f['mma_us'] / a['mma_us'] - 1):+.1f}%); the hash beside the GEMM "
                            f"{f['gcomp']:.3f} G/s against {h['gcomp']:.3f} G/s alone ({f['gcomp'] / h['gcomp']:.2f}x), alone "
                            f"{h['per_clk'] / BAR_CLK:.2f}x the bar {BAR_CLK} an SM-clock")
        if ho and h:
            verdicts.append(f"VERDICT beside handoff {k}: {ho['ms']:.4f} ms, {ho['tops']:.0f} TOPS, the GEMM and every row's "
                            f"digest; {ho['ms'] / max(a['ms'], h['ms']):.3f}x the larger of its GEMM alone ({a['ms']:.4f} ms) "
                            f"and its hash alone ({h['ms']:.4f} ms)"
                            + (f"; {ho['ms'] / ref['ms']:.3f}x wgmma.linear's shape alone ({ref['ms']:.4f} ms, "
                               f"{ref['tops']:.0f} TOPS)" if ref else ""))
    for k in kernels:
        a = med.get((k, "mma"))
        if ref and a and k != REF:
            form = f"n{a['nform']}" + (f" x {256 // a['nform']}" if a["nform"] < 256 else "")
            obuf = {0: "no O buffer", 1: "the O tile", 2: "the half O tile"}[a["ot"]]
            verdicts.append(f"VERDICT beside GEMM form {k} ({a['nwg']} warpgroups, {a['nm']} MMA, {64 * a['nm']}-row tiles, "
                            f"{form}, {a['stages']} stages, {obuf}, Y {a['Y']}): alone {a['mma_us']:.1f} us, "
                            f"{a['tops']:.0f} TOPS, "
                            f"{a['mma_us'] / ref['mma_us']:.3f}x wgmma.linear's shape ({ref['mma_us']:.1f} us)")
    fits = [(med[(k, "handoff")]["ms"], k) for k in kernels if (k, "handoff") in med]
    if fits:
        ms, k = min(fits)
        m = med[(k, "handoff")]
        verdicts.append(f"VERDICT beside budget: the fastest handoff is {k}, {budget(m)}, {ms:.4f} ms"
                        + (f", {ms / ref['ms']:.3f}x wgmma.linear's shape without the hash" if ref else ""))
    served = {k: s for k in kernels if (s := served_call(head, med, k, ref))}
    summary["served_model"] = served
    for k, s in served.items():
        verdicts.append(f"VERDICT beside served model {k} ({budget(med[(k, 'mma')])}): {s['comp'] / 1e6:.2f} M compressions "
                        f"owed; {s['beside']:.0f} done beside the GEMM's {s['gemm_us']:.1f} us, the rest at "
                        f"{s['alone_gs']:.3f} G/s: {s['us']:.1f} us against serial {s['serial_us']:.1f} us "
                        f"({s['us'] / s['serial_us']:.3f}x; model, not a timed call)")
    if served:
        k = min(served, key=lambda x: served[x]["us"])
        s = served[k]
        verdicts.append(f"VERDICT beside budget, served model: {k}, {budget(med[(k, 'mma')])}: {s['us']:.1f} us modeled "
                        f"against serial {s['serial_us']:.1f} us ({s['us'] / s['serial_us']:.3f}x)")
    ncu = sorted(o.glob("ncu-*.csv"))
    if ncu:
        lines += ["", "### Nsight Compute: % of peak sustained over elapsed cycles", "",
                  "| kernel:arm | " + " | ".join(label for label, _ in NCU_COLS) + " |", "|---" * (len(NCU_COLS) + 1) + "|"]
        summary["ncu"], stalls = {}, {}
        for p in ncu:
            v = ncu_row(p)
            name = p.stem[len("ncu-"):]
            summary["ncu"][name] = v
            lines.append(f"| {name} | " + " | ".join(f"{v[label]:.1f}" if label in v else "" for label, _ in NCU_COLS) + " |")
            if s := ncu_stalls(p):
                stalls[name] = s
        if stalls:
            reasons = sorted({r for s in stalls.values() for r in s if r != "active"},
                             key=lambda r: -max(s.get(r, 0.0) for s in stalls.values()))
            reasons = [r for r in reasons if max(s.get(r, 0.0) for s in stalls.values()) >= 0.01]
            lines += ["", "### Nsight Compute: warps a scheduler in each stall reason, an active cycle", "",
                      "| kernel:arm | active | " + " | ".join(reasons) + " |", "|---" * (len(reasons) + 2) + "|"]
            lines += [f"| {n} | {s.get('active', 0.0):.2f} | " + " | ".join(f"{s.get(r, 0.0):.3f}" for r in reasons) + " |"
                      for n, s in stalls.items()]
            summary["ncu_stalls"] = stalls
            verdicts += stall_verdicts(stalls)
    return lines + [""] + verdicts


def main(o: Path) -> int:
    summary: dict = {}
    out = lowreg(o, summary) + beside(o, summary)
    print("\n".join(out))
    (o / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
