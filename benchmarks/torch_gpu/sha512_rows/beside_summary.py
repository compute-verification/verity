"""beside.sh's tables and VERDICT lines, from its output directory.

lowreg: per variant, register cap and warpgroups an SM, the median over rounds of compressions a second and an SM-clock,
against BAR (sx::compress<0, 1>'s full-occupancy rate on the H200, 4.355 G/s at 1830 MHz: 0.01803 an SM-clock), the
registers ptxas gave and the local memory (spills). The verdict: for one warpgroup an SM, the fewest registers at which a
variant reaches 90% of BAR an SM-clock, and the best rate at each cap.

beside: per kernel, the median over rounds of each arm. The co-issue: the GEMM's span (%globaltimer, first MMA to last
epilogue) beside a hash warpgroup that hashes throughout (free) against the GEMM alone (mma), and the hash's rate beside the
GEMM against alone; the handoff (the pipeline) against the larger of its GEMM and its hash alone, and against wgmma.linear's
own shape (wgl_n256_s4, three warpgroups, n256, no O tile) alone. Nsight Compute's pipe shares for NCU_RUNS beside them.

    python3 beside_summary.py OUT_DIR    # markdown tables, then VERDICT lines; summary.json beside them
"""
from __future__ import annotations

import csv
import json
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
NCU_COLS = (("tensor", ("sm__pipe_tensor_op_gmma_cycles_active", "sm__pipe_tensor_cycles_active")),
            ("alu", ("sm__pipe_alu_cycles_active",)), ("fma", ("sm__pipe_fma_cycles_active",)),
            ("fmaheavy", ("sm__pipe_fmaheavy_cycles_active",)), ("alu inst", ("sm__inst_executed_pipe_alu",)),
            ("fma inst", ("sm__inst_executed_pipe_fma",)), ("lsu inst", ("sm__inst_executed_pipe_lsu",)),
            ("shared", ("sm__pipe_shared_cycles_active",)), ("issue", ("smsp__issue_active",)))


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
            if col in head and last[head.index(col)].strip():
                try:
                    out[label] = float(last[head.index(col)].replace(",", ""))
                except ValueError:
                    continue
                break
    return out


def beside(o: Path, summary: dict) -> list[str]:
    lines_in = jsonl(o / "beside-time.jsonl")
    head = next((r for r in lines_in if "grid" in r), None)
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
        m["gcomp"] = m["compressions"] / (m["hash_us"] * 1e3) if m["hash_us"] else 0.0
        m["per_clk"] = m["compressions"] / (m["hash_us"] * m["mhz"] * grid) if m["hash_us"] and m["mhz"] else 0.0
        m["rounds"] = len(rs)
        med[k] = m
    summary["beside"] = {f"{k}/{a}": m for (k, a), m in med.items()}
    summary["beside_head"] = head
    kernels = list(dict.fromkeys(k for k, _ in med))
    ref = med.get((REF, "mma"))
    lines = [f"\n### beside, M {head['M']} N {head['N']} K {head['K']}, grid {grid} on {head['sms']} SMs ({head['device']})",
             "", "| kernel | P / Y / H | hash | regs, spill B | GEMM alone us (TOPS) | GEMM beside the hash us (+%) | "
             "hash alone G/s (an SM-clock, x bar) | hash beside the GEMM G/s (x alone) | handoff ms (TOPS) | handoff / "
             "max(alone) | handoff / wgmma.linear's |", "|---" * 11 + "|"]
    verdicts = []
    for k in kernels:
        a, h, f, ho = (med.get((k, x)) for x in ("mma", "hash", "free", "handoff"))
        if a is None:
            continue
        regs = f"{a['regs']}" + (f", {a['local_bytes']}" if a["local_bytes"] else "")
        roles = f"{a['P']} / {a['Y']} / {a['H'] if a['nwg'] == 4 else '-'}"
        cols = [k, roles, a["var"] if a["nwg"] == 4 else "-", regs, f"{a['mma_us']:.1f} ({a['tops']:.0f})"]
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
            verdicts.append(f"VERDICT beside co-issue {k} (P {a['P']}, Y {a['Y']}, H {a['H']}, {a['var']}): the GEMM beside a "
                            f"hash warpgroup that hashes throughout takes {f['mma_us']:.1f} us against {a['mma_us']:.1f} us "
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
            verdicts.append(f"VERDICT beside GEMM form {k} ({a['nwg']} warpgroups, n{a['nform']}{' x 2' if a['nform'] == 128 else ''}, "
                            f"{a['stages']} stages, Y {a['Y']}): alone {a['mma_us']:.1f} us, {a['tops']:.0f} TOPS, "
                            f"{a['mma_us'] / ref['mma_us']:.3f}x wgmma.linear's shape ({ref['mma_us']:.1f} us)")
    fits = [(med[(k, "handoff")]["ms"], k) for k in kernels if (k, "handoff") in med]
    if fits:
        ms, k = min(fits)
        m = med[(k, "handoff")]
        verdicts.append(f"VERDICT beside budget: the fastest handoff is {k}, P {m['P']} + 2 x Y {m['Y']} + H {m['H']} = "
                        f"{m['P'] + 2 * m['Y'] + m['H']} <= 512 registers a thread across the warpgroups (128 x that <= "
                        f"64K an SM), {ms:.4f} ms" + (f", {ms / ref['ms']:.3f}x wgmma.linear's shape without the hash"
                                                       if ref else ""))
    ncu = sorted(o.glob("ncu-*.csv"))
    if ncu:
        lines += ["", "### Nsight Compute: % of peak sustained over elapsed cycles", "",
                  "| kernel:arm | " + " | ".join(label for label, _ in NCU_COLS) + " |", "|---" * (len(NCU_COLS) + 1) + "|"]
        summary["ncu"] = {}
        for p in ncu:
            v = ncu_row(p)
            name = p.stem[len("ncu-"):]
            summary["ncu"][name] = v
            lines.append(f"| {name} | " + " | ".join(f"{v[label]:.1f}" if label in v else "" for label, _ in NCU_COLS) + " |")
    return lines + [""] + verdicts


def main(o: Path) -> int:
    summary: dict = {}
    out = lowreg(o, summary) + beside(o, summary)
    print("\n".join(out))
    (o / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
