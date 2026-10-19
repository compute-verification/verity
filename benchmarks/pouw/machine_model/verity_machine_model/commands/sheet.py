"""machine-model sheet [--lean]   a device's data sheet (sheet.toml): its DS lines against the census, its peaks at their clock"""
from __future__ import annotations

from verity_machine_model.sheet import differences, figures, lean, lines, sheet


def add(sub) -> None:
    c = sub.add_parser("sheet")
    c.add_argument("--lean", action="store_true", help="print the Lean module sheet.toml's `lean` names")


def run(args) -> int:
    s = sheet(args.data)
    if args.lean:
        print(lean(s, args.data), end="")
        return 0
    diff, fs = differences(s, args.data), figures(s)
    n = sum(len(rs) for rs in lines(s).values())
    print(f"{s.gpu}, {s.vector}: {n} lines from sheet.toml, " + ("the census's" if not diff else "differing:"))
    for d in diff:
        print(f"  {d}")
    print(f"peaks, f_max {float(s.fmax.value):g} MHz, {float(s.sms.value):g} SMs:")
    for f in fs:
        p = s.peaks[f.type]
        print(f"  {f.type:5} {float(p.macs):6g} MACs/SM-clk at {p.clock} {float(s.clocks[p.clock].value):g} MHz: "
              f"{float(f.value):.2f} against {float(f.dense):g} ± {float(f.tol):g}, "
              f"{'holds' if f.holds else 'FAILS'} (from {float(f.low):.2f} to {float(f.high):.2f} MHz)")
    for c in s.clocks:
        at = [f for f in fs if s.peaks[f.type].clock == c]
        low, high = max(f.low for f in at), min(f.high for f in at)
        print(f"every figure at {c} holds from {float(low):.2f} to {float(high):.2f} MHz" if low <= high
              else f"no one {c} clock reproduces every figure quoted at it")
    return 0 if not diff and all(f.holds for f in fs) else 1
