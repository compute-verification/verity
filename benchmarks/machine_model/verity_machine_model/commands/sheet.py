"""machine-model sheet [--lean | --eps [--uni COST ...]]
                                          a device's data sheet (sheet.toml): its DS lines against the census and its
                                          peaks at their clock; --lean prints its Lean module; --eps prices the
                                          headline design's sheet reading on the sheet's lines, with A3 read per
                                          second at f_max, and how much slower the tensor pipe can be before ε moves;
                                          --uni with a uniform instruction at each COST of an issue slot (A2u)"""
from __future__ import annotations

from verity_machine_model.sheet import at_fmax, differences, figures, lean, lines, moves, reading, sheet, uni


def add(sub) -> None:
    c = sub.add_parser("sheet")
    g = c.add_mutually_exclusive_group()
    g.add_argument("--lean", action="store_true", help="print the Lean module sheet.toml's `lean` names")
    g.add_argument("--eps", action="store_true", help="the headline design's ε on the sheet's lines")
    c.add_argument("--uni", type=float, nargs="+", metavar="COST",
                   help="with --eps: also ε with a uniform instruction at each COST of an issue slot (A2u; the "
                        "sheet's is 1)")


def run(args) -> int:
    s = sheet(args.data)
    if args.uni and not args.eps:
        raise SystemExit("sheet --uni prices the headline's sheet reading, so it needs --eps")
    if args.lean:
        print(lean(s, args.data), end="")
        return 0
    if args.eps:
        r = reading(s, args.data)
        m = moves(s, args.data)
        same = "the same" if r.sheet == r.headline else "DIFFERS"
        print(f"{s.gpu}, {s.vector}: {r.design} at q* = {r.q:g}, operands free, credit ♭")
        print(f"  on the sheet's lines: ε {100 * r.sheet:.4f}% (headline {100 * r.headline:.4f}%: {same})")
        print(f"  A3 as a bound a second at f_max, each figure over 2 · SMs · f_max (not a line a clock: "
              f"{', '.join(f'{t} {float(v):.0f}' for t, v in at_fmax(s).items())} MACs/SM-clk): ε {100 * r.fmax:.4f}%")
        print(f"  ε first moves with every tensor line {m:.3f}× the sheet's" if m is not None
              else "  ε doesn't move with every tensor line up to 4× the sheet's")
        for cost in args.uni or ():
            print(f"  a uniform instruction at {cost:g} of an issue slot: ε {100 * uni(s, args.data, cost):.4f}%")
        return 0 if r.sheet == r.headline else 1
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
