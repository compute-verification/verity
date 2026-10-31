"""The tensor-duty sweep's verdict on the H200 card's tensor clock (`clock.tensor`), and its reading as Markdown, over
one or more runs' out/ directories. The sweep is that clock's falsification test: a point whose MACs over the host's
wall clock exceed the card's tensor rate (`verity.hardware.rate`, 132 SMs x 4096 x clock.tensor) refutes the card.

duty.cu prints its host stamps to the millisecond (`STAMP_S`), so a point's rate is known only between its MACs over
its interval lengthened and shortened by a stamp: the verdict is REFUTED when some point exceeds the card's rate even
over the longer interval, UNDECIDED when some point does only over the shorter, and PASS when none does over the
shorter. Only PASS exits 0. Throughput rests on the host's wall clock alone; the GPU's own timers (cudaEvent,
%globaltimer and clock64) are diagnostics. Each rate is also given against the tensor at clock.sm, the card's second
reading. The tables give the maximum by clock, arm and operand class, each clock's and arm's duty x class table, and
the fills and bursts compared; each point is joined to the nvidia-smi samples taken inside its host interval.

    PYTHONPATH=. python3 benchmarks/tensor_duty/summarize.py OUT [OUT...] [--json FILE]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from verity.hardware import load, rate

SKU = "h200-sxm-141gb"
#: the resolution of duty.cu's host stamps, which it prints to the millisecond
STAMP_S = 1e-3


def readings(sku: str = SKU) -> tuple[float, float, float]:
    """The card's tensor rate at clock.tensor (the bound), at clock.sm (the second reading), and a MHz of clock's."""
    card = load(sku)
    return float(rate(card)), float(rate(card, mhz=card["clock"]["sm"]["value"])), float(rate(card, mhz=1))


def host_rate(r: dict, pad: float = 0.0) -> float:
    """MACs a second over the point's host interval lengthened by ``pad`` seconds (shortened, when negative)."""
    span = r["host_t1"] - r["host_t0"] + pad
    return r["macs"] / span if span > 0 else float("inf")


def verdict(pts: list[dict], bound: float) -> tuple[str, list[dict], list[dict]]:
    """PASS, UNDECIDED or REFUTED (module docstring), with the points that refute and those left undecided."""
    refuting = [r for r in pts if host_rate(r, STAMP_S) > bound]
    undecided = [r for r in pts if host_rate(r, -STAMP_S) > bound and host_rate(r, STAMP_S) <= bound]
    return ("REFUTED" if refuting else "UNDECIDED" if undecided else "PASS"), refuting, undecided


def load_points(out: Path) -> list[dict]:
    rows = []
    for name in ("warm.jsonl", "points.jsonl"):
        p = out / name
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if line.startswith("{"):
                r = json.loads(line)
                if not r.get("calibrate"):
                    r["warm"] = name == "warm.jsonl"
                    r.setdefault("clock", "-")
                    rows.append(r)
    return rows


def samples(out: Path) -> tuple[list[tuple[float, float, float, int]], list[str]]:
    """(unix time, SM MHz, W, clock-event reasons) per nvidia-smi line, and the lines that aren't one (smi.csv also takes
    nvidia-smi's errors)"""
    p, f = out / "smi.csv", out / "found.json"
    if not p.exists():
        return [], []
    tz = json.loads(f.read_text()).get("tz", "+0000") if f.exists() else "+0000"
    rows, unread = [], []
    for line in p.read_text().splitlines():
        v = [x.strip() for x in line.split(",")]
        try:
            t = datetime.strptime(f"{v[0]} {tz}", "%Y/%m/%d %H:%M:%S.%f %z").timestamp()
            rows.append((t, float(v[1]), float(v[3]), int(v[6], 16)))
        except (ValueError, IndexError) as e:
            unread.append(f"{p}: {line[:80]!r}: {e}")
    return rows, unread


def join(r: dict, smi: list) -> None:
    s = [x for x in smi if r["host_t0"] <= x[0] <= r["host_t1"]]
    r["smi_n"] = len(s)
    r["smi_mhz_max"] = max((x[1] for x in s), default=None)
    r["smi_w_mean"] = sum(x[2] for x in s) / len(s) if s else None
    reasons = 0
    for x in s:
        reasons |= x[3]
    r["smi_reasons"] = f"0x{reasons:x}"


def fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def at(r) -> str:
    return f"{r['fill']} / {r['burst']} / {r['duty']:.0f}{' warm' if r['warm'] else ''}"


def where(r) -> str:
    return f"clock {r['clock']}, {r['arm']} {r['class']}, fill / burst / duty {at(r)}"


def report(pts: list[dict], sku: str = SKU) -> tuple[str, list[str], dict]:
    """The verdict, the Markdown lines and the JSON summary of ``pts``."""
    bound, second, per_mhz = readings(sku)
    tm = lambda r: host_rate(r) * 1e-12  # noqa: E731
    v, refuting, undecided = verdict(pts, bound)
    top = max(pts, key=host_rate)
    hi = max(pts, key=lambda r: host_rate(r, -STAMP_S))
    L = [f"**{v}**: of {len(pts)} points, {len(refuting)} exceed the card's tensor rate, 132 x 4096 x clock.tensor = "
         f"{bound:.4e} MACs a second, over the host's wall clock at its longest reading, and {len(undecided)} more "
         f"only at its shortest (the host stamps' {STAMP_S * 1e3:g} ms either way). The highest is "
         f"{tm(top):.2f} T MAC/s "
         f"({host_rate(top) / bound:.5f} of the card's rate and {host_rate(top) / second:.5f} of the tensor at "
         f"clock.sm, {second:.4e}; {host_rate(top) / per_mhz:.1f} MHz-equivalent) at {where(top)}. At its shortest "
         f"reading the highest is {host_rate(hi, -STAMP_S) / bound:.5f} of the card's rate, at {where(hi)}."]
    tmax = max(pts, key=lambda r: r["tensor_mhz"])
    gap = [r["host_t1"] - r["host_t0"] - r["event_s"] for r in pts]
    L.append(f"Diagnostics, the GPU's own timers: over cudaEvents the highest is "
             f"{max(r['macs'] / r['event_s'] for r in pts) * 1e-12:.2f} T MAC/s, over %globaltimer "
             f"{max(r['macs'] / r['gt_s'] for r in pts) * 1e-12:.2f}; the host's interval minus the cudaEvent time is "
             f"{sum(gap) / len(gap) * 1e3:.3f} ms on average (from {min(gap) * 1e3:.3f} to {max(gap) * 1e3:.3f}); the "
             f"highest clock64 over tensor phases is {tmax['tensor_mhz']:.1f} MHz ({where(tmax)}, "
             f"{tm(tmax):.2f} T MAC/s).")
    L += ["", "## Maximum by clock, arm and class (T MAC/s over the host's wall clock; clock64 MHz a diagnostic)", ""]
    L.append("| clock | arm | class | max T MAC/s | of the card | of clock.sm's | MHz-equiv | at fill / burst / duty | "
             "duty achieved | clock64 MHz | tensor-phase MHz | max tensor-phase MHz (class) | smi MHz max | "
             "smi W mean | reasons |")
    L.append("| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    by = defaultdict(list)
    for r in pts:
        by[(r["clock"], r["arm"], r["class"])].append(r)
    summary = []
    for (clock, arm, cls), rs in sorted(by.items()):
        m = max(rs, key=host_rate)
        tmh = max(r["tensor_mhz"] for r in rs)
        L.append(f"| {clock} | {arm} | {cls} | {tm(m):.2f} | {host_rate(m) / bound:.4f} | "
                 f"{host_rate(m) / second:.4f} | "
                 f"{host_rate(m) / per_mhz:.1f} | {at(m)} | {m['duty_achieved']:.4f} | {m['clock64_mhz']:.1f} | "
                 f"{m['tensor_mhz']:.1f} | {tmh:.1f} | {fmt(m.get('smi_mhz_max'), 0)} | "
                 f"{fmt(m.get('smi_w_mean'), 0)} | "
                 f"{m.get('smi_reasons', '-')} |")
        summary.append({"clock": clock, "arm": arm, "class": cls, "max_tmacs_per_s": tm(m),
                        "of_bound": host_rate(m) / bound, "of_clock_sm": host_rate(m) / second,
                        "at": {k: m[k] for k in ("fill", "burst", "duty")}, "clock64_mhz": m["clock64_mhz"],
                        "tensor_mhz": m["tensor_mhz"], "max_tensor_mhz": tmh, "points": len(rs)})
    for clock in sorted({r["clock"] for r in pts}):
        for arm in sorted({r["arm"] for r in pts}):
            rs = [r for r in pts if r["clock"] == clock and r["arm"] == arm and not r["warm"]]
            if not rs:
                continue
            classes = sorted({r["class"] for r in rs})
            L += ["", f"## Clock {clock}, {arm}: T MAC/s by duty and class (the maximum over fills and bursts)", ""]
            L.append("| duty % | " + " | ".join(classes) + " | clock64 MHz at the row's max | tensor-phase MHz there |")
            L.append("| ---: | " + " | ".join("---:" for _ in classes) + " | ---: | ---: |")
            for d in sorted({r["duty"] for r in rs}):
                cells, best = [], None
                for c in classes:
                    m = max((r for r in rs if r["class"] == c and r["duty"] == d), key=host_rate, default=None)
                    cells.append(fmt(m and tm(m), 2))
                    if m and (best is None or host_rate(m) > host_rate(best)):
                        best = m
                L.append(f"| {d:.0f} | " + " | ".join(cells) + f" | {fmt(best and best['clock64_mhz'])} | "
                         f"{fmt(best and best['tensor_mhz'])} |")
            L += ["", f"## Clock {clock}, {arm}: fills and bursts at duties 90-99 (the maximum T MAC/s, and the mean "
                  "clock64 MHz)", ""]
            L.append("| fill | burst (SM-cycles) | max T MAC/s | mean clock64 MHz | mean tensor-phase MHz | "
                     "mean fill-phase MHz |")
            L.append("| --- | ---: | ---: | ---: | ---: | ---: |")
            for f in sorted({r["fill"] for r in rs}):
                for b in sorted({r["burst"] for r in rs}):
                    c = [r for r in rs if r["fill"] == f and r["burst"] == b and 90 <= r["duty"] < 100]
                    if c:
                        mean = lambda k: sum(r[k] for r in c) / len(c)  # noqa: E731
                        L.append(f"| {f} | {b} | {max(tm(r) for r in c):.2f} | {mean('clock64_mhz'):.1f} | "
                                 f"{mean('tensor_mhz'):.1f} | {mean('fill_mhz'):.1f} |")
    doc = {"verdict": v, "sku": sku, "bound": bound, "clock_sm_rate": second, "stamp_s": STAMP_S, "points": len(pts),
           "refuting": len(refuting), "undecided": len(undecided),
           "top": {**{k: top[k] for k in ("clock", "arm", "class", "fill", "burst", "duty", "clock64_mhz",
                                          "tensor_mhz")},
                   "tmacs_per_s": tm(top), "of_bound": host_rate(top) / bound},
           "by_class": summary}
    return v, L, doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("outs", nargs="+", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    pts, unread = [], []
    for out in a.outs:
        smi, skipped = samples(out)
        unread += skipped
        for r in load_points(out):
            join(r, smi)
            pts.append(r)
    if not pts:
        print("no points")
        return 2
    v, lines, doc = report(pts)
    doc["smi_unread"] = len(unread)
    print("\n".join(lines))
    if unread:
        print(f"\n{len(unread)} nvidia-smi line(s) not read as samples:", *unread[:5], sep="\n", file=sys.stderr)
    if a.json:
        a.json.write_text(json.dumps(doc, indent=1) + "\n")
    return 0 if v == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
