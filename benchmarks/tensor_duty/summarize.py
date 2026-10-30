"""The tensor-duty sweep's reading of one or more runs' out/ directories, as Markdown: the verdict (whether any point's
MACs a second over the host's wall clock exceeds 132 x 4096 x 1830 MHz), the maximum by arm and operand class with
both readings (MACs over the wall clock, and clock64), each arm's duty x class table, and the fills and periods
compared. Each point is joined to the nvidia-smi samples taken inside its host interval.

    python3 benchmarks/tensor_duty/summarize.py OUT [OUT...] [--json FILE]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

BOUND = 132 * 4096 * 1830e6


def load(out: Path) -> list[dict]:
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
                    rows.append(r)
    return rows


def samples(out: Path) -> list[tuple[float, float, float, int]]:
    """(unix time, SM MHz, W, clock-event reasons) per nvidia-smi line"""
    p, f = out / "smi.csv", out / "found.json"
    if not p.exists():
        return []
    tz = json.loads(f.read_text()).get("tz", "+0000") if f.exists() else "+0000"
    rows = []
    for line in p.read_text().splitlines():
        v = [x.strip() for x in line.split(",")]
        try:
            t = datetime.strptime(f"{v[0]} {tz}", "%Y/%m/%d %H:%M:%S.%f %z").timestamp()
            rows.append((t, float(v[1]), float(v[3]), int(v[6], 16)))
        except (ValueError, IndexError):
            continue
    return rows


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


def key(r):
    return (r["arm"], r["class"], r["fill"], r["period"], r["duty"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("outs", nargs="+", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    pts = []
    for out in a.outs:
        smi = samples(out)
        for r in load(out):
            join(r, smi)
            pts.append(r)
    if not pts:
        print("no points")
        return 1
    over = [r for r in pts if r["tmacs_per_s"] * 1e12 > BOUND]
    top = max(pts, key=lambda r: r["tmacs_per_s"])
    L = []
    verdict = "FAIL" if over else "PASS"
    L.append(f"**{verdict}**: {len(over)} of {len(pts)} points exceed 132 x 4096 x 1830 MHz = {BOUND:.4e} MACs a second. "
             f"The highest is {top['tmacs_per_s']:.2f} T MAC/s ({top['tmacs_per_s'] * 1e12 / BOUND:.5f} of the bound, "
             f"{top['mhz_equiv']:.1f} MHz-equivalent) at {top['arm']} {top['class']} {top['fill']} period {top['period']} "
             f"duty {top['duty']:.0f}%{' (warm-up)' if top['warm'] else ''}; clock64 there {top['clock64_mhz']:.1f} MHz "
             f"(tensor phases {top['tensor_mhz']:.1f} MHz).")
    tmax = max(pts, key=lambda r: r["tensor_mhz"])
    L.append(f"The highest clock64 over tensor phases is {tmax['tensor_mhz']:.1f} MHz ({tmax['arm']} {tmax['class']} "
             f"{tmax['fill']} period {tmax['period']} duty {tmax['duty']:.0f}%, {tmax['tmacs_per_s']:.2f} T MAC/s).")
    L.append("")
    L.append("## Maximum by arm and class")
    L.append("")
    L.append("| arm | class | max T MAC/s | of bound | MHz-equiv | at fill / period / duty | duty achieved | clock64 MHz | "
             "tensor-phase MHz | max tensor-phase MHz (class) | smi MHz max | smi W mean | reasons |")
    L.append("| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    by = defaultdict(list)
    for r in pts:
        by[(r["arm"], r["class"])].append(r)
    summary = []
    for (arm, cls), rs in sorted(by.items()):
        m = max(rs, key=lambda r: r["tmacs_per_s"])
        tm = max(r["tensor_mhz"] for r in rs)
        L.append(f"| {arm} | {cls} | {m['tmacs_per_s']:.2f} | {m['tmacs_per_s'] * 1e12 / BOUND:.4f} | {m['mhz_equiv']:.1f} | "
                 f"{m['fill']} / {m['period']} / {m['duty']:.0f}{' warm' if m['warm'] else ''} | {m['duty_achieved']:.4f} | "
                 f"{m['clock64_mhz']:.1f} | {m['tensor_mhz']:.1f} | {tm:.1f} | {fmt(m['smi_mhz_max'], 0)} | "
                 f"{fmt(m['smi_w_mean'], 0)} | {m['smi_reasons']} |")
        summary.append({"arm": arm, "class": cls, "max_tmacs_per_s": m["tmacs_per_s"], "of_bound": m["tmacs_per_s"] * 1e12 / BOUND,
                        "at": {k: m[k] for k in ("fill", "period", "duty")}, "clock64_mhz": m["clock64_mhz"],
                        "tensor_mhz": m["tensor_mhz"], "max_tensor_mhz": tm, "points": len(rs)})
    for arm in sorted({r["arm"] for r in pts}):
        rs = [r for r in pts if r["arm"] == arm and not r["warm"]]
        classes = sorted({r["class"] for r in rs})
        duties = sorted({r["duty"] for r in rs})
        L += ["", f"## {arm}: T MAC/s by duty and class (the maximum over fills and periods)", ""]
        L.append("| duty % | " + " | ".join(classes) + " | clock64 MHz at the row's max | tensor-phase MHz there |")
        L.append("| ---: | " + " | ".join("---:" for _ in classes) + " | ---: | ---: |")
        for d in duties:
            cells, best = [], None
            for c in classes:
                cand = [r for r in rs if r["class"] == c and r["duty"] == d]
                m = max(cand, key=lambda r: r["tmacs_per_s"], default=None)
                cells.append(fmt(m and m["tmacs_per_s"], 2))
                if m and (best is None or m["tmacs_per_s"] > best["tmacs_per_s"]):
                    best = m
            L.append(f"| {d:.0f} | " + " | ".join(cells) + f" | {fmt(best and best['clock64_mhz'])} | "
                     f"{fmt(best and best['tensor_mhz'])} |")
        L += ["", f"## {arm}: fills and periods at duties 90-99 (the maximum T MAC/s, and the mean clock64 MHz)", ""]
        L.append("| fill | period | max T MAC/s | mean clock64 MHz | mean tensor-phase MHz | mean fill-phase MHz |")
        L.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for f in sorted({r["fill"] for r in rs}):
            for p in sorted({r["period"] for r in rs}):
                c = [r for r in rs if r["fill"] == f and r["period"] == p and 90 <= r["duty"] < 100]
                if c:
                    mean = lambda k: sum(r[k] for r in c) / len(c)  # noqa: E731
                    L.append(f"| {f} | {p} | {max(r['tmacs_per_s'] for r in c):.2f} | {mean('clock64_mhz'):.1f} | "
                             f"{mean('tensor_mhz'):.1f} | {mean('fill_mhz'):.1f} |")
    print("\n".join(L))
    if a.json:
        a.json.write_text(json.dumps({"verdict": verdict, "bound": BOUND, "points": len(pts), "over": len(over),
                                      "top": {k: top[k] for k in ("arm", "class", "fill", "period", "duty", "tmacs_per_s",
                                                                   "clock64_mhz", "tensor_mhz")},
                                      "by_class": summary}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
