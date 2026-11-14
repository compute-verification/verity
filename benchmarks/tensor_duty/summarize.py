"""The tensor-duty sweep's verdict on the H200 card's tensor clock (`clock.tensor`), and its reading as Markdown, over
one or more runs' out/ directories. The sweep is that clock's falsification test: a point whose MACs over the host's
wall clock exceed the card's tensor rate (`verity.device.rate`, 132 SMs x 4096 x clock.tensor) refutes the card.

duty.cu prints its host stamps to the millisecond (`STAMP_S`), so a point's rate is known only between its MACs over
its interval lengthened and shortened by a stamp: the verdict is REFUTED when some point exceeds the card's rate even
over the longer interval, UNDECIDED when some point does only over the shorter, and PASS when none does over the
shorter. Only PASS exits 0. Throughput rests on the host's wall clock alone; the GPU's own timers (cudaEvent,
%globaltimer and clock64) are diagnostics. Each rate is also given against the tensor at clock.sm, the card's second
reading. The tables give the maximum by clock, arm and operand class, each clock's and arm's duty x class table, and
the fills and bursts compared; each point is joined to the nvidia-smi samples taken inside its host interval.

`--arms fp32-beside` reads run.sh's FP32-beside-tensor points (beside.jsonl, duty.cu's arms A to D) instead, and judges
each of the card's two clock readings of an FP32 add's price (`price_readings`: `price.fp32.sm`, 32, both pipes at
clock.sm, the card's `clock.default`; `price.fp32.tensor`, 29.58, the tensor at clock.tensor) under the unit CMP-438
rules: each operation charged in seconds at its own pipe's data-sheet ceiling. A reading is refuted by a point, of any
arm, whose MACs over its host interval lengthened by a stamp exceed the tensor's ceiling at that reading's tensor clock,
or whose adds over it exceed the FP32 pipe's, 32 x rate.issue at clock.sm; it is undecided when a point does so only
over its interval shortened by a stamp, and survives otherwise. It exits 0 only when every reading is decided. Each
point's clock is the one it held (nvidia-smi's mean over its interval, else clock64), never the one the run asked nvidia-smi to lock.

Beside the verdicts, a diagnostic judged against nothing: p = R_t,alone / R_f,mixed for each point of arms B, C and D,
the time price of an add in the tensor pipe's own time, the unit CMP-438 refused. R_t,alone is the highest arm A point's
MACs over its host interval at the point's clock setting and class; R_f,mixed is the point's adds over the part of its
host interval its FP32 warps spent adding (`fp32_share`, of the kernel's %globaltimer time). An arm B point adding under
95% of the loop's ideal, 32 x rate.issue x 64 / 67 a SM-clock, is marked `undersaturated` (R_f is then under-measured
and p reads high).

    PYTHONPATH=. python3 benchmarks/tensor_duty/summarize.py OUT [OUT...] [--arms fp32-beside] [--json FILE]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from verity.device import READINGS, load_card, number, rate
from verity.device import readings as card_readings

SKU = "h200-sxm-141gb"
#: the resolution of duty.cu's host stamps, which it prints to the millisecond
STAMP_S = 1e-3
#: the FP32 loop: adds and instructions an iteration; arm B under SATURATED of its ideal rate is undersaturated
LOOP_ADDS, LOOP_INSNS, SATURATED = 64, 67, 0.95


def readings(sku: str = SKU) -> tuple[float, float, float]:
    """The card's tensor rate at clock.tensor (the bound), at clock.sm (the second reading), and a MHz of clock's."""
    card = load_card(sku)
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
    r["smi_mhz_mean"] = sum(x[1] for x in s) / len(s) if s else None
    r["smi_w_mean"] = sum(x[2] for x in s) / len(s) if s else None
    reasons = 0
    for x in s:
        reasons |= x[3]
    r["smi_reasons"] = f"0x{reasons:x}"


def fmt(x: float | None, nd: int = 1) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def at(r: dict) -> str:
    return f"{r['fill']} / {r['burst']} / {r['duty']:.0f}{' warm' if r['warm'] else ''}"


def where(r: dict) -> str:
    return f"clock {r['clock']}, {r['arm']} {r['class']}, fill / burst / duty {at(r)}"


def report(pts: list[dict], sku: str = SKU) -> tuple[str, list[str], dict]:
    """The verdict, the Markdown lines and the JSON summary of ``pts``."""
    bound, second, per_mhz = readings(sku)

    def tm(r: dict) -> float:
        return host_rate(r) * 1e-12

    def mean(rs: list[dict], k: str) -> float:
        return sum(r[k] for r in rs) / len(rs)

    v, refuting, undecided = verdict(pts, bound)
    top = max(pts, key=host_rate)
    hi = max(pts, key=lambda r: host_rate(r, -STAMP_S))
    L = [(f"**{v}**: of {len(pts)} points, {len(refuting)} exceed the card's tensor rate, 132 x 4096 x clock.tensor = "
          f"{bound:.4e} MACs a second, over the host's wall clock at its longest reading, and {len(undecided)} more "
          f"only at its shortest (the host stamps' {STAMP_S * 1e3:g} ms either way). The highest is "
          f"{tm(top):.2f} T MAC/s "
          f"({host_rate(top) / bound:.5f} of the card's rate and {host_rate(top) / second:.5f} of the tensor at "
          f"clock.sm, {second:.4e}; {host_rate(top) / per_mhz:.1f} MHz-equivalent) at {where(top)}. At its shortest "
          f"reading the highest is {host_rate(hi, -STAMP_S) / bound:.5f} of the card's rate, at {where(hi)}.")]
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
            L += ["", (f"## Clock {clock}, {arm}: fills and bursts at duties 90-99 (the maximum T MAC/s, and the mean "
                       "clock64 MHz)"), ""]
            L.append("| fill | burst (SM-cycles) | max T MAC/s | mean clock64 MHz | mean tensor-phase MHz | "
                     "mean fill-phase MHz |")
            L.append("| --- | ---: | ---: | ---: | ---: | ---: |")
            for f in sorted({r["fill"] for r in rs}):
                for b in sorted({r["burst"] for r in rs}):
                    c = [r for r in rs if r["fill"] == f and r["burst"] == b and 90 <= r["duty"] < 100]
                    if c:
                        L.append(f"| {f} | {b} | {max(tm(r) for r in c):.2f} | {mean(c, 'clock64_mhz'):.1f} | "
                                 f"{mean(c, 'tensor_mhz'):.1f} | {mean(c, 'fill_mhz'):.1f} |")
    doc = {"verdict": v, "sku": sku, "bound": bound, "clock_sm_rate": second, "stamp_s": STAMP_S, "points": len(pts),
           "refuting": len(refuting), "undecided": len(undecided),
           "top": {**{k: top[k] for k in ("clock", "arm", "class", "fill", "burst", "duty", "clock64_mhz",
                                          "tensor_mhz")},
                   "tmacs_per_s": tm(top), "of_bound": host_rate(top) / bound},
           "by_class": summary}
    return v, L, doc


def load_beside(out: Path) -> list[dict]:
    p = out / "beside.jsonl"
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.startswith("{")] if p.exists() else []
    for r in rows:
        r.setdefault("clock", "-")
    return rows


def fp32_rate(r: dict, pad: float = 0.0) -> float:
    """R_f,mixed: adds a second over the part of the point's host interval, lengthened by ``pad`` seconds, that its FP32
    warps spent adding."""
    span = (r["host_t1"] - r["host_t0"] + pad) * r["fp32_share"]
    return r["adds"] / span if span > 0 else float("inf")


def price_readings(sku: str = SKU) -> list[dict]:
    """The card's clock readings of an FP32 add's price in tensor units (`price.fp32.<reading>`, by `READINGS`), its
    `clock.default` first, each with its pipes' ceilings: the tensor's e4m3 MACs a second under the reading
    (`verity.device.readings`), and the FP32 pipe's adds a second, 32 x rate.issue at clock.sm."""
    card = load_card(sku)
    clock, default = card["clock"], card["clock"]["default"]
    sm = clock["sm"]["value"]
    adds = 32 * float(rate(card, "issue"))
    peaks = card_readings(card, fmt="e4m3")
    source = {"sm": f"both pipes at clock.sm's {sm} MHz",
              "tensor": f"the tensor at clock.tensor's {clock['tensor']['value']} MHz beside FP32 at clock.sm's {sm} "
                        f"MHz; it holds only if the tensor pipe never exceeds {clock['tensor']['value']} MHz"}
    order = [default, *(r for r in READINGS if r != default)]
    return [{"reading": r, "price": float(number(card["price"]["fp32"][r])), "default": r == default,
             "tensor_mhz": clock[r]["value"], "tensor_ceiling": float(peaks[r]), "fp32_ceiling": adds,
             "source": ("the default: " if r == default else "") + source[r]} for r in order]


def over(r: dict, rd: dict, pad: float) -> bool:
    """Whether the point's MACs or adds over its host interval, lengthened by ``pad`` seconds, exceed the reading's
    tensor or FP32 ceiling."""
    span = r["host_t1"] - r["host_t0"] + pad
    return span <= 0 or r["macs"] / span > rd["tensor_ceiling"] or r["adds"] / span > rd["fp32_ceiling"]


def ideal_adds(sku: str = SKU) -> float:
    """The FP32 loop's adds a SM-clock at the card's issue rate: 32 lanes x rate.issue x LOOP_ADDS / LOOP_INSNS."""
    return 32 * float(load_card(sku)["rate"]["issue"]["value"]) * LOOP_ADDS / LOOP_INSNS


def setting(clock: str) -> str:
    return "default" if clock in ("default", "-") else f"lgc {clock}"


def held(r: dict) -> float:
    """The clock a point held: nvidia-smi's mean inside its interval, else its kernel's clock64 over %globaltimer."""
    return r.get("smi_mhz_mean") or r["clock64_mhz"]


def beside(pts: list[dict], sku: str = SKU) -> tuple[bool, list[str], dict]:
    """Whether every reading is decided, and the Markdown lines and summary.json of ``pts`` (module docstring)."""
    ideal = ideal_adds(sku)
    alone: dict[tuple[str, str], dict] = {}
    for r in pts:
        k = (r["clock"], r["class"])
        if r["arm"] == "A" and (k not in alone or host_rate(r) > host_rate(alone[k])):
            alone[k] = r
    rows = []
    for r in sorted(pts, key=lambda r: (r["clock"], r["class"], r["arm"], r["x"])):
        a = alone.get((r["clock"], r["class"]))
        priced = r["arm"] != "A" and a is not None and r["fp32_share"] > 0
        span = r["host_t1"] - r["host_t0"]
        rows.append({
            "set": setting(r["clock"]), "held_mhz": held(r), "arm": r["arm"], "class": r["class"], "x": r["x"],
            "macs_per_s": host_rate(r), "adds_per_s": r["adds"] / span,
            "r_t_alone": a and host_rate(a), "r_f_mixed": fp32_rate(r) if r["fp32_share"] > 0 else None,
            "p": host_rate(a) / fp32_rate(r) if priced else None,
            "p_lo": host_rate(a, STAMP_S) / fp32_rate(r, -STAMP_S) if priced else None,
            "p_hi": host_rate(a, -STAMP_S) / fp32_rate(r, STAMP_S) if priced else None,
            "saturation": r["adds_per_smclk"] / ideal if r["arm"] == "B" else None,
            "undersaturated": r["arm"] == "B" and r["adds_per_smclk"] < SATURATED * ideal,
            **{k: r[k] for k in ("clock64_mhz", "fp32_busy_mhz", "fp32_share", "macs_per_smclk", "adds_per_smclk")},
            **{f"{m}_mhz": r[f"{m}_mhz"] or None for m in ("tensor", "fp32", "both")},
            **{k: r.get(k) for k in ("smi_mhz_mean", "smi_mhz_max", "smi_w_mean")}})
    under = [x for x in rows if x["undersaturated"]]
    low = min((x for x in rows if x["arm"] in "CD" and x["p"] is not None), key=lambda x: x["p"], default=None)
    verdicts = []
    for rd in price_readings(sku):
        refuted = [r for r in pts if over(r, rd, STAMP_S)]
        undecided = [r for r in pts if over(r, rd, -STAMP_S) and not over(r, rd, STAMP_S)]
        v = "refuted" if refuted else "undecided" if undecided else "survives"
        verdicts.append({**rd, "verdict": v, "refuting": len(refuted), "undecided": len(undecided)})
    decided = all(v["verdict"] != "undecided" for v in verdicts)

    def where(x: dict) -> str:
        return f"{x['set']} (held {x['held_mhz']:.0f} MHz), arm {x['arm']} {x['class']}, X {x['x']:g}"

    L = [f"Of {len(pts)} points, each reading's verdict: refuted by a point whose MACs or adds over its host interval, "
         f"at its longest reading (the host stamps' {STAMP_S * 1e3:g} ms), exceed that reading's tensor or FP32 "
         "ceiling (CMP-438's unit: each operation charged in seconds at its own pipe's data-sheet ceiling).", ""]
    L += ["| reading | price (tensor units) | tensor ceiling (T MAC/s) | FP32 ceiling (G add/s) | verdict | "
          "points refuting | undecided | source |", "| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |"]
    for v in verdicts:
        L.append(f"| {v['reading']} | {v['price']:.3f} | {v['tensor_ceiling'] * 1e-12:.2f} | "
                 f"{v['fp32_ceiling'] * 1e-9:.1f} | **{v['verdict']}** | {v['refuting']} | {v['undecided']} | "
                 f"{v['source']} |")
    b_rates = ", ".join(f"{x['adds_per_smclk']:.2f}" for x in rows if x["arm"] == "B") or "nothing"
    L += ["", "A diagnostic, judged against nothing: p, an add's price in the tensor pipe's own time, the unit CMP-438 "
          "refused. Its lowest over arms C and D is "
          + (f"p = {low['p']:.3f} tensor units (from {low['p_lo']:.3f} to {low['p_hi']:.3f} at the host stamps' "
             f"{STAMP_S * 1e3:g} ms), at {where(low)}." if low else "unread."),
          f"Arm B adds {b_rates} a SM-clock against the loop's ideal {ideal:.2f}"
          + (f": **undersaturated** ({len(under)} under {SATURATED:.0%}), so R_f is under-measured and p reads high."
             if under else ".")]
    L += ["", "## The clocks the arms held (MHz; a setting is what the run asked nvidia-smi for, not what held)", "",
          "| setting | arm | smi mean | smi max | clock64 mean |", "| --- | --- | ---: | ---: | ---: |"]
    for s in sorted({x["set"] for x in rows}):
        for arm in "ABCD":
            xs = [x for x in rows if x["set"] == s and x["arm"] == arm]
            if xs:
                smi = [x["smi_mhz_mean"] for x in xs if x["smi_mhz_mean"] is not None]
                L.append(f"| {s} | {arm} | {fmt(sum(smi) / len(smi) if smi else None, 0)} | "
                         f"{fmt(max((x['smi_mhz_max'] for x in xs if x['smi_mhz_max'] is not None), default=None), 0)} | "
                         f"{sum(x['clock64_mhz'] for x in xs) / len(xs):.0f} |")
    L += ["", "## Points (rates over the host's wall clock; clock64 MHz by mode a diagnostic)", ""]
    L.append("| setting | held MHz | class | arm | X | T MAC/s | G add/s | R_f,mixed G add/s | p | p range | "
             "clock64 MHz | tensor MHz | fp32 MHz | both MHz | FP32 busy MHz | FP32 share | adds/SM-clk | smi MHz max | "
             "smi W mean |")
    L.append("| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
             "---: | ---: | ---: |")
    for x in rows:
        span = "-" if x["p"] is None else f"{x['p_lo']:.3f}-{x['p_hi']:.3f}"
        L.append(f"| {x['set']} | {x['held_mhz']:.0f} | {x['class']} | {x['arm']} | {x['x']:g} | "
                 f"{x['macs_per_s'] * 1e-12:.2f} | {x['adds_per_s'] * 1e-9:.1f} | "
                 f"{fmt(x['r_f_mixed'] and x['r_f_mixed'] * 1e-9)} | {fmt(x['p'], 3)} | {span} | "
                 f"{x['clock64_mhz']:.1f} | {fmt(x['tensor_mhz'])} | {fmt(x['fp32_mhz'])} | {fmt(x['both_mhz'])} | "
                 f"{fmt(x['fp32_busy_mhz'] or None)} | {x['fp32_share']:.4f} | {x['adds_per_smclk']:.2f}"
                 f"{' (under)' if x['undersaturated'] else ''} | {fmt(x['smi_mhz_max'], 0)} | "
                 f"{fmt(x['smi_w_mean'], 0)} |")
    doc = {"decided": decided, "verdicts": verdicts, "stamp_s": STAMP_S, "ideal_adds_per_smclk": ideal,
           "saturated_at": SATURATED, "undersaturated": len(under) > 0, "points": len(rows),
           "lowest": low and {k: low[k] for k in ("set", "held_mhz", "arm", "class", "x", "p", "p_lo", "p_hi")},
           "rows": rows}
    return decided, L, doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("outs", nargs="+", type=Path)
    ap.add_argument("--arms", choices=("duty", "fp32-beside"), default="duty")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    pts, unread = [], []
    for out in a.outs:
        smi, skipped = samples(out)
        unread += skipped
        for r in load_beside(out) if a.arms == "fp32-beside" else load_points(out):
            join(r, smi)
            pts.append(r)
    if not pts:
        print("no points")
        return 2
    if a.arms == "fp32-beside":
        ok, lines, doc = beside(pts)
    else:
        v, lines, doc = report(pts)
        ok = v == "PASS"
    doc["smi_unread"] = len(unread)
    print("\n".join(lines))
    if unread:
        print(f"\n{len(unread)} nvidia-smi line(s) not read as samples:", *unread[:5], sep="\n", file=sys.stderr)
    if a.json:
        a.json.write_text(json.dumps(doc, indent=1) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
