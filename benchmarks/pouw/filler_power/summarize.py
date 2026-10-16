"""Tables from pwr.cu's arms (pwr.sh's arms.jsonl): the filler alone per variant, and per suite point the step's slowdown
split into its SM-share part (the step on its partition, filler idle) and the filler's added part (the same arm with the
filler running), with each arm's board power, SM clock and clock-event reasons.

    python summarize.py RUN_OUT_DIR... [--json]

Tensor time is the MMA pipe's: one m16n8k32 int8/FP8 or m16n8k64 NVFP4 mma.sync occupies a sub-partition's tensor pipe
for 16 clocks at full rate, so an SM retires 0.25 a clock and a GPU of n SMs 0.25 n. Shares are at the clock NVML read in
the same window and at the locked 2,092 MHz.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

SMS = 188
LOCK_MHZ = 2092.0
MMA_PER_SM_CLK = 0.25


def rows(paths: list[Path]) -> list[dict]:
    out = []
    for p in paths:
        f = p / "arms.jsonl" if p.is_dir() else p
        for line in f.read_text().splitlines():
            if line.startswith("{"):
                d = json.loads(line)
                d["_run"] = str(p)
                out.append(d)
    return out


def share(mma_per_s: float, mhz: float) -> float:
    return mma_per_s / (MMA_PER_SM_CLK * SMS * mhz * 1e6) if mhz > 0 else float("nan")


def variant(d: dict) -> str:
    kind = {0: "int8", 1: "fp8", 2: "nvfp4"}[d["kind"]]
    s = f"{kind} p{d['pat']} W{d['W']} NC{d['NC']}"
    return s + (f" sleep{d['sleep_ns']}" if d.get("sleep_ns") else "")


def main(argv: list[str]) -> int:
    as_json = "--json" in argv
    paths = [Path(a) for a in argv if not a.startswith("--")]
    rs = rows(paths)
    idle = [d["watt_energy"] for d in rs if d["mode"] == "pidle" and d["watt_energy"] > 0]
    w_idle = statistics.median(idle) if idle else float("nan")
    out: dict = {"idle_watt": w_idle, "alone": [], "points": []}
    print(f"idle board: {w_idle:.1f} W (energy counter, {len(idle)} windows)")
    print("\nfiller alone (sustained):")
    print(f"{'variant':28s} {'SMs':>4s} {'W':>6s} {'W-idle':>6s} {'MHz':>6s} {'pcap':>5s} {'share':>6s} {'@2092':>6s} "
          f"{'tensor-SM-s/kJ':>14s} {'incr':>8s} {'bad':>4s}")
    for d in rs:
        if d["mode"] != "palone":
            continue
        sh = share(d["fill_mma_per_s"], d["mhz"])
        sh_lock = share(d["fill_mma_per_s"], LOCK_MHZ)
        tsm = sh_lock * SMS  # tensor-SM-seconds a second, at the lock
        per_kj = tsm / d["watt_energy"] * 1e3
        incr = tsm / (d["watt_energy"] - w_idle) * 1e3
        row = dict(variant=variant(d), sms=d["fill_sms"], watt=d["watt_energy"], mhz=d["mhz"], pcap=d["frac_pcap"],
                   share=sh, share_lock=sh_lock, tsm_per_kj=per_kj, tsm_per_kj_incr=incr, rec_bad=d["rec_bad"],
                   reasons=d["reasons_or"], temp=d["temp"])
        out["alone"].append(row)
        print(f"{row['variant']:28s} {d['fill_sms']:4d} {d['watt_energy']:6.1f} {d['watt_energy'] - w_idle:6.1f} "
              f"{d['mhz']:6.0f} {d['frac_pcap']:5.2f} {sh:6.3f} {sh_lock:6.3f} {per_kj:14.1f} {incr:8.1f} "
              f"{d['rec_bad']:4d}")
    pts: dict = defaultdict(lambda: defaultdict(list))
    for d in rs:
        if d["mode"] == "pfill":
            key = (d["B"], d["L"], d["fill_sms"], d["_run"])
            pts[key][d["arm"]].append(d)
            if d["arm"] == "split_fill":
                pts[key]["_variant"] = variant(d)
            if d["arm"] == "split_ctl":
                pts[key]["_ctl"] = variant(d)
    print("\nbeside the decode step (medians over rounds; slowdowns over the mean of full and full2):")
    print(f"{'B':>4s} {'L':>6s} {'fSMs':>4s} {'variant':22s} {'split':>6s} {'+fill':>6s} {'+ctl':>6s} {'fill/sp':>7s} "
          f"{'W full':>6s} {'W sp':>6s} {'W sp+f':>6s} {'W fill':>6s} {'MHz full':>8s} {'MHz sp+f':>8s} {'pcap sp+f':>9s} "
          f"{'f share':>7s} {'bad':>4s}")
    for (b, l, f, run), arms in sorted(pts.items()):
        def m(arm: str, k: str) -> float:
            v = [d[k] for d in arms.get(arm, [])]
            return statistics.median(v) if v else float("nan")

        full = statistics.mean([m("full", "step_ms"), m("full2", "step_ms")])
        sp, spf, spc = m("split", "step_ms") / full, m("split_fill", "step_ms") / full, m("split_ctl", "step_ms") / full
        fshare = share(m("split_fill", "fill_mma_per_s"), m("split_fill", "mhz"))
        fshare_lock = share(m("split_fill", "fill_mma_per_s"), LOCK_MHZ)
        bad = sum(d["rec_bad"] for a in ("split_fill", "split_ctl", "fill") for d in arms.get(a, []) if d["rec_bad"] > 0)
        row = dict(B=b, L=l, fill_sms=f, variant=arms.get("_variant"), ctl=arms.get("_ctl"), split=sp, split_fill=spf,
                   split_ctl=spc, fill_over_split=spf / sp, power_part=(spf - spc) / sp if spc == spc else None,
                   watt={a: m(a, "watt_energy") for a in ("full", "split", "split_fill", "split_ctl", "fill", "full2")},
                   mhz={a: m(a, "mhz") for a in ("full", "split", "split_fill", "split_ctl", "fill", "full2")},
                   mhz_min={a: m(a, "mhz_min") for a in ("full", "split", "split_fill", "split_ctl", "fill")},
                   pcap={a: m(a, "frac_pcap") for a in ("full", "split", "split_fill", "split_ctl", "fill")},
                   fill_share=fshare, fill_share_lock=fshare_lock, rec_bad=bad, run=run)
        out["points"].append(row)
        print(f"{b:4d} {l:6d} {f:4d} {str(row['variant']):22s} {sp:6.3f} {spf:6.3f} {spc:6.3f} {spf / sp:7.4f} "
              f"{row['watt']['full']:6.1f} {row['watt']['split']:6.1f} {row['watt']['split_fill']:6.1f} "
              f"{row['watt']['fill']:6.1f} {row['mhz']['full']:8.0f} {row['mhz']['split_fill']:8.0f} "
              f"{row['pcap']['split_fill']:9.2f} {fshare:7.3f} {bad:4d}")
        if arms.get("co_fill"):
            row["co"] = dict(slowdown=m("co_fill", "step_ms") / full, watt=m("co_fill", "watt_energy"),
                             mhz=m("co_fill", "mhz"), mhz_min=m("co_fill", "mhz_min"), pcap=m("co_fill", "frac_pcap"),
                             fill_share=share(m("co_fill", "fill_mma_per_s"), m("co_fill", "mhz")))
            c = row["co"]
            print(f"{'':16s} co-resident on all SMs: step {c['slowdown']:.3f}x, {c['watt']:.1f} W, {c['mhz']:.0f} MHz "
                  f"(min {c['mhz_min']:.0f}), pcap {c['pcap']:.2f}, filler share {c['fill_share']:.3f}")
    pre: dict = defaultdict(lambda: defaultdict(list))
    for d in rs:
        if d["mode"] == "pprefill":
            pre[(d["M"], d["B"], d["L"], d["fill_sms"], variant(d), d["_run"])][d["arm"]].append(d)
    if pre:
        print("\nprefill right after heat (decode on its partition alone, or beside the filler):")
        print(f"{'M':>5s} {'B':>4s} {'L':>6s} {'fSMs':>4s} {'variant':18s} {'heat':>10s} {'W heat':>6s} {'temp':>9s} "
              f"{'pre ms':>8s} {'pre W':>6s} {'pre MHz':>7s} {'MHz min':>7s} {'pcap':>5s}")
    for (mm, b, l, f, var, run), arms in sorted(pre.items()):
        out.setdefault("prefill", [])
        for h, p in (("heat_split", "pre_after_split"), ("heat_fill", "pre_after_fill")):
            def m2(arm: str, k: str) -> float:
                v = [d[k] for d in arms.get(arm, [])]
                return statistics.median(v) if v else float("nan")

            temps = [d["temp"][1] for d in arms.get(h, [])]
            row = dict(M=mm, B=b, L=l, fill_sms=f, variant=var, heat=h, heat_watt=m2(h, "watt_energy"),
                       heat_temp_end=statistics.median(temps) if temps else None, pre_ms=m2(p, "step_ms"),
                       pre_watt=m2(p, "watt_energy"), pre_mhz=m2(p, "mhz"), pre_mhz_min=m2(p, "mhz_min"),
                       pre_pcap=m2(p, "frac_pcap"), run=run)
            out["prefill"].append(row)
            print(f"{mm:5d} {b:4d} {l:6d} {f:4d} {var:18s} {h:>10s} {row['heat_watt']:6.1f} {str(temps):>9s} "
                  f"{row['pre_ms']:8.2f} {row['pre_watt']:6.1f} {row['pre_mhz']:7.0f} {row['pre_mhz_min']:7.0f} "
                  f"{row['pre_pcap']:5.2f}")
    if as_json:
        print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
