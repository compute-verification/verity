"""Where the card's capacity goes in a HIGHUTIL run, from period.py's summary.json and the protocol arm's kernel timeline
(timeline.py over the same settings), at θ under each reading of the card's clock side by side (the H200's tensor clock
and the boost clock nvidia-smi reports: `h200_step.CLOCKS_MHZ`; any other card, its card file's clock). At each reading
the terms sum to 1:

- accounted: the declared MACs at θ over the wall, the declared utilization;
- the plain GEMM's floor: the same run's plain cuBLASLt arm (at whatever clock the card holds there) past the MACs
  at θ, what no protocol could recover;
- the protocol: the protocol block's GPU time past the plain arm's, split into the GEMM beside the hashing (its GEMMs'
  time past the plain arm's), off the GEMM (the coins, forming and hashing with no GEMM running) and idle in a block;
- host between blocks: the protocol wall the GPU spends outside its blocks (coin round trips, the host's turn).

ε = ε_S + (1 - accounted), ε_S the sampling's (ε_skip). `credited` says what the accounted share counts. The certified
MACs, the declared MACs plus the peels the honest route computes (Z_X U^T and V Y_W^T, 2 m n L' a call), are reported
apart. With --periods and --smi, each period's SM clock (the blocks' samples) and nvidia-smi's over its wall-clock
window: the window is pinned on the challenger session's id (its UTC second) and trimmed a second at each end.

    terms.py SUMMARY TIMELINE [--periods PERIODS.jsonl --smi SMI.csv]      prints one JSON object

`terms.py served` reads a served step instead, from kernels' breakdown of its profile: the JSON
`benchmarks/hash/exactlin/tensor_util.py` writes for one run of `benchmarks/one_stage/served_util.sh` (VLLM-33), the one
per-kernel source P1a's table reads. It adds only what that breakdown leaves out:

- **the certified MACs** beside the declared: the declared plus the peels this route computes (2 m n L' a call), at the
  L' `period.lp_of` gives each of the model's linears at the word interval L (`--L`). That models this route's honest
  work on the served shapes; it measures nothing in the run, whose scheme may have no L' (H1 + fx1's noise is a keyed
  per-row word), so no share of the step is given for it.
- **the step itemized to 1 in SM-cycles**: each class's exclusive time at the SM clock sampled under it (the class's
  `clock_mhz`, else the steps'), over the step's cycles. The declared int8 linears' MACs at 4,096 a SM-cycle on each SM are
  the accounted share, and the int8 GEMM's other cycles are its shortfall below θ.

u (the int8 linears at θ_int8, the bf16 attention and LM head at half that) and the accounted share are given over the
step's SM-cycles, the data sheet's 1,830 MHz beside them. `--credit` says what the accounted share credits and
`--covers` what kind of run it is.

    terms.py served [LABEL=]KERNELS.json ... [--L 16384] [--credit TEXT] [--covers TEXT] [--text OUT.txt]
                                  prints a JSON list, one object a run, and, with --text, a table each
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "accel_v31"))

import h200_step

#: the SM clock of each reading by card SKU; a card not named here has its card file's clock as its one reading
READINGS = {"h200-sxm-141gb": h200_step.CLOCKS_MHZ}

CREDITED = ("the declared MACs alone: each accepted call's m n k, the served or filler product. Tensor-only accounting "
            "credits nothing the protocol does (the peels, the hashing, the coins, the forming), the host's time or the "
            "GEMM's shortfall below θ: an accepted prover can run hidden tensor work beside each. The certified "
            "MACs (the declared plus the peels) are reported apart and not credited.")


def readings(card: dict) -> dict[str, float]:
    return dict(READINGS.get(card["sku"], {"card": card["clock_mhz"]}))


def macs_per_step(workload: dict) -> dict:
    m, lp = workload["m"], workload["lp"]
    declared = workload["step_macs"]
    peels = sum(2 * m * n * (lp.get(name) or 0) for name, n, _ in workload["layers"])
    return {"declared": declared, "peels": peels, "certified": declared + peels,
            "certified_over_declared": (declared + peels) / declared}


def terms(summary: dict, timeline: dict, periods: list[dict] | None = None, smi: list | None = None) -> dict:
    card = summary["card"]
    if not card:
        raise SystemExit("terms.py: summary.json has no card (period.py run --card)")
    lt = summary.get("baselines", {}).get("lt", {})
    wall, gpu = summary["proto_s_per_step"], summary["proto_gpu_s_per_step"]
    plain = lt.get("plain_gpu_s_per_step") or summary["plain_gpu_s_per_step"]
    gemm, busy = timeline["gemm_share"] * gpu, timeline["busy_share"] * gpu
    macs = macs_per_step(summary["workload"])
    out = {"card": card["sku"], "credited": CREDITED, "macs_per_step": macs, "readings": {},
           "eps_skip": summary["eps_skip"], "delta": summary["delta"],
           "ms_per_step": {"wall": wall * 1e3, "gpu": gpu * 1e3, "plain_gpu": plain * 1e3,
                           "protocol_gemm": gemm * 1e3, "protocol_busy": busy * 1e3}}
    for name, mhz in readings(card).items():
        theta = card["tensor"] * card["sms"] * mhz * 1e6
        ideal = macs["declared"] / theta
        proto = {"GEMM beside the hashing": (gemm - plain) / wall, "off the GEMM": (busy - gemm) / wall,
                 "idle in a block": (gpu - busy) / wall}
        t = {"accounted": ideal / wall, "the plain GEMM's floor": (plain - ideal) / wall,
             "the protocol": sum(proto.values()), "host between blocks": (wall - gpu) / wall}
        out["readings"][name] = {
            "clock_mhz": mhz, "theta_macs_s": theta, "terms": t, "protocol": proto, "sum": sum(t.values()),
            "eps_undecl": 1 - t["accounted"], "eps": summary["eps_skip"] + 1 - t["accounted"],
            "certified_share": macs["certified"] / theta / wall, "plain_share": ideal / plain,
            "ideal_ms_per_step": ideal * 1e3}
    if periods is not None:
        out["clocks"] = clocks(periods, smi or [])
    return out


def smi_samples(text: str) -> list[tuple[datetime, dict[str, float]]]:
    """nvidia-smi's CSV (run.sh's: timestamp, clocks.sm, clocks.gr, clocks.mem, ...) as (UTC time, values)."""
    rows = list(csv.reader(text.splitlines()))
    if not rows:
        return []
    head = [h.strip() for h in rows[0]]
    col = {k: next((i for i, h in enumerate(head) if h.startswith(p)), None)
           for k, p in (("sm_mhz", "clocks.current.sm"),)}
    out = []
    for r in rows[1:]:
        if len(r) != len(head) or not re.fullmatch(r"\d{4}/\d\d/\d\d \d\d:\d\d:\d\d\.\d+", r[0].strip()):
            continue
        vals = {k: re.match(r"\s*(\d+(?:\.\d+)?)\b", r[i]) for k, i in col.items() if i is not None}
        if vals and all(vals.values()):
            t = datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=UTC)
            out.append((t, {k: float(v.group(1)) for k, v in vals.items()}))
    return out


def windows(periods: list[dict]) -> list[tuple[datetime, datetime] | None]:
    """Each period's wall-clock window: a protocol period's opens at its session's id plus the session's opening, and
    the plain periods before it end where the next begins; None where nothing pins it."""
    out: list[tuple[datetime, datetime] | None] = [None] * len(periods)
    for i, r in enumerate(periods):
        m = re.match(r"s(\d{8}T\d{6})Z", (r.get("coins") or {}).get("session") or "")
        if r["arm"] != "proto" or not m:
            continue
        t = datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
        start = t + timedelta(seconds=r["session_open_s"])
        out[i] = (start, start + timedelta(seconds=r["wall_s"]))
        end, j = t, i - 1
        while j >= 0 and periods[j]["arm"] == "plain":
            out[j] = (end - timedelta(seconds=periods[j]["wall_s"]), end)
            end, j = out[j][0], j - 1
    return out


def clocks(periods: list[dict], smi: list) -> list[dict]:
    out = []
    for r, w in zip(periods, windows(periods)):
        mhz = [b["sm_mhz"] for b in r["blocks"] if b.get("sm_mhz")]
        row = {"period": r["period"], "arm": r["arm"], "baseline": r.get("baseline"), "cheat": r.get("cheat"),
               "wall_s": r["wall_s"], "blocks_sm_mhz_median": statistics.median(mhz) if mhz else None, "smi": None}
        if w:
            lo, hi = w[0] + timedelta(seconds=1), w[1] - timedelta(seconds=1)
            got = [v for t, v in smi if lo <= t <= hi]
            if got:
                row["smi"] = {"samples": len(got), "clocks_sm_mhz": statistics.median(v["sm_mhz"] for v in got)}
        out.append(row)
    return out


H200_SMS, INT8_PER_SM_CLOCK = 132, 4096
#: the H200's dense bf16 tensor rate over its int8 rate (989.4 against 1,978.9 T/s at 1,830 MHz)
BF16_OVER_INT8 = 0.5
SHEET_MHZ = h200_step.CLOCKS_MHZ["tensor"]
UNCREDITED = "uncredited: development key, credit not established"
#: what a util80 run is (Daniel, 7:44 PM PDT on CMP-231: full-utilization prefill is a serving target, not a certification one)
SERVED = ("a served run: no certifier in front, PoUW uncredited (development key); V3's certified run is a small declared "
          "workload sized to the network certifier")
#: period.py's defaults for its honest route: the least L', the longest, and the word interval L
ROUTE_LP, ROUTE_LP_MAX, ROUTE_L = 512, 1024, 16384
GEMM = "int8 GEMM"
ACCOUNTED, BELOW = "accounted: the declared int8 linears at θ", "the int8 GEMM below θ"


def certified_ratio(linears: dict[str, list[int]], L: int = ROUTE_L) -> dict:
    """#5724's certified over declared MACs on these linears ({name: [N, K]}): each call's L' by `period.lp_of` at L (a
    linear no L' covers runs as `period.calls`' K-slices), its peels 2 m n L' over its m n K."""
    import period as P
    saved = P.LAYERS, P.M, P.NB, P.LP, P.L, P.LP_MAX
    try:
        P.configure({"layers": [[name, n, k] for name, (n, k) in linears.items()], "m": 1, "nb": 1, "lp": ROUTE_LP,
                     "lp_max": ROUTE_LP_MAX, "L": L})
        cs = P.calls(P.LAYERS)
        lp = {c: P.lp_of(k) for c, _, k, _, _ in cs}
    finally:
        P.LAYERS, P.M, P.NB, P.LP, P.L, P.LP_MAX = saved
    wl = {"m": 1, "lp": lp, "layers": [(c, n, k) for c, n, k, _, _ in cs], "step_macs": sum(n * k for _, n, k, _, _ in cs)}
    return {"L": L, "lp": lp, "certified_over_declared": macs_per_step(wl)["certified_over_declared"]}


def _mhz(v: float | None) -> float | None:
    """A clock in MHz; one of kernels' exports (r20261028-023843-627b) holds nsys's Hz under the MHz key."""
    return v / 1e6 if v and v > 1e5 else v


def served(doc: dict, label: str, L: int = ROUTE_L, credit: str = UNCREDITED, covers: str = SERVED) -> dict:
    """One run from kernels' `tensor_util.py` JSON: u and the accounted share over the step's SM-cycles, the certified MACs
    beside the declared, and the step's cycles by class, summing to 1."""
    if "error" in doc or "summary" not in doc:
        raise SystemExit(f"terms.py served: {label}: kernels' breakdown has no steps ({doc.get('error')})")
    s, model = doc["summary"], doc["model"]
    clock = _mhz(s.get("clock_mhz"))
    if clock is None:
        raise SystemExit(f"terms.py served: {label}: kernels' breakdown has no measured SM clock (GPU metrics)")
    cycles, unsampled = {}, []
    for r in doc["classes"]:
        mhz = _mhz(r.get("clock_mhz"))
        if mhz is None and r["exclusive_ms"]:
            unsampled.append(r["class"])
        cycles[r["class"]] = r["exclusive_ms"] * 1e3 * (mhz or clock)
    total = sum(cycles.values())
    per_cycle = INT8_PER_SM_CLOCK * H200_SMS
    macs = s["macs"]
    int8 = macs["int8"] / per_cycle
    bf16 = (macs["attn"] + macs["head"]) / (BF16_OVER_INT8 * per_cycle)
    sheet = per_cycle * SHEET_MHZ * 1e6 * s["wall_ms"] / 1e3
    ratio = certified_ratio(model["linears"], L)
    shares = {ACCOUNTED: int8 / total, BELOW: (cycles.pop(GEMM, 0.0) - int8) / total}
    shares |= {c: v / total for c, v in sorted(cycles.items(), key=lambda kv: -kv[1])}
    return {
        "run": label, "T": doc.get("tokens"), "steps": s["steps"], "dropped": s.get("dropped"), "step_wall_ms": s["wall_ms"],
        "covers": covers, "credit": credit, "clock_mhz": clock, "sheet_mhz": SHEET_MHZ,
        "sm_cycles": total, "sm_cycles_over_clock_wall": total / (clock * 1e3 * s["wall_ms"]),
        "unsampled_classes": unsampled,
        "u": (int8 + bf16) / total, "u_sheet": (int8 + bf16) * per_cycle / sheet,
        "accounted": int8 / total, "accounted_sheet": macs["int8"] / sheet,
        "macs": {"declared": macs["int8"], "certified": ratio["certified_over_declared"] * macs["int8"],
                 "attention": macs["attn"], "lm_head": macs["head"]},
        "certified": ratio | {"of": "#5724's peels on the served shapes, a model: not what the run's scheme computes"},
        "tensor_active": s.get("tensor_active"), "shares": shares, "shares_sum": sum(shares.values()),
    }


def served_text(r: dict) -> str:
    c = r["certified"]
    lines = [f"{r['run']}: T {r['T']:,}, {r['steps']} steps of {r['step_wall_ms']:.2f} ms at {r['clock_mhz']:,.0f} MHz "
             f"measured ({r['covers']})",
             f"u {r['u']:.2%} at the run's clock ({r['u_sheet']:.2%} at the data sheet's {r['sheet_mhz']:,} MHz); "
             f"accounted {r['accounted']:.2%} ({r['accounted_sheet']:.2%}) [{r['credit']}]"
             + (f"; Tensor Active {r['tensor_active']:.1f}%" if r.get("tensor_active") is not None else ""),
             f"declared {r['macs']['declared']:.4g} MACs a step; certified {r['macs']['certified']:.4g} at "
             f"{c['certified_over_declared']:.4f}× ({c['of']}; L' {c['lp']} at L {c['L']:,})",
             f"the step in SM-cycles at the clock sampled under each class ({r['sm_cycles']:.4g} a step, "
             f"{r['sm_cycles_over_clock_wall']:.4f} of the mean clock × wall; sums to {r['shares_sum']:.4f}):"]
    lines += [f"  {v:7.2%}  {k}" for k, v in r["shares"].items() if abs(v) >= 5e-5]
    if r["unsampled_classes"]:
        lines.append(f"at the steps' clock (no sample under them): {', '.join(r['unsampled_classes'])}")
    return "\n".join(lines)


def served_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="terms.py served", description="a served step's certified MACs and its SM-cycles by "
                                 "class, from kernels' tensor_util.py breakdown")
    ap.add_argument("kernels", nargs="+", metavar="[LABEL=]KERNELS.json", help="tensor_util.py's JSON, one a run")
    ap.add_argument("--L", type=int, default=ROUTE_L, help="the word interval the certified MACs' L' is taken at")
    ap.add_argument("--credit", default=UNCREDITED)
    ap.add_argument("--covers", default=SERVED)
    ap.add_argument("--text", type=Path, help="also write the tables here")
    a = ap.parse_args(argv)
    rs = []
    for arg in a.kernels:
        label, _, path = arg.rpartition("=") if "=" in arg else (arg, "", arg)
        rs.append(served(json.loads(Path(path).read_text()), label, a.L, a.credit, a.covers))
    print(json.dumps(rs, indent=1))
    if a.text:
        a.text.write_text("\n\n".join(served_text(r) for r in rs) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["served"]:
        return served_main(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("summary")
    ap.add_argument("timeline")
    ap.add_argument("--periods", help="periods.jsonl, for each period's clock")
    ap.add_argument("--smi", help="run.sh's smi.csv")
    a = ap.parse_args(argv)
    s, tl = (json.loads(Path(x).read_text()) for x in (a.summary, a.timeline))
    periods = [json.loads(x) for x in Path(a.periods).read_text().splitlines() if x.strip()] if a.periods else None
    smi = smi_samples(Path(a.smi).read_text()) if a.smi and Path(a.smi).exists() else []
    print(json.dumps(terms(s, tl, periods, smi), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
