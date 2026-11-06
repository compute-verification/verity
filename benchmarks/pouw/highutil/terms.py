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


def main(argv: list[str] | None = None) -> int:
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
