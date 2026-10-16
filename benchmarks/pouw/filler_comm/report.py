#!/usr/bin/env python3
"""tpcomm.sh's output as one table: per set, TP, B and L, the step's split and what the phase filler adds.

    python3 benchmarks/pouw/filler_comm/report.py RUN_OUT_DIR [--json]

Every rank times the same lockstep step, so a time is the median over the arm's reps of the slowest rank's rep. base and
phase are the medians over their (interleaved) repeats. Shares are of the tensor capacity of one GPU at the clock read:
served = the step's int8 MACs, fill = the phase filler's; idle = 1 - served; comm = (base - nocomm) / base, the exposed
collective time; slowdown = phase / base. The filler's certified fraction is not applied here (the filler spec's
provisional 0.99 under CH, its proved floor otherwise). "KV wrapped": the batch's KV cache doesn't fit beside the shards,
so the attention re-reads the pool: the point's time is the right size, but no server could hold it.
"""
import argparse
import glob
import json
import os
import statistics
from collections import defaultdict


def rows(out_dir):
    for sd in sorted(glob.glob(os.path.join(out_dir, "set*"))):
        settings = open(os.path.join(sd, "settings.txt")).read().strip() if os.path.exists(os.path.join(sd, "settings.txt")) else ""
        arms = defaultdict(lambda: defaultdict(list))  # (tp, B, L) -> arm -> [per-repeat [rank records]]
        gates = []
        wrapped = set()
        for f in sorted(glob.glob(os.path.join(sd, "rank*.jsonl"))):
            seen = defaultdict(int)
            for line in open(f):
                if not line.startswith("{"):
                    continue
                r = json.loads(line)
                if r.get("mode") == "pf_gate":
                    gates.append(r["ok"])
                if r.get("mode") == "point" and r.get("kv_wrapped"):
                    wrapped.add((r["tp"], r["B"], r["L"]))
                if r.get("mode") != "tp":
                    continue
                key = (r["tp"], r["B"], r["L"])
                i = seen[(key, r["arm"])]
                seen[(key, r["arm"])] += 1
                lst = arms[key][r["arm"]]
                while len(lst) <= i:
                    lst.append([])
                lst[i].append(r)
        for key, by_arm in sorted(arms.items()):
            def t(arm):
                reps = by_arm.get(arm)
                if not reps:
                    return None
                meds = []
                for ranks in reps:
                    n = min(len(x["reps"]) for x in ranks)
                    meds.append(statistics.median(max(x["reps"][j][0] for x in ranks) for j in range(n)))
                return statistics.median(meds)

            def first(arm, k):
                reps = by_arm.get(arm)
                return statistics.median(r[k] for ranks in reps for r in ranks) if reps else None

            base, nocomm, comm, phase, commphase = (t(a) for a in ("base", "nocomm", "comm", "phase", "commphase"))
            row = {"set": os.path.basename(sd), "settings": settings, "tp": key[0], "B": key[1], "L": key[2],
                   "base_ms": base, "nocomm_ms": nocomm, "comm_ms": comm, "phase_ms": phase, "commphase_ms": commphase,
                   "gates_ok": all(gates) if gates else None, "kv_wrapped": key in wrapped}
            if base:
                mhz = first("base", "mhz")
                macs = first("base", "served_macs")
                row["served"] = macs / (base * 1e-3) / (1024 * 188 * mhz * 1e6)
                row["idle"] = 1 - row["served"]
                if nocomm:
                    row["comm_exposed"] = (base - nocomm) / base
                if phase:
                    row["slowdown"] = phase / base
                    fm = first("phase", "fill_macs_rep")
                    pm = first("phase", "mhz")
                    row["fill"] = fm / (phase * 1e-3) / (1024 * 188 * pm * 1e6)
                    if by_arm["phase"][0][0].get("fill_steps_full_rep") is not None:
                        row["fill_finished"] = row["fill"] * first("phase", "fill_steps_full_rep") / first("phase", "fill_steps_rep")
                    row["served_phase"] = macs / (phase * 1e-3) / (1024 * 188 * pm * 1e6)
                    row["rec_bad"] = sum(r.get("rec_bad", 0) for ranks in by_arm["phase"] for r in ranks)
                    row["rec_checked"] = sum(r.get("rec_checked", 0) for ranks in by_arm["phase"] for r in ranks)
            if comm and commphase:
                row["comm_slowdown"] = commphase / comm
                fm = first("commphase", "fill_macs_rep")
                row["fill_in_comm"] = fm / (commphase * 1e-3) / (1024 * 188 * first("commphase", "mhz") * 1e6)
            yield row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    rs = list(rows(a.out_dir))
    if a.json:
        for r in rs:
            print(json.dumps(r))
        return
    f = lambda v, p=4: "-" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))
    print("| set | TP | B | L | base ms | nocomm ms | comm ms | comm exposed | served | idle | phase ms | slowdown | fill | "
          "fill (finished tiles) | served+fill | comm+filler/comm | receipts bad/checked |")
    print("|" + "---|" * 17)
    for r in rs:
        tot = r["served_phase"] + r["fill"] if "fill" in r else None
        L = f"{r['L']} (KV wrapped)" if r["kv_wrapped"] else r["L"]
        print(f"| {r['set']} | {r['tp']} | {r['B']} | {L} | {f(r['base_ms'], 2)} | {f(r['nocomm_ms'], 2)} | "
              f"{f(r['comm_ms'], 2)} | {f(r.get('comm_exposed'), 3)} | {f(r.get('served'), 4)} | {f(r.get('idle'), 3)} | "
              f"{f(r['phase_ms'], 2)} | {f(r.get('slowdown'), 4)} | {f(r.get('fill'), 4)} | {f(r.get('fill_finished'), 4)} | "
              f"{f(tot, 4)} | "
              f"{f(r.get('comm_slowdown'), 4)} | {r.get('rec_bad', '-')}/{r.get('rec_checked', '-')} |")
    for s in sorted({(r["set"], r["settings"]) for r in rs}):
        print(f"- {s[0]}: `{s[1]}`")


if __name__ == "__main__":
    main()
