#!/usr/bin/env python3
"""tpcomm.sh's output as one table: per set, TP, B and L, the step's split and what the phase filler adds; then the run's
gates.

    python3 benchmarks/pouw/filler_comm/report.py RUN_OUT_DIR [--json]

Every rank times the same lockstep step, so a time is the median over the arm's reps of the slowest rank's rep. base and
phase are the medians over their (interleaved) repeats. Shares are of the tensor capacity of one GPU at the clock read:
served = the step's int8 MACs, fill = the phase filler's; idle = 1 - served; comm = (base - nocomm) / base, the exposed
collective time; off-comm idle = idle - comm, the idle tensor time the collective windows don't reach; slowdown = phase /
base. fill K = fill × K_FRACTION, pfill-k's proved fraction under Theorem K (c = 1 at φ = 0.9). coverage is filler-scorer
v3's per GPU at the slowest rank's step: the scorer's serving fraction × served (at base alone, at phase with the
filler) + fill K. The fill counts every MMA
a warp ran (partial tiles included: each word is committed as it leaves its MMA); "fill (whole tiles)" counts only tiles
all of whose warps finished. "KV wrapped": the batch's KV cache doesn't fit beside the shards, so the attention re-reads
the pool: the point's time is the right size, but no server could hold it.

Gates (`gates`): the filler's pre-timing gates on every rank (pf_gate), its records after every arm (receipts and shape),
slowdown <= the scorer's LIMIT at every point, kfill_k's SASS (IMMA.16832.S8.S8 only), the alone probe's receipts, ncu's
DRAM bytes against the scorer's DRAM_LIMIT with bw_read's 1 GiB control, and clocks read only (application clocks at
their defaults in every nvidia-smi row).
"""
import argparse
import glob
import importlib.util
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

K_FRACTION = 0.9
SASS_OK = {"IMMA.16832.S8.S8"}

_spec = importlib.util.spec_from_file_location("filler_scoring", Path(__file__).resolve().parents[1] / "filler" / "scoring.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)


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
                    row["idle_off_comm"] = row["idle"] - row["comm_exposed"]
                if phase:
                    row["slowdown"] = phase / base
                    fm = first("phase", "fill_macs_rep")
                    pm = first("phase", "mhz")
                    row["fill"] = fm / (phase * 1e-3) / (1024 * 188 * pm * 1e6)
                    row["fill_k"] = row["fill"] * K_FRACTION
                    p0 = by_arm["phase"][0][0]
                    unit = "mmas" if "fill_mmas_rep" in p0 else "steps"
                    if p0.get(f"fill_{unit}_full_rep") is not None:
                        row["fill_finished"] = row["fill"] * first("phase", f"fill_{unit}_full_rep") / first("phase", f"fill_{unit}_rep")
                    row["served_phase"] = macs / (phase * 1e-3) / (1024 * 188 * pm * 1e6)
                    cs = S.CERTIFIED["serving"]["fraction"]
                    row["coverage_base"] = cs * row["served"]
                    row["coverage"] = cs * row["served_phase"] + K_FRACTION * row["fill"]
                    pr = [r for ranks in by_arm["phase"] for r in ranks]
                    row["rec_bad"] = sum(r.get("rec_bad", 0) for r in pr)
                    row["rec_checked"] = sum(r.get("rec_checked", 0) for r in pr)
                    row["whole_checked"] = sum(r.get("whole_checked", 0) for r in pr)
                    row["warps_checked"] = sum(r.get("warps_checked", 0) for r in pr)
                    row["shape_bad"] = sum(r.get("shape_bad", 0) for r in pr)
                    row["cut_warps"] = statistics.median(r["cut_warps"] for r in pr) if "cut_warps" in p0 else None
            if comm and commphase:
                row["comm_slowdown"] = commphase / comm
                fm = first("commphase", "fill_macs_rep")
                row["fill_in_comm"] = fm / (commphase * 1e-3) / (1024 * 188 * first("commphase", "mhz") * 1e6)
            yield row


def _read(p):
    return open(p).read() if os.path.exists(p) else ""


def gates(out_dir, rs):
    """The run's gates, each {"ok": bool, ...}, and all_ok."""
    pf = []
    for fn in glob.glob(os.path.join(out_dir, "set*", "rank*.jsonl")):
        pf += [json.loads(ln) for ln in open(fn) if ln.startswith("{") and '"pf_gate"' in ln]
    out = {"pf_gate": {"ok": bool(pf) and all(r["ok"] for r in pf), "records": len(pf),
                       "kinds": sorted({r.get("kind", "") for r in pf})}}
    ph = [r for r in rs if "fill" in r]
    out["receipts"] = {"ok": bool(ph) and all(r["rec_bad"] == 0 and r.get("shape_bad", 0) == 0 and r["rec_checked"] > 0
                                              for r in ph),
                       "checked": sum(r["rec_checked"] for r in ph), "bad": sum(r["rec_bad"] for r in ph),
                       "whole": sum(r.get("whole_checked", 0) for r in ph), "cut_warps": sum(r.get("warps_checked", 0) for r in ph),
                       "shape_bad": sum(r.get("shape_bad", 0) for r in ph)}
    worst = max((r["slowdown"] for r in ph), default=None)
    out["slowdown"] = {"ok": worst is not None and worst <= S.LIMIT, "worst": worst, "limit": S.LIMIT}
    ops = {ln.split()[1] for ln in _read(os.path.join(out_dir, "sass.txt")).splitlines() if len(ln.split()) == 2}
    out["sass"] = {"ok": bool(ops) and ops <= SASS_OK, "mma": sorted(ops)}
    al = [json.loads(ln) for ln in _read(os.path.join(out_dir, "alone.jsonl")).splitlines() if ln.startswith("{")]
    out["alone"] = {"ok": bool(al) and all(r["ok"] for r in al), "share_gpu": al[0]["share_gpu"] if al else None,
                    "share_fill_sms": al[0]["share_fill_sms"] if al else None}
    cb = S.ncu_metric(_read(os.path.join(out_dir, "ncu-control.csv")), "dram__bytes.sum", "bw_read")
    ct = S.ncu_metric(_read(os.path.join(out_dir, "ncu-control.csv")), "gpu__time_duration.sum", "bw_read")
    fb = S.ncu_metric(_read(os.path.join(out_dir, "ncu-filler.csv")), "dram__bytes.sum", "kfill_k")
    ft = S.ncu_metric(_read(os.path.join(out_dir, "ncu-filler.csv")), "gpu__time_duration.sum", "kfill_k")
    if cb and ct and fb and ft:
        ratio = statistics.median(cb) / S.CONTROL_BYTES
        hbm = statistics.median(cb) / statistics.median(ct)
        rate = max(b / t for b, t in zip(fb, ft))
        valid = S.CONTROL_RANGE[0] <= ratio <= S.CONTROL_RANGE[1]
        out["dram"] = {"ok": valid and rate <= S.DRAM_LIMIT * hbm, "control_bytes_ratio": ratio,
                       "filler_bytes_per_launch": max(fb), "share_of_hbm": rate / hbm, "limit": S.DRAM_LIMIT}
    else:
        out["dram"] = {"ok": False, "why": f"an ncu metric is missing ({len(cb)}/{len(ct)}, {len(fb)}/{len(ft)})"}
    smi = [[x.strip() for x in ln.split(",")] for ln in _read(os.path.join(out_dir, "gpu.txt")).splitlines()]
    smi = [f for f in smi if len(f) >= 9]
    out["clocks"] = {"ok": bool(smi) and all(f[6] == f[7] for f in smi), "rows": len(smi),
                     "app_mhz": sorted({f[6] for f in smi}), "default_app_mhz": sorted({f[7] for f in smi})}
    out["all_ok"] = all(v["ok"] for v in out.values())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    rs = list(rows(a.out_dir))
    g = gates(a.out_dir, rs)
    if a.json:
        for r in rs:
            print(json.dumps(r))
        print(json.dumps({"gates": g}))
        return
    f = lambda v, p=4: "-" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))
    print("| set | TP | B | L | base ms | comm exposed | served | idle | off-comm idle | phase ms | slowdown | fill | "
          f"fill K (x{K_FRACTION}) | fill (whole tiles) | coverage base → phase | comm+filler/comm | receipts bad/checked |")
    print("|" + "---|" * 17)
    for r in rs:
        L = f"{r['L']} (KV wrapped)" if r["kv_wrapped"] else r["L"]
        print(f"| {r['set']} | {r['tp']} | {r['B']} | {L} | {f(r['base_ms'], 2)} | {f(r.get('comm_exposed'), 3)} | "
              f"{f(r.get('served'), 4)} | {f(r.get('idle'), 3)} | {f(r.get('idle_off_comm'), 3)} | {f(r['phase_ms'], 2)} | "
              f"{f(r.get('slowdown'), 4)} | {f(r.get('fill'), 4)} | {f(r.get('fill_k'), 4)} | {f(r.get('fill_finished'), 4)} | "
              f"{f(r.get('coverage_base'), 4)} → {f(r.get('coverage'), 4)} | {f(r.get('comm_slowdown'), 4)} | {r.get('rec_bad', '-')}/{r.get('rec_checked', '-')} |")
    for s in sorted({(r["set"], r["settings"]) for r in rs}):
        print(f"- {s[0]}: `{s[1]}`")
    print(f"- gates {'all ok' if g['all_ok'] else 'NOT all ok'}: " + "; ".join(
        f"{k} {'ok' if v['ok'] else 'FAIL'} " + json.dumps({x: y for x, y in v.items() if x != "ok"})
        for k, v in g.items() if k != "all_ok"))


if __name__ == "__main__":
    main()
