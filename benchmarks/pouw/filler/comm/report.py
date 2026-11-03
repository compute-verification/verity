#!/usr/bin/env python3
"""tpcomm.sh's output as one table: per set, TP, B and L, the step's split and what the phase filler adds; then the run's
gates.

    python3 benchmarks/pouw/filler/comm/report.py RUN_OUT_DIR [--json]

Every rank times the same lockstep step, so a time is the median over the arm's reps of the slowest rank's rep. base and
phase are the medians over their (interleaved) repeats. Shares are of the tensor capacity of one GPU at the clock read:
served = the step's int8 MACs, fill = the phase filler's; idle = 1 - served; comm = (base - nocomm) / base, the exposed
collective time; off-comm idle = idle - comm, the idle tensor time the collective windows don't reach; slowdown = phase /
base. fill K = fill × the fold's proved fraction under Theorem K: duo (NC 2) binds every word, PHI = 0.9,
provisional: its binding is proved for runs of one to four words (Lean Pouw.SecurityProofs.Receipt.DuoQuadNoKernel), and
five or more words and the served receipt's projected form are open; the ram fold (NC 4) binds word pairs only, 0.45;
r9's XOR-family (NC 4, -DPFILL_LI/-DPFILL_XIN13) is provisional, credited at the same proved half 0.45.
coverage is filler-scorer
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

PHI = 0.9  # Theorem K's per-word proved fraction for a word-binding fold (c = 1 at φ 0.9)
K_FRACTION = PHI  # the fraction of a run with no fold recorded (an old binary): the word-binding reading
SASS_OK = {"IMMA.16832.S8.S8"}


def proved_fraction(nc, xin, S):
    """The fold's proved fill fraction under Theorem K (the filler receipt break). duo (NC 2) binds every
    word: PHI, provisional (binding proved for runs of one to four words, DuoQuadNoKernel; longer runs and the served
    receipt's projected form open). The ram fold (NC 4) binds a word PAIR only through Q = a0·d0 + d1, so it certifies half the words
    (PHI/2); round 3's rem fill K, taken at PHI under ram, was 2× this. The r9 XOR-family folds (NC 4 with -DPFILL_LI
    or -DPFILL_XIN13) read at the same proved half (a run may pick a
    complement flip from the data, so Theorem K certifies only >= 1/2 of the words); their fuller reading waits on r10's
    select-charging lemma, so it isn't credited here."""
    if nc is None:
        return PHI
    if nc == 2:
        return PHI
    return PHI * 0.5


def fold_label(nc, xin):
    if nc is None:
        return None
    if nc == 2:
        return "duo"
    if nc == 4 and xin:
        return "xin13/li(prov)"
    return "ram"

_spec = importlib.util.spec_from_file_location("filler_scoring", Path(__file__).resolve().parents[1] / "scoring.py")
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
                if not reps:
                    return None
                vals = [r[k] for ranks in reps for r in ranks if k in r]
                return statistics.median(vals) if vals else None

            def fold_of(arm):
                nc, xin, S = first(arm, "fold_nc"), first(arm, "fold_xin"), first(arm, "fold_s")
                nc = int(nc) if nc is not None else None
                return nc, bool(xin), int(S) if S is not None else None

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
                    nc, xin, Sf = fold_of("phase")
                    frac = proved_fraction(nc, xin, Sf)
                    row["fold"], row["fill_k"] = fold_label(nc, xin), row["fill"] * frac
                    p0 = by_arm["phase"][0][0]
                    unit = "mmas" if "fill_mmas_rep" in p0 else "steps"
                    if p0.get(f"fill_{unit}_full_rep") is not None:
                        row["fill_finished"] = row["fill"] * first("phase", f"fill_{unit}_full_rep") / first("phase", f"fill_{unit}_rep")
                    row["served_phase"] = macs / (phase * 1e-3) / (1024 * 188 * pm * 1e6)
                    cs = S.CERTIFIED["serving"]["fraction"]
                    row["coverage_base"] = cs * row["served"]
                    row["coverage"] = cs * row["served_phase"] + frac * row["fill"]
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
            # the deferred pass: defer (deferred work in the gaps, overflow drained after the step) and dalone (the
            # whole step's deferred work drained with no gaps)
            defer, dalone = t("defer"), t("dalone")
            row["defer_ms"], row["dalone_ms"] = defer, dalone
            if defer and base:
                dm = first("defer", "defer_demand_macs")
                gap = first("defer", "defer_gap_macs_rep")
                drain = first("defer", "defer_drain_macs_rep")
                pm = first("defer", "mhz")
                row["defer_gap_share"] = gap / dm if dm else None      # the step's deferred work run beside the collectives
                row["defer_overflow_share"] = drain / dm if dm else None  # the overflow, drained after the step
                # the true step-time cost is defer_slowdown; a gap share of 1 with a slowdown > 1 means it ran in the gap
                # windows but did not hide (the DRAM-bound load contends with the step)
                row["defer_slowdown"] = defer / base
                fm = first("defer", "fill_macs_rep")
                row["defer_fill"] = fm / (defer * 1e-3) / (1024 * 188 * pm * 1e6)
                nc, xin, Sf = fold_of("defer")  # the filler after the deferred quota uses the same fold
                row["defer_fold"] = fold_label(nc, xin)
                row["defer_fill_k"] = row["defer_fill"] * proved_fraction(nc, xin, Sf)  # the filler's remaining proved fill
                store = first("defer", "defer_store_bytes")
                row["defer_store_gib"] = store / 2**30 if store else None  # per-shard stored operands for the sub-round
                dr = [r for ranks in by_arm["defer"] for r in ranks]
                row["dp_whole_bad"] = sum(r.get("dp_whole_bad", 0) for r in dr) + sum(r.get("dp_whole_control_equal", 0) for r in dr)
                row["dp_whole_checked"] = sum(r.get("dp_whole_checked", 0) for r in dr)
                # the remaining fill's counters against its records (kf_shape); a record without the field wasn't checked
                row["defer_fill_bad"] = sum(r.get("fill_shape_bad", 1) for r in dr)
                row["dcap"], row["dSd"], row["dT"] = first("defer", "dcap"), first("defer", "dSd"), first("defer", "dT")
            if dalone and base:
                row["dalone_cost"] = dalone / base  # the deferred pass's decode cost a step without the gaps
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
    # the deferred pass: the pre-timing dp_gate on every rank, and the deferred receipts after the defer arm
    dp = []
    for fn in glob.glob(os.path.join(out_dir, "set*", "rank*.jsonl")):
        dp += [json.loads(ln) for ln in open(fn) if ln.startswith("{") and '"dp_gate"' in ln]
    if dp:
        out["dp_gate"] = {"ok": all(r["ok"] for r in dp), "records": len(dp),
                          "bad": sum(r["bad"] + r["control_equal"] + r["q_bad"] for r in dp)}
        df = [r for r in rs if r.get("dp_whole_checked")]
        out["defer_receipts"] = {"ok": bool(df) and all(r["dp_whole_bad"] == 0 and r["dp_whole_checked"] > 0
                                                        and r.get("defer_fill_bad") == 0 for r in df),
                                 "checked": sum(r["dp_whole_checked"] for r in df), "bad": sum(r["dp_whole_bad"] for r in df),
                                 "fill_bad": sum(r.get("defer_fill_bad", 0) for r in df)}
    sass_lines = _read(os.path.join(out_dir, "sass.txt")).splitlines()
    ops = {ln.split()[-1] for ln in sass_lines if len(ln.split()) == 2 and "MMA" in ln.split()[-1]}
    nonrz = sum(int(ln.split()[-1]) for ln in sass_lines if "non-RZ IMMA:" in ln)
    out["sass"] = {"ok": bool(ops) and ops <= SASS_OK and nonrz == 0, "mma": sorted(ops), "non_rz": nonrz}
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
    if any(r.get("defer_ms") for r in rs):
        print()
        print("Decision 36's deferred decode pass: demand a step = the step's served MACs; gap share = run beside the "
              "collectives; overflow = drained after the step; slowdown = the true step-time cost (gap share 1 with "
              "slowdown > 1 means it ran in the gap windows but did not hide); store GiB/shard = the served-side operands "
              "kept over the sub-round (per shard, scales with T); fold = the receipt fold (duo 0.9, provisional: "
              "binding proved for runs of one to four words; ram 0.45; xin provisional); rem. fill K = the filler's remaining proved fill under that fold; dalone = the whole "
              "step's deferred work with no gaps.")
        print("| set | TP | B | L | T | store GiB/shard | base ms | defer ms | defer slowdown | gap share | overflow "
              "share | fold | rem. fill K | dalone ms | dalone cost (xbase) | dp bad/checked |")
        print("|" + "---|" * 16)
        for r in rs:
            if not r.get("defer_ms"):
                continue
            L = f"{r['L']} (KV wrapped)" if r["kv_wrapped"] else r["L"]
            print(f"| {r['set']} | {r['tp']} | {r['B']} | {L} | {f(r.get('dT'))} | {f(r.get('defer_store_gib'), 1)} | "
                  f"{f(r['base_ms'], 2)} | {f(r.get('defer_ms'), 2)} | {f(r.get('defer_slowdown'), 4)} | "
                  f"{f(r.get('defer_gap_share'), 4)} | {f(r.get('defer_overflow_share'), 4)} | {r.get('defer_fold') or '-'} | "
                  f"{f(r.get('defer_fill_k'), 4)} | {f(r.get('dalone_ms'), 2)} | "
                  f"{f(r.get('dalone_cost'), 4)} | {r.get('dp_whole_bad', '-')}/{r.get('dp_whole_checked', '-')} |")
    for s in sorted({(r["set"], r["settings"]) for r in rs}):
        print(f"- {s[0]}: `{s[1]}`")
    print(f"- gates {'all ok' if g['all_ok'] else 'NOT all ok'}: " + "; ".join(
        f"{k} {'ok' if v['ok'] else 'FAIL'} " + json.dumps({x: y for x, y in v.items() if x != "ok"})
        for k, v in g.items() if k != "all_ok"))


if __name__ == "__main__":
    main()
