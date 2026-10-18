"""A8's cost per opening, from its runs' records (`a8.py --points`, each session's ``cost/<arm>-k<k>-r<rep>/prove.log``),
and what that costs a window of the remote bandwidth audit.

    python benchmarks/one_stage/a8_cost.py OUT/a8 [OUT/a8 ...] [--openings 33273] [--window-s 60] [--delta-log2 40]
      [--statement 512] [--gpus 1,8,64] [--horizon 10]

For each session it reads the prover's LIVE line: ``e2e_s``, ``wait_e2e_s`` (its wait on the verifier inside e2e), the
proof reps' ``buckets``, and from them the device phase (each rep's ``t.total``: what the prover's device, the GPU under
``--gpu``, spends on the proof), the host setup (``h.setup``) and the firewall's host recheck (``cpu.firewall_recheck``);
and the process's wall and CPU seconds from cost.jsonl. Per run and arm it fits each time on k, the drawn units (least
squares): the slope is an opening's, the intercept the session's. An opening's cost at a full statement of
``--statement`` units (C-Flock's ceiling, m = 35 at the unit's k_log 26) is (intercept + slope · statement) / statement:
the device phase alone (the floor: a prover that overlaps every host step), and the cold session's e2e and wall (today's
prover, one process a statement). A window's ``--openings`` at that cost, in GPU-s, and the GPUs that proves them in
``--window-s``; then what ``--gpus`` GPUs prove a window, and the ε that buys at δ = 2^-``--delta-log2``: ε = ln(1/δ)/k for
the window's k proved openings (each window's own reading), and pooled over ``--horizon`` windows (§10's reading).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def sessions(out: Path) -> list[dict]:
    """Every cost session under ``out`` (an a8 output directory): cost.jsonl's row with its LIVE line's breakdown."""
    rows = []
    for line in (out / "cost" / "cost.jsonl").read_text().splitlines():
        r = json.loads(line)
        log = out / "cost" / f"{r['arm']}-k{r['k']}-r{r['rep']}" / "prove.log"
        live = next((json.loads(x[5:]) for x in log.read_text().splitlines() if x.startswith("LIVE\t")), {}) if log.exists() else {}
        b = live.get("buckets") or []
        rows.append({**r, "device_s": sum(x.get("t.total", 0.0) for x in b), "setup_s": sum(x.get("h.setup", 0.0) for x in b),
                     "recheck_s": sum(x.get("cpu.firewall_recheck", 0.0) for x in b),
                     "prover_s": r["e2e_s"] - (live.get("wait_e2e_s") or 0.0), "proof_bytes": sum(live.get("proof_bytes") or [])})
    return rows


TIMES = ("device_s", "prover_s", "e2e_s", "wall_s", "cpu_s")


def fit(rows: list[dict]) -> dict:
    ks = np.array([r["k"] for r in rows], dtype=float)
    out = {"points": sorted({r["k"] for r in rows}), "sessions": len(rows)}
    for x in TIMES:
        y = np.array([r[x] for r in rows], dtype=float)
        slope, icpt = np.polyfit(ks, y, 1) if len(set(ks)) > 1 else (float("nan"), float(y.mean()))
        out[x] = {"per_opening": float(slope), "fixed": float(icpt)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("outs", nargs="+")
    ap.add_argument("--openings", type=int, default=33273)
    ap.add_argument("--window-s", type=float, default=60.0)
    ap.add_argument("--delta-log2", type=float, default=40.0)
    ap.add_argument("--statement", type=int, default=512)
    ap.add_argument("--gpus", default="1,8,64")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    ln = a.delta_log2 * math.log(2)
    report = {}
    for o in a.outs:
        rows = [r for r in sessions(Path(o)) if r["accepted"]]
        for arm in sorted({r["arm"] for r in rows}):
            f = fit([r for r in rows if r["arm"] == arm])
            per = {x: (f[x]["fixed"] + f[x]["per_opening"] * a.statement) / a.statement for x in TIMES}
            window = {x: per[x] * a.openings for x in TIMES}
            plan = {}
            for g in (int(x) for x in a.gpus.split(",")):
                k = {x: a.window_s * g / per[x] for x in ("device_s", "e2e_s")}
                plan[g] = {x: {"openings": int(k[x]), "eps": ln / k[x], "eps_pooled": ln / (k[x] * a.horizon)} for x in k}
            report[f"{o}:{arm}"] = {"fit": f, "per_opening_at_statement": per, "window_gpu_s": window,
                                    "gpus_for_window": {x: window[x] / a.window_s for x in TIMES}, "plan": plan}
    if a.json:
        print(json.dumps(report, indent=1))
        return 0
    for name, r in report.items():
        f = r["fit"]
        print(f"{name}: points {f['points']} ({f['sessions']} sessions)")
        for x in TIMES:
            print(f"  {x:9s} per opening {f[x]['per_opening']:8.4f} s, fixed {f[x]['fixed']:7.2f} s; "
                  f"an opening at {a.statement} a statement {r['per_opening_at_statement'][x]:.4f} s; "
                  f"{a.openings} a window {r['window_gpu_s'][x]:9.0f} s = {r['gpus_for_window'][x]:7.1f} devices for {a.window_s:.0f} s")
        for g, p in r["plan"].items():
            print(f"  {g} GPU(s) a window: " + "; ".join(f"{x}: {v['openings']} openings, eps {100 * v['eps']:.2f}% "
                                                         f"({100 * v['eps_pooled']:.3f}% pooled over {a.horizon})" for x, v in p.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
