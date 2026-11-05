"""REDS levers over fix_min {240 (REDS measured median), 360 (DESIGN calibration)} and three break draws (load's salt)."""
import json
import statistics
import sys

import sim_reds as s

V = {
    "current": {},
    "L2 revert": {"revert": True},
    "L1 serial 30": {"precheck": True, "catch": 15 / 24, "pre_min": 30},
    "L1 async 30": {"precheck": True, "catch": 15 / 24, "pre_min": 0, "async_pre": 30, "lean_min": 92},
    "L1 Lean only": {"precheck": True, "catch": 15 / 24, "pre_min": 0, "lean_min": 92, "lean_only": True},
    "L1 async + L2": {"precheck": True, "catch": 15 / 24, "pre_min": 0, "async_pre": 30, "lean_min": 92, "revert": True},
    "lever3 serial 8": {"precheck": True, "catch": 4 / 24, "pre_min": 8},
    "D throughput": {"pin": False, "set_landing": True, "admit": None, "pass0": 10, "pass1": 0.0},
    "D throughput + L2": {"pin": False, "set_landing": True, "admit": None, "pass0": 10, "pass1": 0.0, "revert": True},
    "D throughput + L1 async + L2": {"pin": False, "set_landing": True, "admit": None, "pass0": 10, "pass1": 0.0,
                                     "revert": True, "precheck": True, "catch": 15 / 24, "pre_min": 0, "async_pre": 30,
                                     "lean_min": 92},
}
orig = s.load
out = []
for salt in ("b", "c", "d"):
    s.load = lambda pb, seed_salt=salt: orig(pb, seed_salt)
    for fix in (240, 360):
        for name, over in V.items():
            p = dict(s.CURRENT)
            p.update(over)
            p["fix_min"] = fix
            prs, moves, _ = s.run("current", p, 0.04)
            r = s.report(f"{salt} fix{fix} {name}", prs, moves)
            r.update(salt=salt, fix_min=fix, lever=name)
            out.append(r)
json.dump(out, open(sys.argv[1], "w"), indent=1)
print()
for fix in (240, 360):
    base = {r["salt"]: r["mean_h"] for r in out if r["fix_min"] == fix and r["lever"] == "current"}
    for name in V:
        rows = [r for r in out if r["fix_min"] == fix and r["lever"] == name]
        m = [r["mean_h"] for r in rows]
        d = [100 * (r["mean_h"] / base[r["salt"]] - 1) for r in rows]
        print(f"fix{fix} {name:32s} mean {min(m):5.1f}-{max(m):5.1f} h  vs current {min(d):+4.0f}..{max(d):+4.0f}%  "
              f"p90 {statistics.mean(r['p90_h'] for r in rows):5.1f}  to-main/day {statistics.mean(r['to_main_per_day'] for r in rows):4.0f}")
