"""REDS levers through DESIGN's landing simulator (sim_reds.py: sim.py plus a `precheck` lever of the current model).

L1 precheck: the changed suites on a landing batch's merged tip, `pre_min` of wall time added to the pass that lands it;
a breaker is caught with probability `catch` (15 of 24 incidents in REDS's table are catchable that way; ci's lever 3,
repository invariants only, catches 4 of 24) and leaves next, back fixed `fix_min` later. L2: DESIGN's `revert`.
"""
import json
import sys

import sim_reds as s

VARIANTS = [
    ("current (calibrated)", {}),
    ("L2 revert on red", {"revert": True}),
    ("ci lever 3: invariants only (catch 4/24, 8 min)", {"precheck": True, "catch": 4 / 24, "pre_min": 8}),
    ("L1 changed suites (catch 15/24, 30 min)", {"precheck": True, "catch": 15 / 24, "pre_min": 30}),
    ("L1, 40 min", {"precheck": True, "catch": 15 / 24, "pre_min": 40}),
    ("L1, catch 0.4 (pessimistic)", {"precheck": True, "catch": 0.4, "pre_min": 30}),
    ("L1 + L2", {"precheck": True, "catch": 15 / 24, "pre_min": 30, "revert": True}),
    ("L1 async (30 min from joining next, Lean 92)", {"precheck": True, "catch": 15 / 24, "pre_min": 0,
                                                     "async_pre": 30, "lean_min": 92}),
    ("L1 async + L2", {"precheck": True, "catch": 15 / 24, "pre_min": 0, "async_pre": 30, "lean_min": 92,
                       "revert": True}),
    ("L1 Lean part only (lean-suites in the Lean check)", {"precheck": True, "catch": 15 / 24, "pre_min": 0,
                                                           "async_pre": 0, "lean_min": 92, "lean_only": True}),
    ("DESIGN no pin + admission + pass", {"pin": False, "set_landing": True, "admit": None, "pass0": 10, "pass1": 0.0}),
    ("DESIGN no pin + admission + pass + L2", {"pin": False, "set_landing": True, "admit": None, "pass0": 10,
                                               "pass1": 0.0, "revert": True}),
    ("DESIGN no pin + admission + pass + L1 + L2", {"pin": False, "set_landing": True, "admit": None, "pass0": 10,
                                                    "pass1": 0.0, "revert": True, "precheck": True, "catch": 15 / 24,
                                                    "pre_min": 30}),
    ("DESIGN no pin + admission + pass + L1 async + L2", {"pin": False, "set_landing": True, "admit": None,
                                                          "pass0": 10, "pass1": 0.0, "revert": True, "precheck": True,
                                                          "catch": 15 / 24, "pre_min": 0, "async_pre": 30,
                                                          "lean_min": 92}),
]

out = {}
for pb in (0.04, 0.047):
    print(f"p_break={pb}")
    rows = []
    for name, over in VARIANTS:
        params = dict(s.CURRENT)
        params.update(over)
        prs, moves, cores = s.run("current", params, pb)
        row = s.report(name, prs, moves)
        row["params"] = params
        rows.append(row)
    out[str(pb)] = rows
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/reds/reds_sim.json", "w"), indent=1)
