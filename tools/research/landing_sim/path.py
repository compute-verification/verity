"""The levers in shipping order, each added to the ones before, under the calibrated current model, then the tree queue:
each step's ready-to-main p50/p90 and PR-hours a week, and what it removed, at two break rates (`cumulative.json`)."""
import importlib.util
import json
import os

DATA = os.environ.get("LANDING_STUDY_DIR", "/tmp/ld")
spec = importlib.util.spec_from_file_location("sim", os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim.py"))
sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)

STEPS = [("current (calibrated)", {}),
         ("1 + revert at once on red", {"revert": True}),
         ("2 + no prefix block (set landing)", {"pin": False, "set_landing": True}),
         ("3 + review <=2h (dispatch, verdict carry)", {"review_cap": 120}),
         ("4 + admission off the tick, 10-min pass", {"admit": None, "pass0": 10, "pass1": 0.0}),
         ("5 + executor after-landing run 60 min", {"run_min": 60}),
         ("6 + flakes deleted", {"flake": 0.0}),
         ("7 + Lean chain 45 min", {"lean_min": 45})]

out = []
for pb in (0.02, 0.05):
    p, prev = dict(sim.CURRENT), None
    print(f"== p_break {pb}")
    for name, o in STEPS:
        p.update(o)
        prs, moves, _ = sim.run("current", dict(p), pb)
        r = sim.report(name, prs, moves)
        r["delta"], r["p_break"], prev = None if prev is None else prev - r["pr_hours_per_week"], pb, r["pr_hours_per_week"]
        out.append(r)
    prs, moves, _ = sim.run("tree", dict(sim.TREE, review_cap=120, own_lean_min=45, int_lean_min=30), pb)
    r = sim.report("8 tree queue (pre-landing gate, no checked)", prs, moves)
    r["delta"], r["p_break"] = prev - r["pr_hours_per_week"], pb
    out.append(r)
json.dump(out, open(f"{DATA}/cumulative.json", "w"), indent=1)
