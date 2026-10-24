"""10 s bins (p50, share over 150 and 300 µs) of an irqpin run's concurrent on/off flows, per phase.

    python bins_irqpin.py RUN_DIR --json bins.json"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from align import binned, series  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--json", required=True)
ns = ap.parse_args()
rep = {}
for ph in ("phase1", "phase2"):
    s = {role: series(f"{ns.run}/{ph}/{role}") for role in ("on", "off")}
    t0 = max(v[2]["start"] for v in s.values())
    t1 = min(v[2]["end"] for v in s.values())
    rep[ph] = {role: {"cpu": m["cpu"], "bins": binned(t, us, t0, t1, 10.0)} for role, (t, us, m) in s.items()}
    on = [b["p50"] for b in rep[ph]["on"]["bins"] if b]
    off = [b["p50"] for b in rep[ph]["off"]["bins"] if b]
    print(f"{ph}: on p50 per 10 s {min(on):.1f}-{max(on):.1f} µs; off {min(off):.1f}-{max(off):.1f} µs")
json.dump(rep, open(ns.json, "w"), indent=1)
