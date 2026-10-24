"""Swapped in as counted.py (the analyzer moved to counted_real.py) in run r20261024-194056-b967's scratch at 21:10Z: run B's
last step (no ft3e-4 snapshot: that slot ran watch.py) writes $OUT/F_beta.json over every B1 run on this node."""
import os, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
a = sys.argv[1:]
if len(a) == 3 and not os.path.exists(a[1]) and a[2].endswith("ft3e-4_s*"):
    runs = ["r20261024-194029-24eb", "r20261024-194056-b967", "r20261024-195533-e468", "r20261024-201854-2f7d"]
    dirs = [d for d in (f"/workspace/research/runs/{r}/out" for r in runs) if os.path.isdir(d)]
    print("summarize over", dirs, flush=True)
    sys.exit(subprocess.call([sys.executable, os.path.join(here, "summarize.py"), os.environ["OUT"], *dirs]))
os.execv(sys.executable, [sys.executable, os.path.join(here, "counted_real.py"), *a])
