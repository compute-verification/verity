"""Swapped in as counted.py (the analyzer moved to counted_real.py) in run r20261024-194056-b967's scratch at 21:10Z: run B's
last step (no ft3e-4 snapshot: that slot ran watch.py) writes $OUT/F_beta.json over every B1 run on this node, once the
sweep and watcher v2 have ended or at 22:42Z, whichever is first."""
import datetime, json, os, subprocess, sys, time
here = os.path.dirname(os.path.abspath(__file__))
a = sys.argv[1:]
if len(a) == 3 and not os.path.exists(a[1]) and a[2].endswith("ft3e-4_s*"):
    root = "/workspace/research/runs"
    runs = ["r20261024-194029-24eb", "r20261024-194056-b967", "r20261024-195533-e468", "r20261024-201854-2f7d"]
    until = datetime.datetime(2026, 10, 24, 22, 42, tzinfo=datetime.timezone.utc)

    def live(r):
        try:
            return json.load(open(f"{root}/{r}/status.json")).get("state") == "running"
        except (OSError, json.JSONDecodeError):
            return False
    while datetime.datetime.now(datetime.timezone.utc) < until and any(live(r) for r in (runs[0], runs[2])):
        time.sleep(30)
    dirs = [d for d in (f"{root}/{r}/out" for r in runs) if os.path.isdir(d)]
    print("summarize over", dirs, flush=True)
    sys.exit(subprocess.call([sys.executable, os.path.join(here, "summarize.py"), os.environ["OUT"], *dirs]))
os.execv(sys.executable, [sys.executable, os.path.join(here, "counted_real.py"), *a])
