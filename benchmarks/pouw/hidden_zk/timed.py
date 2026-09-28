"""timed.py OUT_FILE CMD...: run CMD, write its wall seconds and peak RSS (KB) as JSON to OUT_FILE, exit with its code. A SIGTERM
is passed on to CMD, so a server run under it stops when it is killed."""
import json
import resource
import signal
import subprocess
import sys
import time

out, cmd = sys.argv[1], sys.argv[2:]
t0 = time.monotonic()
child = subprocess.Popen(cmd)
signal.signal(signal.SIGTERM, lambda *_: child.terminate())
rc = child.wait()
wall = time.monotonic() - t0
rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
with open(out, "w") as f:
    json.dump({"cmd": cmd[-1][:300], "exit": rc, "wall_s": round(wall, 2), "maxrss_kb": rss}, f)
sys.exit(rc if rc >= 0 else 128 - rc)
