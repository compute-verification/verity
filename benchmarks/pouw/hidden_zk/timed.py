"""timed.py OUT_FILE CMD...: run CMD, write its wall seconds, peak RSS (KB) and CPU seconds (user + system of CMD and every
descendant it waited for) as JSON to OUT_FILE, exit with its code. A SIGTERM is passed on to CMD, so a server run under it
stops when it is killed."""
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
ru = resource.getrusage(resource.RUSAGE_CHILDREN)
with open(out, "w") as f:
    json.dump({"cmd": cmd[-1][:300], "exit": rc, "wall_s": round(wall, 2), "maxrss_kb": ru.ru_maxrss,
               "cpu_s": round(ru.ru_utime + ru.ru_stime, 2)}, f)
sys.exit(rc if rc >= 0 else 128 - rc)
