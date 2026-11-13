"""Scratch for MEM-310 (never merged): elaborate each probe with `lake env lean` from this package and print its marks,
wall time and peak RSS. Run from the package directory: `python3 PrimeCertProbe/measure.py PrimeCertProbe/X.lean ...`."""
import resource
import subprocess
import sys
import time

for path in sys.argv[1:]:
    t = time.monotonic()
    r = subprocess.run(["lake", "env", "lean", path], capture_output=True, text=True)
    wall = time.monotonic() - t
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024
    print(f"== {path}: exit {r.returncode}, wall {wall:.1f} s, peak RSS so far {rss} MiB", flush=True)
    print(r.stdout[-4000:], r.stderr[-2000:], flush=True)
