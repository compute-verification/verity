"""Memory hog fixture: allocate anonymous memory in steps until killed (or until --stop-mib).  Sets oom_score_adj so that in a SHARED pod
cgroup the kernel's victim is this process (the fixture must never take sshd or the sampler with it).

    hog.py [--step-mib 128] [--sleep 0.2] [--stop-mib N] [--hold S] [--oom-score-adj 1000]
"""
import argparse
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--step-mib", type=int, default=128)
ap.add_argument("--sleep", type=float, default=0.2)
ap.add_argument("--stop-mib", type=int, default=0, help="stop (exit 0) after this many MiB instead of dying")
ap.add_argument("--oom-score-adj", type=int, default=1000)
ap.add_argument("--hold", type=float, default=0.0, help="after --stop-mib, keep the memory resident this long before exiting (a surviving bystander)")
a = ap.parse_args()
try:
    with open("/proc/self/oom_score_adj", "w") as f:
        f.write(str(a.oom_score_adj))
except OSError as e:
    print(f"oom_score_adj not set: {e}", file=sys.stderr)
blocks = []
total = 0
while True:
    b = bytearray(a.step_mib << 20)
    for i in range(0, len(b), 4096):   # touch every page so it is resident (anon)
        b[i] = 1
    blocks.append(b)
    total += a.step_mib
    print(f"hog {total} MiB", flush=True)
    if a.stop_mib and total >= a.stop_mib:
        if a.hold:
            print(f"hog holding {total} MiB for {a.hold}s", flush=True)
            time.sleep(a.hold)
        print("hog done", flush=True)
        sys.exit(0)
    time.sleep(a.sleep)
