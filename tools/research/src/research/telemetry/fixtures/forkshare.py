"""Fork/share fixture (brief §12): the parent maps SHARED_MIB of MAP_SHARED anonymous memory and touches it, then forks N children that
(a) touch the shared map (shared pages: RSS counts them in every process, PSS divides them) and (b) each allocate PRIVATE_MIB of private
anon (duplicated, counted once per child by both).  Holds for --hold seconds so the sampler sees the steady state.

    forkshare.py [--shared-mib 512] [--private-mib 128] [--children 3] [--hold 12]

Expected in the resources.jsonl `pss` records: rss_sum_bytes ~ (N+1)*SHARED + N*PRIVATE, pss_sum_bytes ~ SHARED + N*PRIVATE.
"""
import argparse
import mmap
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--shared-mib", type=int, default=512)
ap.add_argument("--private-mib", type=int, default=128)
ap.add_argument("--children", type=int, default=3)
ap.add_argument("--hold", type=float, default=12.0)
a = ap.parse_args()

shared = mmap.mmap(-1, a.shared_mib << 20, flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS)
for i in range(0, len(shared), 4096):
    shared[i] = 1
print(f"parent {os.getpid()}: shared {a.shared_mib} MiB touched", flush=True)
kids = []
for k in range(a.children):
    pid = os.fork()
    if pid == 0:
        s = 0
        for i in range(0, len(shared), 4096):    # read the shared pages -> mapped in this child too
            s += shared[i]
        priv = bytearray(a.private_mib << 20)
        for i in range(0, len(priv), 4096):
            priv[i] = 2
        print(f"child {os.getpid()}: shared read ({s} pages), private {a.private_mib} MiB", flush=True)
        time.sleep(a.hold)
        os._exit(0)
    kids.append(pid)
time.sleep(a.hold)
for pid in kids:
    os.waitpid(pid, 0)
print("forkshare done", flush=True)
sys.exit(0)
