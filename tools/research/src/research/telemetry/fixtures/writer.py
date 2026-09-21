"""Controlled buffered writer (brief §12): write --mib MiB to a file in --chunk-mib chunks with NO fsync, sleep --hold, then delete.
The sampler should see cgroup memory.stat file_dirty / file_writeback rise and fall, io.stat wbytes and /proc/<pid>/io write_bytes
increase; the classifier must report these as observations, not as a cause of anything.

    writer.py --path FILE [--mib 1024] [--chunk-mib 64] [--hold 8] [--fsync]
"""
import argparse
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--path", required=True)
ap.add_argument("--mib", type=int, default=1024)
ap.add_argument("--chunk-mib", type=int, default=64)
ap.add_argument("--hold", type=float, default=8.0)
ap.add_argument("--fsync", action="store_true")
a = ap.parse_args()
chunk = os.urandom(1 << 20) * a.chunk_mib
written = 0
with open(a.path, "wb") as f:
    while written < a.mib:
        f.write(chunk)
        written += a.chunk_mib
        print(f"writer {written} MiB (buffered, no fsync)", flush=True)
    if a.fsync:
        f.flush()
        os.fsync(f.fileno())
        print("fsync done", flush=True)
time.sleep(a.hold)
os.unlink(a.path)
print("writer done", flush=True)
sys.exit(0)
