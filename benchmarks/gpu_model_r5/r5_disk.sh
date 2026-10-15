#!/usr/bin/env bash
# R5 (CMP-12 red team of M5): can vy-nebius-2's virtio disk answer a PoUS challenge inside its ~1.2 ms deadline?
# 4 KiB O_DIRECT random reads of a 1 GiB file on the run's disk, at queue depth 1 and 16, 3 s each. No GPU, so no
# lease. A 1 GiB span favours the disk (backend caching), so its latencies are lower bounds for a 5 TB store.
#   research run --on vy-nebius-2 --project verity --campaign pouw --source <tree> --cwd clone \
#     --send r5_disk.sh -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/r5_disk.sh"'
set -uo pipefail
OUT=${OUT:?run under research run}
D=$(mktemp -d -p "${RESEARCH_RUN_DIR:-/workspace}")
trap 'rm -rf "$D"' EXIT
df -h "$D" >"$OUT/df.txt" 2>&1
lsblk -o NAME,SIZE,ROTA,TYPE,MOUNTPOINT >"$OUT/lsblk.txt" 2>&1
dd if=/dev/urandom of="$D/blob" bs=16M count=64 oflag=direct status=none
cat >"$D/rr.py" <<'EOF'
import json, mmap, os, random, sys, threading, time
path = sys.argv[1]
size = os.path.getsize(path)
blocks = size // 4096
def worker(n, lat, seed):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    buf = mmap.mmap(-1, 4096)
    rng = random.Random(seed)
    end = time.perf_counter_ns() + n
    while time.perf_counter_ns() < end:
        off = rng.randrange(blocks) * 4096
        t0 = time.perf_counter_ns()
        os.preadv(fd, [buf], off)
        lat.append(time.perf_counter_ns() - t0)
    os.close(fd)
for qd in (1, 16):
    lats = [[] for _ in range(qd)]
    ts = [threading.Thread(target=worker, args=(3_000_000_000, lats[i], 1000 + i)) for i in range(qd)]
    t0 = time.perf_counter_ns()
    for t in ts: t.start()
    for t in ts: t.join()
    wall = time.perf_counter_ns() - t0
    allv = sorted(v for l in lats for v in l)
    q = lambda p: allv[min(len(allv) - 1, int(p * len(allv)))] / 1000
    print(json.dumps({"probe": "disk-4k-random", "qd": qd, "reads": len(allv), "MB_s": round(len(allv) * 4096 / wall * 1e3, 1),
                      "lat_us_p50": q(0.5), "lat_us_p90": q(0.9), "lat_us_p99": q(0.99), "lat_us_max": allv[-1] / 1000}))
EOF
python3 "$D/rr.py" "$D/blob" >"$OUT/disk.jsonl" 2>"$OUT/disk.err"
cat "$OUT/df.txt" "$OUT/disk.jsonl" "$OUT/disk.err"
