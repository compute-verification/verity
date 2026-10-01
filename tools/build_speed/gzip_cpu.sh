#!/usr/bin/env bash
# CPU seconds of instances.json.gz at gzip levels 9 and 6, which wall time under a loaded node can't separate: one kept instances file
# (FILE) recompressed at each level in this process, then the two inflated in turn ROUNDS times (default 3), so load drift reaches
# both alike. Measurement only.
set -euo pipefail
FILE=$1 ROUNDS=${2:-3}
SRC=$(pwd)
RID=${RESEARCH_RUN_ID:?}
SCR=/workspace/research/build-speed/runs/$RID
mkdir -p "$SCR" "$RESEARCH_RUN_DIR/out"
trap 'rm -rf "$SCR"' EXIT
/workspace/jobs/venv312/bin/python - "$FILE" "$SCR" "$ROUNDS" << 'EOF' | tee "$RESEARCH_RUN_DIR/out/gzip_cpu.json"
import gzip, hashlib, json, os, sys, time

src, scr, rounds = sys.argv[1], sys.argv[2], int(sys.argv[3])
CHUNK = 8 << 20


def text(path):
    with gzip.open(path, "rb") as f:
        while b := f.read(CHUNK):
            yield b


def inflate(path):
    t, h = time.process_time(), hashlib.sha256()
    for b in text(path):
        h.update(b)
    return round(time.process_time() - t, 2), h.hexdigest()


out = {"file": src, "levels": {}}
paths = {}
for level in (9, 6):
    paths[level] = p = os.path.join(scr, f"l{level}.json.gz")
    busy = 0.0
    with gzip.open(p, "wb", compresslevel=level) as f:
        for b in text(src):
            t = time.process_time()
            f.write(b)
            busy += time.process_time() - t
    out["levels"][level] = {"compress_cpu_s": round(busy, 1), "gz_bytes": os.path.getsize(p), "inflate_cpu_s": []}
shas = set()
for _ in range(rounds):
    for level in (9, 6):
        s, h = inflate(paths[level])
        out["levels"][level]["inflate_cpu_s"].append(s)
        shas.add(h)
out["same_text"] = len(shas) == 1
print(json.dumps(out, indent=1))
EOF
