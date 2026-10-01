#!/usr/bin/env bash
# The cost of instances.json.gz's compression level: one kept instances file (FILE) decompressed once, then recompressed at each
# LEVEL in its own process, as build.py's writer compresses it (gzip, text written in chunks); then each output read back through
# the manifest's reader (`program_view.iter_instance_rows`). Measurement only.
set -euo pipefail
FILE=$1
shift
SRC=$(pwd)
RID=${RESEARCH_RUN_ID:?}
SCR=/workspace/research/build-speed/runs/$RID
mkdir -p "$SCR" "$RESEARCH_RUN_DIR/out"
trap 'rm -rf "$SCR"' EXIT
export PY=/workspace/jobs/venv312/bin/python PYTHONPATH=$SRC/integrations/vllm:$SRC/packages/verity/src
"$PY" - "$FILE" "$SCR" "$RESEARCH_RUN_DIR/out/gzip_levels.json" "$@" << 'EOF'
import gzip, hashlib, json, os, sys, time, zlib
from concurrent.futures import ProcessPoolExecutor

src, scr, out, levels = sys.argv[1], sys.argv[2], sys.argv[3], [int(x) for x in sys.argv[4:]]
CHUNK = 8 << 20


def text():
    with gzip.open(src, "rb") as f:
        while b := f.read(CHUNK):
            yield b


def decompress_only():
    t, n, h = time.perf_counter(), 0, hashlib.sha256()
    for b in text():
        n += len(b)
        h.update(b)
    return {"bytes": n, "sha256": h.hexdigest(), "secs": round(time.perf_counter() - t, 1)}


def recompress(level):
    path = os.path.join(scr, f"l{level}.json.gz")
    t, busy = time.perf_counter(), 0.0
    with gzip.open(path, "wb", compresslevel=level) as f:
        for b in text():
            t0 = time.perf_counter()
            f.write(b)
            busy += time.perf_counter() - t0
    wall = time.perf_counter() - t
    t, h, n = time.perf_counter(), hashlib.sha256(), 0
    with gzip.open(path, "rb") as f:
        while b := f.read(CHUNK):
            h.update(b)
            n += len(b)
    inflate = time.perf_counter() - t
    from verity_vllm.query.program_view import iter_instance_rows
    t, rows = time.perf_counter(), 0
    for _ in iter_instance_rows(path):
        rows += 1
    read = time.perf_counter() - t
    size = os.path.getsize(path)
    os.remove(path)
    return {"level": level, "compress_secs": round(busy, 1), "write_wall_secs": round(wall, 1), "gz_bytes": size, "text_sha256": h.hexdigest(),
            "text_bytes": n, "inflate_secs": round(inflate, 1), "rows": rows, "read_rows_secs": round(read, 1)}


with ProcessPoolExecutor(len(levels) + 1) as ex:
    base = ex.submit(decompress_only)
    res = list(ex.map(recompress, levels))
doc = {"file": src, "source_gz_bytes": os.path.getsize(src), "text": base.result(), "levels": res,
       "same_text": all(r["text_sha256"] == base.result()["sha256"] for r in res)}
json.dump(doc, open(out, "w"), indent=1)
print(json.dumps(doc, indent=1))
EOF
