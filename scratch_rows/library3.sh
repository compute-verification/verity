#!/usr/bin/env bash
# Scratch, never lands: the library's rows for the held entries and their honest config, in three units
# (TS_ROWS 0: gates as main prints them; 1: AND gates of the Definitions; 2: rows m of each Definition lowered alone),
# one process per entry and unit, $1 at a time. Each writes $OUT/UNIT/ENTRY/library.json.
set -euo pipefail
cd benchmarks/training_slowdown
entries=$(python3 - <<'EOF'
import tomllib
held = tomllib.load(open("training_slowdown/of_record.toml", "rb"))["held"]
print("\n".join(sorted(set(held) | {"s-t20-1.1x-ref-optim"})))
EOF
)
for u in 0 1 2; do for e in $entries; do echo "$u $e"; done; done |
  xargs -P "${1:-8}" -L 1 bash -c 'mkdir -p "$OUT/$0/$1" && OMP_NUM_THREADS=1 TS_ROWS=$0 PYTHONPATH=. \
    python -m training_slowdown.library "$1" --out "$OUT/$0/$1" > "$OUT/$0/$1/stdout.txt" 2> "$OUT/$0/$1/stderr.txt" \
    || echo "failed: $0 $1" >> "$OUT/failed.txt"'
echo done
