#!/usr/bin/env bash
set -euo pipefail
d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
export TMPDIR="$d"
status=0
python3 tools/verity/lean/lean_changed.py --records Security --update --out "$OUT/security" > "$OUT/build.log" 2>&1 || status=$?
grep "depends on axioms" "$OUT/build.log" > "$OUT/probes.txt" || true
tail -n 200 "$OUT/build.log"
exit "$status"
