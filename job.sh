#!/usr/bin/env bash
set -euo pipefail
d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
export TMPDIR="$d"
set +e
python3 tools/verity/lean/lean_changed.py --records Security --update --out "$OUT/security" 2>&1 | tee "$d/build.log"
rc=${PIPESTATUS[0]}
set -e
grep "depends on axioms" "$d/build.log" > "$OUT/probes.txt" || true
cp "$d/build.log" "$OUT/build.log"
exit "$rc"
