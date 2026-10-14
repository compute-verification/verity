#!/usr/bin/env bash
set -uo pipefail
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT; export TMPDIR="$d"
python3 tools/verity/lean/lean_changed.py --records verity/core/service/flock --update --out "$OUT/flock"; rf=$?
uv run --locked --extra torch-cpu python tools/verity/check/suites.py flock --fresh > "$OUT/flock-suite.log" 2>&1; rt=$?
grep -rhE --include=build.log -A12 "^error: .*\.lean:[0-9]+:[0-9]+:" "$OUT" > "$OUT/errors.txt" || true
echo "flock=$rf suite=$rt" > "$OUT/rcs.txt"
exit $(( rf || rt ))
