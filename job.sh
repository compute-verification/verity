#!/usr/bin/env bash
set -uo pipefail
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT; export TMPDIR="$d"
python3 tools/verity/lean/lean_changed.py --records verity/core/service/flock --update --out "$OUT/flock"; rf=$?
python3 tools/verity/lean/lean_changed.py --records Security --update --out "$OUT/security"; rs=$?
uv run tools/verity/check/suites.py flock --fresh > "$OUT/flock-suite.log" 2>&1; rt=$?
grep -rhE --include=build.log "HPortTie\.lean:[0-9]+:[0-9]+: .*(axioms|depend)" "$OUT" | sort -u > "$OUT/probes.txt" || true
grep -rhE --include=build.log -A12 "^(error|warning): .*\.lean:[0-9]+:[0-9]+:" "$OUT" > "$OUT/errors.txt" || true
echo "flock=$rf security=$rs suite=$rt" > "$OUT/rcs.txt"
exit $(( rf || rs || rt ))
