#!/usr/bin/env bash
set -uo pipefail
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT; export TMPDIR="$d"
python3 tools/verity/lean/lean_changed.py --records Security --update --out "$OUT/security"; rc=$?
grep -rhE --include=build.log "ValueInner\.lean:[0-9]+:[0-9]+: .*(axioms|depend)" "$OUT" | sort -u > "$OUT/probes.txt" || true
grep -rhE --include=build.log -A12 "^(error|warning): .*(ValueSource|ValueInner)\.lean:[0-9]+:[0-9]+:" "$OUT" > "$OUT/errors.txt" || true
exit $rc
