#!/usr/bin/env bash
set -uo pipefail

export PATH="$HOME/.elan/bin:$PATH"
rc=0

python3 tools/verity/lean/lean_changed.py --records Security --update --out "$OUT/security" \
  > "$OUT/security.log" 2>&1 || rc=1
grep -rh "depends on axioms" "$OUT" --exclude=probes.txt > "$OUT/probes.txt" || true

exit "$rc"
