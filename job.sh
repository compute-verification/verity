#!/usr/bin/env bash
set -uo pipefail
export PATH="$HOME/.elan/bin:$PATH"
git rev-parse HEAD > "$OUT/head.txt" 2>/dev/null || true
uv run --locked python tools/verity/lean/lean_changed.py --records Security --update --out "$OUT/security" \
  > "$OUT/records-security.log" 2>&1
rc=$?
echo "$rc" > "$OUT/rc-records-security"
grep -nE "axioms|error" "$OUT/records-security.log" > "$OUT/probes.txt" || true
echo "records-security rc $rc"
exit "$rc"
