#!/usr/bin/env bash
set -uo pipefail
M=tools/move/draw_map_lean_moves.json
P=verity/protocols/resource_accounting/compute/pouw/verifier/lean
git update-ref refs/heads/main 354e54f88b61faffef840b5ff5f1c82c3d4ace0e || exit 2
git merge-base HEAD main || exit 2
rc=0
for pkg in security $P/check $P/commit $P/pearlplus $P/window; do
  echo "== records $pkg"
  python3 tools/verity/lean/lean_changed.py --records "$pkg" --update --moved "$M"
  r=$?
  echo "== records $pkg exit $r"
  [ "$r" -gt "$rc" ] && rc=$r
done
echo "== all exit $rc"
exit "$rc"
