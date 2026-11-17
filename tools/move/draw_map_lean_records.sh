#!/usr/bin/env bash
set -uo pipefail
P=verity/protocols/resource_accounting/compute/pouw/verifier/lean
M=$(mktemp --suffix=.json)
python3 -c 'import json, sys; a = json.load(open(sys.argv[1])); a.update(json.load(open(sys.argv[2]))); json.dump(a, open(sys.argv[3], "w"))' \
  tools/move/draw_map_lean_moves.json tools/move/draw_map_lean_pass_moves.json "$M" || exit 2
git update-ref refs/heads/main 70da3f978c8bf9d8dd26bf7e9424cf013efe4147 || exit 2
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
