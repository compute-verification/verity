#!/usr/bin/env bash
set -uo pipefail
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT; export TMPDIR="$d"
python3 tools/verity/lean/lean_changed.py --records Security --update --out "$OUT/security"; rc=$?
grep -rh -A3 -E "valueAt_of_some|valueColl_of_tie|value_tie_coll|ind_valueColl_le|prob_valueColl_le|prob_valueColl_fork|prob_regTopValue_le" "$OUT" \
  | grep -E "depends on axioms|does not depend" > "$OUT/probes.txt" || true
exit $rc
