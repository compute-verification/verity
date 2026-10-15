#!/usr/bin/env bash
# Least tree-plus-branch charge at each saving, for flip-graph schemes of small <p,q,r> over F2 (Question 6.3).
# One core; about 30 minutes. Each line: shape, seed, gamma, {saving: (accepted states, least charge)}, best state.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
for shape in "2 2 3" "2 2 4" "2 3 3" "3 2 3" "3 3 2" "3 3 3" "3 2 4" "4 2 3" "4 2 4"; do
  for gamma in 2 2.5; do
    for seed in 1 2 3 4; do
      timeout 900 "$PY" crossing.py $shape 150000 "$seed" - "$gamma" | cut -c1-2000
    done
  done
done
