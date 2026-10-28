#!/usr/bin/env bash
# The synthesis queries (li_window_sat.py), one process a (width, query), on a CPU node. From a pushed
# commit:
#
#   research run --on vy-nebius-cpu-N --project verity --campaign eps1 --source <tree> --cwd source --mem-gb 16 \
#     --max-min 50 -- bash benchmarks/pouw/rt2_hopper/sat.sh [LIMIT_S]
#
# Outputs under $OUT: one JSON and one log a query (low-closed forms at w = 4, 6, 8; with the high-to-low forms at
# w = 32), and summary.txt, every query's verdict lines. LIMIT_S (default 2400) caps one z3 check; past it a query reads
# "unknown", never "unsat". Exit: 0 every query ran (its verdicts are in summary.txt), 2 no OUT.
set -uo pipefail
OUT=${OUT:?run under research run}
LIMIT=${1:-2400}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONDONTWRITEBYTECODE=1
P=$(nproc)
jobs=()
for w in 4 6 8; do
  for q in half1 half2 half3 pair pair_xor half1_enc half2_enc; do jobs+=("$w $q low"); done
done
for q in half1 half2 pair half1_enc; do jobs+=("32 $q high"); done
# ONLY="W:QUERY:KIND ..." (e.g. "4:half2:low 32:half2:high") runs just those
if [ -n "${ONLY:-}" ]; then
  jobs=()
  for j in $ONLY; do IFS=: read -r w q k <<<"$j"; jobs+=("$w $q $k"); done
fi
printf '%s\n' "${jobs[@]}" | xargs -P "$P" -L 1 bash -c '
  w=$0 q=$1 kind=$2
  flags=(--width "$w"); [ "$kind" = high ] && flags=(--high)
  uv run -q --no-project --with z3-solver python "'"$HERE"'/li_window_sat.py" "${flags[@]}" --only "$q" \
    --limit-s "'"$LIMIT"'" --json "'"$OUT"'/$kind-w$w-$q.json" >"'"$OUT"'/$kind-w$w-$q.log" 2>&1
  echo "$kind w=$w $q rc=$?"' >"$OUT/rcs.txt"
grep -h -E '^(PASS|FAIL)' "$OUT"/*.log | sort >"$OUT/summary.txt"
nproc >"$OUT/nproc.txt"
cat "$OUT/summary.txt"
