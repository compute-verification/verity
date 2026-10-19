#!/usr/bin/env bash
# prefill_breakdown.py over served nsys traces already on this node (CPU only, no lease): each REP exported to sqlite in scratch,
# its prefill steps' GPU time by kernel class into $OUT/prefill-<n>.json (the source path in $OUT/sources.txt).
#
#   research run --on vy-dc-h200 --project verity --campaign hash --source . --cwd source -- \
#     bash benchmarks/hash/overlap/prefill_nsys.sh /workspace/research/runs/<run>/out/served.nsys-rep ...
set -uo pipefail
O=${OUT:?}
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
i=0
for rep in "$@"; do
  i=$((i + 1))
  echo "$i $rep" >> "$O/sources.txt"
  nsys export --type sqlite -f true -o "$W/t$i.sqlite" "$rep" > "$O/export-$i.log" 2>&1 || { echo "export $i failed"; continue; }
  python3 benchmarks/hash/overlap/prefill_breakdown.py "$W/t$i.sqlite" --run "${RUN_LABEL:-run:stock-prof}" > "$O/prefill-$i.json"
  echo "breakdown $i rc $?"
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps({k: d[k] for k in ("prefill_steps", "tokens", "mean")}))' "$O/prefill-$i.json"
  rm -f "$W/t$i.sqlite"
done
