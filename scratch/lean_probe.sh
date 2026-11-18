set -eo pipefail
# $1: the built package dir on this machine; then CLASS=PROGRAM.gz pairs
P=$1; shift
cd "$P"
for kv in "$@"; do
  C=${kv%%=*}; G=${kv#*=}
  gunzip -c "$G" > "$OUT/$C.program.json"
  python3 - "$OUT/$C.program.json" "$OUT/$C.case.json" <<'PY'
import json, sys
t = open(sys.argv[1]).read()
json.dump({"instance": {"program": t, "record": "{}", "gamma": 1, "alpha": [1, 2], "pins": []}, "trusted": {}, "count": {}}, open(sys.argv[2], "w"))
PY
  echo "== $C probe"; /usr/bin/time -f "probe: %e s, %M KB" .lake/build/bin/wired-probe "$OUT/$C.program.json" || echo "probe rc=$?"
  echo "== $C inference-only-record-check (stand-in record)"; /usr/bin/time -f "record-check: %e s, %M KB" .lake/build/bin/inference-only-record-check "$OUT/$C.case.json" 2>&1 | cut -c1-600 || echo "record-check rc=$?"
  rm -f "$OUT/$C.program.json" "$OUT/$C.case.json"
done
