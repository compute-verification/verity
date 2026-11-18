set -eo pipefail
P=$1; shift
cd "$P"
for G in "$@"; do
  C=$(basename "$G" .case.json.gz)
  gunzip -c "$G" > "$OUT/$C.case.json"
  echo "== $C inference-only-record-check (stand-in record and log)"
  /usr/bin/time -f "record-check: %e s, %M KB" .lake/build/bin/inference-only-record-check "$OUT/$C.case.json" 2>&1 | cut -c1-900 || echo "record-check rc=$?"
  rm -f "$OUT/$C.case.json"
done
