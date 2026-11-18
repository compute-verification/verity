# scratch (lean/WIRED): build the record check in a private tree, then run it on each case sent
set -eo pipefail
bash "$RESEARCH_RUN_DIR/inputs/lean_build2.sh"
P=$RESEARCH_RUN_DIR/tree/verity/protocols/workload_classification/training_detection/inference_only/verifier/lean
cd "$P"
for G in "$RESEARCH_RUN_DIR"/inputs/*.case.json.gz; do
  C=$(basename "$G" .case.json.gz); gunzip -c "$G" > "$OUT/$C.case.json"
  echo "== $C"
  /usr/bin/time -f "record-check: %e s, %M KB" timeout 30m .lake/build/bin/inference-only-record-check "$OUT/$C.case.json" > "$OUT/$C.rc.txt" 2>&1 || echo "record-check rc=$?"
  cut -c1-1200 "$OUT/$C.rc.txt"; rm -f "$OUT/$C.case.json"
done
