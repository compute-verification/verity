# scratch (lean/WIRED): one class through circuits' record_case_files.py, the own-rows diagnostic, then the Python twin
set -eo pipefail
CLASS=$1; IN=$RESEARCH_RUN_DIR/inputs
{ [ -L .venv ] && [ ! -e .venv ] && rm .venv; } || true
uv sync -q --all-packages --extra torch-cpu
export LEANPKG=$PWD/verity/protocols/workload_classification/training_detection/inference_only/verifier/lean
S=$LEANPKG/scripts
( cd tests/adapters/vllm && uv run --extra torch-cpu python "$IN/record_case_files.py" "$CLASS" "$OUT/$CLASS" )
D=$OUT/$CLASS
echo "RECORD $CLASS: program sha256 $(sha256sum "$D/program.json" | cut -c1-16), circuit $(stat -c %s "$D/circuit.json") B, in-order variant $(cmp -s "$D/circuit.json" "$D/circuit_po.json" && echo same || echo differs)"
uv run --extra torch-cpu python "$IN/own_diag.py" "$S" "$D/program.json" 2>&1 | cut -c1-1500
for V in "" _po; do
  uv run --extra torch-cpu python "$S/served_record_case.py" --program "$D/program.json" --record "$D/circuit$V.json" \
    --log "$D/log0.json" "$D/log1$V.json" --limits "$D/limits.json" --out "$OUT" --name "$CLASS$V" --case-only 2>&1 | tail -3
  gzip -kf "$OUT/$CLASS$V.case.json" && rm "$OUT/$CLASS$V.case.json"
  echo "CASE $CLASS$V written"
done
gzip -f "$D/program.json" "$D/circuit.json" "$D/circuit_po.json"
S0=$(date +%s)
gunzip -c "$OUT/$CLASS.case.json.gz" > /tmp/wired-$CLASS.case.json
uv run --extra torch-cpu python - "$S" /tmp/wired-$CLASS.case.json <<'PY'
import json, sys; sys.path.insert(0, sys.argv[1])
import served_record_case as SR
got = SR.verdicts(json.loads(open(sys.argv[2]).read()))
print("PY_TWIN check:", json.dumps(got["check"], sort_keys=True)[:900]); print("PY_TWIN count:", json.dumps(got["count"], sort_keys=True)[:300])
PY
rm -f /tmp/wired-$CLASS.case.json
echo "PY_TWIN $CLASS in $(( $(date +%s) - S0 )) s"
