# scratch (lean/WIRED): one class's case and twin (round4.sh), then the timed probe and the Lean record check (lean_time.sh)
set -eo pipefail
CLASS=$1; IN=$RESEARCH_RUN_DIR/inputs
bash "$IN/round4.sh" "$CLASS"
echo "== LEAN $CLASS"
bash "$IN/lean_time.sh" "$OUT/$CLASS.case.json.gz"
