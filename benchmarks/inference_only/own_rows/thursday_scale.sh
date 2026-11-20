#!/usr/bin/env bash
# Thursday's served case at several model sizes: its record case without and with declared nodes (`thursday_case.py`), the
# reference's verdicts, then `inference-only-record-check` on each case, timed. Declared nodes grow with the layers
# (about 6 a layer), so the layers set the declared nodes; the class sets the units.
#   bash benchmarks/inference_only/own_rows/thursday_scale.sh "LAYERS..." R,P,O [LEAN_LIMIT_S]
# Under `research run` the cases and the table go to $OUT.
set -uo pipefail
TREE=$PWD
OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}
LAYERS=${1:?LAYERS: the layer counts, e.g. "1 4 16"}
CLASS=${2:-2,9,2}
LIMIT=${3:-900}
mkdir -p "$OUT"
uv sync -q --all-packages --extra torch-cpu || exit 2
LEAN=$TREE/verity/protocols/workload_classification/training_detection/inference_only/verifier/lean
(cd "$LEAN" && lake build inference-only-record-check 2>&1 | tail -2)
EXE=$LEAN/.lake/build/bin/inference-only-record-check
for L in $LAYERS; do
  D=$OUT/L$L
  mkdir -p "$D"
  .venv/bin/python -u benchmarks/inference_only/own_rows/thursday_case.py "$TREE" "$D" "$L" "$CLASS" 2>&1 | cut -c1-900 | tee "$D/thursday.txt"
  for c in "$D"/*.case.json; do
    echo "=== lean L=$L $(basename "$c") ($(stat -c %s "$c") bytes)"
    /usr/bin/time -f "wall %e s, peak %M KB" timeout "$LIMIT" "$EXE" "$c" 2>&1 | cut -c1-600
  done | tee -a "$D/thursday.txt"
done
