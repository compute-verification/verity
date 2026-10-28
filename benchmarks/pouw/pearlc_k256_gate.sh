#!/usr/bin/env bash
# TT_OUT's gate at k = 256 on a CPU node holding GLM-5.3's snapshot (SNAP), a `research run
# --cwd source` job writing under $OUT:
#   * the census with the replayed debit, every family (the standing ones and the cross-group cancellation ones), the
#     Hopper atom, s5 at line norm 16, δ = 1, tiles 64 x 64 and 64 x 16, runs 1-3, the cross-group search on every word
#     (census_*.json);
#   * pearlc_k256_gate.py on the served scheme: the exhaustive cross-group and peel searches, then again with real w2 slices
#     as B (gate*.json);
#   * glm53_slices.py: the per-row rules on every TP8 rank's 256-wide down_proj slice (slices.json);
# then `pearlc_k256_gate.py --verdict`, whose status is the run's.
set -uo pipefail
SNAP=${SNAP:-/workspace/hf/hub/models--zai-org--GLM-5.3/snapshots/aca966e4e02791568aa6a4ced368624b3d897f42}
uv sync -q --all-packages --extra torch-cpu || exit 2
PY="uv run --no-sync python"
D=benchmarks/pouw
FAMS=$($PY -c "import sys; sys.path.insert(0, '$D'); import pearlc_census as C
print(' '.join(C.FAMILIES + C.ACTIVATION_FAMILIES + C.OUTLIER_FAMILIES + C.FREEZE_FAMILIES + C.XCANCEL_FAMILIES))")
$PY $D/pearl_c_sm90/glm53_slices.py "$SNAP" "$OUT/slices.json" --samples "$OUT/glm_w2_samples.npz" --jobs 40 \
  > "$OUT/slices.log" 2>&1 &
slices=$!
$PY $D/pearlc_k256_gate.py --runs 1,2,3 --jobs 40 --out "$OUT/gate.json" > "$OUT/gate.log" 2>&1 &
gate=$!
for run in 1 2 3; do
  for tile in 64x64 64x16; do
    words=$(( ${tile%x*} * ${tile#*x} ))
    for fam in $FAMS; do
      echo "$run $tile $words $fam"
    done
  done
done | xargs -P 40 -L 1 bash -c '$0 benchmarks/pouw/pearlc_census.py --atom hopper --forming s5 --line-norm 16 --deltas 1 \
  --ks 256 --tile "$2" --families "$4" --run "$1" --debit --cross-words "$3" --transfer-words 64 \
  --out "$OUT/census_${2}_run${1}_$4.json" > "$OUT/census_${2}_run${1}_$4.log" 2>&1 || echo "census $* rc $?"' "uv run --no-sync python"
wait "$slices"; echo "slices rc $?"
$PY $D/pearlc_k256_gate.py --families "" --b-samples "$OUT/glm_w2_samples.npz" --runs 1,2,3 --jobs 24 \
  --out "$OUT/gate_glm.json" > "$OUT/gate_glm.log" 2>&1; echo "gate_glm rc $?"
wait "$gate"; echo "gate rc $?"
$PY $D/pearlc_k256_gate.py --verdict "$OUT"
