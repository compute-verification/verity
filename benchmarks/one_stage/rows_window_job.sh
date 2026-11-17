set -u
# CACHE: the compose cache warm filled; WARM_LOG: warm's output, optional.
ROOT=$PWD
IO=verity/protocols/workload_classification/training_detection/inference_only/verifier/lean
git log -1 --format='head %h'
uv sync --all-packages --extra torch-cpu -q || exit 3
(cd $IO && lake build inference-only-rows 2>&1 | grep -E "error|Build completed" | head -30) || exit 4
EXE=$ROOT/$IO/.lake/build/bin/inference-only-rows
mkdir -p $OUT/cases
uv run --extra torch-cpu python benchmarks/one_stage/rows_case_probe.py $OUT/cases --ran > $OUT/probe.log 2>&1 || { tail -20 $OUT/probe.log; exit 5; }
for name in pop pop-unregistered; do
  uv run --extra torch-cpu python benchmarks/one_stage/rows_from_case.py $OUT/cases/$name.case.json $CACHE $OUT/$name \
    ${WARM_LOG:+--warm-log $WARM_LOG} ${GAMMA:+--gamma $GAMMA} || exit 6
done
rm -f $OUT/cases/*.case.json
for name in pop pop-unregistered; do
  echo "== $name"
  /usr/bin/time -f '%e s %M KB' $EXE check $OUT/$name/params.json $OUT/$name/commitment.json $OUT/$name/evidence.json \
    $(cat $OUT/$name/blobs.txt)
  echo "rc $?"
done
