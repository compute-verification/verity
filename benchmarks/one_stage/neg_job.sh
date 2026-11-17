set -u
ROOT=$PWD
IO=verity/protocols/workload_classification/training_detection/inference_only/verifier/lean
git log -1 --format='head %h'
uv sync --all-packages --extra torch-cpu -q || exit 3
(cd $IO && lake build inference-only-record-check 2>&1 | grep -E "error|Build completed" | head -30) || exit 4
mkdir -p $OUT/cases
uv run --extra torch-cpu python benchmarks/one_stage/rows_case_probe.py $OUT/cases --ran > $OUT/probe.log 2>&1 || { tail -20 $OUT/probe.log; exit 5; }
grep -E "cases written" $OUT/probe.log
rm -f $OUT/cases/pop.case.json
/usr/bin/time -f 'record-check %e s %M KB' $ROOT/$IO/.lake/build/bin/inference-only-record-check $OUT/cases/pop-unregistered.case.json
rm -f $OUT/cases/*.case.json
