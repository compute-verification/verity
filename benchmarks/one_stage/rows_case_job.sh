set -u
ROOT=$PWD
IO=verity/protocols/workload_classification/training_detection/inference_only/verifier/lean
git log -1 --format='head %h'
uv sync --all-packages --extra torch-cpu -q || exit 3
(cd $IO && lake build inference-only-record-check 2>&1 | tail -2) || exit 4
EXE=$ROOT/$IO/.lake/build/bin/inference-only-record-check
mkdir -p $OUT/cases
/usr/bin/time -f 'build %e s %M KB' uv run --extra torch-cpu python benchmarks/one_stage/rows_case_probe.py $OUT/cases ${RAN:-} 2>&1 | cut -c1-2000 || exit 5
cat $OUT/cases/templates.txt
for f in $OUT/cases/*.case.json; do
  echo "== $(basename $f) $(stat -c %s $f) bytes"
  /usr/bin/time -f '%e s %M KB' $EXE $f 2>&1 | cut -c1-3000
done
