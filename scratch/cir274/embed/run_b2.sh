set -u
I=$RESEARCH_RUN_DIR/inputs
git rev-parse HEAD | tee $OUT/head.txt
sha256sum $I/multimem-join.diff | tee $OUT/diff_sha.txt
git apply $I/multimem-join.diff && git diff --stat | tee $OUT/applied.txt
uv sync -q --all-packages --extra torch-cpu
PY=$PWD/.venv/bin/python
cd adapters/vllm
bash $I/watch.sh values $PY -u $I/values_run_join.py $OUT --ids 43 --layers 4 --name join_4l_43 > $OUT/stdout_values.txt 2>&1
rc=$?
rg -v "^\[.*\] (AllGather|[A-Z][A-Za-z0-9]+_v[0-9])" $OUT/stdout_values.txt | cut -c1-2500 | tail -30
exit $rc
