set -u
I=$RESEARCH_RUN_DIR/inputs
uv sync -q --all-packages --extra torch-cpu
PY=$PWD/.venv/bin/python
cd adapters/vllm
bash $I/watch.sh values4c $PY -u $I/values_run_mm.py $OUT --ids 43 --layers 4 --name mm_4l_43 > $OUT/stdout_values.txt 2>&1
rc=$?
cat $OUT/watch_values4c.txt
rg -v "^\[.*\] (AllGather|[A-Z][A-Za-z0-9]+_v[0-9])" $OUT/stdout_values.txt | cut -c1-1500 | tail -40
exit $rc
