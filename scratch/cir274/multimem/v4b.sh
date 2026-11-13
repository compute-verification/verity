set -u
I=$RESEARCH_RUN_DIR/inputs
uv sync -q --all-packages --extra torch-cpu
bash $I/watch.sh values4b .venv/bin/python -u $I/mm_values2.py $OUT 4 7 > $OUT/stdout_values.txt 2>&1
rc=$?
rg "^MM|rc=|Error|Traceback" $OUT/stdout_values.txt | cut -c1-2000
exit $rc
