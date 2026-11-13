set -u
I=$RESEARCH_RUN_DIR/inputs
uv sync -q --all-packages --extra torch-cpu
bash $I/watch.sh values4 .venv/bin/python $I/mm_values.py $OUT 4 7 2>&1 | rg -v "^AllGather|^[A-Z][A-Za-z0-9]+_v[0-9]" | tail -60
