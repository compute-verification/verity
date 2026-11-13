set -u
I=$RESEARCH_RUN_DIR/inputs
uv sync -q --all-packages --extra torch-cpu
bash $I/watch.sh census4 .venv/bin/python $I/mm_census.py $OUT 4 2>&1 | tail -30
