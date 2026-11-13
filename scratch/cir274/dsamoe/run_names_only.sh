uv sync -q --all-packages --extra torch-cpu </dev/null && git rev-parse HEAD
.venv/bin/python $RESEARCH_RUN_DIR/inputs/p3_names.py $OUT 4; echo names rc=$?
