# the unpatched tree, for comparison
I=$RESEARCH_RUN_DIR/inputs
git rev-parse HEAD; git status --short | head -3
uv sync -q --all-packages --extra torch-cpu </dev/null || exit 9
.venv/bin/python $I/p3_fixcheck.py $OUT 2 43 64 65 86 127 128 129; echo fix2 rc=$?
.venv/bin/python $I/p3_fixcheck.py $OUT 3 127 128; echo fix3 rc=$?
