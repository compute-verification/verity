# the description's fields before and after fix-chain-ranges.diff, in one clone
I=$RESEARCH_RUN_DIR/inputs
git rev-parse HEAD
uv sync -q --all-packages --extra torch-cpu </dev/null || exit 9
A="2:43 3:43 2:64 2:86 3:127"
.venv/bin/python $I/p3_parts.py $OUT before $A || exit 8
git apply $I/fix-chain-ranges.diff && git diff --stat
.venv/bin/python $I/p3_parts.py $OUT after $A; echo parts rc=$?
