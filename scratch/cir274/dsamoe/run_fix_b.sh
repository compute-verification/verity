# the patched tree: the value check at 4 layers
I=$RESEARCH_RUN_DIR/inputs
git rev-parse HEAD && git apply $I/fix-chain-ranges.diff && git diff --stat
uv sync -q --all-packages --extra torch-cpu </dev/null || exit 9
for lp in 150 43; do /usr/bin/time -v .venv/bin/python $I/p3_values.py $OUT 4 $lp 2> $OUT/time_values_$lp.txt | tail -4; echo values $lp rc=$?; rg "Maximum resident|Elapsed" $OUT/time_values_$lp.txt; tail -5 $OUT/time_values_$lp.txt | rg -v "^\s" ; done
