set -u
S=$(mktemp -d -p /workspace/research/scratch)
trap 'rm -rf "$S"' EXIT
export TMPDIR=$S
git rev-parse HEAD
uv sync -q --all-packages --extra torch-cpu </dev/null || exit 1
/usr/bin/time -v .venv/bin/python $RESEARCH_RUN_DIR/inputs/p4_describe.py $OUT 4 43 2 3 2> $OUT/time_describe.txt
echo describe rc=$?
rg "Maximum resident|Elapsed" $OUT/time_describe.txt
ls -l $OUT
