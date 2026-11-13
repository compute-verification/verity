set -u
I=$RESEARCH_RUN_DIR/inputs
git rev-parse HEAD | tee $OUT/head.txt
sha256sum $I/multimem-join.diff | tee $OUT/diff_sha.txt
git apply $I/multimem-join.diff && git diff --stat | tee $OUT/applied.txt
uv sync -q --all-packages --extra torch-cpu
bash $I/watch.sh census78 .venv/bin/python $I/embed_census.py $OUT 78 5 > $OUT/stdout_census78.txt 2>&1
rc=$?
tail -4 $OUT/stdout_census78.txt | cut -c1-3000
exit $rc
