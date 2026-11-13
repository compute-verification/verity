set -u
git rev-parse HEAD | tee $OUT/head.txt; git status --short | tee $OUT/status.txt
uv sync -q --all-packages --extra torch-cpu
uv run tools/verity/check/suites.py verity-vllm --fresh -- tests/lint > $OUT/lint.txt 2>&1; echo "lint rc=$?" | tee -a $OUT/rc.txt
tail -15 $OUT/lint.txt
uv run tools/verity/check/suites.py repository --fresh > $OUT/repository.txt 2>&1; echo "repository rc=$?" | tee -a $OUT/rc.txt
tail -15 $OUT/repository.txt
