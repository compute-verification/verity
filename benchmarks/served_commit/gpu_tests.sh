#!/usr/bin/env bash
# The committer's GPU byte-identity tests on a node (TESTS, default the windowed rows-sha512 suite, the window-tree test and the
# windowed fail-closed test).
set -uo pipefail
. benchmarks/harness/env.sh
OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}
mkdir -p "$OUT"
TESTS=${TESTS:-integrations/vllm/tests/commit/test_windowed_rows_sha512.py integrations/vllm/tests/commit/test_p0_footprint.py integrations/vllm/tests/commit/test_windowed_fail_closed.py}
"$VENV/bin/python" -m pytest -p no:cacheprovider -rA -v $TESTS --junitxml "$OUT/junit.xml" 2>&1 | tee "$OUT/pytest.log"
echo "pytest rc ${PIPESTATUS[0]}"
