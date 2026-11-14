#!/usr/bin/env bash
# CIR-286: the capture's CPU tests and the property and records lints, to $OUT.
set -uo pipefail
date -u +"sync %FT%TZ"; git log -1 --format="head %H"
uv sync -q --all-packages --extra torch-cpu || exit 2
cd adapters/vllm || exit 2
date -u +"pytest %FT%TZ"
uv run pytest -q -p no:cacheprovider -rA tests/properties/test_glm53_index_share_capture.py tests/properties/test_record.py \
  tests/properties/test_glm53_mtp_captures.py tests/lint/test_p05_properties.py tests/lint/test_typed_records.py > "$OUT/pytest.log" 2>&1
rc=$?
echo "pytest rc=$rc"; grep -E "passed|failed|PASSED|FAILED|ERROR|Error" "$OUT/pytest.log" | tail -40
date -u +"end %FT%TZ"
exit $rc
