#!/usr/bin/env bash
# every captured token held to FlashMlaV32Decode_v1's own evaluator, off the card, from a saved capture
# (inputs/capture.tar holds sass/ and capture/ of the card run's $OUT)
set -uo pipefail
OUT=${OUT:-$RESEARCH_RUN_DIR/out}; mkdir -p "$OUT"
S=$(mktemp -d)
tar -xf "$RESEARCH_RUN_DIR/inputs/capture.tar" -C "$S" || exit 2
sha256sum "$RESEARCH_RUN_DIR/inputs/capture.tar" > "$OUT/capture.tar.sha256"
git rev-parse HEAD > "$OUT/head.txt"
date -u +"start %FT%TZ"
uv sync -q --all-packages --extra torch-cpu || exit 2
PY=$PWD/.venv/bin/python
cd adapters/vllm || exit 2
n=$(nproc)
"$PY" -m tests.properties.glm53_flashmla_fp8_capture_gpu compare --out "$S" \
  --workers "${WORKERS:-$(( n > 1 ? n - 1 : 1 ))}" --definition "${DEF_TOKENS:-100000}" > "$OUT/compare.txt" 2>&1; rc=$?
tail -n 80 "$OUT/compare.txt"
cp "$S/glm53_flashmla_fp8_capture.json" "$OUT/" 2> /dev/null
rm -rf "$S"
date -u +"end %FT%TZ rc=$rc"
exit $rc
