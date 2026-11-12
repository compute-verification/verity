#!/usr/bin/env bash
# on the final head, in a clone:
#  1. the capture of record (inputs/capture.tar: sass/ and capture/ of the card run's $OUT) compared again by this head's
#     script, its record held byte for byte to the card run's own record (inputs/glm53_flashmla_fp8_capture.json);
#  2. circuit-check's own pins update pass on the module's sidecar (what it wrote is saved, then undone), then
#     circuit-check on the sidecar, on heads at served sizes and on one whole token;
#  3. the suites the change from BASE touches (quick tier) and the repository suite whole.
set -uo pipefail
OUT=${OUT:-$RESEARCH_RUN_DIR/out}; mkdir -p "$OUT"
I=$RESEARCH_RUN_DIR/inputs
git rev-parse HEAD > "$OUT/head.txt"
date -u +"start %FT%TZ head $(cat "$OUT/head.txt")"
uv sync -q --all-packages --extra torch-cpu || exit 2
PY=$PWD/.venv/bin/python
S=$(mktemp -d)
tar -xf "$I/capture.tar" -C "$S" || exit 2
sha256sum "$I/capture.tar" "$I/glm53_flashmla_fp8_capture.json" > "$OUT/recompare-inputs.sha256"
n=$(nproc)
( cd adapters/vllm && "$PY" -m tests.properties.glm53_flashmla_fp8_capture_gpu compare --out "$S" \
    --workers "$(( n > 1 ? n - 1 : 1 ))" --definition "${DEF_TOKENS:-2}" ) > "$OUT/recompare.txt" 2>&1; rr=$?
tail -n 45 "$OUT/recompare.txt" | cut -c1-400
cp "$S/glm53_flashmla_fp8_capture.json" "$OUT/recompare-record.json" 2> /dev/null
if cmp "$OUT/recompare-record.json" "$I/glm53_flashmla_fp8_capture.json"; then
  echo "recompare: rc=$rr, this head's record of the saved capture is byte for byte the card run's record" | tee "$OUT/recompare-verdict.txt"
else
  echo "recompare: rc=$rr, this head's record DIFFERS from the card run's record" | tee "$OUT/recompare-verdict.txt"
  rr=1
fi
rm -rf "$S"
date -u +"recompared %FT%TZ rc=$rr"

mkdir -p "$OUT/pins" "$OUT/cc"
OUT="$OUT/pins" CC_ARGS=--update-pins SERVED=0 bash "$I/cc2.sh"; rp=$?
git status --short > "$OUT/git-status-after-pins.txt"
git diff > "$OUT/update-pins.diff"
git ls-files --others --exclude-standard | grep -v '^\.venv' > "$OUT/untracked-after-pins.txt" || true
echo "pins: rc=$rp, $(wc -l < "$OUT/git-status-after-pins.txt") path(s) changed, $(wc -c < "$OUT/update-pins.diff") bytes of diff"
git checkout -q -- . && git clean -fdq -e .venv
git status --short | head -5
OUT="$OUT/cc" SERVED=1 bash "$I/cc2.sh"; rc_cc=$?
bash "$I/suites.sh"; rs=$?
echo "rc recompare=$rr pins=$rp circuit-check=$rc_cc suites=$rs" >> "$OUT/rc.txt"
cat "$OUT/rc.txt"
[ "$rr" = 0 ] || exit "$rr"
[ "$rp" = 0 ] || exit "$rp"
[ "$rc_cc" = 0 ] || exit "$rc_cc"
exit "$rs"
