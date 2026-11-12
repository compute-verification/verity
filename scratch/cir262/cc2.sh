#!/usr/bin/env bash
# circuit-check on the V32 module's sidecar and on served-size heads and a token, in a clone with the working diff applied
set -uo pipefail
OUT=${OUT:-$RESEARCH_RUN_DIR/out}; mkdir -p "$OUT"
date -u +"start %FT%TZ"
git rev-parse HEAD > "$OUT/head.txt"
if [ -f "$RESEARCH_RUN_DIR/inputs/v2.patch" ]; then
  git apply "$RESEARCH_RUN_DIR/inputs/v2.patch" || exit 3
  sha256sum "$RESEARCH_RUN_DIR/inputs/v2.patch" > "$OUT/patch.sha256"
fi
git status --short | head -20 > "$OUT/git-status.txt" 2>&1 || true
uv sync -q --all-packages --extra torch-cpu || exit 2
date -u +"synced %FT%TZ"
H=HopperBF16WgmmaDot16_v1
HD="DK=576,DV=512,BN=64,DOT=$H,SCALE=0.0625"
S32="(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32)"
S16="(0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32)"
rc=0
uv run circuit-check ${CC_ARGS:-} --out "$OUT/circuit-check.json" \
  verity/definitions/registry/flashmla_dsv32.circuit-check.py > "$OUT/circuit-check.txt" 2>&1 || rc=$?
tail -n 60 "$OUT/circuit-check.txt"
date -u +"sidecar %FT%TZ rc=$rc"
if [ "${SERVED:-1}" = 1 ]; then
  r2=0
  timeout 40m uv run circuit-check ${CC_ARGS:-} --out "$OUT/circuit-check-heads.json" \
    "FlashMlaV32DecodeHead_v1{SPLITS=$S32,$HD}" "FlashMlaV32DecodeHead_v1{SPLITS=$S16,$HD}" \
    "FlashMlaV32DecodeHead_v1{SPLITS=(0,32),$HD}" "FlashMlaV32CombineValues_v1{NS=16,G=256}" \
    "FlashMlaCombine_v1{NS=16,D=512}" \
    > "$OUT/circuit-check-heads.txt" 2>&1 || r2=$?
  tail -n 60 "$OUT/circuit-check-heads.txt"
  date -u +"heads %FT%TZ rc=$r2"
  r3=0
  timeout 45m uv run circuit-check ${CC_ARGS:-} --out "$OUT/circuit-check-token.json" \
    "FlashMlaV32Decode_v1{NH=8,SPLITS=$S16,DN=512,DR=64,QT=128,BN=64,DOT=$H,SCALE=0.0625}" \
    > "$OUT/circuit-check-token.txt" 2>&1 || r3=$?
  tail -n 40 "$OUT/circuit-check-token.txt"
  date -u +"token %FT%TZ rc=$r3"
  echo "rc sidecar=$rc heads=$r2 token=$r3" > "$OUT/rc.txt"
  [ $rc = 0 ] && rc=$r2
  [ $rc = 0 ] && rc=$r3
fi
date -u +"end %FT%TZ rc=$rc"
exit $rc
