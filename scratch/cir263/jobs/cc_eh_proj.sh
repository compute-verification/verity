#!/usr/bin/env bash
# CIR-263: circuit-check's update pass and report on eh_proj's two bindings by their Definition ids (a sidecar's path checks only
# the Definitions its own module defines, and these are core's Gemm_v2 and cublas_bf16's CublasSplitKGemm_v1), the pins' diff to
# $OUT.  Runs in a private clone (--cwd clone).
set -uo pipefail
date -u +"sync %FT%TZ"; git log -1 --format="head %H"
uv sync -q --all-packages --extra torch-cpu || exit 2
IDS=('Gemm_v2{K=12288,N=6144,DOT=HopperBF16WgmmaDot16_v1}' 'CublasSplitKGemm_v1{K=12288,N=6144,SPLIT_K=2,DOT=HopperBF16WgmmaDot16_v1}')
date -u +"update %FT%TZ"
uv run circuit-check --update-pins --mem-gb 28 --out "$OUT/update.json" "${IDS[@]}" > "$OUT/update.log" 2>&1
echo "update rc=$?"
tail -30 "$OUT/update.log"
git add -N -A verity
git diff > "$OUT/pins.diff"
git status --porcelain > "$OUT/status.txt"
echo "status:"; cat "$OUT/status.txt"
date -u +"report %FT%TZ"
uv run circuit-check --mem-gb 28 --out "$OUT/report.json" "${IDS[@]}" > "$OUT/report.log" 2>&1
rc=$?
echo "report rc=$rc"
tail -40 "$OUT/report.log"
date -u +"end %FT%TZ"
exit $rc
