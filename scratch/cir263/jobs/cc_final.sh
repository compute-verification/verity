#!/usr/bin/env bash
# CIR-263: circuit-check at the PR's final head. Update pass on both sidecars and eh_proj's two N = 1 slices (by id: they are
# core's Gemm_v2 and cublas_bf16's CublasSplitKGemm_v1, which a sidecar path does not reach), the pins' diff to $OUT; then one
# report on those, the norm's apply DsaRMSNormApply_v1{G=64}, and eh_proj's two whole rows.  Runs in a private clone (--cwd clone).
set -uo pipefail
date -u +"sync %FT%TZ"; git log -1 --format="head %H"
uv sync -q --all-packages --extra torch-cpu || exit 2
SIDECARS=(verity/definitions/registry/mtp_draft.circuit-check.py verity/definitions/registry/boolean_mtp_draft.circuit-check.py)
SLICES=('Gemm_v2{K=12288,N=1,DOT=HopperBF16WgmmaDot16_v1}' 'CublasSplitKGemm_v1{K=12288,N=1,SPLIT_K=2,DOT=HopperBF16WgmmaDot16_v1}')
ROWS=('Gemm_v2{K=12288,N=6144,DOT=HopperBF16WgmmaDot16_v1}' 'CublasSplitKGemm_v1{K=12288,N=6144,SPLIT_K=2,DOT=HopperBF16WgmmaDot16_v1}')
date -u +"update %FT%TZ"
uv run circuit-check --update-pins --mem-gb 28 --out "$OUT/update.json" "${SIDECARS[@]}" "${SLICES[@]}" > "$OUT/update.log" 2>&1
urc=$?
echo "update rc=$urc"
tail -40 "$OUT/update.log"
git add -N -A verity
git diff > "$OUT/pins.diff"
git status --porcelain > "$OUT/status.txt"
echo "status:"; cat "$OUT/status.txt"
date -u +"report %FT%TZ"
uv run circuit-check --mem-gb 28 --out "$OUT/report.json" "${SIDECARS[@]}" 'DsaRMSNormApply_v1{G=64}' "${SLICES[@]}" "${ROWS[@]}" \
  > "$OUT/report.log" 2>&1
rc=$?
echo "report rc=$rc"
tail -60 "$OUT/report.log"
date -u +"end %FT%TZ"
exit $rc
