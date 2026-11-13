#!/usr/bin/env bash
# CIR-263: circuit-check's update pass over mtp_draft's and boolean_mtp_draft's sidecars (their pins, written by the tool), the pins' diff to $OUT, then
# circuit-check's report over the sidecars and over DsaRMSNormApply_v1{G=64}, the apply unit of MtpEhNorm_v1's cut (report only:
# its pin is dsa_norm's sidecar's).  Runs in a private clone (--cwd clone).
set -uo pipefail
date -u +"sync %FT%TZ"; git log -1 --format="head %H"
uv sync -q --all-packages --extra torch-cpu || exit 2
SIDE="verity/definitions/registry/mtp_draft.circuit-check.py verity/definitions/registry/boolean_mtp_draft.circuit-check.py"
APPLY='DsaRMSNormApply_v1{G=64}'
date -u +"update %FT%TZ"
uv run circuit-check --update-pins --mem-gb 28 --out "$OUT/update.json" $SIDE > "$OUT/update.log" 2>&1
echo "update rc=$?"
tail -30 "$OUT/update.log"
git add -N -A verity/definitions/registry
git diff > "$OUT/pins.diff"
git status --porcelain > "$OUT/status.txt"
cat "$OUT/status.txt"
date -u +"report %FT%TZ"
uv run circuit-check --mem-gb 28 --out "$OUT/report.json" $SIDE "$APPLY" > "$OUT/report.log" 2>&1
rc=$?
echo "report rc=$rc"
tail -60 "$OUT/report.log"
date -u +"end %FT%TZ"
exit $rc
