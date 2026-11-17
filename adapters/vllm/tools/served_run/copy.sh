#!/usr/bin/env bash
# A deferred window's files sent from its serve dir to the same path on AUDIT_NODE (`run.sh copy`).
set -uo pipefail
export D=/dev/shm/vllm-served-run/${SERVED_RUN:?start it with \`run.sh copy\`} OUT=${RESEARCH_RUN_DIR:?run under research run}/out
mkdir -p "$OUT"
[ -f "$D/private/deferred/deferred.env" ] || { echo "copy.sh: $D holds no deferral: nothing sent"; exit 2; }
echo "copy.sh: the deferral's manifest, inputs.sha256: $(sha256sum < "$D/private/deferred/inputs.sha256" | cut -d' ' -f1)"
exec bash benchmarks/one_stage/served_verdicts.sh copy deferred "${AUDIT_NODE:?start it with \`run.sh copy\`}"
