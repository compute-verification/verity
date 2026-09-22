#!/usr/bin/env bash
# S5 of `scripts/derive_e2e.sh` (lane CAPCORR, GUIDANCE §43.5): prove a NONEMPTY selection of the certified record's
# obligations FROM THE PROGRAM'S GATES (S4's address map), witness reconstructed from the record's authenticated
# interfaces, proved with the record's own pinned SP1 host (ELF/vk = manifest pins) on the pod GPU, verified, then
# the verifier-side negatives (a different program, a proof from a different record, a foreign proof, tampered
# proof records) — all must be REFUSED.
#
#   PROGRAM=<spec> RECORD=<dir> S4=<S4 out dir> BUNDLE=<commit dir> CKPT=<dir> HOST=<veritor-zk-host> OUT=<dir> \
#     [MODES=core,compressed] [SP1_PROVER=cuda] [MAX_UNITS=0] bash scripts/derive_prove.sh
#
# Defaults are the pod layout of `capcorr_pod_setup.sh`.  Exit: 0 proved+verified+bound and every negative refused;
# 2 binding refused; 3 proof/witness/negative failure.
set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
cd "$REPO"
PY=${PY:-/workspace/capcorr-env/venv/bin/python}
[ -x "$PY" ] || PY=python
export PYTHONPATH=.:src:vllm-poc:scripts
PROGRAM=${PROGRAM:-registry:vllm-poc/profiles/b0-a100-eager}
RECORD=${RECORD:-fixtures/results/INT-PAD-2/INTPAD2-B0-20260909T013519Z}
S4=${S4:-/workspace/capcorr/DERIVE-CORR/harness-b0-s4}
BUNDLE=${BUNDLE:-/workspace/capcorr/bundle/INT-B0-INTPAD2-B0-20260909T013519Z/bundle}
CKPT=${CKPT:-/workspace/capcorr-env/ckpt}
HOST=${HOST:-$RECORD/bin/veritor-zk-host}
OUT=${OUT:-/workspace/capcorr/DERIVE-CORR/harness-b0-s5}
MODES=${MODES:-core,compressed}
export SP1_PROVER=${SP1_PROVER:-cuda}
MAX_UNITS=${MAX_UNITS:-0}
mkdir -p "$OUT"
stage() { echo; echo "=== [$(date -u +%FT%TZ)] S5 $*"; }

stage "prove ($MODES, SP1_PROVER=$SP1_PROVER, host $HOST)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
# shared 4090: two lanes' sp1-gpu-servers on device 0 (/tmp/sp1-cuda-0.sock) fail each other's setup ("allocation failed"); wait for
# 60 s of no sp1-gpu-server and < 1 GB used before proving (WAIT_GPU_IDLE=0 to skip; gives up after WAIT_GPU_MAX seconds)
if [ "${WAIT_GPU_IDLE:-1}" = "1" ] && [ "$SP1_PROVER" = "cuda" ] && command -v nvidia-smi >/dev/null; then
  idle=0; waited=0
  while [ $idle -lt 6 ] && [ $waited -lt "${WAIT_GPU_MAX:-1800}" ]; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    if pgrep -f "sp1-gpu-serve[r]" >/dev/null || [ "${used:-99999}" -gt 1000 ]; then idle=0; else idle=$((idle+1)); fi
    sleep 10; waited=$((waited+10))
  done
  [ $idle -ge 6 ] && echo "GPU idle after ${waited}s" || echo "WARNING: GPU still busy after ${waited}s; proving anyway"
fi
"$PY" scripts/derive_prove.py --program "$PROGRAM" --record "$RECORD" --s4 "$S4" --bundle "$BUNDLE" --checkpoint-dir "$CKPT" \
  --host "$HOST" --out "$OUT" --modes "$MODES" --sp1-prover "$SP1_PROVER" --max-units "$MAX_UNITS"
rc=$?
echo "prove exit $rc"
[ $rc -eq 0 ] || exit $rc

if [ "${SUBSET:-1}" = "1" ] && [ "$MAX_UNITS" = "0" ]; then
  stage "subset proof (1 GEMM coordinate, ${MODES%%,*}) — the 'missing required proof' negative's input"
  "$PY" scripts/derive_prove.py --program "$PROGRAM" --record "$RECORD" --s4 "$S4" --bundle "$BUNDLE" --checkpoint-dir "$CKPT" \
    --host "$HOST" --out "$OUT/subset" --modes "${MODES%%,*}" --sp1-prover "$SP1_PROVER" --max-units 1 --label "subset proof (negative input only)"
  echo "subset exit $?"
fi

stage "verifier-side refusals"
"$PY" scripts/derive_prove.py --program "$PROGRAM" --record "$RECORD" --s4 "$S4" --bundle "$BUNDLE" --checkpoint-dir "$CKPT" \
  --host "$HOST" --out "$OUT" --modes "$MODES" --self-test-refusals
rc2=$?
echo "refusals exit $rc2"
echo "derived_proof_record: $OUT/derived_proof_record.json"
exit $rc2
