#!/usr/bin/env bash
# [E8 NIGHT 2026-09-12] Pod-side AUTONOMOUS orchestration of the 32x16 one-request protocol run (INTEGRATOR D37: the laptop may die ~09:40Z;
# everything below runs under nohup on night-e8-prove and needs nothing from the laptop after it starts).
#   phase A  (now):            launch_32x16_pod.sh <tag> freeze instance coverage serve     -> the run-private tree exists; its hash is what E3's census reads
#   wait     (rows):           E1's producer `done` event with rows_mismatch == 0 (else stop: a boundary mismatch is a finding, not a run)
#   wait     (census):         /workspace/a9/GO-<tag> (touched from the laptop when E3's verdict `census 0 on <hash>` is posted), until GO_DEADLINE (UTC HH:MM);
#                              after the deadline the run proceeds WITHOUT the verdict and says so (GO-<tag>.autonomous) -- the record is then labelled
#                              "census gate NOT confirmed" and the morning reader decides; a record that exists beats none (D37).
#   phase B:                   launch_32x16_pod.sh <tag> protocol nonfinite obligations prove verify costlog report   (steps 1-5, real proofs, live driver)
#   finish:                    archive_v5 pack/check + sha256, reverify_clean.sh (fresh venv, shipped e2e.verifier_v3.reverify) on this pod, slim tarball
# usage: nohup bash orchestrate_32x16_pod.sh n1 > /workspace/a9/rec/orchestrate-n1.out 2>&1 &
set -u
TAG=${1:?tag}
RUN_ID=${RUN_ID:-V5-N32x16-1req-bex0-inc}
E1OUT=${E1OUT:-/workspace/e1/out/n32x16-7cc7c3bb}
GO=/workspace/a9/GO-$TAG
GO_DEADLINE=${GO_DEADLINE:-11:15}
REC=/workspace/a9/rec/$RUN_ID-$TAG
POP=${POP:-/workspace/a9/pop/32x16/population.v2.json}
HOST=${HOST:-/workspace/a9/veritor-zk-host}
LOG=/workspace/a9/rec/orchestrate-$TAG.log
HERE=$(cd "$(dirname "$0")" && pwd)
log() { echo "[orch $(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
mkdir -p /workspace/a9/rec /workspace/a9/archive /workspace/a9/reverify

log "phase A: freeze instance coverage serve (tag $TAG, fresh serve)"
# E1OUT=/nonexistent in phase A: the rows are still being produced; run_night_pod.sh would otherwise stage the PARTIAL rows now and mark them .staged-ok
SHARED_SERVE= E1OUT=/nonexistent bash "$HERE/launch_32x16_pod.sh" "$TAG" freeze instance coverage serve >> /workspace/a9/rec/launch-$TAG.out 2>&1; log "phase A rc=$? ($(tail -1 /workspace/a9/rec/launch-$TAG.out | cut -c1-160))"
[ -f "$REC/freeze.json" ] || { log "FATAL: no freeze.json after phase A"; exit 2; }
log "tree for the census: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print({k: (str(v)[:72]) for k, v in d.items() if k in ('tree_sha256','manifest_sha256','source_digest','revision','commit','dirty_patch_sha256','snapshot_sha256')})" "$REC/freeze.json")"
cp "$REC/freeze.json" /workspace/a9/rec/freeze-$TAG.json

log "waiting for E1's rows ($E1OUT)"
while ! grep -q '"event": "done"' "$E1OUT/progress.jsonl" 2>/dev/null; do sleep 60; done
log "rows done: $(tail -1 "$E1OUT/progress.jsonl" | cut -c1-240)"
python3 - "$E1OUT/progress.jsonl" <<'EOF' || { log "ROWS MISMATCH or no verdict -- the run stops here (finding, not a record)"; exit 3; }
import json, sys
last = json.loads(open(sys.argv[1]).read().strip().split("\n")[-1])
ok = last.get("event") == "done" and int(last.get("rows_mismatch", 1)) == 0 and int(last.get("rows_compared", 0)) > 0
print("rows verdict:", last.get("verdict"), "ok" if ok else "NOT OK"); sys.exit(0 if ok else 1)
EOF

log "waiting for GO ($GO) until ${GO_DEADLINE}Z"
while [ ! -f "$GO" ]; do
  now=$(date -u +%H:%M)
  if [[ "$now" > "$GO_DEADLINE" ]]; then
    echo "no GO by ${GO_DEADLINE}Z; proceeding autonomously (D37); the census gate is NOT confirmed for this tree -- label the record so" | tee "$GO.autonomous" | tee -a "$LOG"
    break
  fi
  sleep 30
done
[ -f "$GO" ] && log "GO: $(cat "$GO")"

log "phase B: protocol nonfinite obligations prove verify costlog report"
SHARED_SERVE= STAGED=/workspace/a9/interior-values-$TAG bash "$HERE/launch_32x16_pod.sh" "$TAG" protocol nonfinite obligations prove verify costlog report >> /workspace/a9/rec/launch-$TAG.out 2>&1; log "phase B rc=$?"
log "live verdict: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('verdict') or d.get('status'))" "$REC/verify/verify_v5.json" 2>/dev/null || echo 'no verify_v5.json')"

REPO=/workspace/a9/run-$TAG-veritor
PY=$REPO/.venv/bin/python; export PYTHONPATH=$REPO:$REPO/vllm-poc
cd "$REPO" || exit 3
log "finish: archive"
$PY -m record_v5.archive_v5 pack "$REC" /workspace/a9/archive/$RUN_ID-$TAG.tgz 2>&1 | tail -3 | tee -a "$LOG"
$PY -m record_v5.archive_v5 check /workspace/a9/archive/$RUN_ID-$TAG.tgz 2>&1 | tail -2 | tee -a "$LOG"
sha256sum /workspace/a9/archive/$RUN_ID-$TAG.tgz | tee /workspace/a9/archive/$RUN_ID-$TAG.tgz.sha256 | tee -a "$LOG"
log "finish: reverify_clean (fresh venv; the shipped e2e.verifier_v3.reverify --instance-format record_v5)"
/usr/bin/time -v -o /workspace/a9/reverify/$TAG.time bash record_v5/reverify_clean.sh /workspace/a9/veritor "$REC" "$POP" "$HOST" /workspace/a9/reverify/$TAG 2>&1 | tail -8 | tee -a "$LOG"
log "reverify rc=${PIPESTATUS[0]}; $(grep -E 'Elapsed|Maximum resident' /workspace/a9/reverify/$TAG.time | tr '\n' ' ')"
# slim tarball for the laptop / T1: the record without the staged rows and the large trees' leaf files, plus reverify + logs
tar czf /workspace/a9/archive/$RUN_ID-$TAG-slim.tgz -C /workspace/a9 --exclude='*.jsonl' --exclude='interior-values*' --exclude='*/leaves.bin' --exclude='*/values.bin' rec/$RUN_ID-$TAG reverify/$TAG rec/launch-$TAG.out rec/orchestrate-$TAG.log 2>/dev/null
sha256sum /workspace/a9/archive/$RUN_ID-$TAG-slim.tgz | tee -a "$LOG"
log "DONE: archive /workspace/a9/archive/$RUN_ID-$TAG.tgz; slim /workspace/a9/archive/$RUN_ID-$TAG-slim.tgz; reverify /workspace/a9/reverify/$TAG"
