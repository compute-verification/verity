#!/usr/bin/env bash
# [E8 D40 T-3 2026-09-12] Pod-side AUTONOMOUS orchestration of the 32x16 rerun on ct-e8-32x16 (RTX 4090 + network volume h5uk3omqsl at /vol).
# Nothing here needs the laptop after it starts (setsid nohup).  Every phase end writes the record (DURABLE_DIR mirror) and REPORT-e8.md to the volume.
#   wait      bootstrap:  /workspace/a9/setup/{host,vllm,venv}.done + weights_authorized.json (E10's pod_bootstrap_a9.sh)
#   phase A:              launch_32x16_pod.sh <tag> freeze instance coverage serve   (run-private tree copy; E6's pin; fresh serve)
#   wait      rows:       $E1OUT.DONE (ct-e8-rows' mirror marker on the volume) + progress.jsonl `done` with rows_mismatch == 0
#   GO:                   pre-written by the launcher (D40: the gate carries from d8208239 for the same population; E3 census 08:49Z on this instance)
#   phase B:              launch_32x16_pod.sh <tag> protocol nonfinite obligations prove verify costlog report   (steps 1-5, real SP1 proofs, live driver)
#   finish:               archive_v5 pack/check (+ --allow-missing-proofs only when refused before prove), reverify_clean.sh (fresh venv, shipped reverify),
#                         slim tarball + sha256 -> /vol/e8/archive, record -> /vol/e8/rec (mirror), DONE marker /vol/e8/DONE-ct-e8-32x16
# usage: cd /workspace/a9 && nohup setsid bash orchestrate_ct_pod.sh n3 > /workspace/a9/rec/orchestrate-n3.out 2>&1 < /dev/null &
set -u
TAG=${1:?tag}
RUN_ID=${RUN_ID:-V5-N32x16-1req-bex0-inc}
VOL=${VOL:-/vol/e8}
E1OUT=${E1OUT:-$VOL/rows/n32x16-7cc7c3bb}
GO=/workspace/a9/GO-$TAG
REC=/workspace/a9/rec/$RUN_ID-$TAG
POP=${POP:-/workspace/a9/pop/32x16/population.v2.json}
HOST=${HOST:-/workspace/a9/veritor-zk-host}
LOG=/workspace/a9/rec/orchestrate-$TAG.log
HERE=$(cd "$(dirname "$0")" && pwd)
log() { echo "[orch $(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
mkdir -p /workspace/a9/rec /workspace/a9/archive /workspace/a9/reverify "$VOL/rec" "$VOL/archive" "$VOL/reverify"
export DURABLE_DIR=$VOL/rec
export POD_NAME=${POD_NAME:-ct-e8-32x16} POD_ID=${POD_ID:-9ywl098yqc40k7}
ROWS_NOTE_DEFAULT="ct-e8-rows nc17112naqm52x (CPU cpu5g 8 vCPU), E1 wave-5 producer 5e016f0e/60cd3930, --only-required --parallel 8 --parallel-min-gates 2000, mirrored to $E1OUT"
export ROWS_NOTE="${ROWS_NOTE:-$ROWS_NOTE_DEFAULT}"
export REQUIRED_FROM_RECORD=${REQUIRED_FROM_RECORD:-$VOL/rec-32x15}

log "waiting for the bootstrap (host/vllm/venv .done + weights_authorized.json)"
while ! { [ -f /workspace/a9/setup/host.done ] && [ -f /workspace/a9/setup/vllm.done ] && [ -f /workspace/a9/setup/venv.done ] && [ -f /workspace/a9/weights/weights_authorized.json ]; }; do sleep 30; done
log "bootstrap: host $(cat /workspace/a9/setup/host.done) vllm $(cat /workspace/a9/setup/vllm.done) venv $(cat /workspace/a9/setup/venv.done); host sha256 $(sha256sum $HOST | cut -c1-16)"
grep -q "rc=0" /workspace/a9/setup/host.done || { log "FATAL: host build failed"; exit 2; }

log "phase A: freeze instance coverage serve (tag $TAG, fresh serve)"
SHARED_SERVE= E1OUT=/nonexistent bash "$HERE/launch_32x16_pod.sh" "$TAG" freeze instance coverage serve >> /workspace/a9/rec/launch-$TAG.out 2>&1; log "phase A rc=$? ($(tail -1 /workspace/a9/rec/launch-$TAG.out | cut -c1-160))"
[ -f "$REC/freeze.json" ] || { log "FATAL: no freeze.json after phase A"; exit 2; }
log "tree: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print({k: (str(v)[:72]) for k, v in d.items() if k in ('commit','dirty_patch_sha256','snapshot_sha256')})" "$REC/freeze.json")"
cp "$REC/freeze.json" /workspace/a9/rec/freeze-$TAG.json; rsync -a --exclude 'interior-values*' "$REC/" "$DURABLE_DIR/$(basename "$REC")/" 2>/dev/null

# two producers race for the same rows (D40: the 4090 pod idles while it waits): ct-e8-rows (8 vCPU, -> the volume, E1OUT) and this pod's
# own producer (E1OUT_ALT, 23.8-core quota, local disk); the first `.DONE` marker wins and names the rows the record commits (ROWS_NOTE).
log "waiting for a rows marker: $E1OUT.DONE (ct-e8-rows mirror)${E1OUT_ALT:+ or $E1OUT_ALT.DONE (this pod)}"
while :; do
  [ -f "$E1OUT.DONE" ] && break
  [ -n "${E1OUT_ALT:-}" ] && [ -f "$E1OUT_ALT.DONE" ] && { E1OUT=$E1OUT_ALT; export ROWS_NOTE="${ROWS_NOTE_ALT:-this pod ct-e8-32x16 $POD_ID (23.8-core quota), E1 wave-5 producer, --only-required --parallel 20 --parallel-min-gates 2000, $E1OUT_ALT}"; break; }
  sleep 60
done
export E1OUT
log "rows done ($E1OUT): $(tail -1 "$E1OUT/progress.jsonl" | cut -c1-240)"
python3 - "$E1OUT/progress.jsonl" <<'EOF' || { log "ROWS MISMATCH or no verdict -- the run stops here (finding, not a record)"; exit 3; }
import json, sys
last = json.loads(open(sys.argv[1]).read().strip().split("\n")[-1])
ok = last.get("event") == "done" and int(last.get("rows_mismatch", 1)) == 0 and int(last.get("rows_compared", 0)) > 0
print("rows verdict:", last.get("verdict"), "ok" if ok else "NOT OK"); sys.exit(0 if ok else 1)
EOF
[ -f "$GO" ] && log "GO: $(cat "$GO")" || log "no GO file (D40: the gate carries; proceeding)"

# [wave 6, E9 c160e55b] V6.gather / V6.checkpoint: the verifier's OWN copy of the pinned checkpoint (sha256 vs manifests/checkpoints.json) --
# live path through verify_v5's reverify_argv (--reverify-args), shipped reverify_clean.sh through its extra args
CKPT=${CKPT:-$VOL/ckpt}
# one shell word each (run_v5.sh expands ${VERIFY_ARGS:-} unquoted): --reverify-args=<extra>; verify_v5 splits <extra> on whitespace -> --checkpoint=<dir>
[ -f "$CKPT/model.safetensors" ] && export VERIFY_ARGS="--reverify-args=--checkpoint=$CKPT" || log "WARNING: no verifier checkpoint copy at $CKPT (V6.gather would refuse on a sampled Gather unit)"
# [wave 7, E10 b9c96a62 / E1 fcaae287] V4.elf_archived: the archived bytes of the host's guest (666e937f) -- explicit for both entry points
ELF=${ELF:-/workspace/a9/veritor/fixtures/results/GUEST-666e937f-e7-ftz-f32/veritor-zk-guest.elf}
if [ -f "$ELF" ]; then export VERITOR_REVERIFY_ELF=$ELF; log "archived guest ELF for V4.elf_archived: $ELF sha256 $(sha256sum "$ELF" | cut -c1-16)"; else log "WARNING: no archived guest ELF at $ELF (V4.elf_archived would FAIL under guest 666e937f)"; ELF=; fi
log "phase B: protocol nonfinite obligations prove verify costlog report (VERIFY_ARGS=${VERIFY_ARGS:-none}; VERITOR_REVERIFY_ELF=${VERITOR_REVERIFY_ELF:-none})"
SHARED_SERVE= STAGED=/workspace/a9/interior-values-$TAG bash "$HERE/launch_32x16_pod.sh" "$TAG" protocol nonfinite obligations prove verify costlog report >> /workspace/a9/rec/launch-$TAG.out 2>&1; log "phase B rc=$?"
VERDICT=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('verdict') or d.get('status'))" "$REC/verify/verify_v5.json" 2>/dev/null || echo 'no verify_v5.json')
log "live verdict: $VERDICT"

REPO=/workspace/a9/run-$TAG-veritor
PY=$REPO/.venv/bin/python; export PYTHONPATH=$REPO:$REPO/vllm-poc
cd "$REPO" || exit 3
log "finish: archive"
if [ -d "$REC/proofs" ]; then
  $PY -m record_v5.archive_v5 pack "$REC" /workspace/a9/archive/$RUN_ID-$TAG.tgz 2>&1 | tail -3 | tee -a "$LOG"
else
  log "no proofs/ (refused before prove): archive_v5 pack --allow-missing-proofs"
  $PY -m record_v5.archive_v5 pack --allow-missing-proofs "$REC" /workspace/a9/archive/$RUN_ID-$TAG.tgz 2>&1 | tail -3 | tee -a "$LOG"
fi
[ -f /workspace/a9/archive/$RUN_ID-$TAG.tgz ] && { $PY -m record_v5.archive_v5 check /workspace/a9/archive/$RUN_ID-$TAG.tgz 2>&1 | tail -2 | tee -a "$LOG"; sha256sum /workspace/a9/archive/$RUN_ID-$TAG.tgz | tee /workspace/a9/archive/$RUN_ID-$TAG.tgz.sha256 | tee -a "$LOG"; }
log "finish: reverify_clean (fresh venv; the shipped e2e.verifier_v3.reverify --instance-format record_v5)"
/usr/bin/time -v -o /workspace/a9/reverify/$TAG.time bash record_v5/reverify_clean.sh /workspace/a9/veritor "$REC" "$POP" "$HOST" /workspace/a9/reverify/$TAG ${CKPT:+--checkpoint "$CKPT"} ${ELF:+--elf "$ELF"} 2>&1 | tail -8 | tee -a "$LOG"
log "reverify rc=${PIPESTATUS[0]}; $(grep -E 'Elapsed|Maximum resident' /workspace/a9/reverify/$TAG.time | tr '\n' ' ')"
tar czf /workspace/a9/archive/$RUN_ID-$TAG-slim.tgz -C /workspace/a9 --exclude='*.jsonl' --exclude='interior-values*' --exclude='*/leaves.bin' --exclude='*/values.bin' rec/$RUN_ID-$TAG reverify/$TAG rec/launch-$TAG.out rec/orchestrate-$TAG.log 2>/dev/null
sha256sum /workspace/a9/archive/$RUN_ID-$TAG-slim.tgz | tee -a "$LOG"
log "finish: to the volume ($VOL)"
cp /workspace/a9/archive/$RUN_ID-$TAG*.tgz* "$VOL/archive/" 2>/dev/null; rsync -a /workspace/a9/reverify/$TAG/ "$VOL/reverify/$TAG/" 2>/dev/null; cp /workspace/a9/reverify/$TAG.time "$VOL/reverify/" 2>/dev/null
rsync -a --exclude 'interior-values*' "$REC/" "$DURABLE_DIR/$(basename "$REC")/" 2>/dev/null; cp /workspace/a9/rec/launch-$TAG.out /workspace/a9/rec/orchestrate-$TAG.log "$VOL/rec/" 2>/dev/null
log "DONE: archive /workspace/a9/archive/$RUN_ID-$TAG.tgz; slim /workspace/a9/archive/$RUN_ID-$TAG-slim.tgz (volume copies under $VOL/archive); reverify /workspace/a9/reverify/$TAG ($VOL/reverify/$TAG); live verdict $VERDICT"
sleep 150  # the row / report watchers write their final files after DONE
echo "$(date -u +%FT%TZ) $POD_NAME $POD_ID DONE: live verdict $VERDICT; record $VOL/rec/$RUN_ID-$TAG; slim $VOL/archive/$RUN_ID-$TAG-slim.tgz; report $VOL/REPORT-e8.md -- integrator: terminate this pod" | tee "$VOL/DONE-$POD_NAME" | tee -a "$LOG"
