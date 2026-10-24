#!/bin/bash
# PRF-131 mainv keeper: a share's completed lanes kept past the share's exit, and a mainv run ended once it has written
# the verdicts it should.
#   research run --on NODE --project verity --source . --cwd source --timeout 600m --send keep.sh \
#     --env SHARE=RUN --env TAGS='lane17 ...' [--env TARGET=MAINV_RUN --env RELEASE_KEYS='RUN.TAG ...' --env REASON=...] \
#     -- bash -c 'exec bash $RESEARCH_RUN_DIR/inputs/keep.sh'
# Once SHARE marks a tag's session complete (its out/TAG/sessions.du), the tag's sessions and verifier inputs (its plan's
# dir/sessions, pub, registered, program and partition, all in the share's scratch, which its exit removes) are
# hard-linked into KEEP/SHARE.TAG in mainv.sh's snapshot layout, with KEEP/SHARE.TAG.inputs.sha256; a tag the share never
# completes gets KEEP/SHARE.TAG.missing. mainv.sh reads them as a share `SHARE~KEEPER@HOST=TAGS` and removes each after its
# verify. TARGET, if set, is cancelled once each of RELEASE_KEYS has its out/v/KEY.rc or .missing; its out/v is copied first.
set -uo pipefail
OUT=${RESEARCH_RUN_DIR:?}/out RUNS=${SHARE_RUNS:-/workspace/research/runs}
KEEP=${KEEP:-/workspace/research/proofs-mainv/keep} SHARE=${SHARE:?} TAGS=${TAGS:?} POLL=${KEEP_POLL:-5}
mkdir -p "$OUT" "$KEEP"
log() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$OUT/job.log"; }
st() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['state'])" "$RUNS/$1/status.json" 2>/dev/null; }

release() {
  local RD=$RUNS/$TARGET V=$RUNS/$TARGET/out/v ok k pgid tool
  log "release $TARGET once these have verdicts: ${RELEASE_KEYS:?}"
  while :; do
    [ "$(st "$TARGET")" = running ] || { log "$TARGET is $(st "$TARGET"): nothing to cancel"; return 0; }
    ok=1
    for k in $RELEASE_KEYS; do [ -e "$V/$k.rc" ] || [ -e "$V/$k.missing" ] || ok=0; done
    [ $ok = 1 ] && break
    sleep "$POLL"
  done
  cp -r "$V" "$OUT/v-at-cancel"
  pgid=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['pgid'])" "$RD/status.json")
  tool=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['run']['remote']['tool'])" "$RD/job.json")
  log "every key has a verdict; cancelling $TARGET (pgid $pgid)"
  env PYTHONPATH="$tool" python3 -m research.telemetry cancel-intent --actor manual \
    --reason "${REASON:?} (cancelled by proofs-mainv)" --target-pgid "$pgid" --expect-run-dir "$RD" \
    --attempt-dir "$RD" --grace-s 20 --deliver > "$OUT/cancel.log" 2>&1
  log "cancel-intent exit $?: $(tail -c 600 "$OUT/cancel.log")"
}
RP=
[ -n "${TARGET:-}" ] && { release & RP=$!; }

SR=$RUNS/$SHARE
until [ -s "$SR/out/plan.json" ]; do
  [ "$(st "$SHARE")" = running ] || { log "$SHARE ended without a plan.json"; for t in $TAGS; do
    echo "the share ended without a plan.json" > "$KEEP/$SHARE.$t.missing"; done; [ -n "$RP" ] && wait "$RP"; exit 0; }
  sleep "$POLL"
done
cp "$SR/out/plan.json" "$KEEP/$SHARE.plan.json"
field() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['checks'][sys.argv[2]][sys.argv[3]])" \
  "$KEEP/$SHARE.plan.json" "$1" "$2"; }
lnk() { cp -al "$1" "$2" 2>/dev/null || cp -a "$1" "$2"; }   # a hard link survives the share's rm; a copy if across filesystems
keep() {   # keep TAG
  local t=$1 P=$KEEP/$SHARE.$1.partial
  rm -rf "$P" && mkdir -p "$P" || return 1
  lnk "$(field "$t" dir)/sessions" "$P/sessions" && lnk "$(field "$t" pub)" "$P/pub.bin" \
    && lnk "$(field "$t" registered)" "$P/registered.json" && lnk "$(field "$t" program)" "$P/program.json" \
    && lnk "$(field "$t" partition)" "$P/partition.json" || { rm -rf "$P"; return 1; }
  (cd "$P" && find . -type f | sort | xargs sha256sum) > "$KEEP/$SHARE.$t.inputs.sha256" && mv "$P" "$KEEP/$SHARE.$t"
}
log "keeping $SHARE's $TAGS in $KEEP as each completes"
pending=" $TAGS "
while [ -n "${pending// /}" ]; do
  alive=$(st "$SHARE")
  for t in $pending; do
    [ -e "$SR/out/$t/sessions.du" ] || continue
    if keep "$t"; then
      log "kept $SHARE $t: $(ls "$KEEP/$SHARE.$t/sessions" | tr '\n' ' ')$(du -sh "$KEEP/$SHARE.$t" | cut -f1)"
      pending=${pending/ $t / }
    else log "keep $SHARE $t FAILED"; fi
  done
  if [ "$alive" != running ] && [ -n "${pending// /}" ]; then
    for t in $pending; do echo "the share ended without a complete session" > "$KEEP/$SHARE.$t.missing"; done
    log "$SHARE ended; no session for:$pending"; break
  fi
  sleep "$POLL"
done
[ -n "$RP" ] && wait "$RP"
log done
