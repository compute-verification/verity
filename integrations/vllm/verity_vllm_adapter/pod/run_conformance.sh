#!/bin/bash
# GPU stages of the configuration-conformance harness for one profile port (lane CH; HP/TP call this on their pods).
# usage: run_conformance.sh <PORT> <TAG> <production|replay|all|conformance> [--rev REV] [--workload W] [--challenge security|diagnostic]
#                           [--seed N] [--weights-values store|hash|checkpoint] [--arena-gb N] [--limit N] [--production-inductor-cache DIR]
#                           [--skip-second-production] [-- <extra run_case.sh args>]
# env: PY (python of the pinned venv), WHEEL, VLLM_COMMIT (as run_case.sh), VERITY_REPO (default /workspace/veritor),
#      VERITY_RESULTS_ROOT (default /workspace/results).
#
# Bootstraps nothing. Reads the port's declared flags (`python -m verity_vllm_adapter.conformance flags --port P`) and calls the
# existing stage scripts with them:
#   production   run_case.sh <CASE> <TAG> <REV> --production-only <declared flags>        -> $R/production
#                run_case.sh <CASE> <TAG>-r2 ... --production-only (a second, independent process) -> $R/production-r2
#   replay       eager / compiled-nograph: run_case.sh <CASE> <TAG> <REV> <declared flags> --replay-exec <exec> [--rank-tag r per rank]
#                                          -> $R/{ref1,ref2,commit}[/rank<r>], compare[.rank<r>].json, verify-<challenge>[/rank<r>]/
#                compiled-fx:              compiled_fx/pod_run.sh <CASE> <TAG> <REV> $R/production --production-inductor-cache DIR
#   conformance  python -m verity_vllm_adapter.conformance run --port P --out $R/conformance --from-results $R [...]
#   all          production, replay, conformance
# Per-rank captures (a port with parallelism.tensor_parallel_size > 1 and adapter.capture_per_rank true): established on hardware
# by lane TP — with TP > 1 the model lives in vLLM's worker processes (the driver holds no model), so the replay stage runs ONCE
# and replay.py installs the capture inside every worker (tp_worker.py, worker_extension_cls); rank r's stream is written to
# <mode>/rank<r>/ with s=<r> on every id and tp_merge joins the ranks into <mode>/ (one commitments.json over both ranks). The
# compare/verify stages run on the joined store. (--rank-tag r remains the single-process tag for a one-rank capture.)
set -u
PORT=$1; TAG=$2; STAGE=$3; shift 3
REV=""; WL=""; CHALLENGE=security; SEED=""; WV=store; ARENA=""; LIMIT=""; PIC=""; SECOND=1; EXTRA=""
while [ $# -gt 0 ]; do case "$1" in
  --rev) REV=$2; shift;; --workload) WL=$2; shift;; --challenge) CHALLENGE=$2; shift;; --seed) SEED=$2; shift;;
  --weights-values) WV=$2; shift;; --arena-gb) ARENA="--arena-gb $2"; shift;; --limit) LIMIT="--limit $2"; shift;;
  --production-inductor-cache) PIC=$2; shift;; --skip-second-production) SECOND=0;;
  --) shift; EXTRA="$*"; break;; *) echo "unknown argument $1"; exit 2;; esac; shift; done
REPO=${VERITY_REPO:-/workspace/veritor}
RESULTS=${VERITY_RESULTS_ROOT:-/workspace/results}
PY=${PY:-/workspace/venv/bin/python}
cd $REPO/vllm-poc || exit 1
export PYTHONPATH=.:..:../src
[ -n "$REV" ] || REV=$(git -C $REPO rev-parse --short HEAD)
eval "$($PY -m verity_vllm_adapter.conformance flags --port $PORT)" || { echo "port $PORT invalid"; exit 2; }
for kv in $CONF_ENV; do export "$kv"; done
R=$RESULTS/$CONF_CASE/$TAG; mkdir -p $R
log() { echo "[$(date -u +%H:%M:%SZ)] conformance $PORT/$TAG: $*" | tee -a $R/conformance.log; }
log "stage=$STAGE case=$CONF_CASE exec=$CONF_REPLAY_EXEC tp=$CONF_TP per_rank=$CONF_PER_RANK production_args='$CONF_PRODUCTION_ARGS' rev=$REV"
RC=verity_vllm_adapter/pod/run_case.sh
WLARG=""; [ -n "$WL" ] && WLARG="--workload $WL"
SEEDARG=""; [ -n "$SEED" ] && SEEDARG="--seed $SEED"

production() {
  bash $RC $CONF_CASE $TAG $REV --production-only $CONF_PRODUCTION_ARGS $WLARG $EXTRA || { log "production FAILED"; exit 1; }
  if [ $SECOND = 1 ]; then
    # Level 1 needs two production processes of the same profile: a second run_case.sh production into <TAG>-r2, linked as production-r2
    bash $RC $CONF_CASE $TAG-r2 $REV --production-only $CONF_PRODUCTION_ARGS $WLARG $EXTRA || { log "second production FAILED"; exit 1; }
    rm -rf $R/production-r2; ln -s $RESULTS/$CONF_CASE/$TAG-r2/production $R/production-r2
  fi
  log "production done: $(grep -h profile_id= /workspace/logs/$CONF_CASE-$TAG-production.log 2>/dev/null | tail -1)"
}

replay() {
  [ -d $R/production ] || { log "no $R/production: run the production stage first"; exit 1; }
  if [ "$CONF_REPLAY_EXEC" = compiled-fx ]; then
    [ -n "$PIC" ] || { log "compiled-fx replay needs --production-inductor-cache DIR (the production's TORCHINDUCTOR_CACHE_DIR)"; exit 1; }
    bash verity_vllm_adapter/compiled_fx/pod_run.sh $CONF_CASE $TAG $REV $R/production --production-inductor-cache $PIC --weights-values $WV $EXTRA || { log "compiled-fx replay FAILED"; exit 1; }
    return
  fi
  # TP > 1 (CONF_PER_RANK=1): one replay per stage; the per-rank capture happens inside the workers (replay.py reads
  # tensor_parallel_size from the production manifest and installs tp_worker.VerityTPWorkerExtension)
  bash $RC $CONF_CASE $TAG $REV $CONF_PRODUCTION_ARGS $CONF_REPLAY_ARGS --weights-values $WV --challenge $CHALLENGE $SEEDARG $ARENA $LIMIT $EXTRA || { log "replay FAILED"; exit 1; }
  log "replay done"
}

conformance() {
  $PY -m verity_vllm_adapter.conformance run --port $PORT --out $R/conformance --from-results $R 2>&1 | tee -a $R/conformance.log
  log "level: $($PY -c "import json; d=json.load(open('$R/conformance/level.json')); print(d['summary'])")"
  # the public C0–C5 verdict (ladder.json is written beside level.json by the ladder-aware harness; absent on older revisions)
  [ -f $R/conformance/ladder.json ] && log "ladder: $($PY -c "import json; d=json.load(open('$R/conformance/ladder.json')); print(d['summary'], '| outcome', d['outcome'])")"
}

case "$STAGE" in
  production) production;;
  replay) replay;;
  conformance) conformance;;
  all) production; replay; conformance;;
  *) echo "stage must be production|replay|conformance|all"; exit 2;;
esac
log "STAGE_DONE $STAGE"
