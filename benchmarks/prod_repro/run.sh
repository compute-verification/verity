#!/usr/bin/env bash
# vLLM at its serving settings, replayed (README.md): serve stock and ours at the same settings and load, replay the
# window ours served, and print the settings and three lines (tokens, cost, advice). From the tree's root, as the
# command of a `research run` on a GPU node, the C4 shard sent with --send (or C4=PATH) unless a ready plan is
# (PLAN_FILE):
#   verity bench prod_repro [--model moonlight] [--stock ARM] [--again ARM] [--ours ARM] [--load burst] [--rate 1000]
#     [--requests 800 | --span-s S] [--reps 3] [--max-model-len N] [--target 1]
#   verity bench prod_repro --client vllm-bench --load NAME[=RUNG+RUNG...] [--reps 6] [--model ...] [--stock ...] ...
#   verity bench prod_repro --report OUT --load NAME [--stock ARM] [--again ARM] [--ours ARM] [--target 1]
# The serve, the load and the replay are the serving cost benchmark's (benchmarks/serve_tax/run.sh, whose header says
# what a node needs and what its other variables do: they pass through, among them C4, PLAN_FILE, PRIVATE, VENV,
# CHECKPOINT, FLASHINFER, LDCONFIG, MAX_MIN and BUDGET_MIN). The arms are by default the arms at vLLM's published
# settings where the benchmark has them, else stock compiled vLLM, stock again and the step record (prod_repro.py
# arms), each a server lifetime of its own. By default the corpus client sends the load: REQUESTS requests of the plan (about 60% of its
# prompts behind one of eight shared prefixes) arriving at RATE a second, faster than the engine serves, each rep from
# an empty prefix cache; then the replay re-runs rep 1 of ours under its recorded schedule, in the mode it was served
# in (REPLAY_MODES, by default compiled). With --client vllm-bench vLLM's own client sends the load at its rungs, which
# gives the benchmark's paired cost and no replay: its passes keep no token ids.
# The last form reads an earlier job's out/ and needs no GPU. Every form ends with the report (prod_repro.py report),
# written to OUT/prod_repro.{json,txt}; the job fails where the serve or the replay does, or where a replayed token
# or step differs from the served one.
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
tax=$here/../serve_tax
CLIENT=corpus; REPORT=
while [ $# -gt 0 ]; do
  flag=$1; shift
  [ $# -gt 0 ] || { echo "prod_repro: $flag takes a value"; exit 2; }
  case $flag in
    --model) MODEL_ID=$1 ;; --stock) STOCK=$1 ;; --again) AGAIN=$1 ;; --ours) OURS=$1 ;; --load) LOAD=$1 ;;
    --rate) RATE=$1 ;; --requests) REQUESTS=$1 ;; --span-s) SPAN_S=$1 ;; --reps) REPS=$1 ;;
    --max-model-len) MAX_MODEL_LEN=$1 ;; --target) TARGET=$1 ;; --client) CLIENT=$1 ;; --report) REPORT=$1 ;;
    *) echo "prod_repro: unknown flag $flag (--model, --stock, --again, --ours, --load, --rate, --requests, --span-s," \
            "--reps, --max-model-len, --target, --client corpus|vllm-bench, --report OUT)"; exit 2 ;;
  esac
  shift
done
case $CLIENT in corpus|vllm-bench) ;; *) echo "prod_repro: --client corpus or vllm-bench, not $CLIENT"; exit 2 ;; esac
[ -d /workspace/cp/venv312 ] && venv=/workspace/cp/venv312 || venv=/workspace/venv312   # as serve_tax/run.sh finds it
VENV=${VENV:-$venv}
[ -x "$VENV/bin/python" ] && PY=$VENV/bin/python || PY=python3
arms=($("$PY" "$here/prod_repro.py" arms)) || exit 2
STOCK=${STOCK:-${arms[0]}}; AGAIN=${AGAIN-${arms[1]}}; OURS=${OURS:-${arms[2]}}
REPLAY_MODES=${REPLAY_MODES:-compiled}
report() {
  "$PY" "$here/prod_repro.py" report "$1" --stock "$STOCK" --again "$AGAIN" --ours "$OURS" --load "${LOAD%%=*}" \
    --replay "$1/replay-${REPLAY_MODES%% *}" --target "${TARGET:-1}"
}
if [ -n "$REPORT" ]; then
  [ -n "${LOAD:-}" ] || { echo "prod_repro: --report OUT --load NAME: the load the job served"; exit 2; }
  report "$REPORT"; exit $?
fi
D=${RESEARCH_RUN_DIR:?run this under research run (or read an earlier job: --report OUT --load NAME)}; OUT=$D/out
export MODEL_ID=${MODEL_ID:-moonlight} CLIENT ${MAX_MODEL_LEN:+MAX_MODEL_LEN}
case $CLIENT in
  corpus)
    LOAD=${LOAD:-burst}
    [ -n "${SPAN_S:-}" ] || REQUESTS=${REQUESTS:-800}
    export ARMS="$STOCK $AGAIN $OURS" RATES="$LOAD=${RATE:-1000}" REPS=${REPS:-3} ${REQUESTS:+REQUESTS} ${SPAN_S:+SPAN_S}
    export REPLAY=$LOAD REPLAY_FROM=$OUT/$OURS REPLAY_MODES
    bash "$tax/run.sh"; rc=$? ;;
  vllm-bench)
    [ -n "${LOAD:-}" ] || { echo "prod_repro: --client vllm-bench --load NAME[=RUNG+RUNG...]: one of the benchmark's"; exit 2; }
    unset REPLAY REPLAY_FROM
    bash "$tax/run.sh" --model "$MODEL_ID" --arms "$STOCK $AGAIN $OURS" --loads "$LOAD" --reps "${REPS:-6}"; rc=$? ;;
esac
[ $rc = 0 ] || echo "prod_repro: the serving cost benchmark's runner ended $rc: an arm or the replay failed (see above)"
[ -d "$OUT/$OURS" ] || { echo "prod_repro: nothing was served (no $OUT/$OURS): no report"; exit $(( rc ? rc : 1 )); }
report "$OUT"; r=$?
exit $(( rc ? rc : r ))
