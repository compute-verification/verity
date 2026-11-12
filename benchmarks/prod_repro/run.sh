#!/usr/bin/env bash
# vLLM at its serving settings, replayed (README.md): serve stock and ours at the same settings and load, replay the
# window ours served, and print the settings and three lines (tokens, cost, advice). From the tree's root, as the
# command of a `research run` on a GPU node, the C4 shard sent with --send (or C4=PATH) unless a ready plan is
# (PLAN_FILE):
#   verity bench prod_repro [--model moonlight] [--stock ARM] [--again ARM|none] [--ours ARM] [--load burst]
#     [--rate 1000] [--requests 800 | --span-s S] [--reps 3] [--replay PASS|none] [--max-model-len N] [--target 1]
#   verity bench prod_repro --client vllm-bench --load NAME[=RUNG+RUNG...] [--reps 6] [--replay NAME@RUNG]
#     [--pass-prompts NAME=N] [--sampled] [--stop-at TIME] [--model ...] [--stock ...] ...
#   verity bench prod_repro --report OUT --load NAME [--replay PASS] [--stock ARM] [--again ARM|none] [--ours ARM]
#     [--target 1]
#   ADVICE_TRACE=DIR [ADVICE_STOPS=ID,...] verity bench prod_repro ...   # the advice line, of a certified serve's trace
# The serve, the load and the replay are the serving cost benchmark's (benchmarks/serve_tax/run.sh, whose header says
# what a node needs and what its other variables do: they pass through, among them C4, PLAN_FILE, PRIVATE, VENV,
# CHECKPOINT, FLASHINFER, LDCONFIG, MAX_MIN and BUDGET_MIN). The arms are by default the arms at vLLM's published
# settings (prod, prod-again, prod-record) for a model whose recipe the benchmark holds, else stock compiled vLLM,
# stock again and the step record (prod_repro.py arms), each a server lifetime of its own; --again none serves stock
# once, and the cost then has no noise floor. By
# default the corpus client sends the load: REQUESTS requests of the plan (about 60% of its
# prompts behind one of eight shared prefixes) arriving at RATE a second, faster than the engine serves, each rep from
# an empty prefix cache; then the replay re-runs rep 1 of ours under its recorded schedule, in the mode it was served
# in (REPLAY_MODES, by default compiled). With --client vllm-bench vLLM's own client sends the load at its rungs, which
# gives the benchmark's paired cost; it replays nothing unless --replay names the pass of ours to re-run, as the
# benchmark names it (NAME@RUNG), which needs an arm of ours that keeps each request's prompt ids and served ids.
# --pass-prompts, --sampled and --stop-at are the benchmark's PASS_PROMPTS (the prompts a pass sends), SAMPLED (each
# request sampled at the recipe's temperature and top-p) and STOP_AT (an arm starts no pass it can't finish by then,
# and no replay starts after it).
# The replay is a step of its own after the arms: the benchmark's runner once more, in its replay-only form (so the
# cards are leased a second time), under what ours' engine has on that the job's environment has not: the arm's own
# vLLM general plugins and environment (prod_repro.py replay-env: the canonical DSA top-k's), which a stock arm's
# server must not inherit. OUT/replay_engine.json says what that was, with the canonical top-k's counters from the
# replay's processes and, with PRIVATE set, from serving's. MAX_MIN, the minutes a lease may last, is here the job's
# in all: the replay's lease gets what the arms left of it, and none starts once it is spent.
# The advice line reads the trace directory of a certified serve (ADVICE_TRACE, under the stop ids ADVICE_STOPS, else
# the served checkpoint's own): no arm of this job is one.
# The --report form reads an earlier job's out/ and needs no GPU. Every form ends with the report (prod_repro.py
# report), written to OUT/prod_repro.{json,txt}, which says how far each arm and the replay got, also where the job is
# ended early (SIGTERM: the runner's children stop, no replay starts, and the report is of what was served); the job
# fails where the serve or the replay does, where a replay was asked for and its tokens were not compared, or where a
# replayed token or step differs from the served one: the report then names the first token and the first step that
# differ.
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
tax=$here/../serve_tax
CLIENT=corpus; REPORT=; REPLAYED=
while [ $# -gt 0 ]; do
  flag=$1; shift
  [ "$flag" != --sampled ] || { SAMPLED=1; continue; }
  [ $# -gt 0 ] || { echo "prod_repro: $flag takes a value"; exit 2; }
  case $flag in
    --model) MODEL_ID=$1 ;; --stock) STOCK=$1 ;; --again) AGAIN=$1 ;; --ours) OURS=$1 ;; --load) LOAD=$1 ;;
    --rate) RATE=$1 ;; --requests) REQUESTS=$1 ;; --span-s) SPAN_S=$1 ;; --reps) REPS=$1 ;;
    --max-model-len) MAX_MODEL_LEN=$1 ;; --target) TARGET=$1 ;; --client) CLIENT=$1 ;; --report) REPORT=$1 ;;
    --replay) REPLAYED=$1 ;; --pass-prompts) PASS_PROMPTS=$1 ;; --stop-at) STOP_AT=$1 ;;
    *) echo "prod_repro: unknown flag $flag (--model, --stock, --again, --ours, --load, --rate, --requests, --span-s," \
            "--reps, --replay PASS|none, --pass-prompts NAME=N, --sampled, --stop-at TIME, --max-model-len, --target," \
            "--client corpus|vllm-bench, --report OUT)"; exit 2 ;;
  esac
  shift
done
case $CLIENT in corpus|vllm-bench) ;; *) echo "prod_repro: --client corpus or vllm-bench, not $CLIENT"; exit 2 ;; esac
[ -z "${SAMPLED:-}" ] || [ "$CLIENT" = vllm-bench ] || { echo "prod_repro: --sampled is a pass of vLLM's client (--client vllm-bench): the corpus client's sampling is its arm's"; exit 2; }
[ -d /workspace/cp/venv312 ] && venv=/workspace/cp/venv312 || venv=/workspace/venv312   # as serve_tax/run.sh finds it
VENV=${VENV:-$venv}
[ -x "$VENV/bin/python" ] && PY=$VENV/bin/python || PY=python3
arms=($("$PY" "$here/prod_repro.py" arms --model "${MODEL_ID:-moonlight}" ${REPORT:+--out "$REPORT"})) || exit 2
[ "${AGAIN-}" != none ] || AGAIN=
STOCK=${STOCK:-${arms[0]}}; AGAIN=${AGAIN-${arms[1]}}; OURS=${OURS:-${arms[2]}}
served="$STOCK${AGAIN:+ $AGAIN} $OURS"
REPLAY_MODES=${REPLAY_MODES:-compiled}
# the report of a job's out/: with a replayed pass named, the job asked for a replay, and one that did not end fails
report() {
  local more=()
  [ -z "$REPLAYED" ] || [ "$REPLAYED" = none ] || more+=(--replayed "$REPLAYED" --expect-replay)
  [ -z "${ADVICE_TRACE:-}" ] || more+=(--advice-trace "$ADVICE_TRACE" ${ADVICE_STOPS:+--advice-stops "$ADVICE_STOPS"})
  "$PY" "$here/prod_repro.py" report "$1" --stock "$STOCK" --again "$AGAIN" --ours "$OURS" --load "${LOAD%%=*}" \
    --replay "$1/replay-${REPLAY_MODES%% *}" --target "${TARGET:-1}" ${more[@]+"${more[@]}"}
}
if [ -n "$REPORT" ]; then
  [ -n "${LOAD:-}" ] || { echo "prod_repro: --report OUT --load NAME: the load the job served"; exit 2; }
  report "$REPORT"; exit $?
fi
D=${RESEARCH_RUN_DIR:?run this under research run (or read an earlier job: --report OUT --load NAME)}; OUT=$D/out
export MODEL_ID=${MODEL_ID:-moonlight} CLIENT ${MAX_MODEL_LEN:+MAX_MODEL_LEN} ${PASS_PROMPTS:+PASS_PROMPTS} ${STOP_AT:+STOP_AT} \
  ${SAMPLED:+SAMPLED}
[ -z "${STOP_AT:-}" ] || stop_at=$(date -d "$STOP_AT" +%s) || { echo "prod_repro: --stop-at $STOP_AT is not a time"; exit 2; }
unset REPLAY REPLAY_FROM   # the replay is this script's own step, below
t0=$(date +%s)
# SIGTERM (the job's time ending) reaches the runner and its arm too, which stop; this script goes on to the report
termed=0; trap 'termed=1' TERM

# The replay, a step of its own: the benchmark's runner in its replay-only form, under ours' own general plugins and
# environment (replay-env), whose site and files are kept outside the run's directory and removed
replay_step() {
  local kept rc=0
  [ -f "$OUT/$OURS/$REPLAYED/rep-1/meta.json" ] || { echo "prod_repro: no replay: $OURS served no rep of $REPLAYED"; return 1; }
  if [ -n "${STOP_AT:-}" ] && [ "$(date +%s)" -ge "$stop_at" ]; then
    echo "prod_repro: no replay: --stop-at $STOP_AT has passed"; return 1
  fi
  local left=${MAX_MIN:+$(( MAX_MIN - ($(date +%s) - t0) / 60 ))}
  [ "${left:-1}" -ge 1 ] || { echo "prod_repro: no replay: the job's MAX_MIN=$MAX_MIN minutes are spent"; return 1; }
  kept=$(mktemp -d) || return 2
  "$PY" "$here/prod_repro.py" replay-env --ours "$OURS" --kept "$kept" > "$kept/env" || { rm -rf "$kept"; return 2; }
  [ ! -f "$OUT/engine_env.txt" ] || cp "$OUT/engine_env.txt" "$OUT/engine_env.arms.txt"   # the runner writes it anew
  (
    while IFS= read -r kv; do export "$kv"; done < "$kept/env"
    unset ARMS LOADS RATES VLLM_PLUGINS
    export CLIENT=corpus REPLAY=$REPLAYED REPLAY_FROM=$OUT/$OURS REPLAY_MODES ${left:+MAX_MIN=$left}
    bash "$tax/run.sh"
  ) || rc=$?
  "$PY" "$here/prod_repro.py" replay-ran --ours "$OURS" --kept "$kept" ${PRIVATE:+--private "$PRIVATE"} --out "$OUT" || rc=1
  rm -rf "$kept"
  return $rc
}

case $CLIENT in
  corpus)
    LOAD=${LOAD:-burst}; REPLAYED=${REPLAYED:-$LOAD}
    [ -n "${SPAN_S:-}" ] || REQUESTS=${REQUESTS:-800}
    ( export ARMS="$served" RATES="$LOAD=${RATE:-1000}" REPS=${REPS:-3} ${REQUESTS:+REQUESTS} ${SPAN_S:+SPAN_S}
      bash "$tax/run.sh" ); rc=$? ;;
  vllm-bench)
    [ -n "${LOAD:-}" ] || { echo "prod_repro: --client vllm-bench --load NAME[=RUNG+RUNG...]: one of the benchmark's"; exit 2; }
    REPLAYED=${REPLAYED:-none}
    bash "$tax/run.sh" --model "$MODEL_ID" --arms "$served" --loads "$LOAD" --reps "${REPS:-6}"; rc=$? ;;
esac
[ $rc = 0 ] || echo "prod_repro: the serving cost benchmark's runner ended $rc: an arm or its own cost step failed (see above; the report below says how far each got)"
[ -d "$OUT" ] || { echo "prod_repro: nothing was served (no $OUT): no report"; exit $(( rc ? rc : 1 )); }
if [ $termed = 1 ]; then
  echo "prod_repro: ended early (SIGTERM): no replay starts; the report is of what was served"
  # an arm that was serving stops its server and then writes its record: wait for it, a little
  for _ in $(seq "${TERM_WAIT_S:-30}"); do
    grep -L '"server_rc"' "$OUT"/*/meta.json 2>/dev/null | grep -q . || break
    sleep 1
  done
  rc=$(( rc ? rc : 143 ))
elif [ "$REPLAYED" != none ]; then
  replay_step; r=$?
  [ $r = 0 ] || echo "prod_repro: the replay step ended $r (see above; the report below says how far it got)"
  rc=$(( rc ? rc : r ))
fi
report "$OUT"; r=$?
exit $(( rc ? rc : r ))
