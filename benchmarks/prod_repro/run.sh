#!/usr/bin/env bash
# vLLM at its serving settings, replayed (README.md): serve stock and ours at the same settings and load, replay the
# window ours served, and print the settings and three lines (tokens, cost, advice). From the tree's root, as the
# command of a `research run` on a GPU node, the C4 shard sent with --send (or C4=PATH) unless a ready plan is
# (PLAN_FILE):
#   verity bench prod_repro [--model moonlight] [--stock ARM] [--again ARM|none] [--ours ARM] [--load burst]
#     [--rate 1000] [--requests 800 | --span-s S] [--reps 3] [--replay PASS|none] [--max-model-len N] [--target 1]
#   verity bench prod_repro --client vllm-bench --load NAME[=RUNG+RUNG...] [--reps 6] [--replay NAME@RUNG]
#     [--pass-prompts NAME=N] [--sampled] [--stop-at TIME] [--lifetimes L] [--leave-out N] [--arms "ARM ARM ..."]
#     [--model ...] [--stock ...] ...
#   verity bench prod_repro --report OUT --load NAME [--replay PASS] [--stock ARM] [--again ARM|none] [--ours ARM]
#     [--arms "ARM ARM ..."] [--target 1]
#   verity bench prod_repro --replay-of OUT --load NAME --replay PASS [--model ...] [--ours ARM]
# The serve, the load and the replay are the serving cost benchmark's (benchmarks/serve_tax/run.sh, whose header says
# what a node needs and what its other variables do: they pass through, among them C4, PLAN_FILE, PRIVATE, VENV,
# CHECKPOINT, FLASHINFER, LDCONFIG, MAX_MIN and BUDGET_MIN). The arms are by default the arms at vLLM's published
# settings (prod, prod-again, prod-record) for a model whose recipe the benchmark holds, else stock compiled vLLM,
# stock again and the step record (prod_repro.py arms), each a server lifetime of its own; --again none serves stock
# once, and the cost then has no noise floor; --stock none serves ours alone, and there is then no cost. --arms names
# every arm the job serves, in the order it serves them: ours and, unless --stock none, stock are among them; stock
# again is served only where it is among them; any other is served beside them, an arm that differs from stock or
# from ours in one thing (the benchmark's prod-seeded and prod-trace), and is in no line of the three: the report
# says how far it got and lists its passes with every arm's. The context
# limit is --max-model-len's for every arm, by default the benchmark's for the arm (at the recipe's settings, the
# recipe's for the cards); limits.json holds what is known of leaving it to vLLM, which the report's first line says.
# By default the corpus client sends the load: REQUESTS requests of the plan (about 60% of its
# prompts behind one of eight shared prefixes) arriving at RATE a second, faster than the engine serves, each rep from
# an empty prefix cache; then the replay re-runs rep 1 of ours under its recorded schedule, in the mode it was served
# in (REPLAY_MODES, by default compiled). With --client vllm-bench vLLM's own client sends the load at its rungs, which
# gives the benchmark's paired cost; it replays nothing unless --replay names the pass of ours to re-run, as the
# benchmark names it (NAME@RUNG), which needs an arm of ours that keeps each request's prompt ids and served ids.
# --pass-prompts, --sampled and --stop-at are the benchmark's PASS_PROMPTS (the prompts a pass sends), SAMPLED (each
# request sampled at the recipe's temperature and top-p) and STOP_AT (an arm starts no pass it can't finish by then;
# the replay runs after the arms whatever the time, and the job's end is its lease's, MAX_MIN).
# --lifetimes and --leave-out are the benchmark's LIFETIMES and LEAVE_OUT: the arms are served L times, each time a
# server of its own, lifetime K of an arm in OUT/<arm>.K, and each lifetime's first N passes, its warm-up, are left
# out of every figure. A sessions load's cost has its interval over the lifetimes and so its verdict only with two
# at least; the replay re-runs the named pass of ours's first lifetime.
# The replay runs under the arms' lease, after them. An arm of ours that serves on the canonical DSA top-k (prod-record)
# is replayed on it, and keeps files on the node: PRIVATE names a directory for them outside every run's (the corpus
# client needs it for such an arm), from which the report reads the top-k's counters and keeps them in
# OUT/topk_counters.json.
# The advice line is of the pass the replay re-runs: what its schedule costs as advice, by the scheduler replay's
# loader (benchmarks/network_e2e/encodings/replay.py).
# The --replay-of form serves nothing: it copies ours's directory from an earlier job's out/ into this job's and
# re-runs the replay of the named pass from it, under a lease the benchmark's runner takes for the replay alone; its
# report is of ours alone, with no cost.
# The --report form reads an earlier job's out/ and needs no GPU. Every form ends with the report (prod_repro.py
# report), written to OUT/prod_repro.{json,txt}, which says how far each arm and the replay got, also where the job is
# ended early (SIGTERM: the runner's children stop, and the report is of what was served and replayed by then); the
# job fails where the serve or the replay does, where a replay was asked for and its tokens were not compared, or
# where a replayed token or step differs from the served one: the report then names the first token and the first
# step that differ.
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
tax=$here/../serve_tax
CLIENT=corpus; REPORT=; REPLAYED=; REPLAY_OF=
while [ $# -gt 0 ]; do
  flag=$1; shift
  [ "$flag" != --sampled ] || { SAMPLED=1; continue; }
  [ $# -gt 0 ] || { echo "prod_repro: $flag takes a value"; exit 2; }
  case $flag in
    --model) MODEL_ID=$1 ;; --stock) STOCK=$1 ;; --again) AGAIN=$1 ;; --ours) OURS=$1 ;; --load) LOAD=$1 ;;
    --rate) RATE=$1 ;; --requests) REQUESTS=$1 ;; --span-s) SPAN_S=$1 ;; --reps) REPS=$1 ;;
    --max-model-len) MAX_MODEL_LEN=$1 ;; --target) TARGET=$1 ;; --client) CLIENT=$1 ;; --report) REPORT=$1 ;;
    --replay) REPLAYED=$1 ;; --pass-prompts) PASS_PROMPTS=$1 ;; --stop-at) STOP_AT=$1 ;; --replay-of) REPLAY_OF=$1 ;;
    --lifetimes) LIFETIMES=$1 ;; --leave-out) LEAVE_OUT=$1 ;; --arms) SERVE=$1 ;;
    *) echo "prod_repro: unknown flag $flag (--model, --stock, --again, --ours, --arms, --load, --rate, --requests, --span-s," \
            "--reps, --replay PASS|none, --pass-prompts NAME=N, --sampled, --stop-at TIME, --lifetimes L, --leave-out N," \
            "--max-model-len, --target, --client corpus|vllm-bench, --report OUT, --replay-of OUT)"; exit 2 ;;
  esac
  shift
done
case $CLIENT in corpus|vllm-bench) ;; *) echo "prod_repro: --client corpus or vllm-bench, not $CLIENT"; exit 2 ;; esac
[ -z "${SAMPLED:-}" ] || [ "$CLIENT" = vllm-bench ] || { echo "prod_repro: --sampled is a pass of vLLM's client (--client vllm-bench): the corpus client's sampling is its arm's"; exit 2; }
[ -d /workspace/cp/venv312 ] && venv=/workspace/cp/venv312 || venv=/workspace/venv312   # as serve_tax/run.sh finds it
VENV=${VENV:-$venv}
[ -x "$VENV/bin/python" ] && PY=$VENV/bin/python || PY=python3
arms=($("$PY" "$here/prod_repro.py" arms --model "${MODEL_ID:-moonlight}" ${REPORT:+--out "$REPORT"})) || exit 2
# --stock none serves ours alone: a first start of it and its replay, with no stock arm, so no cost
alone=0; [ "${STOCK-}" != none ] || { alone=1; AGAIN=none; }
[ "${AGAIN-}" != none ] || AGAIN=
if [ $alone = 1 ]; then STOCK=; else STOCK=${STOCK:-${arms[0]}}; fi
AGAIN=${AGAIN-${arms[1]}}; OURS=${OURS:-${arms[2]}}
served="${STOCK:+$STOCK }${AGAIN:+$AGAIN }$OURS"
# --arms: every arm served, in its order; the ones that are none of the three are served beside them
BESIDE=()
if [ -n "${SERVE:-}" ]; then
  [ -z "$REPLAY_OF" ] || { echo "prod_repro: --replay-of serves nothing: --arms is of a job that serves its arms"; exit 2; }
  for a in $OURS $STOCK; do
    case " $SERVE " in *" $a "*) ;; *) echo "prod_repro: --arms \"$SERVE\" holds no $a: it names every arm served, ours and stock among them (--stock, --ours; --stock none for ours alone)"; exit 2 ;; esac
  done
  [ -z "$AGAIN" ] || case " $SERVE " in *" $AGAIN "*) ;; *) AGAIN= ;; esac
  for a in $SERVE; do
    case " $STOCK $AGAIN $OURS " in *" $a "*) ;; *) BESIDE+=("$a") ;; esac
  done
  served=$SERVE
fi
REPLAY_MODES=${REPLAY_MODES:-compiled}
# the report of a job's out/: with a replayed pass named, the job asked for a replay, and one that did not end fails;
# the job's own report also reads what its processes kept in PRIVATE
report() {
  local more=()
  [ "${2:-}" != job ] || [ -z "${PRIVATE:-}" ] || more+=(--private "$PRIVATE")
  [ -z "$REPLAYED" ] || [ "$REPLAYED" = none ] || more+=(--replayed "$REPLAYED" --expect-replay)
  [ ${#BESIDE[@]} = 0 ] || more+=(--beside "${BESIDE[@]}")
  "$PY" "$here/prod_repro.py" report "$1" --stock "$STOCK" --again "$AGAIN" --ours "$OURS" --load "${LOAD%%=*}" \
    --replay "$1/replay-${REPLAY_MODES%% *}" --target "${TARGET:-1}" ${more[@]+"${more[@]}"}
}
if [ -n "$REPORT" ]; then
  [ -n "${LOAD:-}" ] || { echo "prod_repro: --report OUT --load NAME: the load the job served"; exit 2; }
  report "$REPORT"; exit $?
fi
D=${RESEARCH_RUN_DIR:?run this under research run (or read an earlier job: --report OUT --load NAME)}; OUT=$D/out
MODEL_ID=${MODEL_ID:-moonlight}
export MODEL_ID CLIENT ${MAX_MODEL_LEN:+MAX_MODEL_LEN} ${PASS_PROMPTS:+PASS_PROMPTS} ${STOP_AT:+STOP_AT} ${SAMPLED:+SAMPLED} \
  ${LIFETIMES:+LIFETIMES} ${LEAVE_OUT:+LEAVE_OUT}
[[ "${LIFETIMES:-1}" =~ ^[1-9][0-9]*$ ]] || { echo "prod_repro: --lifetimes $LIFETIMES: how many times the arms are served, 1 or more"; exit 2; }
[ -z "$REPLAY_OF" ] || [ "${LIFETIMES:-1}" = 1 ] || { echo "prod_repro: --replay-of serves nothing: --lifetimes is of a job that serves its arms"; exit 2; }
# ours's directory the replay reads: its first lifetime's where the arms are served in lifetimes
ours_dir=$OURS; [ "${LIFETIMES:-1}" = 1 ] || ours_dir=$OURS.1
[ -z "${STOP_AT:-}" ] || date -d "$STOP_AT" +%s > /dev/null || { echo "prod_repro: --stop-at $STOP_AT is not a time"; exit 2; }
# the pass of ours the replay re-runs: REPLAY, REPLAY_FROM and REPLAY_MODES are the benchmark's runner's
replays() {
  if [ "$REPLAYED" = none ]; then unset REPLAY REPLAY_FROM; else export REPLAY=$REPLAYED REPLAY_FROM=$OUT/$ours_dir REPLAY_MODES; fi
}
# SIGTERM (the job's time ending) reaches the runner and its arm too, which stop; this script goes on to the report
termed=0; trap 'termed=1' TERM
if [ -n "$REPLAY_OF" ]; then
  # no arm is served: the benchmark's runner replays alone where it is given no arms and the directory of an arm that
  # kept each request's ids, which here is the earlier job's, copied so that this job's out/ holds what its report reads
  [ -n "${LOAD:-}" ] && [ -n "$REPLAYED" ] && [ "$REPLAYED" != none ] || {
    echo "prod_repro: --replay-of OUT --load NAME --replay PASS: the load the earlier job served and the pass of it to re-run"; exit 2; }
  [ -f "$REPLAY_OF/$OURS/meta.json" ] || {
    echo "prod_repro: --replay-of $REPLAY_OF holds no $OURS/meta.json: name the out/ of a job that served $OURS"; exit 2; }
  if [ -e "$OUT/$OURS" ]; then
    cmp -s "$REPLAY_OF/$OURS/meta.json" "$OUT/$OURS/meta.json" || {
      echo "prod_repro: $OUT/$OURS is there already and is not $REPLAY_OF's: --replay-of is a job of its own"; exit 2; }
  else
    mkdir -p "$OUT" && cp -r "$REPLAY_OF/$OURS" "$OUT/$OURS" || { echo "prod_repro: could not copy $REPLAY_OF/$OURS into $OUT"; exit 2; }
  fi
  # the canonical top-k's counters of the arm's serving are the earlier job's; its replay's are not this replay's
  [ ! -f "$REPLAY_OF/topk_counters.json" ] || "$PY" -c 'import json, sys
was = json.load(open(sys.argv[1]))
json.dump({"replay": None, "served": was.get("served")} | {k: was[k] for k in ("dirs",) if k in was}, open(sys.argv[2], "w"), indent=1)' \
    "$REPLAY_OF/topk_counters.json" "$OUT/topk_counters.json" || { echo "prod_repro: could not read $REPLAY_OF/topk_counters.json"; exit 2; }
  STOCK=; AGAIN=; alone=0
  ( unset ARMS LOADS; export CLIENT=corpus REPLAY=$REPLAYED REPLAY_FROM=$OUT/$OURS REPLAY_MODES; bash "$tax/run.sh" ); rc=$?
else
case $CLIENT in
  corpus)
    LOAD=${LOAD:-burst}; REPLAYED=${REPLAYED:-$LOAD}; replays
    [ -n "${SPAN_S:-}" ] || REQUESTS=${REQUESTS:-800}
    export ARMS="$served" RATES="$LOAD=${RATE:-1000}" REPS=${REPS:-3} ${REQUESTS:+REQUESTS} ${SPAN_S:+SPAN_S}
    bash "$tax/run.sh"; rc=$? ;;
  vllm-bench)
    [ -n "${LOAD:-}" ] || { echo "prod_repro: --client vllm-bench --load NAME[=RUNG+RUNG...]: one of the benchmark's"; exit 2; }
    REPLAYED=${REPLAYED:-none}; replays
    bash "$tax/run.sh" --model "$MODEL_ID" --arms "$served" --loads "$LOAD" --reps "${REPS:-6}"; rc=$? ;;
esac
fi
# with no stock arm, the benchmark's runner under vLLM's client still ends on its own cost step, which then has
# nothing to be over and fails after everything else: where ours served its load without an error, the job's exit is
# the report's, which fails where the replay did not end or differs
if [ $alone = 1 ] && [ "$CLIENT" = vllm-bench ] && [ $rc = 1 ] && "$PY" "$here/prod_repro.py" served "$OUT" "$OURS" --load "${LOAD%%=*}"; then
  echo "prod_repro: no stock arm was served (--stock none), so the benchmark's own cost step had nothing to be over and" \
    "ended 1; $OURS served its load without an error, and the job's exit is the report's"
  rc=0
fi
[ $rc = 0 ] || echo "prod_repro: the serving cost benchmark's runner ended $rc: an arm, the replay or its own cost step failed (see above; the report below says how far each got)"
[ -d "$OUT" ] || { echo "prod_repro: nothing was served (no $OUT): no report"; exit $(( rc ? rc : 1 )); }
if [ $termed = 1 ]; then
  echo "prod_repro: ended early (SIGTERM): the report is of what was served and replayed by then"
  # an arm that was serving stops its server and then writes its record: wait for it, a little
  for _ in $(seq "${TERM_WAIT_S:-30}"); do
    grep -L '"server_rc"' "$OUT"/*/meta.json 2>/dev/null | grep -q . || break
    sleep 1
  done
  rc=$(( rc ? rc : 143 ))
fi
report "$OUT" job; r=$?
exit $(( rc ? rc : r ))
