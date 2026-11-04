#!/usr/bin/env bash
# PoUW's piece with Pearl+, `served_verdicts.sh`'s POUW_CMD under POUW=pearlplus (H1 off): Pearl+'s own served run on the
# card after the window's serve, every pass retained, and Pearl+'s verdict over all of them:
#   POUW_CMD='bash benchmarks/one_stage/served_pearlplus.sh "$1" "$2" "$3"'      (served_verdicts.sh runs it in the tree)
#   bash benchmarks/one_stage/served_pearlplus.sh SERVE_DIR WINDOW PIECE_DIR
# Everything but PIECE_DIR's files goes to SERVE_DIR/private/pearlplus, each step's log among it:
#   1. benchmarks/harness/setup/pearl_c_sm90.sh builds pearl_c_sm90's cubin and its debit cubin (debit/debit_sm90.cubin)
#      from this tree, with the SASS gates build.sh enforces; the cubin line also names the harness's .FTZ and FP32 MUFU
#      gate's verdict (harness_gate.json), which build.sh records and doesn't enforce, its pins being sm_120a's;
#   2. under `gpu-lease 1` (PEARLPLUS_LEASE_MIN, default 30), `served_passes.py serve` serves PEARLPLUS_SPEC's Pearl-C arm
#      (default benchmarks/specs/llama32-1b-1gpu-pouw-protocol-served-h200.toml: pouw:pearl-c-h100-v1-h3 over Llama 3.2 1B's
#      BF16 checkpoint) a warm-up, then PEARLPLUS_REQUESTS requests (default 32) of PEARLPLUS_PROMPT_TOKENS seeded random
#      tokens (default 1,024) and PEARLPLUS_OUTPUT_TOKENS outputs (default 1) at PEARLPLUS_T tokens a step (default 8,192:
#      a prefill-heavy load, 8 prompts a step), each engine step one pass retained in passes/, its epoch's salt coin a
#      live session's on vy-challenger-1 where PEARLC_CHALLENGER_GATE and PEARLC_CHALLENGER_PEER are set (served_verdicts.sh
#      sets them under POUW=pearlplus; `served_passes.py serve`'s, `bench_arm.live_coins`), else an in-process
#      challenger's; before it, unless PEARLPLUS_UNRETAINED is set empty, the same load under its own lease with no pass
#      retained (`serve --no-retain`, unretained/), whose wall is the option's without writing the passes, its coins and
#      salt key seeded (`serve --seeded`), since no verdict reads them; then, unless PEARLPLUS_SAME is set empty, that
#      serve once more (same/) and `served_passes.py same` holding the two to the same commitments pass by pass (each
#      pass's transcript root, activation roots and counts), which they give only where the served path commits the same
#      bytes for the same load (PIECE_DIR/same.json: the fields that differ, never their values); every serve with
#      sm_90's BF16 rows off unless PEARLC_BF16_ROWS says otherwise (e2e.BF16_RACE), as each serve line says. The
#      retained serve's passes/ is a symlink to a 0700 `mktemp -d` directory in RAM, under PEARLPLUS_PASSES_RAM (default
#      /dev/shm; set empty, they go to disk), which the serve, the give-up, retain and verify all read and write through,
#      where that filesystem has room for PEARLPLUS_PASSES_GB (default 2 GB a 1,024 tokens of the load) and the node's
#      available memory holds that beside PEARLPLUS_RAM_HEADROOM_GB (default 100: the serve's and the give-up's own),
#      else on disk, the passes line saying which; after the piece, off the serve's clock, they are copied back to disk
#      (`cp -a`, their modes kept, the symlink replaced once the copy is whole) and the RAM copy removed, a stopped piece
#      leaving them in RAM, its line naming where;
#   3. after the serve's lease and before the verifier's coin, `served_passes.py volunteer` gives up, in the passes'
#      records, the rows over a debit cap (V-EX), which the serve's screen can't see, from every tile's words' counts
#      over the whole run, not a sample's. With PEARLPLUS_VOLUNTEER=1 (the default) the counts are the card's, under a
#      lease of its own: debit.py first checks step 1's debit cubin against fast_tile on the leased card, and only if
#      it passed the volunteer takes them from it (`--device`), every PEARLPLUS_DEVICE_CHECK-th tile (default
#      volunteer_pass.CROSS_CHECK) and each tile with a word the card marks non-finite replayed on the CPU too and held
#      equal. With PEARLPLUS_VOLUNTEER=cpu it replays every tile on the CPU in PEARLPLUS_JOBS processes instead, which
#      at the default load runs far past POUW_TIMEOUT; set empty, it doesn't run and the verdict may draw a tile over
#      its cap. It stops at PEARLPLUS_VOLUNTEER_MIN (default 60), and `served_passes.py retain` then drops, still
#      before the coin, each pass it didn't finish or left over a cap: those credit nothing and stay in the declared
#      share;
#   4. `served_passes.py verify` takes Pearl+'s verdict over every retained pass on the CPU in PEARLPLUS_JOBS processes (default
#      nproc): one draw of PEARLPLUS_DRAWS work-weighted tiles (default its own, 459) over all the passes, and each pass's
#      exclusion tiles, replayed on fast_tile's batch forms, the first and every 256th after it on the scalar reference
#      too (a scalar replay takes minutes a tile at the 1B's K), or with PEARLPLUS_FAST set empty on the scalar reference
#      alone. Its document goes to PIECE_DIR/verify.json and its core answer to PIECE_DIR/pearlplus-answer.json, not
#      answer.json, which the record would compose as the window's PoUW answer. Its coin is, where PEARLC_CHALLENGER_GATE
#      is set, a draw session's of its own reserved withheld there as PEARLC_CHALLENGER_PEER and opened on the run's
#      commitment (`--draw-gate`, `pearl_c.draw`), else the node's; recorded in passes/coin.json before the replay, it
#      leaves the passes the moment it is drawn (`served_verdicts.py coin`: PIECE_DIR/coin.json, put in the evidence
#      store, preserved where the node holds a store key and else in the node's own store under the same id, its answer
#      in PIECE_DIR/coin-put.json);
#   5. `served_verdicts.py pearlplus` writes the piece (PIECE_DIR/development.json): its verdict, with each pass's salt
#      held to the serve's live session (`pearlplus_coins`: the session fetched through PEARLC_CHALLENGER_GATE, under
#      H1_KEY_SHA512 and MAIN_FILE's line; with in-process coins it claims no verdict) and an accept to the published
#      coin (`pearlplus_coin`) and that coin to its draw session, reserved for PEARLC_CHALLENGER_LANE
#      (`pearlplus_draw`), its load, the replay's node and the rounds dropped, Pearl+'s verdict's credit,
#      its shares (with the unretained serve's beside them) and why it credits nothing (served_verdicts.PEARLPLUS_WHY: its
#      passes are not the window sampled proofs audited, so WINDOW names nothing here).
# Under served_verdicts.sh's PEARL_TIE=1 the window's own serve is Pearl+'s served run (serve-openai's Pearl+ arm, its
# passes the window's steps in SERVE_DIR/pearl/wWINDOW/), and the steps go three ways:
#   bash benchmarks/one_stage/served_pearlplus.sh arm DIR                    (before the window's serve: step 1 in DIR,
#        DIR/setup.paths the kernel and the cubin, then `served_passes.py arm` writes DIR/protocols-config.json from
#        PEARLPLUS_SPEC's Pearl-C arm, serve-openai's --protocols-config for its Pearl+ arm; exit 0 iff it wrote it)
#   bash benchmarks/one_stage/served_pearlplus.sh give-up SERVE_DIR WINDOW   (once the window is picked, before any coin of
#        Pearl+'s: step 3 on the window's passes, its files in SERVE_DIR/private/pearl-give-up, the card's counts with
#        the debit cubin beside PEARLC_CUBIN, and then retain, run whatever the volunteer did, writes Pearl+'s window
#        record W, pearl/wWINDOW/window-record.json, over the rounds it keeps; a window whose arm wrote no served.json or
#        an error there gets none; exit 0 iff it wrote W)
# and the piece is steps 4 and 5 alone over those passes: verify holds them to W (`--tie W`, no verify without W), and
# `served_verdicts.py pearlplus --tie W` holds its verdict to W's digest (`pearlplus`'s ``tie``).
# The passes hold the run's activations, so they stay in SERVE_DIR/private, under served_verdicts.sh's `research keep`,
# and the piece runs under umask 077, so every file it or a step writes there is 0600.
# env: served_verdicts.sh's (VENV, HF_HOME, CUDA_HOME, PEARL_TIE), PEARLPLUS_* above. Its lines name no window. Exit 0
# once step 5 wrote the piece.
set -uo pipefail
MODE=piece
case ${1:-} in arm | give-up) MODE=$1; shift ;; esac
case $MODE in
  arm) mkdir -p "${1:?DIR}" && Q=$(cd "$1" && pwd) || exit 1 ;;
  give-up) SD=$(cd "${1:?SERVE_DIR}" && pwd) && Q=$SD/private/pearl-give-up PASSES=$SD/pearl/w${2:?WINDOW} || exit 1 ;;
  *) mkdir -p "${3:?PIECE_DIR}" && SD=$(cd "${1:?SERVE_DIR}" && pwd) && P=$(cd "$3" && pwd) || exit 1
     Q=$SD/private/pearlplus PASSES=$SD/private/pearlplus/passes
     [ -z "${PEARL_TIE:-}" ] || PASSES=$SD/pearl/w${2:?WINDOW} ;;
esac
# the piece's own serve (steps 1-3 in full) only untied; the arm has no passes
UNTIED=$([ "$MODE" = piece ] && [ -z "${PEARL_TIE:-}" ] && echo 1)
[ "$MODE" = arm ] || SERVED=$PASSES/served.json W=$PASSES/window-record.json
[ -z "$UNTIED" ] || SERVED=$Q/served.json
cd "$(dirname "$0")/../.." || exit 1
TREE=$PWD TPY=$PWD/.venv/bin/python
[ -x "$TPY" ] || TPY=${PY:-python3}
SPEC=${PEARLPLUS_SPEC:-benchmarks/specs/llama32-1b-1gpu-pouw-protocol-served-h200.toml}
FAST=${PEARLPLUS_FAST-1}
# every file the piece writes is its owner's alone, the passes' files a step rewrites too: a manifest holds the salt key
umask 077
[ "$MODE" = arm ] || { rm -rf "$Q" && mkdir -p "$Q"; } || exit 1
say() { echo "POUW $*"; }
took() { echo "$(( ($(date +%s) - $1) / 60 )) min"; }
export VERITY_BENCH_VENV=${VERITY_BENCH_VENV:-${VENV:-/workspace/cp/venv312}}
# shellcheck source=/dev/null
. benchmarks/harness/env.sh
# Pearl+ served as vLLM serves it, not under the window serve's batch invariance
unset VLLM_BATCH_INVARIANT
# sm_90's BF16-rows path off unless PEARLC_BF16_ROWS says otherwise: a whole capture forking each call's A tree and A
# scales races on it (e2e.BF16_RACE); the protocol arm served here captures no forward whole, the one path that reads it
export PEARLC_BF16_ROWS=${PEARLC_BF16_ROWS:-0}
BF16="sm_90's BF16 rows $([ "$PEARLC_BF16_ROWS" = 0 ] && echo off || echo on) (PEARLC_BF16_ROWS=$PEARLC_BF16_ROWS)"

cubin() {  # step 1 in $Q: PEARLC_KERNEL and PEARLC_CUBIN exported and in $Q/setup.paths; 0 iff built
  local t rc harness
  t=$(date +%s)
  rm -f "$Q/setup.paths"
  ( export RESEARCH_RUN_DIR=$Q
    # shellcheck source=/dev/null
    . benchmarks/harness/setup/pearl_c_sm90.sh && printf '%s\n' "$PEARLC_KERNEL" "$PEARLC_CUBIN" > "$Q/setup.paths" ) \
    > "$Q/setup.log" 2>&1
  rc=$?
  if [ "$rc" != 0 ] || [ ! -s "$Q/setup.paths" ]; then
    say "cubin: pearl_c_sm90.sh failed (exit $rc after $(took "$t")): $(tail -1 "$Q/setup.log" | cut -c1-400)"
    return 1
  fi
  { read -r PEARLC_KERNEL; read -r PEARLC_CUBIN; } < "$Q/setup.paths"
  export PEARLC_KERNEL PEARLC_CUBIN
  harness=$("$TPY" -c 'import json, sys
d = json.load(open(sys.argv[1]))
print("pass" if d.get("ok") else "FAIL, " + str(d.get("error")).split(".cubin: ", 1)[-1][:160])' \
    "${PEARLC_CUBIN%/*}/harness_gate.json" 2>/dev/null) || harness="no harness_gate.json read"
  say "cubin: pearl_c_sm90.sh built it and the SASS gates build.sh enforces passed after $(took "$t") (cubin sha256" \
    "$(sha256sum "$PEARLC_CUBIN" | cut -c1-16), debit cubin $(sha256sum "${PEARLC_CUBIN%/*}/debit/debit_sm90.cubin" | cut -c1-16));" \
    "the harness's .FTZ and FP32 MUFU gate, recorded and not enforced on sm_90a: $harness"
}
if [ "$MODE" = arm ]; then
  rm -f "$Q/protocols-config.json"
  cubin || exit 1
  "$VENV/bin/python" benchmarks/pouw/pearl_c_vllm/served_passes.py arm --spec "$SPEC" --out "$Q/protocols-config.json" \
    > "$Q/arm.log" 2>&1
  rc=$? line=$(tail -1 "$Q/arm.log")
  say "arm: served_passes.py arm exit $rc: $(echo "${line//"$Q/"/}" | cut -c1-400)"
  [ "$rc" = 0 ] && [ -s "$Q/protocols-config.json" ]
  exit $?
fi
[ -z "$UNTIED" ] || cubin

serve() {  # LOG ARGS...: served_passes.py serve of the piece's load under its own lease
  local log=$1; shift
  GPU_LEASE_WHO=${GPU_LEASE_WHO:-vllm-pearlplus} gpu-lease 1 --wait --max-min "${PEARLPLUS_LEASE_MIN:-30}" -- \
    "$VENV/bin/python" benchmarks/pouw/pearl_c_vllm/served_passes.py serve --spec "$SPEC" \
    --requests "${PEARLPLUS_REQUESTS:-32}" --prompt-tokens "${PEARLPLUS_PROMPT_TOKENS:-1024}" \
    --output-tokens "${PEARLPLUS_OUTPUT_TOKENS:-1}" --max-num-batched-tokens "${PEARLPLUS_T:-8192}" "$@" > "$Q/$log" 2>&1
}

RAM=${PEARLPLUS_PASSES_RAM-/dev/shm} R='' STOPPED=''
NEED=${PEARLPLUS_PASSES_GB:-$(( (${PEARLPLUS_REQUESTS:-32} * (${PEARLPLUS_PROMPT_TOKENS:-1024} + ${PEARLPLUS_OUTPUT_TOKENS:-1}) * 2 + 1023) / 1024 ))}
HEADROOM=${PEARLPLUS_RAM_HEADROOM_GB:-100}
passes_ram() {  # the retained serve's passes/ a symlink into RAM where the node holds them beside the headroom, else on disk
  [ -n "$RAM" ] || { say "passes: on disk (PEARLPLUS_PASSES_RAM empty)"; return; }
  local shm mem
  shm=$(df -B1 --output=avail "$RAM" 2>/dev/null | tail -1)
  shm=${shm//[^0-9]/}
  shm=$(( ${shm:-0} / 1000000000 ))
  mem=$(awk '/^MemAvailable:/ {print int($2 * 1024 / 1e9)}' /proc/meminfo)
  if [ "$shm" -ge "$NEED" ] && [ "${mem:-0}" -ge $(( NEED + HEADROOM )) ] && R=$(mktemp -d -p "$RAM" pearlplus-passes.XXXXXX) \
    && ln -s "$R" "$Q/passes"; then
    say "passes: in RAM at $R, symlinked as passes/ ($RAM has $shm GB free and the node $mem GB of memory available, for" \
      "about $NEED GB of passes beside $HEADROOM GB of headroom); copied back to disk after the piece"
  else
    [ -n "$R" ] && rmdir "$R"
    R=''
    say "passes: on disk, not in RAM: $RAM has ${shm} GB free and the node ${mem:-no} GB of memory available, short of" \
      "about $NEED GB of passes (PEARLPLUS_PASSES_GB) beside $HEADROOM GB of headroom (PEARLPLUS_RAM_HEADROOM_GB)"
  fi
}
copy_back() {  # passes/ from RAM to disk, modes kept, the symlink replaced only once the copy is whole
  [ -n "$R" ] && [ -L "$Q/passes" ] || return 0
  local t
  t=$(date +%s)
  if [ -n "$STOPPED" ]; then
    rm -rf "$Q/passes.disk"
    say "passes: left in RAM at $R, symlinked as passes/: the piece was stopped before copying them back"
  elif cp -a "$R" "$Q/passes.disk"; then
    rm "$Q/passes" && mv "$Q/passes.disk" "$Q/passes" && rm -rf "$R"
    say "passes: copied from RAM to disk in $(( $(date +%s) - t )) s ($(du -sh "$Q/passes" | cut -f1)), off the serve's clock"
  else
    rm -rf "$Q/passes.disk"
    say "passes: copying them from RAM to disk failed, so they stay in RAM at $R, symlinked as passes/"
  fi
  R=''
}
trap copy_back EXIT
trap 'STOPPED=1; exit 143' TERM HUP
trap 'STOPPED=1; exit 130' INT

if [ -n "$UNTIED" ] && [ -n "${PEARLC_CUBIN:-}" ]; then
  if [ -n "${PEARLPLUS_UNRETAINED-1}" ]; then
    t=$(date +%s)
    ( unset PEARLC_CHALLENGER_GATE PEARLC_CHALLENGER_PEER
      serve serve-unretained.log --no-retain --seeded --passes "$Q/unretained" --out "$Q/served-unretained.json" )
    rc=$?
    say "serve, no pass retained: served_passes.py serve --no-retain --seeded exit $rc after $(took "$t"), $BF16: $(grep -m1 '^served ' "$Q/serve-unretained.log" | cut -c1-300)"
    if [ -n "${PEARLPLUS_SAME-1}" ]; then
      t=$(date +%s)
      ( unset PEARLC_CHALLENGER_GATE PEARLC_CHALLENGER_PEER
        serve serve-same.log --no-retain --seeded --passes "$Q/same" --out "$Q/served-same.json" )
      rc=$?
      "$VENV/bin/python" benchmarks/pouw/pearl_c_vllm/served_passes.py same "$Q/served-unretained.json" \
        "$Q/served-same.json" --out "$P/same.json" > "$Q/same.log" 2>&1
      sc=$?
      say "same: served_passes.py serve --no-retain --seeded once more exit $rc and served_passes.py same exit $sc after" \
        "$(took "$t"), $BF16: $(tail -1 "$Q/same.log" | cut -c1-400)"
    fi
  fi
  passes_ram
  t=$(date +%s)
  serve serve.log --passes "$Q/passes" --out "$Q/served.json"
  rc=$?
  say "serve: served_passes.py serve exit $rc after $(took "$t"), $BF16: $(grep -m1 '^served ' "$Q/serve.log" | cut -c1-300)"
fi

VOLUNTEER=${PEARLPLUS_VOLUNTEER-1} VMIN=${PEARLPLUS_VOLUNTEER_MIN:-60} JOBS=${PEARLPLUS_JOBS:-$(nproc)}
if [ "$MODE" = give-up ]; then  # the window's epoch as serve-openai's arm closed it: one Pearl+ round a Commit step
  why=$("$TPY" -c 'import json, sys
d = json.load(open(sys.argv[1]))
print("" if d.get("ok", True) else d.get("error") or "its served.json says it failed")' "$SERVED" 2>&1) \
    || why="no served passes of the window (serve-openai's Pearl+ arm wrote no served.json for it)"
  [ -z "$why" ] || { say "give-up: none, and no W: $(echo "$why" | tail -1 | cut -c1-400)"; exit 1; }
fi
if [ -n "$UNTIED" ] || [ "$MODE" = give-up ]; then
  if [ -s "$SERVED" ] && [ -n "$VOLUNTEER" ]; then
    t=$(date +%s)
    if [ "$VOLUNTEER" = cpu ]; then
      how="on the CPU"
      timeout "${VMIN}m" "$VENV/bin/python" benchmarks/pouw/pearl_c_vllm/served_passes.py volunteer --passes "$PASSES" \
        --out "$Q/volunteer.json" --jobs "$JOBS" > "$Q/volunteer.log" 2>&1
    else
      how=--device DEBIT=$(dirname "${PEARLC_CUBIN:-.}")/debit/debit_sm90.cubin
      # shellcheck disable=SC2016
      GPU_LEASE_WHO=${GPU_LEASE_WHO:-vllm-pearlplus} gpu-lease 1 --wait --max-min "$VMIN" -- bash -c '
        "$1/bin/python" benchmarks/pouw/pearl_c_sm90/debit.py --cubin "$3" --out "$2/debit.json" > "$2/debit.log" 2>&1 || {
          echo "debit.py refused the debit cubin on the leased card, so no row is given up: $(tail -1 "$2/debit.log")"
          exit 3; }
        exec "$1/bin/python" benchmarks/pouw/pearl_c_vllm/served_passes.py volunteer --passes "$6" \
          --out "$2/volunteer.json" --jobs "$4" --device "$3" ${5:+--device-check "$5"}' \
        _ "$VENV" "$Q" "$DEBIT" "$JOBS" "${PEARLPLUS_DEVICE_CHECK:-}" "$PASSES" > "$Q/volunteer.log" 2>&1
    fi
    rc=$?
    say "volunteer: served_passes.py volunteer $how exit $rc after $(took "$t"): $(tail -1 "$Q/volunteer.log" | cut -c1-400)"
  elif [ -s "$SERVED" ]; then
    say "volunteer: not run (PEARLPLUS_VOLUNTEER empty): no row over a debit cap is given up before the verifier's coin"
  fi
  # untied, only after a volunteer; under the tie always, since retain is what writes W
  RETAIN=()
  [ -s "$Q/volunteer.json" ] && RETAIN=(--volunteer "$Q/volunteer.json")
  [ "$MODE" = give-up ] && RETAIN+=(--window-record "$W")
  if [ -s "$SERVED" ] && [ "${#RETAIN[@]}" -gt 0 ]; then
    "$VENV/bin/python" benchmarks/pouw/pearl_c_vllm/served_passes.py retain --passes "$PASSES" "${RETAIN[@]}" \
      > "$Q/retain.log" 2>&1
    rc=$? line=$(tail -1 "$Q/retain.log")
    say "retain: served_passes.py retain exit $rc: $(echo "${line//"$PASSES/"/}" | cut -c1-400)"
  fi
fi
if [ "$MODE" = give-up ]; then
  [ -s "$W" ] || { say "W: none written, so nothing ties the window's passes to a commit (retain's log in the serve dir)"; exit 1; }
  exit 0
fi

if [ -n "${PEARL_TIE:-}" ] && [ ! -s "$W" ]; then
  say "verdict: served_passes.py verify not run: the window's passes have no Pearl+ window record W to hold them to" \
    "(served_pearlplus.sh give-up wrote none), so no coin is drawn on them"
elif [ -s "$SERVED" ]; then
  t=$(date +%s)
  DRAW=() TIE=()
  [ -n "${PEARLC_CHALLENGER_GATE:-}" ] && DRAW=(--draw-gate "$PEARLC_CHALLENGER_GATE" --draw-peer "${PEARLC_CHALLENGER_PEER:-}")
  [ -n "$UNTIED" ] || TIE=(--tie "$W")
  "$VENV/bin/python" benchmarks/pouw/pearl_c_vllm/served_passes.py verify --passes "$PASSES" --out "$P/verify.json" \
    --answer "$P/pearlplus-answer.json" --jobs "$JOBS" ${PEARLPLUS_DRAWS:+--draws "$PEARLPLUS_DRAWS"} \
    ${FAST:+--fast} ${TIE[@]+"${TIE[@]}"} ${DRAW[@]+"${DRAW[@]}"} > "$Q/verify.log" 2>&1 &
  V=$!
  # the coin record leaves the passes the moment verify draws it, before its replay: a deleted one would draw afresh
  while [ ! -s "$PASSES/coin.json" ] && kill -0 "$V" 2>/dev/null; do sleep 1; done
  if [ -s "$PASSES/coin.json" ]; then
    "$TPY" benchmarks/one_stage/served_verdicts.py coin "$P" --record "$PASSES/coin.json" --run "${RESEARCH_RUN_ID:-}" \
      > "$Q/coin.log" 2>&1
    say "coin: $(tail -1 "$Q/coin.log" | cut -c1-400)"
  else
    say "coin: served_passes.py verify drew none (it refused the run before its coin)"
  fi
  wait "$V"
  rc=$?
  say "verdict: served_passes.py verify exit $rc after $(took "$t"): $(tail -1 "$Q/verify.log" | cut -c1-400)"
fi

UN=()
[ -s "$Q/served-unretained.json" ] && UN=(--unretained "$Q/served-unretained.json")
[ -s "$Q/volunteer.json" ] && UN+=(--volunteer "$Q/volunteer.json")
[ -z "${PEARL_TIE:-}" ] || {
  UN+=(--tie "$W")
  [ ! -s "$SD/private/pearl-give-up/volunteer.json" ] || UN+=(--volunteer "$SD/private/pearl-give-up/volunteer.json"); }
"$TPY" benchmarks/one_stage/served_verdicts.py pearlplus "$P" --served "$SERVED" --verify "$P/verify.json" \
  ${UN[@]+"${UN[@]}"} ${PEARLC_CHALLENGER_GATE:+--gate "$PEARLC_CHALLENGER_GATE"} ${H1_KEY_SHA512:+--key-sha512 "$H1_KEY_SHA512"} \
  ${MAIN_FILE:+--main "$MAIN_FILE"} ${PEARLC_CHALLENGER_LANE:+--lane "$PEARLC_CHALLENGER_LANE"} --coin "$P/coin-put.json" \
  > "$Q/pearlplus.log" 2>&1
rc=$?
copy_back
grep '^POUW-DEV ' "$Q/pearlplus.log"
say "piece: served_verdicts.py pearlplus exit $rc"
exit "$rc"
