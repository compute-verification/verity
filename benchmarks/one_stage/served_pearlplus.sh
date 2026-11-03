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
#      retained (`serve --no-retain`, unretained/), whose wall is the option's without writing the passes, its coins an
#      in-process challenger's, since no verdict reads them; every serve with sm_90's BF16 rows off unless
#      PEARLC_BF16_ROWS says otherwise (e2e.BF16_RACE), as each serve line says;
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
#      answer.json, which the record would compose as the window's PoUW answer. Its coin, recorded in passes/coin.json
#      before the replay, leaves the passes the moment it is drawn (`served_verdicts.py coin`: PIECE_DIR/coin.json, put
#      in the evidence store, preserved where the node holds a store key and else in the node's own store under the same
#      id, its answer in PIECE_DIR/coin-put.json);
#   5. `served_verdicts.py pearlplus` writes the piece (PIECE_DIR/development.json): its verdict, with each pass's salt
#      held to the serve's live session (`pearlplus_coins`: the session fetched through PEARLC_CHALLENGER_GATE, under
#      H1_KEY_SHA512 and MAIN_FILE's line; with in-process coins it claims no verdict) and an accept to the published
#      coin (`pearlplus_coin`), its load, the replay's node and the rounds dropped, Pearl+'s verdict's credit,
#      its shares (with the unretained serve's beside them) and why it credits nothing (served_verdicts.PEARLPLUS_WHY: its
#      passes are not the window sampled proofs audited, so WINDOW names nothing here).
# The passes hold the run's activations, so they stay in SERVE_DIR/private, under served_verdicts.sh's `research keep`.
# env: served_verdicts.sh's (VENV, HF_HOME, CUDA_HOME), PEARLPLUS_* above. Its lines name no window. Exit 0 once step 5
# wrote the piece.
set -uo pipefail
mkdir -p "${3:?PIECE_DIR}" && SD=$(cd "${1:?SERVE_DIR}" && pwd) && P=$(cd "$3" && pwd) || exit 1
cd "$(dirname "$0")/../.." || exit 1
TREE=$PWD TPY=$PWD/.venv/bin/python
[ -x "$TPY" ] || TPY=${PY:-python3}
Q=$SD/private/pearlplus SPEC=${PEARLPLUS_SPEC:-benchmarks/specs/llama32-1b-1gpu-pouw-protocol-served-h200.toml}
FAST=${PEARLPLUS_FAST-1}
rm -rf "$Q" && (umask 077; mkdir -p "$Q") || exit 1
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

t=$(date +%s)
( export RESEARCH_RUN_DIR=$Q
  # shellcheck source=/dev/null
  . benchmarks/harness/setup/pearl_c_sm90.sh && printf '%s\n' "$PEARLC_KERNEL" "$PEARLC_CUBIN" > "$Q/setup.paths" ) \
  > "$Q/setup.log" 2>&1
rc=$?
if [ "$rc" = 0 ] && [ -s "$Q/setup.paths" ]; then
  { read -r PEARLC_KERNEL; read -r PEARLC_CUBIN; } < "$Q/setup.paths"
  export PEARLC_KERNEL PEARLC_CUBIN
  harness=$("$TPY" -c 'import json, sys
d = json.load(open(sys.argv[1]))
print("pass" if d.get("ok") else "FAIL, " + str(d.get("error")).split(".cubin: ", 1)[-1][:160])' \
    "${PEARLC_CUBIN%/*}/harness_gate.json" 2>/dev/null) || harness="no harness_gate.json read"
  say "cubin: pearl_c_sm90.sh built it and the SASS gates build.sh enforces passed after $(took "$t") (cubin sha256" \
    "$(sha256sum "$PEARLC_CUBIN" | cut -c1-16), debit cubin $(sha256sum "${PEARLC_CUBIN%/*}/debit/debit_sm90.cubin" | cut -c1-16));" \
    "the harness's .FTZ and FP32 MUFU gate, recorded and not enforced on sm_90a: $harness"
else
  say "cubin: pearl_c_sm90.sh failed (exit $rc after $(took "$t")): $(tail -1 "$Q/setup.log" | cut -c1-400)"
fi

serve() {  # LOG ARGS...: served_passes.py serve of the piece's load under its own lease
  local log=$1; shift
  GPU_LEASE_WHO=${GPU_LEASE_WHO:-vllm-pearlplus} gpu-lease 1 --wait --max-min "${PEARLPLUS_LEASE_MIN:-30}" -- \
    "$VENV/bin/python" benchmarks/pouw/pearl_c_vllm/served_passes.py serve --spec "$SPEC" \
    --requests "${PEARLPLUS_REQUESTS:-32}" --prompt-tokens "${PEARLPLUS_PROMPT_TOKENS:-1024}" \
    --output-tokens "${PEARLPLUS_OUTPUT_TOKENS:-1}" --max-num-batched-tokens "${PEARLPLUS_T:-8192}" "$@" > "$Q/$log" 2>&1
}

if [ -n "${PEARLC_CUBIN:-}" ]; then
  if [ -n "${PEARLPLUS_UNRETAINED-1}" ]; then
    t=$(date +%s)
    ( unset PEARLC_CHALLENGER_GATE PEARLC_CHALLENGER_PEER
      serve serve-unretained.log --no-retain --passes "$Q/unretained" --out "$Q/served-unretained.json" )
    rc=$?
    say "serve, no pass retained: served_passes.py serve --no-retain exit $rc after $(took "$t"), $BF16: $(grep -m1 '^served ' "$Q/serve-unretained.log" | cut -c1-300)"
  fi
  t=$(date +%s)
  serve serve.log --passes "$Q/passes" --out "$Q/served.json"
  rc=$?
  say "serve: served_passes.py serve exit $rc after $(took "$t"), $BF16: $(grep -m1 '^served ' "$Q/serve.log" | cut -c1-300)"
fi

VOLUNTEER=${PEARLPLUS_VOLUNTEER-1} VMIN=${PEARLPLUS_VOLUNTEER_MIN:-60} JOBS=${PEARLPLUS_JOBS:-$(nproc)}
if [ -s "$Q/served.json" ] && [ -n "$VOLUNTEER" ]; then
  t=$(date +%s)
  if [ "$VOLUNTEER" = cpu ]; then
    how="on the CPU"
    timeout "${VMIN}m" "$VENV/bin/python" benchmarks/pouw/pearl_c_vllm/served_passes.py volunteer --passes "$Q/passes" \
      --out "$Q/volunteer.json" --jobs "$JOBS" > "$Q/volunteer.log" 2>&1
  else
    how=--device DEBIT=$(dirname "${PEARLC_CUBIN:-.}")/debit/debit_sm90.cubin
    # shellcheck disable=SC2016
    GPU_LEASE_WHO=${GPU_LEASE_WHO:-vllm-pearlplus} gpu-lease 1 --wait --max-min "$VMIN" -- bash -c '
      "$1/bin/python" benchmarks/pouw/pearl_c_sm90/debit.py --cubin "$3" --out "$2/debit.json" > "$2/debit.log" 2>&1 || {
        echo "debit.py refused the debit cubin on the leased card, so no row is given up: $(tail -1 "$2/debit.log")"
        exit 3; }
      exec "$1/bin/python" benchmarks/pouw/pearl_c_vllm/served_passes.py volunteer --passes "$2/passes" \
        --out "$2/volunteer.json" --jobs "$4" --device "$3" ${5:+--device-check "$5"}' \
      _ "$VENV" "$Q" "$DEBIT" "$JOBS" "${PEARLPLUS_DEVICE_CHECK:-}" > "$Q/volunteer.log" 2>&1
  fi
  rc=$?
  say "volunteer: served_passes.py volunteer $how exit $rc after $(took "$t"): $(tail -1 "$Q/volunteer.log" | cut -c1-400)"
  if [ -s "$Q/volunteer.json" ]; then
    "$VENV/bin/python" benchmarks/pouw/pearl_c_vllm/served_passes.py retain --passes "$Q/passes" \
      --volunteer "$Q/volunteer.json" > "$Q/retain.log" 2>&1
    say "retain: served_passes.py retain exit $?: $(tail -1 "$Q/retain.log" | cut -c1-400)"
  fi
elif [ -s "$Q/served.json" ]; then
  say "volunteer: not run (PEARLPLUS_VOLUNTEER empty): no row over a debit cap is given up before the verifier's coin"
fi

if [ -s "$Q/served.json" ]; then
  t=$(date +%s)
  "$VENV/bin/python" benchmarks/pouw/pearl_c_vllm/served_passes.py verify --passes "$Q/passes" --out "$P/verify.json" \
    --answer "$P/pearlplus-answer.json" --jobs "${PEARLPLUS_JOBS:-$(nproc)}" ${PEARLPLUS_DRAWS:+--draws "$PEARLPLUS_DRAWS"} \
    ${FAST:+--fast} > "$Q/verify.log" 2>&1 &
  V=$!
  # the coin record leaves the passes the moment verify draws it, before its replay: a deleted one would draw afresh
  while [ ! -s "$Q/passes/coin.json" ] && kill -0 "$V" 2>/dev/null; do sleep 1; done
  if [ -s "$Q/passes/coin.json" ]; then
    "$TPY" benchmarks/one_stage/served_verdicts.py coin "$P" --record "$Q/passes/coin.json" --run "${RESEARCH_RUN_ID:-}" \
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
"$TPY" benchmarks/one_stage/served_verdicts.py pearlplus "$P" --served "$Q/served.json" --verify "$P/verify.json" \
  ${UN[@]+"${UN[@]}"} ${PEARLC_CHALLENGER_GATE:+--gate "$PEARLC_CHALLENGER_GATE"} ${H1_KEY_SHA512:+--key-sha512 "$H1_KEY_SHA512"} \
  ${MAIN_FILE:+--main "$MAIN_FILE"} --coin "$P/coin-put.json" > "$Q/pearlplus.log" 2>&1
rc=$?
grep '^POUW-DEV ' "$Q/pearlplus.log"
say "piece: served_verdicts.py pearlplus exit $rc"
exit "$rc"
