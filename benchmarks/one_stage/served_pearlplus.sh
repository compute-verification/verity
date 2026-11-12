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
#      (default benchmarks/harness/specs/llama32-1b-1gpu-pouw-protocol-served-h200.toml: pouw:pearl-c-h100-v1-h3 over Llama 3.2 1B's
#      BF16 checkpoint) a warm-up, then PEARLPLUS_REQUESTS requests (default 32) of PEARLPLUS_PROMPT_TOKENS seeded random
#      tokens (default 1,024) and PEARLPLUS_OUTPUT_TOKENS outputs (default 1) at PEARLPLUS_T tokens a step (default 8,192:
#      a prefill-heavy load, 8 prompts a step), each engine step one pass retained in passes/, its epoch's salt coin a
#      live session's on vy-challenger-1 where PEARLC_CHALLENGER_GATE and PEARLC_CHALLENGER_PEER are set (served_verdicts.sh
#      sets them under POUW=pearlplus; `served_passes.py serve`'s, `bench_arm.live_coins`), else an in-process
#      challenger's; with the gate, the lane's Pearl-C period PEARLPLUS_PERIOD (default pearlplus-RUN_ID) is opened right
#      before that serve (`served_verdicts.py pearlplus-period`: cap 1, its circuit PIECE_DIR/pearlplus-descriptor.json's
#      SHA-512, its answers in PIECE_DIR/period.json), so the serve's salt session joins it and fills its cap, and the gate
#      refuses every other reserve of the lane while it is open; it is closed right before the verifier's draw (step 4,
#      or once the piece exits), so no other reserve of the lane comes between its close and the draw, which the compose
#      takes only as the lane's first reserve after it; before it, unless PEARLPLUS_UNRETAINED is set empty, the same load under its own lease with no pass
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
#      over the whole run, not a sample's, by default exactly while step 4's verdict rejects a run on a drawn row over
#      its cap: its over-cap rule (PEARLPLUS_OVER_CAP, verify's --over-cap; default the verdict's own) is reject, which
#      `served_passes.py volunteer-needed` says before anything is built or served, and under it a PEARLPLUS_VOLUNTEER
#      set empty is refused (exit 2); under uncredited, where the verdict credits such a row nothing, the give-up runs
#      only where PEARLPLUS_VOLUNTEER is set. With PEARLPLUS_VOLUNTEER=1 (the default where it runs) the counts are the
#      card's, under a lease of its own: debit.py first checks step 1's debit cubin against fast_tile on the leased
#      card, and only if it passed the volunteer takes them from it (`--device`), every PEARLPLUS_DEVICE_CHECK-th tile
#      (default volunteer_pass.CROSS_CHECK) and each tile with a word the card marks non-finite replayed on the CPU too
#      and held equal. With PEARLPLUS_VOLUNTEER=cpu it replays every tile on the CPU in PEARLPLUS_JOBS processes
#      instead, which at the default load runs far past POUW_TIMEOUT. It stops at PEARLPLUS_VOLUNTEER_MIN (default 60),
#      and `served_passes.py retain` then drops, still before the coin, each pass it didn't finish or left over a cap:
#      those credit nothing and stay in the declared share;
#   4. `served_passes.py verify` takes Pearl+'s verdict over every retained pass on the CPU in PEARLPLUS_JOBS processes (default
#      nproc): one draw of PEARLPLUS_DRAWS work-weighted tiles (default its own, 459) over all the passes, and each pass's
#      exclusion tiles, replayed on fast_tile's batch forms, the first and every 256th after it on the scalar reference
#      too (a scalar replay takes minutes a tile at the 1B's K), or with PEARLPLUS_FAST set empty on the scalar reference
#      alone, under PEARLPLUS_OVER_CAP's rule where it is set. Its document goes to PIECE_DIR/verify.json and its core
#      answer to PIECE_DIR/pearlplus-answer.json, not answer.json, which the record would compose as the window's PoUW
#      answer. Under the gate its draw source is a draw session's coin on vy-challenger-1, reserved withheld once the
#      period closed, as the lane's first reserve since, and opened on the run's commitment (`bench_arm.LiveDraw`, which
#      refuses before the coin is asked otherwise; verify_run.py's "The run's draw"), else the node's. Its coin, recorded
#      in passes/coin.json before the replay, leaves the passes the moment it is drawn (`served_verdicts.py coin`:
#      PIECE_DIR/coin.json, put in the evidence store, preserved where the node holds a store key and else in the node's own store under the same
#      id, its answer in PIECE_DIR/coin-put.json, which names the run's Attempt carrying out/'s copy as its custody);
#      the RULE line then reads the ledger again: whether the draw session was the lane's first reserve after the
#      period's close (`served_verdicts.py pearlplus-period rule`, kept as period.json's rule);
#   4b. once verify wrote its document, `served_passes.py lean` takes the Lean verdict of record, PouwCheck.PearlPlus's
#      pouw-pearlplus (PEARLPLUS_LEAN_EXE, else built here as served_pouw.sh builds pouw-period, its log
#      pouw-pearlplus-build.log; one that doesn't build answers nothing), on the same passes, epoch coins, draws, coin
#      record and W (--tie), of record (--of-record) on a live session's coins, within PEARLPLUS_LEAN_TIMEOUT seconds
#      (default 3,600), and writes its answer to PIECE_DIR/pearlplus-lean.json,
#      the file `served_verdicts.pearlplus_of_record` reads, only where it is the Python verdict's; both verdicts, its
#      command line and the executable's SHA-256 go to PIECE_DIR/pearlplus-lean-check.json either way;
#   4c. where the Lean answered, `served_passes.py case` writes the parts of the run's case as remote's composite of
#      record reads it that need no registration record, and the pass files its opened tiles read, to
#      SERVE_DIR/private/pearlplus/case, where served_cases.py makes the case once the window's record R is made;
#   5. `served_verdicts.py pearlplus` writes the piece (PIECE_DIR/development.json): its verdict, with each pass's salt
#      held to the serve's live session (`pearlplus_coins`: the session fetched through PEARLC_CHALLENGER_GATE, under
#      H1_KEY_SHA512 and MAIN_FILE's line; with in-process coins it claims no verdict) and an accept to the published
#      coin (`pearlplus_coin`), its load, the replay's node and the rounds dropped, Pearl+'s verdict's credit,
#      its shares (with the unretained serve's beside them) and why it credits nothing (served_verdicts.PEARLPLUS_WHY: its
#      passes are not the window sampled proofs audited, so WINDOW names nothing here).
# With PEARLPLUS_TIE set (served_verdicts.sh's PEARLPLUS_TIE: the window itself served under serve-openai's Pearl+ arm,
# `serve_pearl`, one Pearl+ epoch a Commit window in SERVE_DIR/pearl/w<k>/), the piece serves nothing of its own: step 1
# takes the window serve's cubin (PEARLC_CUBIN) where it is set; step 2 takes window WINDOW's own epoch, passes/ a
# symlink to SERVE_DIR/pearl/wWINDOW (its served.json, coins.json and owner-only pass-NNNNN/, round s the window's step
# s), and goes no further unless that served.json says ok; steps 3 and 4 read and write there, `retain` writing Pearl+'s
# window record W (`--window-record`, private/pearlplus/window-record.json, rounds dropped as above) and then, before the coin,
# PEARLPLUS_TIE_CMD (`served_verdicts.sh pearl-tie SERVE_DIR WINDOW W PASSES`) committing the window tied to W, its
# registration R chained right after W; step 4 is `verify --tie W`, which holds the passes to W, and under the gate it
# draws only after the window serve's Pearl-C period (SERVE_DIR/pearlplus/period.json, the one its RULE line reads)
# closed with the lane's reserves since its open the window's salt sessions alone. Its salt coin is the
# arm's: a live salt-only session of the window's own under the gate (`serve_pearl.live`), which step 5 holds each pass
# to as above, else in-process, and the piece then claims no verdict. Step 5 gets W (`--tie W`): the piece is then the
# window's own epoch, and why it credits nothing by itself names the Pearl-C period its salt session joined
# (served_verdicts.pearlplus_why_tied).
# The passes hold the run's activations, so they stay in SERVE_DIR/private, under served_verdicts.sh's `research keep`,
# and the piece runs under umask 077, so every file it or a step writes there is 0600.
# env: served_verdicts.sh's (VENV, HF_HOME, CUDA_HOME), PEARLPLUS_* above. Its lines name no window. Exit 0 once step 5
# wrote the piece.
set -uo pipefail
mkdir -p "${3:?PIECE_DIR}" && SD=$(cd "${1:?SERVE_DIR}" && pwd) && P=$(cd "$3" && pwd) || exit 1
cd "$(dirname "$0")/../.." || exit 1
TREE=$PWD TPY=$PWD/.venv/bin/python
[ -x "$TPY" ] || TPY=${PY:-python3}
Q=$SD/private/pearlplus SPEC=${PEARLPLUS_SPEC:-benchmarks/harness/specs/llama32-1b-1gpu-pouw-protocol-served-h200.toml}
FAST=${PEARLPLUS_FAST-1}
# every file the piece writes is its owner's alone, the passes' files a step rewrites too: a manifest holds the salt key
umask 077
rm -rf "$Q" && mkdir -p "$Q" || exit 1
say() { echo "POUW $*"; }
took() { echo "$(( ($(date +%s) - $1) / 60 )) min"; }
export VERITY_BENCH_VENV=${VERITY_BENCH_VENV:-${VENV:-/workspace/cp/venv312}}
# shellcheck source=/dev/null
. benchmarks/harness/env.sh
NEEDED=$("$VENV/bin/python" benchmarks/pouw/pearl_c_vllm/served_passes.py volunteer-needed \
  ${PEARLPLUS_OVER_CAP:+--over-cap "$PEARLPLUS_OVER_CAP"} 2> "$Q/volunteer-needed.log")
RULE=${NEEDED#* } VOLUNTEER=${PEARLPLUS_VOLUNTEER-auto}
case ${NEEDED%% *} in
  yes) [ "$VOLUNTEER" = auto ] && VOLUNTEER=1
    if [ -z "$VOLUNTEER" ]; then
      say "volunteer: refused, so nothing is served: Pearl+'s verdict rejects a run on a drawn row over its debit cap" \
        "(over-cap rule $RULE), so the give-up before its coin can't be skipped (PEARLPLUS_VOLUNTEER is empty)"
      exit 2
    fi ;;
  no) [ "$VOLUNTEER" = auto ] && VOLUNTEER='' ;;
  *) say "volunteer: served_passes.py volunteer-needed answered '$NEEDED', so nothing is served:" \
       "$(tail -1 "$Q/volunteer-needed.log" | cut -c1-300)"
     exit 2 ;;
esac
# Pearl+ served as vLLM serves it, not under the window serve's batch invariance
unset VLLM_BATCH_INVARIANT
# sm_90's BF16-rows path off unless PEARLC_BF16_ROWS says otherwise: a whole capture forking each call's A tree and A
# scales races on it (e2e.BF16_RACE); the protocol arm served here captures no forward whole, the one path that reads it
export PEARLC_BF16_ROWS=${PEARLC_BF16_ROWS:-0}
BF16="sm_90's BF16 rows $([ "$PEARLC_BF16_ROWS" = 0 ] && echo off || echo on) (PEARLC_BF16_ROWS=$PEARLC_BF16_ROWS)"

TIE=${PEARLPLUS_TIE:-} K=${2:?WINDOW} SERVED=$Q/served.json W=$Q/window-record.json GO=1
[ -z "$TIE" ] || SERVED=$Q/passes/served.json

t=$(date +%s)
if [ -n "$TIE" ] && [ -s "${PEARLC_CUBIN:-}" ]; then  # the window serve's, built before its lease
  printf '%s\n' "$PEARLC_KERNEL" "$PEARLC_CUBIN" > "$Q/setup.paths"
  rc=0
else
  ( export RESEARCH_RUN_DIR=$Q
    # shellcheck source=/dev/null
    . benchmarks/harness/setup/pearl_c_sm90.sh && printf '%s\n' "$PEARLC_KERNEL" "$PEARLC_CUBIN" > "$Q/setup.paths" ) \
    > "$Q/setup.log" 2>&1
  rc=$?
fi
if [ -n "$TIE" ] && [ "$rc" = 0 ]; then
  read -r PEARLC_KERNEL < "$Q/setup.paths"
  say "cubin: the window serve's (served_verdicts.sh's PEARLPLUS-CUBIN, cubin sha256 $(sha256sum "$PEARLC_CUBIN" | cut -c1-16))"
elif [ "$rc" = 0 ] && [ -s "$Q/setup.paths" ]; then
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
PERIOD_OPEN=''
period() {  # open|close: the lane's Pearl-C period from the retained serve to the draw (served_verdicts.py pearlplus-period)
  if [ "$1" = open ]; then
    "$TPY" benchmarks/one_stage/served_verdicts.py pearlplus-period open "$P" --gate "$PEARLC_CHALLENGER_GATE" \
      --name "${PEARLPLUS_PERIOD:-pearlplus-${RESEARCH_RUN_ID:-local-$(date -u +%Y%m%dT%H%M%SZ)}}" \
      --scheme "${PEARLC_SCHEME:-pearl-c-h100-v1-h3}" ${PEARLC_CUBIN:+--cubin "$PEARLC_CUBIN"} > "$Q/period.log" 2>&1 \
      && PERIOD_OPEN=1
  else
    [ -n "$PERIOD_OPEN" ] || return 0
    "$TPY" benchmarks/one_stage/served_verdicts.py pearlplus-period close "$P" --gate "$PEARLC_CHALLENGER_GATE" \
      >> "$Q/period.log" 2>&1 && PERIOD_OPEN=''
  fi
  say "period: $(tail -1 "$Q/period.log" | cut -c1-400)"
}
trap 'period close; copy_back' EXIT
trap 'STOPPED=1; exit 143' TERM HUP
trap 'STOPPED=1; exit 130' INT

if [ -n "$TIE" ]; then  # the window's own epoch, which serve-openai's Pearl+ arm served and retained
  E=$SD/pearl/w$K
  if [ -s "$E/served.json" ] && ln -s "$E" "$Q/passes"; then
    line=$("$TPY" -c 'import json, sys
d = json.load(open(sys.argv[1]))
print(d.get("passes_with_calls"), "passes with calls of its Commit window'\''s", d.get("commit_steps"), "steps, coins",
      (d.get("coins") or {}).get("source"), "->", "OK" if d.get("ok") else "FAIL, " + str(d.get("error"))[:300])
sys.exit(0 if d.get("ok") else 1)' "$E/served.json" 2>&1)
    rc=$?
    say "window: the window's own Pearl+ epoch $E (serve_pearl), $BF16: $(tail -1 <<< "$line" | cut -c1-400)"
    [ "$rc" = 0 ] || GO=
  else
    say "window: the window serve wrote no Pearl+ epoch $E/served.json (serve_pearl), so there are no passes to tie"
    GO=
  fi
elif [ -n "${PEARLC_CUBIN:-}" ]; then
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
  [ -n "${PEARLC_CHALLENGER_GATE:-}" ] && period open
  t=$(date +%s)
  serve serve.log --passes "$Q/passes" --out "$Q/served.json"
  rc=$?
  say "serve: served_passes.py serve exit $rc after $(took "$t"), $BF16: $(grep -m1 '^served ' "$Q/serve.log" | cut -c1-300)"
fi

VMIN=${PEARLPLUS_VOLUNTEER_MIN:-60} JOBS=${PEARLPLUS_JOBS:-$(nproc)}
[ -s "$SERVED" ] || GO=
if [ -n "$GO" ] && [ -n "$VOLUNTEER" ]; then
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
elif [ -n "$GO" ]; then
  say "volunteer: not run: Pearl+'s verdict credits a drawn row over its debit cap nothing (over-cap rule $RULE), so" \
    "no row is given up before the verifier's coin"
fi
if [ -n "$GO" ] && { [ -s "$Q/volunteer.json" ] || [ -n "$TIE" ]; }; then
  VA=()
  [ -s "$Q/volunteer.json" ] && VA=(--volunteer "$Q/volunteer.json")
  [ -z "$TIE" ] || VA+=(--window-record "$W")
  "$VENV/bin/python" benchmarks/pouw/pearl_c_vllm/served_passes.py retain --passes "$Q/passes" "${VA[@]}" > "$Q/retain.log" 2>&1
  rc=$?
  say "retain: served_passes.py retain exit $rc: $(grep -m1 '^retain ' "$Q/retain.log" | cut -c1-400)"
  [ "$rc" = 0 ] || [ -z "$TIE" ] || GO=
fi
# the window commit tied to W, before Pearl+'s coin, as H1's is made at its window's close before its sample coin
if [ -n "$GO" ] && [ -n "$TIE" ]; then
  if [ -s "$W" ]; then
    say "W: $(grep -m1 '^retain W: ' "$Q/retain.log" | sed -E 's/^retain W: [^,]*, //' | cut -c1-300)"
    t=$(date +%s)
    (cd "$TREE" && bash -c "${PEARLPLUS_TIE_CMD:?served_verdicts.sh sets PEARLPLUS_TIE_CMD}" _ "$SD" "$K" "$W" "$Q/passes") \
      > "$Q/tie.log" 2>&1
    rc=$?
    say "tie: served_verdicts.sh pearl-tie exit $rc after $(took "$t"): $(tail -1 "$Q/tie.log" | cut -c1-600)"
  else
    say "W: retain wrote no window record, so nothing ties the passes to the window"
  fi
fi

PD=$P
[ -z "$TIE" ] || PD=$SD/pearlplus
if [ -n "$GO" ] && [ -n "$TIE" ] && [ -n "${PEARLC_CHALLENGER_GATE:-}" ]; then  # the window serve closed the period
  held=$("$TPY" -c 'import json, sys
d = json.load(open(sys.argv[1]))
print("ok" if (d.get("reserves") or {}).get("ok") is True and d.get("close") else "")' "$PD/period.json" 2>/dev/null)
  [ "$held" = ok ] || { say "draw: refused before it: the window serve's Pearl-C period didn't close with the lane's" \
    "reserves since its open the window's salt sessions alone (served_verdicts.sh's PEARLPLUS-PERIOD line), so no draw" \
    "follows it"; GO=; }
fi
if [ -n "$GO" ]; then
  period close
  t=$(date +%s)
  TA=()
  [ -n "$TIE" ] && [ -s "$W" ] && TA=(--tie "$W")
  "$VENV/bin/python" benchmarks/pouw/pearl_c_vllm/served_passes.py verify --passes "$Q/passes" --out "$P/verify.json" \
    --answer "$P/pearlplus-answer.json" --jobs "${PEARLPLUS_JOBS:-$(nproc)}" ${PEARLPLUS_DRAWS:+--draws "$PEARLPLUS_DRAWS"} \
    ${PEARLPLUS_OVER_CAP:+--over-cap "$PEARLPLUS_OVER_CAP"} ${FAST:+--fast} ${TA[@]+"${TA[@]}"} > "$Q/verify.log" 2>&1 &
  V=$!
  # the coin record leaves the passes the moment verify draws it, before its replay: a deleted one would draw afresh
  while [ ! -s "$Q/passes/coin.json" ] && kill -0 "$V" 2>/dev/null; do sleep 1; done
  if [ -s "$Q/passes/coin.json" ]; then
    "$TPY" benchmarks/one_stage/served_verdicts.py coin "$P" --record "$Q/passes/coin.json" --run "${RESEARCH_RUN_ID:-}" \
      ${RESEARCH_RUN_DIR:+--run-dir "$RESEARCH_RUN_DIR"} > "$Q/coin.log" 2>&1
    say "coin: $(tail -1 "$Q/coin.log" | cut -c1-400)"
    if [ -n "${PEARLC_CHALLENGER_GATE:-}" ] && [ -s "$PD/period.json" ]; then
      "$TPY" benchmarks/one_stage/served_verdicts.py pearlplus-period rule "$PD" --gate "$PEARLC_CHALLENGER_GATE" \
        --lane "${PEARLC_CHALLENGER_LANE:-vllm}" --coin "$Q/passes/coin.json" >> "$Q/period.log" 2>&1
      say "$(tail -1 "$Q/period.log" | cut -c1-400)"
    fi
  else
    say "coin: served_passes.py verify drew none (it refused the run before its coin)"
  fi
  wait "$V"
  rc=$?
  say "verdict: served_passes.py verify exit $rc after $(took "$t"): $(tail -1 "$Q/verify.log" | cut -c1-400)"
fi

rm -f "$P/pearlplus-lean.json"
if [ -n "$GO" ] && [ -s "$P/verify.json" ]; then
  t=$(date +%s)
  LX=${PEARLPLUS_LEAN_EXE:-}
  if [ -z "$LX" ]; then
    LX=$TREE/verity/protocols/pouw/verifier/lean/pearlplus/.lake/build/bin/pouw-pearlplus
    (cd verity/protocols/pouw/verifier/lean/pearlplus && lake build pouw-pearlplus) > "$Q/pouw-pearlplus-build.log" 2>&1 || {
      say "lean: pouw-pearlplus doesn't build (its log in the serve dir's private/pearlplus), so no Lean answer is written"
      LX=; }
  fi
  [ -n "$LX" ] && [ -x "$LX" ] || LX=
  "$VENV/bin/python" benchmarks/pouw/pearl_c_vllm/served_passes.py lean --passes "$Q/passes" --verify "$P/verify.json" \
    --exe "${LX:-none}" --out "$P/pearlplus-lean.json" --timeout "${PEARLPLUS_LEAN_TIMEOUT:-3600}" > "$Q/lean.log" 2>&1
  say "lean: served_passes.py lean exit $? after $(took "$t"): $(tail -1 "$Q/lean.log" | cut -c1-400)"
fi
if [ -s "$P/pearlplus-lean.json" ]; then
  t=$(date +%s)
  "$VENV/bin/python" benchmarks/pouw/pearl_c_vllm/served_passes.py case --passes "$Q/passes" --verify "$P/verify.json" \
    --lean "$P/pearlplus-lean.json" --out "$Q/case" > "$Q/case.log" 2>&1
  say "case: served_passes.py case exit $? after $(took "$t"): $(tail -1 "$Q/case.log" | cut -c1-400)"
fi

UN=()
[ -s "$Q/served-unretained.json" ] && UN=(--unretained "$Q/served-unretained.json")
[ -s "$Q/volunteer.json" ] && UN+=(--volunteer "$Q/volunteer.json")
[ -z "$TIE" ] || UN+=(--tie "$W")
"$TPY" benchmarks/one_stage/served_verdicts.py pearlplus "$P" --served "$SERVED" --verify "$P/verify.json" \
  ${UN[@]+"${UN[@]}"} ${PEARLC_CHALLENGER_GATE:+--gate "$PEARLC_CHALLENGER_GATE"} ${H1_KEY_SHA512:+--key-sha512 "$H1_KEY_SHA512"} \
  ${MAIN_FILE:+--main "$MAIN_FILE"} --coin "$P/coin-put.json" > "$Q/pearlplus.log" 2>&1
rc=$?
copy_back
grep '^POUW-DEV ' "$Q/pearlplus.log"
say "piece: served_verdicts.py pearlplus exit $rc"
exit "$rc"
