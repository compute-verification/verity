#!/usr/bin/env bash
# PoUW's piece with Pearl+, `served_verdicts.sh`'s POUW_CMD under POUW=pearlplus and PEARLPLUS_TIE (H1 off): Pearl+'s
# verdict over the window's own epoch, which serve-openai's Pearl+ served and retained (attached by name,
# `serve_openai.scheme_settings`, one Pearl+ epoch a Commit window in SERVE_DIR/pearl/w<k>/), held to Pearl+'s window
# record W. The piece serves nothing of its own:
#   POUW_CMD='bash benchmarks/one_stage/served_pearlplus.sh "$1" "$2" "$3"'      (served_verdicts.sh runs it in the tree)
#   bash benchmarks/one_stage/served_pearlplus.sh SERVE_DIR WINDOW PIECE_DIR
# Everything but PIECE_DIR's files goes to SERVE_DIR/private/pearlplus, each step's log among it:
#   1. the cubin: the window serve's (PEARLC_CUBIN, built before its lease) where it is set, else
#      benchmarks/harness/setup/pearl_c_sm90.sh builds pearl_c_sm90's cubin and its debit cubin (debit/debit_sm90.cubin)
#      from this tree, with the SASS gates build.sh enforces; the cubin line also names the harness's .FTZ and FP32 MUFU
#      gate's verdict (harness_gate.json), which build.sh records and doesn't enforce, its pins being sm_120a's;
#   2. window WINDOW's own epoch: passes/ a symlink to SERVE_DIR/pearl/wWINDOW (its served.json, coins.json and
#      owner-only pass-NNNNN/, round s the window's step s); the piece goes no further unless that served.json says ok;
#   3. before the verifier's coin, `served_passes.py volunteer` gives up, in the passes'
#      records, the rows over a debit cap (V-EX), which the serve's screen can't see, from every tile's words' counts
#      over the whole run, not a sample's, by default exactly while step 4's verdict rejects a run on a sampled row over
#      its cap: its over-cap rule (PEARLPLUS_OVER_CAP, verify's --over-cap; default the verdict's own) is reject, which
#      `served_passes.py volunteer-needed` says before anything is built, and under it a PEARLPLUS_VOLUNTEER
#      set empty is refused (exit 2); under uncredited, where the verdict credits such a row nothing, the give-up runs
#      only where PEARLPLUS_VOLUNTEER is set. With PEARLPLUS_VOLUNTEER=1 (the default where it runs) the counts are the
#      card's, under a lease of its own: debit.py first checks step 1's debit cubin against fast_tile on the leased
#      card, and only if it passed the volunteer takes them from it (`--device`), every PEARLPLUS_DEVICE_CHECK-th tile
#      (default volunteer_pass.CROSS_CHECK) and each tile with a word the card marks non-finite replayed on the CPU too
#      and held equal. With PEARLPLUS_VOLUNTEER=cpu it replays every tile on the CPU in PEARLPLUS_JOBS processes
#      instead, which at the default load runs far past POUW_TIMEOUT. It stops at PEARLPLUS_VOLUNTEER_MIN (default 60),
#      and `served_passes.py retain` then drops, still before the coin, each pass it didn't finish or left over a cap:
#      those credit nothing and stay in the declared share. `retain` writes Pearl+'s window record W (`--window-record`,
#      private/pearlplus/window-record.json, rounds dropped as above), chained after the serve's first registration
#      record where the serve wrote one (`--previous-record`: SERVE_DIR/deployment-registration.json, served_verdicts.sh's
#      weights(), which registers the weights before the first request) and naming the day the serve declared where it
#      declared one (`--day-declaration`: SERVE_DIR/declaration.json, serve-openai's --day), and then, before the coin,
#      PEARLPLUS_TIE_CMD (`served_verdicts.sh pearl-tie SERVE_DIR WINDOW W PASSES`) commits the window tied to W, its
#      docket R chained right after W. Where that commit fails, or `retain` wrote no W, the piece goes no further: the
#      commit is what holds the window to its class's fixed load, so without it there is no sample, no verdict and no
#      Pearl+ line, and step 5 writes a piece that rejects for want of one;
#   4. `served_passes.py verify --tie W` takes Pearl+'s verdict over every retained pass on the CPU in PEARLPLUS_JOBS
#      processes (default nproc), holding the passes to W, with PEARLPLUS_CLASS (served_verdicts.sh's: V3C_CLASS, the
#      class the window is committed at) `--class`, which refuses passes that are not that class's fixed load, so the
#      verdict's counts of passes, calls and tables are the class's, and whose accept publishes its work in the grid form
#      (verify_run.grid_work: the class's structural credit less the give-up padded up to H1's window grid; the exact
#      credit is the profile's exact_work, which `publish` holds on the node): one sample of PEARLPLUS_SAMPLES
#      work-weighted tiles (default its own, 459) over all the passes, and each pass's
#      exclusion tiles, replayed on fast_tile's batch forms, the first and every 256th after it on the scalar reference
#      too (a scalar replay takes minutes a tile at the 1B's K), or with PEARLPLUS_FAST set empty on the scalar reference
#      alone, under PEARLPLUS_OVER_CAP's rule where it is set. Its document goes to PIECE_DIR/verify.json and its core
#      answer to PIECE_DIR/pearlplus-answer.json, not answer.json, which the record would compose as the window's PoUW
#      answer. Under the gate it samples only after the window serve's Pearl-C period (SERVE_DIR/pearlplus/period.json,
#      the one its RULE line reads) closed with the lane's reserves since its open the window's salt sessions alone, and
#      its sample source is a sample session's coin on vy-challenger-1, reserved withheld once the
#      period closed, as the lane's first reserve since, and opened on the run's commitment (`bench_arm.LiveDraw`, which
#      refuses before the coin is asked otherwise; verify_run.py's "The run's sample"), else the node's. Its coin, recorded
#      in passes/coin.json before the replay, leaves the passes the moment it is sampled (`served_verdicts.py coin`:
#      PIECE_DIR/coin.json, put in the evidence store, preserved where the node holds a store key and else in the node's own store under the same
#      id, its answer in PIECE_DIR/coin-put.json, which names the run's Attempt carrying out/'s copy as its custody);
#      the RULE line then reads the ledger again: whether the sample session was the lane's first reserve after the
#      period's close (`served_verdicts.py pearlplus-period rule`, kept as period.json's rule);
#   4b. once verify wrote its document, `served_passes.py lean` takes the Lean verdict of record, PouwCheck.PearlPlus's
#      pouw-pearlplus (PEARLPLUS_LEAN_EXE, else built here as served_pouw.sh builds pouw-period, its log
#      pouw-pearlplus-build.log; one that doesn't build answers nothing), on the same passes, epoch coins, samples, coin
#      record and W (--tie), of record (--of-record) on a live session's coins, within PEARLPLUS_LEAN_TIMEOUT seconds
#      (default 3,600), and writes its answer to PIECE_DIR/pearlplus-lean.json,
#      the file `served_verdicts.pearlplus_of_record` reads, only where it is the Python verdict's; both verdicts, its
#      command line and the executable's SHA-256 go to PIECE_DIR/pearlplus-lean-check.json either way;
#   4c. where the Lean answered, `served_passes.py case` writes the parts of the run's case as remote's composite of
#      record reads it that need no docket, and the pass files its opened tiles read, to
#      SERVE_DIR/private/pearlplus/case, where served_cases.py makes the case once the window's record R is made;
#   5. `served_verdicts.py pearlplus --tie W` writes the piece (PIECE_DIR/development.json), the window's own epoch: its
#      verdict, with each pass's salt held to the window's salt session (`pearlplus_coins`: the session fetched through
#      PEARLC_CHALLENGER_GATE, under H1_KEY_SHA512 and MAIN_FILE's line; with in-process coins it claims no verdict) and
#      an accept to the published coin (`pearlplus_coin`), the replay's node and the rounds dropped, the give-up's
#      document (which `served_verdicts.py publish` holds on the node, with its counts in this log's lines), Pearl+'s
#      verdict's credit, and why it credits nothing by itself, naming the Pearl-C period its salt session joined
#      (served_verdicts.pearlplus_why_tied).
# The passes hold the run's activations, so they stay in SERVE_DIR/private, under served_verdicts.sh's `research keep`,
# and the piece runs under umask 077, so every file it or a step writes there is 0600.
# env: served_verdicts.sh's (VENV, HF_HOME, CUDA_HOME, PEARLPLUS_TIE_CMD), PEARLPLUS_* above. Its lines name no window.
# Exit 0 once step 5 wrote the piece.
set -uo pipefail
for o in PEARLPLUS_DRAWS:PEARLPLUS_SAMPLES; do n=${o#*:} o=${o%%:*}; [ -z "${!o+x}" ] || [ -n "${!n+x}" ] \
  || { echo "served-pearlplus: $o is set and $n is not: it is read as $n now; set $n instead" >&2; exit 2; }; done
mkdir -p "${3:?PIECE_DIR}" && SD=$(cd "${1:?SERVE_DIR}" && pwd) && P=$(cd "$3" && pwd) || exit 1
cd "$(dirname "$0")/../.." || exit 1
TREE=$PWD TPY=$PWD/.venv/bin/python
[ -x "$TPY" ] || TPY=${PY:-python3}
Q=$SD/private/pearlplus
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
      say "volunteer: refused, so nothing is checked: Pearl+'s verdict rejects a run on a drawn row over its debit cap" \
        "(over-cap rule $RULE), so the give-up before its coin can't be skipped (PEARLPLUS_VOLUNTEER is empty)"
      exit 2
    fi ;;
  no) [ "$VOLUNTEER" = auto ] && VOLUNTEER='' ;;
  *) say "volunteer: served_passes.py volunteer-needed answered '$NEEDED', so nothing is checked:" \
       "$(tail -1 "$Q/volunteer-needed.log" | cut -c1-300)"
     exit 2 ;;
esac
BF16="sm_90's BF16 rows $([ "${PEARLC_BF16_ROWS:-0}" = 0 ] && echo off || echo on) (PEARLC_BF16_ROWS=${PEARLC_BF16_ROWS:-0})"

K=${2:?WINDOW} SERVED=$Q/passes/served.json W=$Q/window-record.json GO=1

t=$(date +%s) rc=0 OWN=''
if [ ! -s "${PEARLC_CUBIN:-}" ]; then
  OWN=1
  ( export RESEARCH_RUN_DIR=$Q
    # shellcheck source=/dev/null
    . benchmarks/harness/setup/pearl_c_sm90.sh && printf '%s\n' "$PEARLC_KERNEL" "$PEARLC_CUBIN" > "$Q/setup.paths" ) \
    > "$Q/setup.log" 2>&1
  rc=$?
fi
if [ -z "$OWN" ]; then  # the window serve's, built before its lease
  printf '%s\n' "$PEARLC_KERNEL" "$PEARLC_CUBIN" > "$Q/setup.paths"
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

E=$SD/pearl/w$K
if [ -s "$E/served.json" ] && ln -s "$E" "$Q/passes"; then
  line=$("$TPY" -c 'import json, sys
d = json.load(open(sys.argv[1]))
print(d.get("passes_with_calls"), "passes with calls of its Commit window'\''s", d.get("commit_steps"), "steps, coins",
      (d.get("coins") or {}).get("source"), "->", "OK" if d.get("ok") else "FAIL, " + str(d.get("error"))[:300])
sys.exit(0 if d.get("ok") else 1)' "$E/served.json" 2>&1)
  rc=$?
  say "window: the window's own Pearl+ epoch $E (pearl-plus), $BF16: $(tail -1 <<< "$line" | cut -c1-400)"
  [ "$rc" = 0 ] || GO=
else
  say "window: the window serve wrote no Pearl+ epoch $E/served.json (pearl-plus), so there are no passes to tie"
  GO=
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
if [ -n "$GO" ]; then
  VA=()
  [ -s "$Q/volunteer.json" ] && VA=(--volunteer "$Q/volunteer.json")
  VA+=(--window-record "$W")
  [ ! -s "$SD/deployment-registration.json" ] || VA+=(--previous-record "$SD/deployment-registration.json")
  [ ! -s "$SD/declaration.json" ] || VA+=(--day-declaration "$SD/declaration.json")
  "$VENV/bin/python" benchmarks/pouw/pearl_c_vllm/served_passes.py retain --passes "$Q/passes" "${VA[@]}" > "$Q/retain.log" 2>&1
  rc=$?
  say "retain: served_passes.py retain exit $rc: $(grep -m1 '^retain ' "$Q/retain.log" | cut -c1-400)"
  [ "$rc" = 0 ] || GO=
fi
# the window commit tied to W, before Pearl+'s coin, as H1's is made at its window's close before its sample coin
if [ -n "$GO" ]; then
  if [ -s "$W" ]; then
    say "W: $(grep -m1 '^retain W: ' "$Q/retain.log" | sed -E 's/^retain W: [^,]*, //' | cut -c1-300)"
    t=$(date +%s)
    (cd "$TREE" && bash -c "${PEARLPLUS_TIE_CMD:?served_verdicts.sh sets PEARLPLUS_TIE_CMD}" _ "$SD" "$K" "$W" "$Q/passes") \
      > "$Q/tie.log" 2>&1
    rc=$?
    say "tie: served_verdicts.sh pearl-tie exit $rc after $(took "$t"): $(tail -1 "$Q/tie.log" | cut -c1-600)"
    [ "$rc" = 0 ] || { say "draw: refused before it: the window's commit tied to W failed (pearl-tie exit $rc), so" \
      "nothing holds the passes to the window's class and no sample or Pearl+ line follows it"; GO=; }
  else
    say "W: retain wrote no window record, so nothing ties the passes to the window and no sample or Pearl+ line follows"
    GO=
  fi
fi

PD=$SD/pearlplus
if [ -n "$GO" ] && [ -n "${PEARLC_CHALLENGER_GATE:-}" ]; then  # the window serve closed the period
  held=$("$TPY" -c 'import json, sys
d = json.load(open(sys.argv[1]))
print("ok" if (d.get("reserves") or {}).get("ok") is True and d.get("close") else "")' "$PD/period.json" 2>/dev/null)
  [ "$held" = ok ] || { say "draw: refused before it: the window serve's Pearl-C period didn't close with the lane's" \
    "reserves since its open the window's salt sessions alone (served_verdicts.sh's PEARLPLUS-PERIOD line), so no draw" \
    "follows it"; GO=; }
fi
if [ -n "$GO" ]; then
  t=$(date +%s)
  TA=(--tie "$W" ${PEARLPLUS_CLASS:+--class "$PEARLPLUS_CLASS"})
  "$VENV/bin/python" benchmarks/pouw/pearl_c_vllm/served_passes.py verify --passes "$Q/passes" --out "$P/verify.json" \
    --answer "$P/pearlplus-answer.json" --jobs "${PEARLPLUS_JOBS:-$(nproc)}" ${PEARLPLUS_SAMPLES:+--draws "$PEARLPLUS_SAMPLES"} \
    ${PEARLPLUS_OVER_CAP:+--over-cap "$PEARLPLUS_OVER_CAP"} ${FAST:+--fast} "${TA[@]}" > "$Q/verify.log" 2>&1 &
  V=$!
  # the coin record leaves the passes the moment verify samples it, before its replay: a deleted one would sample afresh
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
    LX=$TREE/verity/protocols/resource_accounting/compute/pouw/verifier/lean/pearlplus/.lake/build/bin/pouw-pearlplus
    (cd verity/protocols/resource_accounting/compute/pouw/verifier/lean/pearlplus && lake build pouw-pearlplus) > "$Q/pouw-pearlplus-build.log" 2>&1 || {
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

UN=(--tie "$W")
[ -s "$Q/volunteer.json" ] && UN+=(--volunteer "$Q/volunteer.json")
"$TPY" benchmarks/one_stage/served_verdicts.py pearlplus "$P" --served "$SERVED" --verify "$P/verify.json" \
  "${UN[@]}" ${PEARLC_CHALLENGER_GATE:+--gate "$PEARLC_CHALLENGER_GATE"} ${H1_KEY_SHA512:+--key-sha512 "$H1_KEY_SHA512"} \
  ${MAIN_FILE:+--main "$MAIN_FILE"} --coin "$P/coin-put.json" > "$Q/pearlplus.log" 2>&1
rc=$?
grep '^POUW-DEV ' "$Q/pearlplus.log"
say "piece: served_verdicts.py pearlplus exit $rc"
exit "$rc"
