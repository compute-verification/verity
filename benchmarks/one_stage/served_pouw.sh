#!/usr/bin/env bash
# PoUW's piece of a served H1 window and its period case, `served_verdicts.sh`'s POUW_CMD for a run whose composite reads
# the period (PRODUCT=remote):
#   POUW_CMD='bash benchmarks/one_stage/served_pouw.sh "$1" "$2" "$3"'      (served_verdicts.sh runs it in the tree)
#   bash benchmarks/one_stage/served_pouw.sh SERVE_DIR WINDOW PIECE_DIR
# Everything but PIECE_DIR's files goes to SERVE_DIR/private/pouw, each step's log among it:
#   1. h1_prove/live.sh proves the window's live rounds from serve-openai's --h1-prover dump (SERVE_DIR/private/h1/w<K>/live)
#      into live/: its live-evidence.json, credit's live evidence, and the proof files it names;
#   2. h1_prove/drawn.sh stages and proves the window's drawn units (H1_DRAWN_ONLY=1, and H1_DRAWN_CHECK=0 since steps 3
#      and 4 verify the statements), its scratch kept in drawn-work/ (H1_DRAWN_WORK): it holds the proof files
#      drawn/statements.json names, and the staged witness;
#   3. `served_verdicts.py pouw` writes the piece in PIECE_DIR with those --statements and --live-evidence (the window's
#      gate sessions, ledger and pins among it);
#   4. `pouw_piece.py period` writes the period's case and its Lean answer (pouw-period, built here) in period/, from the
#      records step 3 fetched, where served_composite.sh finds it for remote's Lean.
# Steps 3 and 4 each verify every statement with the verifier of record, one after the other: two lane sessions'
# verifiers side by side don't fit a 16-CPU, 196 GB H200 node beside nothing else. H1_POUW_REUSE=1 skips steps 1 and 2
# when their statements and live evidence are already there (a rerun of 3 and 4 on the kept proofs).
# env: served_verdicts.sh's (H1_GATE, H1_KEY_SHA512, MAIN_FILE, PY), drawn.sh's and live.sh's (H1_CIRCUITS, H1_MODEL,
#      H1_CHECKPOINT or H1_WEIGHTS, H1_DRAWN_MEM_GB, H1_ZK_SERVER_GB, H1_MEM_GB, ...)  HZ_CACHE (both jobs' builds,
#      default /workspace/research/cache/pouw-hidden-zk)  FLOCK_VERIFY (default the tree's, which served_verdicts.sh builds)
# Its lines name no window. Exit 0 once step 3 wrote the piece; a failed step 1, 2 or 4 leaves its evidence or the
# period's case out, which step 3's piece and served_composite.sh then name.
set -uo pipefail
WIN=${2:?WINDOW}
mkdir -p "${3:?PIECE_DIR}" && SD=$(cd "${1:?SERVE_DIR}" && pwd) && P=$(cd "$3" && pwd) || exit 1
cd "$(dirname "$0")/../.." || exit 1
TREE=$PWD TPY=$PWD/.venv/bin/python
Q=$SD/private/pouw ST=$SD/private/pouw/drawn/statements.json LE=$SD/private/pouw/live/out/live/live-evidence.json
(umask 077; mkdir -p "$Q") || exit 1
export HZ_CACHE=${HZ_CACHE:-/workspace/research/cache/pouw-hidden-zk}
LEAN=$TREE/verity/protocols/flock/verifier/.lake/build/bin/flock-verify
[ -n "${FLOCK_VERIFY:-}" ] || { [ -x "$LEAN" ] && export FLOCK_VERIFY=$LEAN; }
say() { echo "POUW $*"; }
took() { echo "$(( ($(date +%s) - $1) / 60 )) min"; }

if [ "${H1_POUW_REUSE:-0}" = 1 ] && [ -s "$ST" ] && [ -s "$LE" ]; then
  say "proofs: the kept statements and live evidence (H1_POUW_REUSE=1)"
else
  rm -rf "$Q/live" "$Q/drawn" "$Q/drawn-work"
  t=$(date +%s)
  RESEARCH_RUN_DIR=$Q/live H1_LIVE=$SD/private/h1/w$WIN/live H1_KEEP=$Q/live/live-proofs \
    bash benchmarks/pouw/h1_prove/live.sh > "$Q/live.log" 2>&1
  rc=$?
  say "live rounds: live.sh exit $rc after $(took "$t")$([ -s "$LE" ] && echo ", live evidence written" || echo ", no live evidence")"
  t=$(date +%s)
  H1_DRAWN_ONLY=1 H1_DRAWN_CHECK=0 H1_DRAWN_WORK=$Q/drawn-work bash benchmarks/pouw/h1_prove/drawn.sh "$SD" "$WIN" "$Q" \
    > "$Q/drawn.log" 2>&1
  rc=$?
  say "drawn units: drawn.sh exit $rc after $(took "$t")$([ -s "$ST" ] && echo ", statements written" || echo ", no statements")"
fi
ev=()
[ -s "$ST" ] && ev+=(--statements "$ST")
[ -s "$LE" ] && ev+=(--live-evidence "$LE")

t=$(date +%s)
"$TPY" benchmarks/one_stage/served_verdicts.py pouw "$SD" "$WIN" "$P" "${ev[@]}" > "$Q/pouw.log" 2>&1
rc=$?
grep '^POUW-DEV ' "$Q/pouw.log"
say "piece: served_verdicts.py pouw exit $rc after $(took "$t")"

t=$(date +%s)
EXE=$TREE/verity/protocols/pouw/verifier/lean/check/.lake/build/bin/pouw-period
(cd verity/protocols/pouw/verifier/lean/check && lake build pouw-period) > "$Q/pouw-period-build.log" 2>&1 \
  || say "period: pouw-period doesn't build (its log in the serve dir's private/pouw), so its case is answered by the Python reference"
[ -x "$EXE" ] || EXE=
rm -rf "$Q/period"
"$TPY" benchmarks/one_stage/pouw_piece.py period --out "$Q/period" ${EXE:+--exe "$EXE"} --serve-run "$SD" --window "$WIN" \
  --pouw "$P" --key-sha512 "${H1_KEY_SHA512:?}" --main "${MAIN_FILE:?}" "${ev[@]}" > "$Q/period.log" 2>&1
prc=$?
case $prc in 0) v=accept ;; 1) v=reject ;; 2) v=missing ;; *) v=error ;; esac
say "period: $v (pouw_piece.py period exit $prc after $(took "$t"); its lines in the serve dir's private/pouw/period.log)"
exit "$rc"
