#!/usr/bin/env bash
# PoUS's piece of the composed served run (served_piece.py's docstring), started as served_verdicts.sh starts a piece,
# `bash -c "$POUS_CMD" _ SERVE_DIR WINDOW DIR` with the tree as cwd:
#
#   POUS_CMD='bash benchmarks/pous/served_piece.sh "$@"'
#
# Before the card: libp2v1.so built for it (verity/protocols/resource_accounting/memory/pous/prover/p2/p2_store/build.sh at ARCH, default sm_90a) into a
# scratch directory removed on exit, unless P2_LIB names one, as a node that refuses `setarch -R` needs, since the build
# doesn't run there; the library loaded once by the sessions' Python, with no card to see; and the two Lean verifiers
# that decide, pous-run-protocol (PoUS's case) and poub-p2-protocol (PoUB-P2's), built in a scratch copy of
# verity/, never in this tree's .lake, which served_verdicts.sh's Lean builds write meanwhile, unless POUS_EXE and
# BW_CASE_EXE name ones built already.
#
# On the card, the two sessions and nothing else (served_piece.py --sessions): under one lease (the node's gpu-lease,
# else the tree's own gpu_lease.sh, for at most LEASE_MIN minutes, default 90, and MEM_GB GiB of host memory, default
# 72; with no lease of its own only in the caller's, when GPU_LEASE_UUID is set), with vLLM's venv (PY, default
# /workspace/cp/venv312/bin/python) and the snapshot under HF_HOME (default /workspace/hf): setup, PoUS's session with
# the memory challenger on vy-auditor-2, and then, in the same process over the store it holds, PoUB-P2's with
# vy-auditor-2's bandwidth party, for at most BW_PARTY_MIN minutes (default 10, the gate's cap). The session it asks for
# is the auditor's (p2_parties.SESSION), which the party holds every request to: with another value in POUS_ARGS's --bw-*
# the party refuses it and the bandwidth answer is undecided. What the sessions left stays in SERVE_DIR/private, C*
# among it, and the lease ends with that process: no answer is waited for on the card.
#
# After the card, the answers (served_piece.py --answers), with no lease and at most ANSWERS_MEM_GB GiB of host memory
# (default 128, in a systemd scope where this user can make one: PoUS's case is held by the Python and written as its
# line, four times C*, to the Lean, which works in a few times that line): PoUB-P2's case answered by
# poub-p2-protocol, its answer to DIR's sibling bandwidth, which BANDWIDTH_CMD collects (served_bandwidth.py
# collect), and its case to SERVE_DIR/private/bandwidth; then PoUS's case answered by pous-run-protocol, which must
# agree with the Python. This part starts only when the sessions' part exits 0. A run that ended in it, or whose Lean
# didn't answer or agree (the window is then undecided), keeps what the sessions left and can be answered again by
# hand (served_piece.py's arguments with --answers), with no new session; once each Lean has answered, C* is removed.
#
# POUS_ARGS adds served_piece.py's options, as shell words (--node, --private, --rounds, --challenger-*, --trusted-sha512, --device,
# --bw-*, --no-bandwidth), which this script evaluates: `verity run` builds it from the configuration and its bindings
# alone, and a caller who exports it by hand runs whatever it holds, on its own machine.
#
# --run-keys among them is a development run's: the memory challenger and the bandwidth party start on this node
# (run_attester.sh, listening on loopback), under one key made for the run in SERVE_DIR/private/run-keys (0700) before
# the card is asked for, which both sessions are trusted under for this run alone, so the record says key_pin run and
# trusted-device rehearsal; the directory is removed when this script exits. It is refused beside a pin or an
# attester's flag (--trusted-sha512, --bw-trusted-sha512, --challenger-*, --bw-host, --bw-party). TPY is the tree's
# Python the attesters run in (default .venv/bin/python), and BW_PARTY_EXE a built poub-verifier for the bandwidth party.
#
# Logs: DIR/build.log, DIR/lake-build.log, DIR/pous.log (both parts' stderr) and
# DIR/stamps.txt, the piece's timeline, a line `UTC EVENT` each: its start, each build's end, the library's load, the
# card asked for and leased, each part's exit, and between them served_piece.py's own (the weights loaded, setup, each
# session, each answer). Exit: served_piece.py's (0 accept, 1 another answer, 2 no answer made), 3 when the library
# doesn't build or PY doesn't load it, 7 when a Lean verifier doesn't build.
set -uo pipefail
SERVE_DIR=${1:?usage: served_piece.sh SERVE_DIR WINDOW DIR}; WINDOW=${2:?WINDOW}; DIR=${3:?DIR}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$DIR"
stamp() { echo "$(date -u +%FT%TZ) $*" >> "$DIR/stamps.txt"; }
rm -f "$DIR/stamps.txt"
stamp start
PY=${PY:-/workspace/cp/venv312/bin/python}
eval "GIVEN=(${POUS_ARGS:-})"
EXTRA=() RUN_KEYS=
for a in ${GIVEN[@]+"${GIVEN[@]}"}; do [ "$a" = --run-keys ] && RUN_KEYS=1 || EXTRA+=("$a"); done
if [ -n "$RUN_KEYS" ]; then
  for a in ${EXTRA[@]+"${EXTRA[@]}"}; do
    case $a in
      --trusted-sha512 | --bw-trusted-sha512 | --challenger-host | --challenger-address | --challenger-command | --bw-host | --bw-party)
        echo "POUS REFUSED: --run-keys starts the memory challenger and the bandwidth party on this node under a key" \
          "made for the run, and POUS_ARGS also names $a: drop $a, or drop --run-keys and bind the auditor's attesters"
        stamp refused; exit 2 ;;
    esac
  done
fi
export PYTHONPATH=$ROOT:$ROOT/adapters/vllm:$ROOT/tools/config:$ROOT/tools/research/src PYTHONDONTWRITEBYTECODE=1
K=
S=$(mktemp -d); trap 'rm -rf "$S" ${K:+"$K"}' EXIT
LIB=${P2_LIB:-}
if [ -z "$LIB" ]; then
  setarch "$(uname -m)" -R true 2> /dev/null \
    || { echo "POUS: no answer: setarch -R is refused on this node, so libp2v1.so doesn't build here: build it where" \
           "it is allowed (verity/protocols/resource_accounting/memory/pous/prover/p2/p2_store/build.sh) and name its directory in P2_LIB"; exit 3; }
  ARCH=${ARCH:-sm_90a} bash "$ROOT/verity/protocols/resource_accounting/memory/pous/prover/p2/p2_store/build.sh" "$S/p2" > "$DIR/build.log" 2>&1 \
    || { tail -20 "$DIR/build.log"; echo "POUS: no answer: libp2v1.so doesn't build (build.log)"; exit 3; }
  LIB=$S/p2/lib
  stamp "built libp2v1.so"
fi
# the sessions open the library only once they hold the card: a library their Python can't load ends the piece here
LOADS=$(CUDA_VISIBLE_DEVICES='' "$PY" -c 'import ctypes, sys; ctypes.CDLL(sys.argv[1])' "$LIB/libp2v1.so" 2>&1) \
  || { echo "POUS: no answer: $PY doesn't load $LIB/libp2v1.so: $(echo "$LOADS" | tail -n 1)"; exit 3; }
stamp "loaded libp2v1.so"
built() {  # the Lean executable $2 of the scratch copy's package $1, built there: its path in EXE
  EXE=$S/$1/.lake/build/bin/$2
  (cd "$S/$1" && lake build "$2") >> "$DIR/lake-build.log" 2>&1 && { stamp "built $2"; return; }
  tail -20 "$DIR/lake-build.log"; echo "POUS: no answer: $2 doesn't build (lake-build.log)"; exit 7
}
POUS_EXE=${POUS_EXE:-} BW_CASE_EXE=${BW_CASE_EXE:-} BW_PARTY_EXE=${BW_PARTY_EXE:-}
if [ -z "$POUS_EXE" ] || [ -z "$BW_CASE_EXE" ] || { [ -n "$RUN_KEYS" ] && [ -z "$BW_PARTY_EXE" ]; }; then
  CHECK=verity/protocols/resource_accounting/memory/pous/verifier/check
  POUB=verity/protocols/resource_accounting/memory/poub/verifier/lean
  tar -C "$ROOT" -c --exclude=.lake --exclude=__pycache__ verity | tar -C "$S" -x
  export PATH="$HOME/.elan/bin:$PATH"
  command -v lake > /dev/null 2>&1 || bash "$ROOT/tools/verity/lean/setup.sh" "$S/$CHECK" > "$DIR/lake-setup.log" 2>&1
  : > "$DIR/lake-build.log"
  [ -n "$POUS_EXE" ] || { built "$CHECK" pous-run-protocol; POUS_EXE=$EXE; }
  [ -n "$BW_CASE_EXE" ] || { built "$POUB" poub-p2-protocol; BW_CASE_EXE=$EXE; }
  [ -z "$RUN_KEYS" ] || [ -n "$BW_PARTY_EXE" ] || { built "$POUB" poub-verifier; BW_PARTY_EXE=$EXE; }
fi
if [ -n "$RUN_KEYS" ]; then
  K=$SERVE_DIR/private/run-keys
  rm -rf "$K" && (umask 077 && mkdir -p "$K/pous")
  TPY=${TPY:-$ROOT/.venv/bin/python}
  SHA=$(cd "$ROOT" && env -u PYTHONPATH "$TPY" -c 'import hashlib, sys
from verity.protocols.resource_accounting.memory.pous.attester import onsite
print(hashlib.sha512(onsite.device_key(sys.argv[1], workers=int(sys.argv[2])).public_key).hexdigest())' \
    "$K/pous" "$(nproc)" 2>> "$DIR/pous.log")
  [[ $SHA =~ ^[0-9a-f]{128}$ ]] || { echo "POUS: no answer: the run's key isn't made in $K (pous.log)"; exit 2; }
  stamp "made the run's key"
  printf -v CC 'bash %q challenger_party %q %q' "$ROOT/benchmarks/pous/run_attester.sh" "$TPY" "$K"
  printf -v BP 'bash %q bw_party %q %q %q' "$ROOT/benchmarks/pous/run_attester.sh" "$TPY" "$K" "$BW_PARTY_EXE"
  EXTRA+=(--challenger-host "$(hostname)" --challenger-address 127.0.0.1:47610 --challenger-command "$CC"
          --trusted-sha512 "$SHA" --bw-party "$BP" --bw-host 127.0.0.1 --bw-trusted-sha512 "$SHA")
fi
ARGS=("$ROOT/benchmarks/pous/served_piece.py" "$SERVE_DIR" "$WINDOW" "$DIR" --hf-home "${HF_HOME:-/workspace/hf}"
      --p2-lib "$LIB" --node "${POUS_NODE:-$(hostname)}" --bw-minutes "${BW_PARTY_MIN:-10}"
      --case-exe "$POUS_EXE" --bw-case-exe "$BW_CASE_EXE" ${EXTRA[@]+"${EXTRA[@]}"})
LEASE=(gpu-lease)
command -v gpu-lease > /dev/null 2>&1 || LEASE=(bash "$ROOT/tools/research/src/research/pods/sh/gpu_lease.sh")
stamp "asked for the card"
if [ -n "${GPU_LEASE_UUID:-}" ]; then
  export CUDA_VISIBLE_DEVICES=$GPU_LEASE_UUID
  "$PY" "${ARGS[@]}" --sessions 2>> "$DIR/pous.log"
else
  # shellcheck disable=SC2016
  GPU_LEASE_WHO=${GPU_LEASE_WHO:-memory-accounting} "${LEASE[@]}" 1 --wait --max-min "${LEASE_MIN:-90}" --mem-gb "${MEM_GB:-72}" -- \
    bash -c 'echo "leased $GPU_LEASE_UUID" >&2; echo "$(date -u +%FT%TZ) leased the card" >> "$1"; shift; export CUDA_VISIBLE_DEVICES=$GPU_LEASE_UUID; exec "$@"' \
    _ "$DIR/stamps.txt" "$PY" "${ARGS[@]}" --sessions 2>> "$DIR/pous.log"
fi
rc=$?
stamp "the sessions' part exit $rc"
[ "$rc" = 0 ] || exit "$rc"
scope=()
command -v systemd-run > /dev/null && systemd-run --user --scope --quiet --collect true > /dev/null 2>&1 \
  && scope=(systemd-run --user --scope --quiet --collect -p "MemoryMax=${ANSWERS_MEM_GB:-128}G" -p MemorySwapMax=0 --)
CUDA_VISIBLE_DEVICES='' "${scope[@]}" "$PY" "${ARGS[@]}" --answers 2>> "$DIR/pous.log"
rc=$?
stamp "the answers' part exit $rc"
exit $rc
