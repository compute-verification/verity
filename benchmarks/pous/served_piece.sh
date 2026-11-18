#!/usr/bin/env bash
# PoUS's piece of the composed served run (served_piece.py's docstring), started as served_verdicts.sh starts a piece,
# `bash -c "$POUS_CMD" _ SERVE_DIR WINDOW DIR` with the tree as cwd:
#
#   POUS_CMD='bash benchmarks/pous/served_piece.sh "$@"'
#
# Before the card: libp2v1.so built for it (verity/protocols/resource_accounting/memory/pous/prover/p2/p2_store/build.sh at ARCH, default the `arch` of this node's card record) into a
# scratch directory removed on exit, unless P2_LIB names one, as a node that refuses `setarch -R` needs, since the build
# doesn't run there; the library loaded once by the sessions' Python, with no card to see; and the two Lean verifiers
# that decide, pous-run-protocol (PoUS's case) and poub-p2-protocol (PoUB-P2's), started building in a scratch copy of
# verity/, never in this tree's .lake, which served_verdicts.sh's Lean builds write meanwhile, unless POUS_EXE and
# BW_CASE_EXE name ones built already. The builds run one after another beside the run's key, the card and the
# sessions, and each program is waited for only where it is needed (lean_ready.sh): poub-verifier by the bandwidth
# party, the other two by the answers. A build that fails stops the piece at once, the sessions included, with its own
# error and exit 7.
#
# On the card, the two sessions and nothing else (served_piece.py --sessions): under one lease (the node's gpu-lease,
# else the tree's own gpu_lease.sh, for at most LEASE_MIN minutes, default 90, and MEM_GB GiB of host memory, default
# 72; with no lease of its own only in the caller's, when GPU_LEASE_UUID is set), with vLLM's venv (PY, default
# /workspace/cp/venv312/bin/python) and the snapshot under HF_HOME (default /workspace/hf): setup, PoUS's session with
# the memory attester on vy-auditor-2, and then, in the same process over the store it holds, PoUB-P2's with
# vy-auditor-2's bandwidth party, for at most BW_PARTY_MIN minutes (default 10, the gate's cap). The session it asks for
# is the auditor's (p2_parties.SESSION), which the party holds every request to: with another value in POUS_ARGS's --bw-*
# the party refuses it and the bandwidth answer is undecided. What the sessions left stays in SERVE_DIR/private, C*
# among it, and the lease ends with that process: no answer is waited for on the card.
#
# After the card, the answers (served_piece.py --answers), with no lease and at most ANSWERS_MEM_GB GiB of host memory
# (default 128, in a systemd scope where this user can make one, else under prlimit's address-space limit, else not
# answered: PoUS's case is held by the Python and written as its
# line, four times C*, to the Lean, which works in a few times that line), all three at once: PoUB-P2's case
# answered by poub-p2-protocol, its answer to DIR's sibling bandwidth, which BANDWIDTH_CMD collects
# (served_piece.py collect-bandwidth), and its case to SERVE_DIR/private/bandwidth; and PoUS's case answered by the Python
# and by pous-run-protocol, which must agree with it. This part starts only when the sessions' part exits 0 and
# those two programs are built. A run that ended in it, or whose Lean
# didn't answer or agree (the window is then undecided), keeps what the sessions left and can be answered again by
# hand (served_piece.py's arguments with --answers), with no new session; once each Lean has answered, C* is removed.
#
# POUS_ARGS adds served_piece.py's options, as shell words (--node, --private, --rounds, --challenger-*, --trusted-sha512, --device,
# --bw-*, --no-bandwidth), which this script evaluates: `verity run` builds it from the configuration and its bindings
# alone, and a caller who exports it by hand runs whatever it holds, on its own machine.
#
# --run-keys among them is a development run's: the memory attester and the bandwidth party start on this node
# (run_attester.sh, listening on loopback), under one key made for the run in SERVE_DIR/private/run-keys (0700) before
# the card is asked for, which both sessions are trusted under for this run alone, so the record says key_pin run and
# trusted-device rehearsal; the directory is removed when this script exits. It is refused beside a pin or an
# attester's flag (--trusted-sha512, --bw-trusted-sha512, --challenger-*, --bw-host, --bw-party). TPY is the tree's
# Python the attesters run in (default .venv/bin/python), and BW_PARTY_EXE a built poub-verifier for the bandwidth party.
#
# Logs: DIR/build.log, DIR/lake-build.log, DIR/pous.log (both parts' stderr) and
# DIR/stamps.txt, the piece's timeline, a line `UTC EVENT` each: its start, the builds' start and each build's end, the
# library's load, the card asked for and leased, each part's exit, and served_piece.py's own (the weights loaded, setup,
# each session, each answer's start and end); steps that overlap each keep their own lines. Exit: served_piece.py's (0 accept, 1 another answer, 2 no answer made), 3 when the library
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
        echo "POUS REFUSED: --run-keys starts the memory attester and the bandwidth party on this node under a key" \
          "made for the run, and POUS_ARGS also names $a: drop $a, or drop --run-keys and bind the auditor's attesters"
        stamp refused; exit 2 ;;
    esac
  done
fi
export PYTHONPATH=$ROOT:$ROOT/adapters/vllm:$ROOT/tools/config:$ROOT/tools/research/src:$ROOT/benchmarks/harness PYTHONDONTWRITEBYTECODE=1
K=
S=$(mktemp -d); trap 'rm -rf "$S" ${K:+"$K"}' EXIT
LIB=${P2_LIB:-}
if [ -z "$LIB" ]; then
  setarch "$(uname -m)" -R true 2> /dev/null \
    || { echo "POUS: no answer: setarch -R is refused on this node, so libp2v1.so doesn't build here: build it where" \
           "it is allowed (verity/protocols/resource_accounting/memory/pous/prover/p2/p2_store/build.sh) and name its directory in P2_LIB"; exit 3; }
  if [ -z "${ARCH:-}" ]; then
    ARCH=$(python3 "$ROOT/benchmarks/kernels/machine.py" arch \
      "$(nvidia-smi -i "${GPU_LEASE_UUID:-0}" --query-gpu=name --format=csv,noheader 2> /dev/null | head -1)" 2>&1) \
      || { echo "POUS: no answer: ARCH is not given and the card's record doesn't give it: $ARCH"; exit 3; }
  fi
  TOOLKIT=$(python3 "$ROOT/benchmarks/kernels/machine.py" toolkit 13.0) \
    || { echo "POUS: no answer: no CUDA 13.0 toolkit to build libp2v1.so with (machine.py, above)"; exit 3; }
  ARCH=$ARCH CUDA_HOME=$TOOLKIT bash "$ROOT/verity/protocols/resource_accounting/memory/pous/prover/p2/p2_store/build.sh" "$S/p2" > "$DIR/build.log" 2>&1 \
    || { tail -20 "$DIR/build.log"; echo "POUS: no answer: libp2v1.so doesn't build (build.log)"; exit 3; }
  LIB=$S/p2/lib
  stamp "built libp2v1.so"
fi
# the sessions open the library only once they hold the card: a library their Python can't load ends the piece here
LOADS=$(CUDA_VISIBLE_DEVICES='' "$PY" -c 'import ctypes, sys; ctypes.CDLL(sys.argv[1])' "$LIB/libp2v1.so" 2>&1) \
  || { echo "POUS: no answer: $PY doesn't load $LIB/libp2v1.so: $(echo "$LOADS" | tail -n 1)"; exit 3; }
stamp "loaded libp2v1.so"
CHECK=verity/protocols/resource_accounting/memory/pous/verifier/check
POUB=verity/protocols/resource_accounting/memory/poub/verifier/lean
POUS_EXE=${POUS_EXE:-} BW_CASE_EXE=${BW_CASE_EXE:-} BW_PARTY_EXE=${BW_PARTY_EXE:-}
B=$S/built BUILDS=() LOCKS=() BUILDING=
READY=$ROOT/benchmarks/pous/lean_ready.sh
[ -n "$POUS_EXE" ] || { BUILDS+=("$CHECK pous-run-protocol"); POUS_EXE=$S/$CHECK/.lake/build/bin/pous-run-protocol; }
[ -z "$RUN_KEYS" ] || [ -n "$BW_PARTY_EXE" ] || { BUILDS+=("$POUB poub-verifier"); BW_PARTY_EXE=$S/$POUB/.lake/build/bin/poub-verifier; }
[ -n "$BW_CASE_EXE" ] || { BUILDS+=("$POUB poub-p2-protocol"); BW_CASE_EXE=$S/$POUB/.lake/build/bin/poub-p2-protocol; }
unlocked() {  # "$@" with none of the builds' locks, which only the builder holds
  for fd in "${LOCKS[@]}"; do eval "exec $fd>&-"; done
  exec "$@"
}
builds() {  # each of BUILDS in order, its lock released as it ends: B/EXE.ok, or B/failed with the build's error
  local i b
  command -v lake > /dev/null 2>&1 || (unlocked bash "$ROOT/tools/verity/lean/setup.sh" "$S/$CHECK") > "$DIR/lake-setup.log" 2>&1
  for i in "${!BUILDS[@]}"; do
    b=${BUILDS[$i]}
    if (cd "$S/${b% *}" && unlocked lake build "${b#* }") >> "$DIR/lake-build.log" 2>&1; then
      : > "$B/${b#* }.ok"
      stamp "built ${b#* }"
      eval "exec ${LOCKS[$i]}>&-"
    else
      { tail -20 "$DIR/lake-build.log"; echo "POUS: no answer: ${b#* } doesn't build (lake-build.log)"; } > "$B/failed.tmp"
      mv "$B/failed.tmp" "$B/failed"
      stamp "${b#* } didn't build"
      [ ! -e "$B/sessions.pid" ] || kill -TERM "$(cat "$B/sessions.pid")" 2> /dev/null
      exit 7
    fi
  done
}
if [ ${#BUILDS[@]} -gt 0 ]; then
  tar -C "$ROOT" -c --exclude=.lake --exclude=__pycache__ verity | tar -C "$S" -x
  export PATH="$HOME/.elan/bin:$PATH"
  : > "$DIR/lake-build.log"
  mkdir -p "$B"
  # each lock is taken here, before the builder starts, so a program asked for before its build begins is waited for
  for b in "${BUILDS[@]}"; do exec {fd}>"$B/${b#* }.lock" && flock -x "$fd" && LOCKS+=("$fd"); done
  stamp "the Lean builds start"
  builds &
  BUILDING=$!
  for fd in "${LOCKS[@]}"; do eval "exec $fd>&-"; done
  trap 'kill "$BUILDING" 2> /dev/null || :; rm -rf "$S" ${K:+"$K"}' EXIT
fi
ready() {  # the built program $1, waited for if it is being built; a failed build ends the piece with its own error
  [ -e "$B/$1.lock" ] || return 0
  bash "$READY" "$B" "$1" true 2>&1 && return
  stamp "the piece stops: $1 didn't build"
  exit 7
}
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
  [ ! -e "$B/poub-verifier.lock" ] || printf -v BP 'bash %q %q poub-verifier %s' "$READY" "$B" "$BP"
  EXTRA+=(--challenger-host "$(hostname)" --challenger-address 127.0.0.1:47610 --challenger-command "$CC"
          --trusted-sha512 "$SHA" --bw-party "$BP" --bw-host 127.0.0.1 --bw-trusted-sha512 "$SHA")
fi
ARGS=("$ROOT/benchmarks/pous/served_piece.py" "$SERVE_DIR" "$WINDOW" "$DIR" --hf-home "${HF_HOME:-/workspace/hf}"
      --p2-lib "$LIB" --node "${POUS_NODE:-$(hostname)}" --bw-minutes "${BW_PARTY_MIN:-10}"
      --case-exe "$POUS_EXE" --bw-case-exe "$BW_CASE_EXE" ${EXTRA[@]+"${EXTRA[@]}"})
LEASE=(gpu-lease)
command -v gpu-lease > /dev/null 2>&1 || LEASE=(bash "$ROOT/tools/research/src/research/pods/sh/gpu_lease.sh")
[ ! -e "$B/failed" ] || { cat "$B/failed"; stamp "the piece stops: a Lean verifier didn't build"; exit 7; }
stamp "asked for the card"
if [ -n "${GPU_LEASE_UUID:-}" ]; then
  CUDA_VISIBLE_DEVICES=$GPU_LEASE_UUID "$PY" "${ARGS[@]}" --sessions 2>> "$DIR/pous.log" &
else
  # shellcheck disable=SC2016
  GPU_LEASE_WHO=${GPU_LEASE_WHO:-memory-accounting} "${LEASE[@]}" 1 --wait --max-min "${LEASE_MIN:-90}" --mem-gb "${MEM_GB:-72}" -- \
    bash -c 'echo "leased $GPU_LEASE_UUID" >&2; echo "$(date -u +%FT%TZ) leased the card" >> "$1"; shift; export CUDA_VISIBLE_DEVICES=$GPU_LEASE_UUID; exec "$@"' \
    _ "$DIR/stamps.txt" "$PY" "${ARGS[@]}" --sessions 2>> "$DIR/pous.log" &
fi
SESSIONS=$!
if [ -n "$BUILDING" ]; then
  # the builder stops the sessions when a build fails after this line, and this line when one failed before it
  echo "$SESSIONS" > "$B/sessions.tmp" && mv "$B/sessions.tmp" "$B/sessions.pid"
  [ ! -e "$B/failed" ] || kill -TERM "$SESSIONS" 2> /dev/null
fi
wait "$SESSIONS"
rc=$?
if [ -e "$B/failed" ]; then
  stamp "the sessions' part stopped: a Lean verifier didn't build"
  cat "$B/failed"
  exit 7
fi
stamp "the sessions' part exit $rc"
[ "$rc" = 0 ] || exit "$rc"
ready pous-run-protocol
ready poub-p2-protocol
if command -v systemd-run > /dev/null && systemd-run --user --scope --quiet --collect true > /dev/null 2>&1; then
  scope=(systemd-run --user --scope --quiet --collect -p "MemoryMax=${ANSWERS_MEM_GB:-128}G" -p MemorySwapMax=0 --)
elif command -v prlimit > /dev/null && prlimit "--as=$((1 << 30))" -- true > /dev/null 2>&1; then
  # RLIMIT_AS binds each process alone: each Lean served_piece.py starts gets the whole limit, not a share of it.
  scope=(prlimit "--as=$((${ANSWERS_MEM_GB:-128} << 30))" --)
else
  echo "POUS: no answer: this host makes no systemd user scope and runs no prlimit: the answers run only under a memory limit of their own"
  exit 2
fi
CUDA_VISIBLE_DEVICES='' "${scope[@]}" "$PY" "${ARGS[@]}" --answers 2>> "$DIR/pous.log"
rc=$?
stamp "the answers' part exit $rc"
exit $rc
