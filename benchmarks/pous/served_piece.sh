#!/usr/bin/env bash
# PoUS's piece of the composed served run (served_piece.py's docstring), started as served_verdicts.sh starts a piece,
# `bash -c "$POUS_CMD" _ SERVE_DIR WINDOW DIR` with the tree as cwd:
#
#   POUS_CMD='bash benchmarks/pous/served_piece.sh "$@"'
#
# Outside the lease: libp2v1.so built for the card (benchmarks/torch_gpu/p2_stage/build.sh at ARCH, default sm_90a) into a
# scratch directory removed on exit, unless P2_LIB names one; and the two Lean verifiers that decide, pous-run-protocol
# (PoUS's case) and remote-audit-p2-protocol (PoUB-P2's), built in a scratch copy of verity/, never in this tree's .lake,
# which served_verdicts.sh's Lean builds write meanwhile, unless POUS_EXE and BW_CASE_EXE name ones built already. Under
# one lease of the card (gpu-lease, at most LEASE_MIN minutes, default 90, and MEM_GB GiB of host memory, default 72; in
# the caller's lease when GPU_LEASE_UUID is set): served_piece.py with vLLM's venv (PY, default
# /workspace/cp/venv312/bin/python), the snapshot under HF_HOME (default /workspace/hf) and the memory challenger on
# vy-auditor-2, its case answered by pous-run-protocol, which must agree with the Python. MEM_GB's default leaves
# about 4 times the case's line while the Lean answers it, beside the snapshot's table
# and the card's runtime; LEASE_MIN's leaves the Lean's answer its time between the two sessions. Once PoUS's session
# completes and its case is answered, the same process runs PoUB-P2's session over the store it holds, against
# vy-auditor-2's bandwidth party for at most BW_PARTY_MIN minutes (default 10, the gate's cap), the case answered by
# remote-audit-p2-protocol; its answer goes to DIR's sibling bandwidth, which BANDWIDTH_CMD collects
# (served_bandwidth.py collect), and its case to SERVE_DIR/private/bandwidth. The session it asks for is the auditor's
# (p2_party.SESSION), which the party holds every request to: with another value in POUS_ARGS's --bw-* the party refuses
# it and the bandwidth answer is undecided. POUS_ARGS adds served_piece.py's options (--node, --private, --rounds,
# --challenger-*, --trusted-sha512, --device, --bw-*, --no-bandwidth). Logs: DIR/build.log, DIR/lake-build.log and
# DIR/pous.log (its stderr). Exit: served_piece.py's (0 accept, 1 another answer, 2 no answer made), 3 when the library
# doesn't build, 7 when a Lean verifier doesn't.
set -uo pipefail
SERVE_DIR=${1:?usage: served_piece.sh SERVE_DIR WINDOW DIR}; WINDOW=${2:?WINDOW}; DIR=${3:?DIR}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$DIR"
PY=${PY:-/workspace/cp/venv312/bin/python}
export PYTHONPATH=$ROOT:$ROOT/adapters/vllm:$ROOT/verity/configuration/config:$ROOT/tools/research/src PYTHONDONTWRITEBYTECODE=1
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
LIB=${P2_LIB:-}
if [ -z "$LIB" ]; then
  ARCH=${ARCH:-sm_90a} bash "$ROOT/benchmarks/torch_gpu/p2_stage/build.sh" "$S/p2" > "$DIR/build.log" 2>&1 \
    || { tail -20 "$DIR/build.log"; echo "POUS: no answer: libp2v1.so doesn't build (build.log)"; exit 3; }
  LIB=$S/p2/lib
fi
built() {  # the Lean executable $2 of the scratch copy's package $1, built there: its path in EXE
  EXE=$S/$1/.lake/build/bin/$2
  (cd "$S/$1" && lake build "$2") >> "$DIR/lake-build.log" 2>&1 && return
  tail -20 "$DIR/lake-build.log"; echo "POUS: no answer: $2 doesn't build (lake-build.log)"; exit 7
}
POUS_EXE=${POUS_EXE:-} BW_CASE_EXE=${BW_CASE_EXE:-}
if [ -z "$POUS_EXE" ] || [ -z "$BW_CASE_EXE" ]; then
  CHECK=verity/protocols/pous/verifier/check
  tar -C "$ROOT" -c --exclude=.lake --exclude=__pycache__ verity | tar -C "$S" -x
  export PATH="$HOME/.elan/bin:$PATH"
  command -v lake > /dev/null 2>&1 || bash "$ROOT/tools/verity/lean/setup.sh" "$S/$CHECK" > "$DIR/lake-setup.log" 2>&1
  : > "$DIR/lake-build.log"
  [ -n "$POUS_EXE" ] || { built "$CHECK" pous-run-protocol; POUS_EXE=$EXE; }
  [ -n "$BW_CASE_EXE" ] || { built verity/protocols/poub/verifier/lean remote-audit-p2-protocol; BW_CASE_EXE=$EXE; }
fi
# shellcheck disable=SC2206
ARGS=("$ROOT/benchmarks/pous/served_piece.py" "$SERVE_DIR" "$WINDOW" "$DIR" --hf-home "${HF_HOME:-/workspace/hf}"
      --p2-lib "$LIB" --node "${POUS_NODE:-$(hostname)}" --bw-minutes "${BW_PARTY_MIN:-10}"
      --case-exe "$POUS_EXE" --bw-case-exe "$BW_CASE_EXE" ${POUS_ARGS:-})
echo "served_piece: asking for the card at $(date -u +%H:%M:%SZ)" >> "$DIR/pous.log"
if [ -n "${GPU_LEASE_UUID:-}" ] || ! command -v gpu-lease > /dev/null 2>&1; then
  [ -n "${GPU_LEASE_UUID:-}" ] && export CUDA_VISIBLE_DEVICES=$GPU_LEASE_UUID
  "$PY" "${ARGS[@]}" 2>> "$DIR/pous.log"
else
  # shellcheck disable=SC2016
  GPU_LEASE_WHO=${GPU_LEASE_WHO:-memory-accounting} gpu-lease 1 --wait --max-min "${LEASE_MIN:-90}" --mem-gb "${MEM_GB:-72}" -- \
    bash -c 'echo "leased $GPU_LEASE_UUID at $(date -u +%H:%M:%SZ)" >&2; export CUDA_VISIBLE_DEVICES=$GPU_LEASE_UUID; exec "$@"' \
    _ "$PY" "${ARGS[@]}" 2>> "$DIR/pous.log"
fi
rc=$?
echo "served_piece: exit $rc at $(date -u +%H:%M:%SZ)" >> "$DIR/pous.log"
exit $rc
