#!/usr/bin/env bash
# PoUS's piece of the composed served run (served_piece.py's docstring), started as served_verdicts.sh starts a piece,
# `bash -c "$POUS_CMD" _ SERVE_DIR WINDOW DIR` with the tree as cwd:
#
#   POUS_CMD='bash benchmarks/pous/served_piece.sh "$@"'
#
# Outside the lease: libp2v1.so built for the card (benchmarks/torch_gpu/p2_stage/build.sh at ARCH, default sm_90a) into a
# scratch directory removed on exit, unless P2_LIB names one. Under one lease of the card (gpu-lease, at most LEASE_MIN
# minutes, default 40, and MEM_GB GiB of host memory, default 32; in the caller's lease when GPU_LEASE_UUID is set):
# served_piece.py with vLLM's venv (PY, default /workspace/cp/venv312/bin/python), the snapshot under HF_HOME (default
# /workspace/hf) and the memory challenger on vy-auditor-2. Once PoUS's session completes, the same process runs PoUB-P2's
# session over the store it holds, against vy-auditor-2's bandwidth party for at most BW_PARTY_MIN minutes (default 10,
# the gate's cap), the case answered by BW_CASE_EXE when it names remote-audit-p2-protocol; its answer goes to DIR's
# sibling bandwidth, which BANDWIDTH_CMD collects (served_bandwidth.py collect), and its case to SERVE_DIR/private/bandwidth.
# POUS_ARGS adds served_piece.py's options (--node, --private, --rounds, --challenger-*, --trusted-sha512, --device, --bw-*,
# --no-bandwidth). Logs: DIR/build.log and DIR/pous.log (its stderr). Exit: served_piece.py's
# (0 accept, 1 another answer, 2 no answer made), 3 when the library doesn't build.
set -uo pipefail
SERVE_DIR=${1:?usage: served_piece.sh SERVE_DIR WINDOW DIR}; WINDOW=${2:?WINDOW}; DIR=${3:?DIR}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$DIR"
PY=${PY:-/workspace/cp/venv312/bin/python}
export PYTHONPATH=$ROOT:$ROOT/integrations/vllm:$ROOT/tools/config:$ROOT/tools/research/src PYTHONDONTWRITEBYTECODE=1
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
LIB=${P2_LIB:-}
if [ -z "$LIB" ]; then
  ARCH=${ARCH:-sm_90a} bash "$ROOT/benchmarks/torch_gpu/p2_stage/build.sh" "$S/p2" > "$DIR/build.log" 2>&1 \
    || { tail -20 "$DIR/build.log"; echo "POUS: no answer: libp2v1.so doesn't build (build.log)"; exit 3; }
  LIB=$S/p2/lib
fi
# shellcheck disable=SC2206
ARGS=("$ROOT/benchmarks/pous/served_piece.py" "$SERVE_DIR" "$WINDOW" "$DIR" --hf-home "${HF_HOME:-/workspace/hf}"
      --p2-lib "$LIB" --node "${POUS_NODE:-$(hostname)}" --bw-minutes "${BW_PARTY_MIN:-10}"
      ${BW_CASE_EXE:+--bw-case-exe "$BW_CASE_EXE"} ${POUS_ARGS:-})
echo "served_piece: asking for the card at $(date -u +%H:%M:%SZ)" >> "$DIR/pous.log"
if [ -n "${GPU_LEASE_UUID:-}" ] || ! command -v gpu-lease > /dev/null 2>&1; then
  [ -n "${GPU_LEASE_UUID:-}" ] && export CUDA_VISIBLE_DEVICES=$GPU_LEASE_UUID
  "$PY" "${ARGS[@]}" 2>> "$DIR/pous.log"
else
  # shellcheck disable=SC2016
  GPU_LEASE_WHO=${GPU_LEASE_WHO:-memory-accounting} gpu-lease 1 --wait --max-min "${LEASE_MIN:-40}" --mem-gb "${MEM_GB:-32}" -- \
    bash -c 'echo "leased $GPU_LEASE_UUID at $(date -u +%H:%M:%SZ)" >&2; export CUDA_VISIBLE_DEVICES=$GPU_LEASE_UUID; exec "$@"' \
    _ "$PY" "${ARGS[@]}" 2>> "$DIR/pous.log"
fi
rc=$?
echo "served_piece: exit $rc at $(date -u +%H:%M:%SZ)" >> "$DIR/pous.log"
exit $rc
