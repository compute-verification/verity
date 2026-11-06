#!/usr/bin/env bash
# PoUS's and the wipe's pieces for the composite, built after a served run from what it kept (served_case.py's
# docstring), with the tree as cwd:
#
#   bash benchmarks/pous/served_case.sh PIECES
#
# PIECES is the run's public pieces directory (served_verdicts.sh's $OUT/pieces). For PIECES/pous, when it holds an
# answer: libp2v1.so built for the card (benchmarks/torch_gpu/p2_stage/build.sh at ARCH, default sm_90a) into a scratch
# directory removed on exit, unless P2_LIB names one, then served_case.py under one lease of the card (gpu-lease, at most
# LEASE_MIN minutes, default 40, and MEM_GB GiB of host memory, default 72; in the caller's lease when GPU_LEASE_UUID is
# set), with vLLM's venv (PY, default /workspace/cp/venv312/bin/python) and the snapshot under HF_HOME (default
# /workspace/hf). For PIECES/wipe, when it holds an answer: served_case.py, no card. POUS_EXE and WIPE_EXE, when set, are
# the components' Lean (pous-run-protocol, wipe-protocol), whose answers are written. MEM_GB's default fits POUS_EXE on a
# served window's case, about 14 GiB of line: pous-run-protocol's serve peaks at about 2 times the line (10.2 GiB on a
# 5.0 GiB case from big_case.py), while served_case.py holds the case and the line it sends, 2 times more, so about 4
# times the line in all, 56 GiB, beside the snapshot's table and the card's runtime. Each piece goes to its private
# directory, and each prints its CASE line. Exit: the first nonzero of served_case.py's exits (0 written, 1 refused, 2 not
# a piece), 3 when the library doesn't build.
set -uo pipefail
PIECES=${1:?usage: served_case.sh PIECES}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PY:-/workspace/cp/venv312/bin/python}
export PYTHONPATH=$ROOT:$ROOT/integrations/vllm:$ROOT/tools/config:$ROOT/tools/research/src PYTHONDONTWRITEBYTECODE=1
CASE=$ROOT/benchmarks/pous/served_case.py
rc=0
if [ -f "$PIECES/pous/answer.json" ]; then
  S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
  LIB=${P2_LIB:-}
  if [ -z "$LIB" ]; then
    ARCH=${ARCH:-sm_90a} bash "$ROOT/benchmarks/torch_gpu/p2_stage/build.sh" "$S/p2" > "$S/build.log" 2>&1 \
      || { tail -20 "$S/build.log"; echo "CASE: libp2v1.so doesn't build"; exit 3; }
    LIB=$S/p2/lib
  fi
  ARGS=("$CASE" --window "$PIECES/pous" --hf-home "${HF_HOME:-/workspace/hf}" --p2-lib "$LIB" ${POUS_EXE:+--exe "$POUS_EXE"})
  if [ -n "${GPU_LEASE_UUID:-}" ] || ! command -v gpu-lease > /dev/null 2>&1; then
    [ -n "${GPU_LEASE_UUID:-}" ] && export CUDA_VISIBLE_DEVICES=$GPU_LEASE_UUID
    "$PY" "${ARGS[@]}" || rc=$?
  else
    # shellcheck disable=SC2016
    GPU_LEASE_WHO=${GPU_LEASE_WHO:-memory-accounting} gpu-lease 1 --wait --max-min "${LEASE_MIN:-40}" --mem-gb "${MEM_GB:-72}" -- \
      bash -c 'export CUDA_VISIBLE_DEVICES=$GPU_LEASE_UUID; exec "$@"' _ "$PY" "${ARGS[@]}" || rc=$?
  fi
fi
if [ -f "$PIECES/wipe/answer.json" ]; then
  w=0
  "$PY" "$CASE" --window "$PIECES/wipe" ${WIPE_EXE:+--exe "$WIPE_EXE"} || w=$?
  [ "$rc" != 0 ] || rc=$w
fi
exit "$rc"
