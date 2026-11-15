#!/usr/bin/env bash
# The window's verify passes judged after the window, CPU only: each retained pass under $WORK/passes/<window run id>
# alone gets Pearl+'s verdict in Lean (PouwCheck.PearlPlus's pouw-pearlplus, through `served_passes.py lean`), the verdict
# this job acts on, beside the Python twin the Lean is held to (`served_passes.py verify --manifest`, J processes), whose
# own verdict nothing here reads; each pass's documents are copied into $OUT (this run's out/).
#   research run --on MACHINE --project verity --source <run tree> --cwd clone \
#     --env GPU_LEASE_WHO=<you> -- bash benchmarks/pouw/pearl_c_vllm/verify.sh <window run id> [J]
# --cwd clone: the Lean step builds pouw-pearlplus in the tree (`served_passes.py lean --exe build`) unless
# PEARLPLUS_LEAN_EXE names a built one.  After window.sh's PEARLC_KERNEL=pearl_c_sm90 window,
# WORK=/workspace/research/pearl-c-sm90 and VERITY_BENCH_VENV names the node's Python (window.sh's).
# Each pass is verified against the epoch's coin commitment, $PASSES/coins.json, at DRAWS samples (default 3) on the node's
# own sample, its coin record $PASSES/<pass>/coin.json (a rerun takes the same sample: a retry never redraws).  A window
# recorded before epoch coins has no coins.json and is refused: the Lean refuses its manifests.
# FAST=1 replays the twin on fast_tile's batch forms (`--fast`), its first replayed tile and every CROSS-th after it in the
# sample's order (default 256) on the scalar reference too.  TEST_INSTANCE=1: the passes are a test instance's
# (`--test-instance`; the verdict credits nothing).
# Writes $OUT/<pass>-verify.json (the twin's document), <pass>-lean-check.json (both verdicts, the Lean's command line and
# its line), <pass>-lean.json (the Lean's answer, where it gave one the twin agrees with) and <pass>-manifest.json.
# Exits 0 only where the Lean accepted the prefill and decode passes and rejected every negative control's pass (control,
# and control-leaves where the window retained it); a pass the Lean gave no verdict on (it refused, disagreed with the twin
# or didn't build) fails it.  An ACCEPT is replay-verified (development shortcut; the protocol's check is a ZK proof,
# PouwCheck).
set -euo pipefail
WORK=${WORK:-/workspace/pouw/mvp-e2e}; PY=${VERITY_BENCH_VENV:-$WORK/venv312}/bin/python
PASSES=$WORK/passes/${1:?the window run id}; JOBS=${2:-48}; OUT=${OUT:-${RESEARCH_RUN_DIR:-.}}
export PYTHONPATH=$PWD:$PWD/tools/config:$PWD/adapters/vllm:$PWD/benchmarks/pouw
SP=benchmarks/pouw/pearl_c_vllm/served_passes.py
if [ ! -f "$PASSES/coins.json" ]; then
  echo "verify.sh: no epoch coin commitment at $PASSES/coins.json: a window recorded before epoch coins has no verdict" \
    "the Lean gives (it refuses its manifests)" >&2
  exit 2
fi
status=0
for pass in prefill decode control control-leaves; do
  D=$PASSES/$pass
  [ "$pass" = control-leaves ] && [ ! -f "$D/manifest.json" ] && continue
  rm -f "$D/pearlplus-verify.json" "$D/pearlplus-lean-check.json"
  "$PY" "$SP" verify --passes "$PASSES" --manifest "$D/manifest.json" --coin-record "$D/coin.json" \
    --out "$D/pearlplus-verify.json" --answer "$D/pearlplus-answer.json" --draws "${DRAWS:-3}" --jobs "$JOBS" \
    ${FAST:+--fast --cross-check ${CROSS:-256}} ${TEST_INSTANCE:+--test-instance} || true
  rc=0
  "$PY" "$SP" lean --passes "$PASSES" --manifest "$D/manifest.json" --verify "$D/pearlplus-verify.json" \
    --exe "${PEARLPLUS_LEAN_EXE:-build}" --out "$D/pearlplus-lean.json" || rc=$?
  case "$pass" in control*) want=1 verdict=REJECT;; *) want=0 verdict=ACCEPT;; esac
  if [ "$rc" = "$want" ] && [ -s "$D/pearlplus-lean.json" ]; then
    echo "verify.sh: $pass: the Lean's $verdict, as a ${pass%%-*} pass must have"
  else
    echo "verify.sh: $pass: the Lean's exit $rc, where a ${pass%%-*} pass must have its $verdict (exit $want)"
    status=1
  fi
  cp "$D/manifest.json" "$OUT/$pass-manifest.json"
  for f in verify lean-check lean; do
    if [ -f "$D/pearlplus-$f.json" ]; then cp "$D/pearlplus-$f.json" "$OUT/$pass-$f.json"; fi
  done
done
exit $status
