#!/usr/bin/env bash
# cost.py on vy-nebius-2's CPU, where the Llama-3.1-8B captures are (cwd = the clone's root).  Stage 1: `cost.py check`,
# then one process per captured decoder layer of the Instruct capture (llama8b-w0).  Stage 2 (A4_BASE=1): the base model
# (unsloth/Llama-3.1-8B at 3f0d51f8) captured on the same tokens by pearlc_capture.py (sent as an input) into
# /workspace/pouw/fp8-security/llama8b-base-w0, then the same per-layer runs.
#
#   research run --on vy-nebius-2 --project verity --campaign pouw --source . --cwd clone --custody-ttl 12h \
#     --send benchmarks/fp8_instantiation/run_node2.sh -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/run_node2.sh"'
set -uo pipefail
O=${RESEARCH_RUN_DIR:?}/out
mkdir -p "$O"
export PYTHONPATH=$PWD:$PWD/benchmarks/fp8_instantiation OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/workspace/jobs/venv312/bin/python
SRC=benchmarks/fp8_instantiation
CAPS=/workspace/pouw/fp8-security
$PY -c 'import sys, numpy; print(sys.version.split()[0], numpy.__version__)' > "$O/python.txt" 2>&1
$PY $SRC/cost.py check > "$O/check.json" 2> "$O/check_stderr.txt"
echo "check $?" | tee "$O/status.txt"

layers() {  # $1 capture dir, $2 tag
  local pids=() l
  for l in $(seq 0 4 28); do
    $PY $SRC/cost.py run "$1" --rows 256 --cols 256 --layers $l --draws 4000 --json "$O/cost_$2_L$l.json" \
      > /dev/null 2> "$O/cost_$2_L${l}_stderr.txt" &
    pids+=($!)
  done
  for l in "${!pids[@]}"; do wait "${pids[$l]}"; echo "cost_$2_L$((4 * l)) $?" | tee -a "$O/status.txt"; done
}
layers $CAPS/llama8b-w0 instruct

if [ "${A4_BASE:-}" = 1 ]; then
  CAP=$CAPS/llama8b-base-w0
  mkdir -p "$CAP"
  cp -n $CAPS/llama8b-w0/ids.npy "$CAP/ids.npy"
  cp "$RESEARCH_RUN_DIR/inputs/pearlc_capture.py" "$O/pearlc_capture.py"
  sed -i 's/^    from datasets import load_dataset$/    try:\n        from datasets import load_dataset\n    except ImportError:\n        load_dataset = None/' "$O/pearlc_capture.py"
  HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 $PY "$O/pearlc_capture.py" --model unsloth/Llama-3.1-8B \
    --revision 3f0d51f8e5640f98f1a96ea9044a0e55c0a83814 --every 4 --threads 32 --out "$CAP" > "$O/capture_base.txt" 2>&1
  rc=$?
  echo "capture_base $rc" | tee -a "$O/status.txt"
  cp "$CAP/manifest.json" "$O/base_manifest.json" 2>/dev/null
  [ $rc = 0 ] && layers "$CAP" base
fi
