#!/usr/bin/env bash
# ppl.py on vy-nebius-2's CPU (cwd = the clone's root): the selftest on a tiny random Llama, then one process per method
# over unsloth/Llama-3.1-8B @ 3f0d51f8 on the capture's 2048 tokens.  pearlc_capture.py (for its streamed forward) is
# sent as an input.
#
#   research run --on vy-nebius-2 --project verity --campaign pouw --source . --cwd clone --custody-ttl 12h \
#     --send benchmarks/fp8_instantiation/run_ppl.sh --send pearlc_capture.py -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/run_ppl.sh"'
set -uo pipefail
O=${RESEARCH_RUN_DIR:?}/out
mkdir -p "$O"
export PYTHONPATH=$PWD:$PWD/benchmarks/fp8_instantiation OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12
PY=/workspace/jobs/venv312/bin/python
CAP="$RESEARCH_RUN_DIR/inputs/pearlc_capture.py"
SNAP=/workspace/hf/hub/models--unsloth--Llama-3.1-8B/snapshots/3f0d51f8e5640f98f1a96ea9044a0e55c0a83814
IDS=/workspace/pouw/fp8-security/llama8b-w0/ids.npy
SRC=benchmarks/fp8_instantiation
$PY $SRC/ppl.py "$CAP" --selftest > "$O/selftest.json" 2> "$O/selftest_stderr.txt"
rc=$?
echo "selftest $rc" | tee "$O/status.txt"
[ $rc = 0 ] || exit $rc
pids=() names=()
for m in ${A4_METHODS:-bf16 fp8 pearl_v1 pearl_fix pearl_v1_split pearl_fix_split int8n6 int8n6_chunk}; do
  $PY $SRC/ppl.py "$CAP" "$SNAP" "$IDS" $m --threads 12 --json "$O/ppl_$m.json" > "$O/ppl_${m}_stdout.txt" 2>&1 &
  pids+=($!) names+=($m)
done
fail=0
for i in "${!pids[@]}"; do wait "${pids[$i]}"; r=$?; echo "ppl_${names[$i]} $r" | tee -a "$O/status.txt"; [ $r = 0 ] || fail=1; done
exit $fail
