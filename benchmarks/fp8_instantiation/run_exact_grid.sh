#!/usr/bin/env bash
# exact_grid.py on one leased GPU of vy-nebius-2 (cwd = the clone's root): the selftest, then every method in one process
# over unsloth/Llama-3.1-8B @ 3f0d51f8 on the capture's 2048 tokens (ppl.py's eval).  Clocks are read, never set.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-12 --source . --cwd clone --custody-ttl 12h \
#     --env GPU_LEASE_WHO=... [--env Q1_METHODS="..."] [--env D4_WINDOWS=N] \
#     --send benchmarks/fp8_instantiation/run_exact_grid.sh -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/run_exact_grid.sh"'
#
# D4_WINDOWS=N evaluates WikiText-2 test windows 0..N-1 (`windows.py`, window 0 checked against the capture's ids).
set -uo pipefail
export O=${RESEARCH_RUN_DIR:?}/out
mkdir -p "$O"
export PY=/workspace/jobs/venv312/bin/python
export SNAP=/workspace/hf/hub/models--unsloth--Llama-3.1-8B/snapshots/3f0d51f8e5640f98f1a96ea9044a0e55c0a83814
export IDS=/workspace/pouw/fp8-security/llama8b-w0/ids.npy
if [ -n "${D4_WINDOWS:-}" ]; then
  WT=/workspace/hf/datasets/Salesforce___wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-test.arrow
  "$PY" benchmarks/fp8_instantiation/windows.py "$SNAP" "$WT" "$IDS" "$D4_WINDOWS" "$O/windows.npy" > "$O/windows.json" || exit 4
  export IDS=$O/windows.npy
fi
export METHODS=${Q1_METHODS:-bf16 int8 e4m3i16 e4m3ik e2m1i fp8_tensor fp8_token fp8_block nvfp4 e4m3i16_rot e4m3ik_rot e2m1i_rot e4m3i448 e2m1i_rot_wclip int8_rot}
LEASE=(gpu-lease 1 --wait --max-min "${Q1_MAX_MIN:-60}" --)
[ -n "${GPU_LEASE_UUID:-}" ] && LEASE=()
# shellcheck disable=SC2016
"${LEASE[@]}" bash -c '
  S=benchmarks/fp8_instantiation/exact_grid.py
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,power.limit --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  "$PY" "$S" --selftest > "$O/selftest.json" 2> "$O/selftest_stderr.txt" || exit 3
  "$PY" "$S" "$SNAP" "$IDS" "$O" $METHODS > "$O/results.jsonl" 2> "$O/stderr.txt"
  r=$?
  nvidia-smi --query-gpu=index,clocks.sm,clocks.mem --format=csv -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  exit $r'
rc=$?
echo "exact_grid $rc" | tee "$O/status.txt"
tail -3 "$O/selftest_stderr.txt" "$O/stderr.txt" 2>/dev/null
exit $rc
