#!/usr/bin/env bash
# attention.py on one leased GPU of vy-nebius-2 (cwd = the clone's root): the selftest, then every method in one process
# over unsloth/Llama-3.1-8B @ 3f0d51f8 at context D4A_CTX (2048: ppl.py's window-0 tokens; longer: window 0 of WikiText-2's
# test split from the node's Hugging Face cache).  Clocks are read, never set.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd clone --custody-ttl 12h \
#     --env GPU_LEASE_WHO=... --env D4A_CTX=2048 [--env D4A_METHODS="..."] --send benchmarks/fp8_instantiation/run_attention.sh \
#     -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/run_attention.sh"'
set -uo pipefail
export O=${RESEARCH_RUN_DIR:?}/out
mkdir -p "$O"
export PY=/workspace/jobs/venv312/bin/python
export SNAP=/workspace/hf/hub/models--unsloth--Llama-3.1-8B/snapshots/3f0d51f8e5640f98f1a96ea9044a0e55c0a83814
export IDS=/workspace/pouw/fp8-security/llama8b-w0/ids.npy
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
export CTX=${D4A_CTX:-2048}
if [ -z "${D4A_METHODS:-}" ]; then
  m="bf16 ref ref-rot"
  for g in int8 e4m3; do for r in "" -rot; do for p in prow pfix; do for v in vchan vhead; do
    m+=" $g-$p-$v-none$r"; done; done; done; done
  for r in "" -rot; do for a in pm1 pm13 pm57 u1 u2 u4 u8 u16; do for p in prow pfix; do for v in vchan vhead; do
    m+=" int8-$p-$v-$a$r"; done; done; done; done
  for r in "" -rot; do for a in pm1 u1 u2 pm13 u4 u8 pm57; do for p in prow pfix; do m+=" e4m3-$p-vchan-$a$r"; done; done; done
  for r in "" -rot; do for a in u1 u4 u16; do for w in qk pv; do m+=" int8-prow-vchan-$a-$w$r"; done; done; done
  D4A_METHODS=$m
fi
export D4A_METHODS
LEASE=(gpu-lease 1 --wait --max-min "${D4A_MAX_MIN:-60}" --)
[ -n "${GPU_LEASE_UUID:-}" ] && LEASE=()
# shellcheck disable=SC2016
"${LEASE[@]}" bash -c '
  S=benchmarks/fp8_instantiation/attention.py
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,power.limit --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  "$PY" "$S" --selftest > "$O/selftest.json" 2> "$O/selftest_stderr.txt" || exit 3
  "$PY" "$S" "$SNAP" "$IDS" "$O" --ctx "$CTX" $D4A_METHODS > "$O/results.jsonl" 2> "$O/stderr.txt"
  r=$?
  nvidia-smi --query-gpu=index,clocks.sm,clocks.mem --format=csv -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  "$PY" "$S" --table "$O/results.jsonl" > "$O/table.md" 2>> "$O/stderr.txt"
  exit $r'
rc=$?
echo "attention $rc" | tee "$O/status.txt"
tail -3 "$O/selftest_stderr.txt" "$O/stderr.txt" 2>/dev/null
exit $rc
