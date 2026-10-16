#!/usr/bin/env bash
# EGEMM's edited GEMM with checkpointed receipts (wgemm.cu) against plain int8 at o_proj (m = 8192, n = k = 4096), on one
# leased sm_120 GPU (cwd = the source tree's root). Builds wgemm.cu outside the lease (its local-memory ops counted), the
# venv by build.sh; lease 1 runs the device gates (256 x 256 x 4096 and o_proj, poisoned buffers) and dumps X, W' and the
# receipts; wgate.py recomputes them with derive.win_receipts; only if both pass, lease 2 times the per-call pairs IL
# (ARM:BASE, wgemm il). Clocks are read, never set. Under $OUT: gate.jsonl, wgate.jsonl, il.jsonl, rc.txt.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd source --env GPU_LEASE_WHO=E2E \
#     -- bash benchmarks/e2e_slice/wgemm.sh
# IL (the pairs), IL_REPS (41).
set -uo pipefail
export O=${OUT:?} D=benchmarks/e2e_slice
export PATH=/usr/local/cuda-13.0/bin:$PATH
export BIN=$(mktemp -d /tmp/e2e-wgemm.XXXXXX)
trap 'rm -rf "$BIN"' EXIT
git rev-parse HEAD > "$O/commit.txt" 2>&1
nvcc -O3 -std=c++17 --no-compress -gencode arch=compute_120a,code=sm_120a -Xptxas -v -o "$BIN/wgemm" $D/wgemm.cu -lcuda -lcublasLt \
  -lnvidia-ml > "$O/build.log" 2>&1
rc=$?
echo "build rc $rc" | tee -a "$O/rc.txt"
[ "$rc" = 0 ] || { tail -40 "$O/build.log"; exit 4; }
cuobjdump -sass "$BIN/wgemm" > "$BIN/wgemm.sass" 2>/dev/null
awk '/Function :/{f=$3} /[[:space:]](LDL|STL)[.[:space:]]/{n[f]++} END{for (f in n) print n[f], f}' "$BIN/wgemm.sass" | sort -k2 \
  > "$O/sass_local_mem_ops.txt" 2>&1
grep -A2 -E 'Compiling entry function .*(w_l|e_l2q|e_l3qm|e_plain)' "$O/build.log" | grep -E 'entry|registers|spill' > "$O/ptxas.txt"
bash $D/build.sh > "$O/venv.log" 2>&1 || { tail -20 "$O/venv.log"; exit 2; }
PY=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["python"])' "$O/builds.json")
export PYTHONPATH=$PWD
export W="$BIN/gates"
mkdir -p "$W/small" "$W/o"
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min 6 -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.limit \
    --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  "$BIN/wgemm" gate "$W/small" 256 256 4096 0x1234 >> "$O/gate.jsonl" 2>> "$O/gate.err"; echo "gate small rc $?" >> "$O/rc.txt"
  "$BIN/wgemm" gate "$W/o" 8192 4096 4096 0x5678 >> "$O/gate.jsonl" 2>> "$O/gate.err"; echo "gate o rc $?" >> "$O/rc.txt"'
echo "lease 1 rc $?" >> "$O/rc.txt"
$PY $D/wgate.py "$W/small" "$W/o" > "$O/wgate.jsonl" 2> "$O/wgate.err"
echo "wgate rc $?" | tee -a "$O/rc.txt"
if grep -q 'rc [1-9]' "$O/rc.txt"; then
  echo "a gate failed: nothing timed"; cat "$O/rc.txt" "$O/gate.jsonl" "$O/wgate.jsonl"; tail -20 "$O/gate.err" "$O/wgate.err"; exit 5
fi
export IL_REPS=${IL_REPS:-41}
export IL=${IL:-"edited:e_l2q:plain edited:w_l2q_s4:plain edited:w_l2q_s8:plain edited:w_l2q_s16:plain edited:w_l2q_s32:plain
  e_l2q:plain w_l2q_s8:plain e_plain:plain edited:w_l3qm_s8:plain edited:w_l3qm_s16:plain editedf:e_l3qm:plainq
  editedf:w_l3qm_s8:plainq editedf:w_l2q_s8:plainq"}
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min 10 -- bash -c '
  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu --format=csv -lms 500 -i "$GPU_LEASE_UUID" > "$O/smi.csv" 2>&1 &
  smi=$!
  for p in $IL; do
    base=${p##*:}; arm=${p%:*}
    "$BIN/wgemm" il "$IL_REPS" "$arm" "$base" 8192 4096 4096 >> "$O/il.jsonl" 2>> "$O/il.err"; echo "il $p rc $?" >> "$O/rc.txt"
  done
  kill $smi'
echo "lease 2 rc $?" >> "$O/rc.txt"
cat "$O/rc.txt"
python3 - "$O/il.jsonl" <<'EOF' | tee "$O/il.md"
import json, sys
print("| arm | base | ms | base ms | ratio | MHz |\n|---|---|---|---|---|---|")
for l in open(sys.argv[1]):
    r = json.loads(l)
    print(f"| {r['il']} | {r['base']} | {r['ms']:.4f} | {r['base_ms']:.4f} | {r['ratio']:.3f} | {r['mhz']:.0f} |")
EOF
