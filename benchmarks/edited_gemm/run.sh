#!/usr/bin/env bash
# The adopted design's edited GEMM against plain int8 at o_proj (m = 8192, n = k = 4096), on one leased sm_120 GPU
# (cwd = the source tree's root). Builds egemm.cu outside the lease and runs the SASS gate on it; then, gates before
# timing: lease 1 runs the device gates (a small shape and o_proj, poisoned buffers, a no-write control) and dumps tiles;
# egemm_ref.py recomputes them on the CPU from x, W and the key; only if both pass, lease 2 times hot items (EG_ARMS), the
# slice's per-call interleaving (IL pairs ARM:BASE) and ncu's issue and pipe counters. Clocks are read, never set.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd source --env GPU_LEASE_WHO=EGEMM \
#     -- bash benchmarks/edited_gemm/run.sh
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
export D=benchmarks/edited_gemm
export BIN=$(mktemp -d /tmp/egemm.XXXXXX)
trap 'rm -rf "$BIN"' EXIT
PY=${EG_PYTHON:-python3}
git rev-parse HEAD > "$O/commit.txt" 2>&1
nvcc -O3 -std=c++17 --no-compress -gencode arch=compute_120a,code=sm_120a -Xptxas -v -o "$BIN/egemm" $D/egemm.cu -lcuda -lcublasLt -lnvidia-ml \
  > "$O/build.log" 2>&1
rc=$?
echo "build rc $rc" | tee -a "$O/rc.txt"
[ "$rc" = 0 ] || { tail -40 "$O/build.log"; exit 4; }
$PY benchmarks/pouw/harness/sass_gate.py "$BIN/egemm" --json "$O/sass_gate.json" > "$O/sass_gate.txt" 2>&1
echo "sass_gate (whole binary) rc $?" | tee -a "$O/sass_gate.txt"
# The binary also holds fgemm.cu's, mgemm.cu's and lpgemm.cu's file-scope kernels, never launched here, and the gate refuses
# an integer-division seed in some of them; the gate passes only if every flagged instruction is in one of those.
$PY - "$O/sass_gate.json" > "$O/sass_scope.txt" 2>&1 <<'EOF'
import json, re, sys
UNLAUNCHED = {"f_plain", "o_plain", "r_plain", "d_duo32", "d_duo2_32", "d_add3_32", "d_mul2_32", "w_duo2_32"}
g = json.load(open(sys.argv[1]))
bad = []
for b in g["binaries"]:
    if b["ptx_bad_n"] or not b["cubins"]:
        bad.append(f"{b['path']}: ptx_bad {b['ptx_bad_n']}, cubins {b['cubins']}")
    for o in b["offending"]:
        m = re.match(r"_Z\d+(\w+?)4Args$", o["function"])
        if not m or m.group(1) not in UNLAUNCHED:
            bad.append(f"{o['function']} {o['address']}: {o['instruction']} ({o.get('why', '')})")
if not g["ok"] and not all(b["offending"] for b in g["binaries"]):
    bad.append(f"the gate failed without a flagged instruction: {g.get('error')}")
print("\n".join(bad) if bad else f"pass: flagged only in unlaunched kernels {sorted(UNLAUNCHED)}" if not g["ok"] else "pass")
sys.exit(7 if bad else 0)
EOF
echo "sass_gate rc $?" | tee -a "$O/rc.txt"
cuobjdump -sass "$BIN/egemm" > "$BIN/egemm.sass" 2>/dev/null
awk '/Function :/{f=$3} /[[:space:]](LDL|STL)[.[:space:]]/{n[f]++} END{for (f in n) print n[f], f}' "$BIN/egemm.sass" | sort -k2 \
  > "$O/sass_local_mem_ops.txt" 2>&1
export W="$BIN/gates"
mkdir -p "$W/small" "$W/o"
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min 6 -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.limit \
    --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  "$BIN/egemm" gate "$W/small" 256 256 512 0x1234 >> "$O/gate.jsonl" 2>> "$O/gate.err"; echo "gate small rc $?" >> "$O/rc.txt"
  "$BIN/egemm" gate "$W/o" 8192 4096 4096 0x5678 >> "$O/gate.jsonl" 2>> "$O/gate.err"; echo "gate o rc $?" >> "$O/rc.txt"'
echo "lease 1 rc $?" >> "$O/rc.txt"
cp "$W/small/meta.json" "$O/gate_small_meta.json" 2>/dev/null; cp "$W/o/meta.json" "$O/gate_o_meta.json" 2>/dev/null
$PY $D/egemm_ref.py "$W/small" "$W/o" > "$O/ref.jsonl" 2> "$O/ref.err"
ref=$?
echo "ref rc $ref" | tee -a "$O/rc.txt"
if [ "$ref" != 0 ] || grep -q 'rc [1-9]' "$O/rc.txt"; then
  echo "a gate failed: nothing timed"; cat "$O/rc.txt"; cat "$O/gate.jsonl" "$O/ref.jsonl"; exit 5
fi
export REPS=${REPS:-9}
export EG_ARMS=${EG_ARMS:-plain,plainq,qplain,qdither,wedit,e_plain,e_plain_nc,t_duo32,e_l1t,e_l1h,e_l2,e_l2h,e_l2h_nc,e_l2q,e_l3h,e_l3q,e_l4h,edited:e_l2h,overlap:e_l2h,editedq:e_l2h,overlapq:e_l2h}
export IL=${IL:-"edited:e_l2h:plain overlap:e_l2h:plain editedq:e_l2h:plainq overlapq:e_l2h:plainq edited:e_l2q:plain edited:e_l3q:plain e_plain:plain e_plain_nc:plain"}
export NCU=${NCU:-"e_plain e_plain_nc e_l2h e_l2h_nc"}
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min 12 -- bash -c '
  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu --format=csv -lms 500 -i "$GPU_LEASE_UUID" > "$O/smi.csv" 2>&1 &
  smi=$!
  "$BIN/egemm" bench "$REPS" 8192 4096 4096 > "$O/bench.jsonl" 2> "$O/bench.err"; echo "bench rc $?" >> "$O/rc.txt"
  for p in $IL; do
    base=${p##*:}; arm=${p%:*}
    "$BIN/egemm" il 41 "$arm" "$base" 8192 4096 4096 >> "$O/il.jsonl" 2>> "$O/il.err"; echo "il $p rc $?" >> "$O/rc.txt"
  done
  kill $smi
  M=gpu__time_duration.sum,sm__cycles_elapsed.avg,sm__inst_executed.sum,sm__inst_executed_pipe_tensor.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active,smsp__issue_active.avg.pct_of_peak_sustained_active,smsp__warps_eligible.avg.per_cycle_active
  for a in $NCU; do
    EG_ARMS=$a ncu --clock-control none --kernel-name "regex:^${a}\$" --launch-skip 4 --launch-count 1 --metrics $M --csv \
      "$BIN/egemm" bench 1 8192 4096 4096 > "$O/ncu-$a.csv" 2>&1 || echo "ncu $a rc $?" >> "$O/rc.txt"
    EG_ARMS=$a ncu --clock-control none --kernel-name "regex:^${a}\$" --launch-skip 4 --launch-count 1 --csv \
      --metrics "regex:^smsp__average_warps_issue_stalled_.*_per_issue_active\.ratio$" \
      "$BIN/egemm" bench 1 8192 4096 4096 > "$O/ncu-stall-$a.csv" 2>&1 || echo "ncu stall $a rc $?" >> "$O/rc.txt"
  done'
echo "lease 2 rc $?" >> "$O/rc.txt"
cat "$O/rc.txt"
$PY $D/summary.py "$O" | tee "$O/summary.md"
