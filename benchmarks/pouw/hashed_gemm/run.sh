#!/usr/bin/env bash
# The edited GEMM with its commitments hashed in its own CTAs (hgemm.cu) against plain int8 at Llama-8B's linears (m = 8192
# tokens; o_proj n = k = 4096, qkv n = 6144, gate_up n = 28672, down n = 4096 and k = 14336), on one leased sm_120 GPU
# (cwd = the source tree's root). Builds hgemm.cu outside the lease and runs the SASS gate on it; then, gates before timing:
# lease 1 runs the device gates (two small shapes with few CTAs, ragged trees and, at the second, the 3-atom peel and a
# short deferral; then every linear at the full grid), each on poisoned buffers over several calls, and hgref.py judges
# every level of every tree and sampled tiles' receipts by the repository's functions; only if all pass, lease 2 times
# each linear's 8-call sequences (HG_ARMS) and ncu's issue and pipe counters at o_proj. Clocks are read, never set.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd source --env GPU_LEASE_WHO=LANE \
#     -- bash benchmarks/pouw/hashed_gemm/run.sh
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
export D=benchmarks/pouw/hashed_gemm
export BIN=$(mktemp -d /tmp/hgemm.XXXXXX)
trap 'rm -rf "$BIN"' EXIT
export PY=${HG_PYTHON:-python3}
export PYTHONPATH=.
git rev-parse HEAD > "$O/commit.txt" 2>&1
nvcc -O3 -std=c++17 --no-compress -gencode arch=compute_120a,code=sm_120a -Xptxas -v -o "$BIN/hgemm" $D/hgemm.cu -lcuda -lcublasLt -lnvidia-ml \
  > "$O/build.log" 2>&1
rc=$?
echo "build rc $rc" | tee -a "$O/rc.txt"
[ "$rc" = 0 ] || { tail -40 "$O/build.log"; exit 4; }
$PY benchmarks/pouw/harness/sass_gate.py "$BIN/hgemm" --json "$O/sass_gate.json" > "$O/sass_gate.txt" 2>&1
echo "sass_gate (whole binary) rc $?" | tee -a "$O/sass_gate.txt"
# The binary also holds fgemm.cu's, mgemm.cu's and lpgemm.cu's file-scope kernels, never launched here, and the gate refuses
# an integer-division seed in some of them; and commit_gpu's salt_kernel (E2E's committer, verbatim), launched only by the
# serial +commit arm, whose u16 division's reciprocal body isn't a pinned one. The gate passes only if every flagged
# instruction is in one of those.
$PY - "$O/sass_gate.json" > "$O/sass_scope.txt" 2>&1 <<'EOF'
import json, re, sys
UNLAUNCHED = {"f_plain", "o_plain", "r_plain", "d_duo32", "d_duo2_32", "d_add3_32", "d_mul2_32", "w_duo2_32"}
COMMIT_ARM = ("_Z11salt_kernelILb1ELb1EE",)
g = json.load(open(sys.argv[1]))
bad = []
for b in g["binaries"]:
    if b["ptx_bad_n"] or not b["cubins"]:
        bad.append(f"{b['path']}: ptx_bad {b['ptx_bad_n']}, cubins {b['cubins']}")
    for o in b["offending"]:
        m = re.match(r"_Z\d+(\w+?)4Args$", o["function"])
        if o["function"].startswith(COMMIT_ARM):
            continue
        if not m or m.group(1) not in UNLAUNCHED:
            bad.append(f"{o['function']} {o['address']}: {o['instruction']} ({o.get('why', '')})")
if not g["ok"] and not all(b["offending"] for b in g["binaries"]):
    bad.append(f"the gate failed without a flagged instruction: {g.get('error')}")
print("\n".join(bad) if bad else f"pass: flagged only in unlaunched kernels {sorted(UNLAUNCHED)}" if not g["ok"] else "pass")
sys.exit(7 if bad else 0)
EOF
echo "sass_gate rc $?" | tee -a "$O/rc.txt"
cuobjdump -sass "$BIN/hgemm" > "$BIN/hgemm.sass" 2>/dev/null
awk '/Function :/{f=$3} /[[:space:]](LDL|STL)[.[:space:]]/{n[f]++} END{for (f in n) print n[f], f}' "$BIN/hgemm.sass" | sort -k2 \
  > "$O/sass_local_mem_ops.txt" 2>&1
# NAME:M:N:K:CALLS:ENV (ENV: HG_GRID and HG_D, comma-separated)
export GATES=${GATES:-"small:2048:1536:4096:3:HG_GRID=8 small3:1024:640:6144:3:HG_GRID=4,HG_D=5 o:8192:4096:4096:3: qkv:8192:6144:4096:2: gate_up:8192:28672:4096:2: down:8192:4096:14336:2:"}
export LAYERS=${LAYERS:-"o_proj:8192:4096:4096 qkv:8192:6144:4096 gate_up:8192:28672:4096 down:8192:4096:14336"}
export W="$BIN/gates"
for g in $GATES; do
  IFS=: read -r name m n k calls env <<< "$g"
  mkdir -p "$W/$name"
  $PY $D/hgref.py params "$W/$name" "$m" "$n" "$k" --ks-seed 7 || echo "params $name rc $?" >> "$O/rc.txt"
done
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min 10 -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.limit \
    --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  for g in $GATES; do
    IFS=: read -r name m n k calls env <<< "$g"
    env ${env//,/ } "$BIN/hgemm" gate "$W/$name" "$m" "$n" "$k" 0x9ABC "$calls" >> "$O/gate.jsonl" 2>> "$O/gate.err"
    echo "gate $name rc $?" >> "$O/rc.txt"
  done'
echo "lease 1 rc $?" >> "$O/rc.txt"
dirs=""
for g in $GATES; do dirs="$dirs $W/${g%%:*}"; cp "$W/${g%%:*}/meta.json" "$O/gate_${g%%:*}_meta.json" 2>/dev/null; done
# shellcheck disable=SC2086
$PY $D/hgref.py check $dirs > "$O/ref.jsonl" 2> "$O/ref.err"
ref=$?
echo "ref rc $ref" | tee -a "$O/rc.txt"
if [ "$ref" != 0 ] || grep -q 'rc [1-9]' "$O/rc.txt"; then
  echo "a gate failed: nothing timed"; cat "$O/rc.txt"; cat "$O/gate.jsonl"; tail -c 4000 "$O/ref.jsonl"; tail -20 "$O/ref.err"; exit 5
fi
rm -rf "$W"
export REPS=${REPS:-21}
export HG_ARMS=${HG_ARMS:-"plain edited:w_d edited:h_d edited:h_d/nohash edited:h_d96 edited:w_x edited:h_x edited:w_d+commit"}
export NCU=${NCU:-"h_d w_d h_d96"}
for l in $LAYERS; do
  IFS=: read -r name m n k <<< "$l"
  $PY $D/hgref.py params "$BIN/p_$name" "$m" "$n" "$k" --ks-seed 11 || echo "params $name rc $?" >> "$O/rc.txt"
done
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min 20 -- bash -c '
  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu --format=csv -lms 500 -i "$GPU_LEASE_UUID" > "$O/smi.csv" 2>&1 &
  smi=$!
  for l in $LAYERS; do
    IFS=: read -r name m n k <<< "$l"
    "$BIN/hgemm" pipe "$REPS" "$m" "$n" "$k" "$BIN/p_$name/px.bin" "$BIN/p_$name/pr.bin" $HG_ARMS | sed "s/^{/{\"layer\": \"$name\", /" \
      >> "$O/pipe.jsonl" 2>> "$O/pipe.err"
    echo "pipe $name rc $?" >> "$O/rc.txt"
  done
  kill $smi
  M=gpu__time_duration.sum,sm__cycles_elapsed.avg,sm__inst_executed.sum,sm__inst_executed_pipe_tensor.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active,smsp__issue_active.avg.pct_of_peak_sustained_active,smsp__warps_eligible.avg.per_cycle_active,l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum,lts__t_bytes.sum
  for a in $NCU; do
    ncu --clock-control none --kernel-name "regex:^${a}\$" --launch-skip 4 --launch-count 1 --metrics $M --csv \
      "$BIN/hgemm" pipe 1 8192 4096 4096 "$BIN/p_o_proj/px.bin" "$BIN/p_o_proj/pr.bin" "edited:$a" > "$O/ncu-$a.csv" 2>&1 || echo "ncu $a rc $?" >> "$O/rc.txt"
  done'
echo "lease 2 rc $?" >> "$O/rc.txt"
cat "$O/rc.txt"
$PY $D/summary.py "$O" | tee "$O/summary.md"
