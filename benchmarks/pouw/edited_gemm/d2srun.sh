#!/usr/bin/env bash
# Dither-2S's linear (d2sgemm.cu) on one leased sm_120 GPU (cwd = the source tree's root). Builds d2sgemm.cu outside the
# lease and runs the SASS gate on it; then, gates before timing: lease 1 runs the device gates (poisoned buffers, Y against
# cuBLASLt's X W'^T, d_l2q's receipts against d_l7qs's, the fused producer's operands, the overlap arm, a no-write control)
# at two small shapes, Llama-3.1-8B's four linears at m = 2048 tokens (qkv, o_proj, gate_up, down) and 8192^3, and dumps
# three tiles of each; d2sgemm_ref.py recomputes them on the CPU from x, W0 and the key; only if every gate passes, lease
# 2 times D2_ARMS at each shape (interleaved hot items) and runs ncu's sections on NCU_ONE at each shape and NCU_VEC (the
# operand passes) at NCU_VEC_SHAPES (out/ncu/, read by eps_profile.py, whose profile and TOML end the run). A shape
# NAME:H:HKV:T:qk is Q K^T with Q and K dithered (qkgate, qkbench, NCU_QK through qkone). Clocks are read, never set.
#
#   research run --on MACHINE --project verity --source . --cwd source --env GPU_LEASE_WHO=LANE \
#     -- bash benchmarks/pouw/edited_gemm/d2srun.sh
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
export D=benchmarks/pouw/edited_gemm
export BIN=$(mktemp -d /tmp/d2sgemm.XXXXXX)
trap 'rm -rf "$BIN"' EXIT
PY=${D2_PYTHON:-python3}
git rev-parse HEAD > "$O/commit.txt" 2>&1
nvcc -O3 -std=c++17 --no-compress -gencode arch=compute_120a,code=sm_120a -Xptxas -v -o "$BIN/d2sgemm" $D/d2sgemm.cu -lcuda -lcublasLt -lnvidia-ml \
  > "$O/build.log" 2>&1
rc=$?
echo "build rc $rc" | tee -a "$O/rc.txt"
[ "$rc" = 0 ] || { tail -40 "$O/build.log"; exit 4; }
$PY benchmarks/pouw/harness/sass_gate.py "$BIN/d2sgemm" --json "$O/sass_gate.json" > "$O/sass_gate.txt" 2>&1
echo "sass_gate (whole binary) rc $?" | tee -a "$O/sass_gate.txt"
# The binary holds egemm.cu's kernels and so fgemm.cu's, mgemm.cu's and lpgemm.cu's, some never launched; the gate refuses
# an integer-division seed in some of those, and passes only if every flagged instruction is in one of them (crun.sh's set)
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
cuobjdump -sass "$BIN/d2sgemm" > "$BIN/d2sgemm.sass" 2>/dev/null
awk '/Function :/{f=$3} /[[:space:]](LDL|STL)[.[:space:]]/{n[f]++} END{for (f in n) print n[f], f}' "$BIN/d2sgemm.sass" | sort -k2 \
  > "$O/sass_local_mem_ops.txt" 2>&1
gzip -c "$BIN/d2sgemm.sass" > "$O/d2sgemm.sass.gz"
# small, small8 (k = 8192: e_l7qs's 3-atom tail beside it), Llama-3.1-8B's linears at 2048 tokens, and 8192^3 (o)
# and Q K^T (NAME:H:HKV:T:qk): a small one and Llama-3.1-8B's at 2048 tokens (32 query heads, 8 KV heads)
export SHAPES=${SHAPES:-"small:1024:256:512 small8:1024:256:8192 q2k:2048:6144:4096 o2k:2048:4096:4096 gu2k:2048:28672:4096 dn2k:2048:4096:14336 o:8192:8192:8192 qks:2:1:1024:qk qk2k:32:8:2048:qk"}
export W="$BIN/gates"
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min 10 -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.limit \
    --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  for s in $SHAPES; do
    IFS=: read -r name m n k kind <<< "$s"
    mkdir -p "$W/$name"
    "$BIN/d2sgemm" ${kind:+qk}gate "$W/$name" "$m" "$n" "$k" 0x9ABC >> "$O/gate.jsonl" 2>> "$O/gate.err"; echo "gate $name rc $?" >> "$O/rc.txt"
  done'
echo "lease 1 rc $?" >> "$O/rc.txt"
dirs=""
for s in $SHAPES; do
  name=${s%%:*}
  dirs="$dirs $W/$name"
  cp "$W/$name/meta.json" "$O/gate_${name}_meta.json" 2>/dev/null
done
# shellcheck disable=SC2086
$PY $D/d2sgemm_ref.py $dirs > "$O/ref.jsonl" 2> "$O/ref.err"
ref=$?
echo "ref rc $ref" | tee -a "$O/rc.txt"
if [ "$ref" != 0 ] || grep -q 'rc [1-9]' "$O/rc.txt"; then
  echo "a gate failed: nothing timed"; cat "$O/rc.txt"; cat "$O/gate.jsonl" "$O/ref.jsonl"; tail -20 "$O/ref.err"
  mkdir -p "$O/gates" && cp -r "$W"/* "$O/gates/" 2>/dev/null
  exit 5
fi
[ "${D2_GATES_ONLY:-0}" = 1 ] && { echo "gates only"; cat "$O/rc.txt"; exit 0; }
export REPS=${REPS:-15}
export D2_ARMS=${D2_ARMS:-"plain,plainq,lt_d2s,e_plains,d_plains,e_l7qs,d_l7qs,d_l2q,editedq:e_l7qs,dedited:d_l7qs,deditedf:d_l7qs,overlapq:e_l7qs,dov:d_l7qs,qplain,qdither,wedit,d2quant,d2wedit,d2ops,d2qwe"}
export NCU_ONE=${NCU_ONE:-"plain lt_d2s e_plains d_plains e_l7qs d_l7qs"}
export NCU_VEC=${NCU_VEC:-"qplain d2quant d2wedit d2qwe"} NCU_VEC_SHAPES=${NCU_VEC_SHAPES:-"o2k gu2k dn2k"}
export NCU_QK=${NCU_QK:-"qk_ltq qk_plains qk_l7qs qk_plains_m qk_l7qs_m qkq_plain qkq_d"}
[ "$NCU_VEC" = - ] && NCU_VEC=""
export BENCH_SHAPES=${BENCH_SHAPES:-"q2k o2k gu2k dn2k o qk2k"}
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min 30 -- bash -c '
  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu --format=csv -lms 500 -i "$GPU_LEASE_UUID" > "$O/smi.csv" 2>&1 &
  smi=$!
  for s in $SHAPES; do
    IFS=: read -r name m n k kind <<< "$s"
    case " $BENCH_SHAPES " in *" $name "*) ;; *) continue ;; esac
    "$BIN/d2sgemm" ${kind:+qk}bench "$REPS" "$m" "$n" "$k" >> "$O/bench.jsonl" 2>> "$O/bench.err"; echo "bench $name rc $?" >> "$O/rc.txt"
  done
  kill $smi
  mkdir -p "$O/ncu"
  S=(--section SpeedOfLight --section ComputeWorkloadAnalysis --section InstructionStats --section WarpStateStats
     --section SchedulerStats --section SourceCounters --section LaunchStats --section Occupancy --section MemoryWorkloadAnalysis)
  for s in $SHAPES; do
    IFS=: read -r name m n k kind <<< "$s"
    case " $BENCH_SHAPES " in *" $name "*) ;; *) continue ;; esac
    arms_=$NCU_ONE
    case " $NCU_VEC_SHAPES " in *" $name "*) arms_="$arms_ $NCU_VEC" ;; esac
    [ -n "$kind" ] && arms_=$NCU_QK
    for a in $arms_; do
      ncu --clock-control none --profile-from-start off --import-source no "${S[@]}" -f -o "$O/ncu/$name-${a//:/_}" \
        "$BIN/d2sgemm" ${kind:+qk}one "$a" "$m" "$n" "$k" > "$O/ncu/$name-${a//:/_}.log" 2>&1 || echo "ncu one $name $a rc $?" >> "$O/rc.txt"
    done
  done'
echo "lease 2 rc $?" >> "$O/rc.txt"
for r in "$O"/ncu/*.ncu-rep; do
  [ -e "$r" ] || continue
  b=${r%.ncu-rep}
  ncu -i "$r" --page raw --csv --print-metric-instances values > "$b.raw.csv" 2> "$b.raw.err" ||
    ncu -i "$r" --page raw --csv > "$b.raw.csv" 2>> "$b.raw.err"
  ncu -i "$r" --page details --csv > "$b.details.csv" 2> "$b.details.err"
  ncu -i "$r" --page source --csv --print-source sass > "$b.sass.csv" 2> "$b.sass.err"
done
cat "$O/rc.txt"
$PY $D/eps_profile.py "$O" --mm benchmarks/pouw/machine_model --json "$O/profile.json" --toml "$O/profile.toml" \
  --toml-arms "${PROFILE_ARMS:-d_l7qs,d_plains,e_l7qs,e_plains,lt_d2s,plain,qk_l7qs,qk_plains,qk_l7qs_m,qk_plains_m}" \
  > "$O/profile.md" 2> "$O/profile.err"
echo "profile rc $?" | tee -a "$O/rc.txt"
