#!/usr/bin/env bash
# Dither-C's linear (cgemm.cu, co32c) on one leased sm_120 GPU (cwd = the source tree's root). Builds cgemm.cu outside
# the lease and runs the SASS gate on it; then, gates before timing: lease 1 runs the device gates (poisoned buffers, O
# against cuBLASLt, c_l2q's receipts against c_l7qs's, a no-write control) at a small shape at m = 2 and m = 3 digits,
# Llama-3.1-8B's four linears at m = 2048 tokens (qkv, o_proj, gate_up, down) and 8192^3, and dumps three tiles of each;
# cgemm_ref.py recomputes them on the CPU through checked_strip's co32c_words from x, W and the key; only if every gate
# passes, lease 2 times CG_ARMS at each shape (interleaved hot items) and runs ncu's sections on NCU_ONE at each shape and
# NCU_VEC (the quantizers and weight editors) at NCU_VEC_SHAPES (out/ncu/, read by eps_profile.py, whose profile and TOML
# end the run). Clocks are read, never set.
#
#   research run --on MACHINE --project verity --source . --cwd source --env GPU_LEASE_WHO=LANE \
#     -- bash benchmarks/pouw/edited_gemm/crun.sh
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
export D=benchmarks/pouw/edited_gemm
export BIN=$(mktemp -d /tmp/cgemm.XXXXXX)
trap 'rm -rf "$BIN"' EXIT
PY=${CG_PYTHON:-python3}
REF=${CG_REF_PYTHON:-"uv run --no-project --with numpy --with python-sat --with-editable benchmarks/pouw/exact_kit python3"}
git rev-parse HEAD > "$O/commit.txt" 2>&1
nvcc -O3 -std=c++17 --no-compress -gencode arch=compute_120a,code=sm_120a -Xptxas -v -o "$BIN/cgemm" $D/cgemm.cu -lcuda -lcublasLt -lnvidia-ml \
  > "$O/build.log" 2>&1
rc=$?
echo "build rc $rc" | tee -a "$O/rc.txt"
[ "$rc" = 0 ] || { tail -40 "$O/build.log"; exit 4; }
$PY benchmarks/pouw/harness/sass_gate.py "$BIN/cgemm" --json "$O/sass_gate.json" > "$O/sass_gate.txt" 2>&1
echo "sass_gate (whole binary) rc $?" | tee -a "$O/sass_gate.txt"
# The binary holds egemm.cu's kernels and so fgemm.cu's, mgemm.cu's and lpgemm.cu's, some never launched; the gate refuses
# an integer-division seed in some of those, and passes only if every flagged instruction is in one of them (run.sh's set)
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
cuobjdump -sass "$BIN/cgemm" > "$BIN/cgemm.sass" 2>/dev/null
awk '/Function :/{f=$3} /[[:space:]](LDL|STL)[.[:space:]]/{n[f]++} END{for (f in n) print n[f], f}' "$BIN/cgemm.sass" | sort -k2 \
  > "$O/sass_local_mem_ops.txt" 2>&1
gzip -c "$BIN/cgemm.sass" > "$O/cgemm.sass.gz"
# small (m = 2 digits), small3 (k > 4096: m = 3), Llama-3.1-8B's linears at 2048 tokens, and 8192^3 (o, PROFILE's shape);
# a shape at k = 128 is cohalf's QK^T (keys x queries x head dimension; h8k's 4,096 tiles are 16 heads at 2,048 tokens),
# its arms CG_ARMS_H and NCU_H. CG_SET=half: cohalf's shapes, and c_l7qs_v's gate and profile beside c_l7qs at o2k, dn2k
case "${CG_SET:-all}" in
  half) SHAPES=${SHAPES:-"hs:1024:256:128 h2k:2048:2048:128 h8k:8192:8192:128 small:1024:256:512 o2k:2048:4096:4096 dn2k:2048:4096:14336"}
    BENCH_SHAPES=${BENCH_SHAPES:-"h2k h8k o2k dn2k"} CG_ARMS=${CG_ARMS:-"plain,e_l7qs,c_l7qs,c_l7qs_o,c_l7qs_v"}
    NCU_ONE=${NCU_ONE:-"c_l7qs c_l7qs_o c_l7qs_v"} NCU_VEC=${NCU_VEC:-"-"} ;;
esac
export SHAPES=${SHAPES:-"small:1024:256:512 small3:1024:256:8192 q2k:2048:6144:4096 o2k:2048:4096:4096 gu2k:2048:28672:4096 dn2k:2048:4096:14336 o:8192:8192:8192"}
export W="$BIN/gates"
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min 10 -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.limit \
    --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  for s in $SHAPES; do
    IFS=: read -r name m n k <<< "$s"
    mkdir -p "$W/$name"
    "$BIN/cgemm" gate "$W/$name" "$m" "$n" "$k" 0x9ABC >> "$O/gate.jsonl" 2>> "$O/gate.err"; echo "gate $name rc $?" >> "$O/rc.txt"
  done'
echo "lease 1 rc $?" >> "$O/rc.txt"
dirs=""
for s in $SHAPES; do
  name=${s%%:*}
  dirs="$dirs $W/$name"
  cp "$W/$name/meta.json" "$O/gate_${name}_meta.json" 2>/dev/null
done
# shellcheck disable=SC2086
$REF $D/cgemm_ref.py $dirs > "$O/ref.jsonl" 2> "$O/ref.err"
ref=$?
echo "ref rc $ref" | tee -a "$O/rc.txt"
if [ "$ref" != 0 ] || grep -q 'rc [1-9]' "$O/rc.txt"; then
  echo "a gate failed: nothing timed"; cat "$O/rc.txt"; cat "$O/gate.jsonl" "$O/ref.jsonl"; tail -20 "$O/ref.err"
  mkdir -p "$O/gates" && cp -r "$W"/* "$O/gates/" 2>/dev/null
  exit 5
fi
export REPS=${REPS:-15}
export CG_ARMS=${CG_ARMS:-"plain,plainq,e_plains,e_l7qs,c_l7qs,c_l7qs_o,editedq:e_l7qs,cedited:c_l7qs,qdither,cquant,cquant_s,cquant_d,wedit,cwedit"}
export NCU_ONE=${NCU_ONE:-"plain e_plains e_l7qs c_l7qs c_l7qs_o"}
export NCU_VEC=${NCU_VEC:-"qdither cquant cquant_s cquant_d wedit cwedit"} NCU_VEC_SHAPES=${NCU_VEC_SHAPES:-"o2k dn2k"}
[ "$NCU_VEC" = - ] && NCU_VEC=""
# cohalf: cquant and cwedit are hquant and hwedit there; egemm.cu's quantizers don't run at k = 128
export CG_ARMS_H=${CG_ARMS_H:-"plain,e_plains,e_l7qs,h_l7qs,cedited:h_l7qs,cquant,cwedit"}
export NCU_H=${NCU_H:-"plain e_plains e_l7qs h_l7qs cquant cwedit"}
export BENCH_SHAPES=${BENCH_SHAPES:-"q2k o2k gu2k dn2k o"}
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min 30 -- bash -c '
  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu --format=csv -lms 500 -i "$GPU_LEASE_UUID" > "$O/smi.csv" 2>&1 &
  smi=$!
  for s in $SHAPES; do
    IFS=: read -r name m n k <<< "$s"
    case " $BENCH_SHAPES " in *" $name "*) ;; *) continue ;; esac
    arms_=$CG_ARMS
    [ "$k" = 128 ] && arms_=$CG_ARMS_H
    CG_ARMS=$arms_ "$BIN/cgemm" bench "$REPS" "$m" "$n" "$k" >> "$O/bench.jsonl" 2>> "$O/bench.err"; echo "bench $name rc $?" >> "$O/rc.txt"
  done
  kill $smi
  mkdir -p "$O/ncu"
  S=(--section SpeedOfLight --section ComputeWorkloadAnalysis --section InstructionStats --section WarpStateStats
     --section SchedulerStats --section SourceCounters --section LaunchStats --section Occupancy)
  for s in $SHAPES; do
    IFS=: read -r name m n k <<< "$s"
    case " $BENCH_SHAPES " in *" $name "*) ;; *) continue ;; esac
    arms_=$NCU_ONE
    case " $NCU_VEC_SHAPES " in *" $name "*) arms_="$arms_ $NCU_VEC" ;; esac
    [ "$k" = 128 ] && arms_=$NCU_H
    for a in $arms_; do
      ncu --clock-control none --profile-from-start off --import-source no "${S[@]}" -f -o "$O/ncu/$name-${a//:/_}" \
        "$BIN/cgemm" one "$a" "$m" "$n" "$k" > "$O/ncu/$name-${a//:/_}.log" 2>&1 || echo "ncu one $name $a rc $?" >> "$O/rc.txt"
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
  --toml-arms "${CG_TOML_ARMS:-c_l7qs,c_l7qs_o,c_l7qs_v,h_l7qs,e_l7qs,e_plains,plain}" > "$O/profile.md" 2> "$O/profile.err"
echo "profile rc $?" | tee -a "$O/rc.txt"
