#!/usr/bin/env bash
# co32c's linear with the weight edit inside the GEMM (cwgemm.cu: cplane, then cw_l7qs forms W' in shared memory) on one
# leased sm_120 GPU (cwd = the source tree's root), crun.sh's protocol: build outside the lease and the SASS gate; lease 1
# runs cwgemm's gates (cgemm.cu's whole gate, then cw_l7qs's O against cuBLASLt and its receipts against c_l7qs's under
# the same key, cplane's P_B against cwedit's, a no-write control) at crun.sh's shapes; cwgemm_ref.py recomputes the dumped
# tiles on the CPU (cgemm_ref.py's check, then cw_l7qs's O, receipts, plane and UB from x, W and the key); only if every
# gate passes, lease 2 times CW_ARMS at each shape (interleaved hot items: the served linear is cwedited:cw_l7qs = cquant,
# cplane, cw_l7qs against cedited:c_l7qs = cquant, cwedit, c_l7qs) and runs ncu on NCU_ONE at each shape and NCU_VEC at
# NCU_VEC_SHAPES (out/ncu/, read by eps_profile.py, whose profile and TOML end the run). Clocks are read, never set.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --env GPU_LEASE_WHO=LANE \
#     -- bash benchmarks/pouw/edited_gemm/cwrun.sh
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
export D=benchmarks/pouw/edited_gemm
export BIN=$(mktemp -d /tmp/cwgemm.XXXXXX)
trap 'rm -rf "$BIN"' EXIT
PY=${CG_PYTHON:-python3}
REF=${CG_REF_PYTHON:-"uv run --no-project --with numpy --with python-sat --with-editable benchmarks/pouw/exact_kit python3"}
git rev-parse HEAD > "$O/commit.txt" 2>&1
nvcc -O3 -std=c++17 --no-compress -gencode arch=compute_120a,code=sm_120a -Xptxas -v -o "$BIN/cwgemm" $D/cwgemm.cu -lcuda -lcublasLt \
  -lnvidia-ml > "$O/build.log" 2>&1
rc=$?
echo "build rc $rc" | tee -a "$O/rc.txt"
[ "$rc" = 0 ] || { tail -40 "$O/build.log"; exit 4; }
$PY benchmarks/pouw/harness/sass_gate.py "$BIN/cwgemm" --json "$O/sass_gate.json" > "$O/sass_gate.txt" 2>&1
echo "sass_gate (whole binary) rc $?" | tee -a "$O/sass_gate.txt"
# crun.sh's scope: flagged instructions pass only in egemm.cu's never-launched kernels (cw_* take WArgs, so never match)
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
cuobjdump -sass "$BIN/cwgemm" > "$BIN/cwgemm.sass" 2>/dev/null
awk '/Function :/{f=$3} /[[:space:]](LDL|STL)[.[:space:]]/{n[f]++} END{for (f in n) print n[f], f}' "$BIN/cwgemm.sass" | sort -k2 \
  > "$O/sass_local_mem_ops.txt" 2>&1
gzip -c "$BIN/cwgemm.sass" > "$O/cwgemm.sass.gz"
export SHAPES=${SHAPES:-"small:1024:256:512 small3:1024:256:8192 q2k:2048:6144:4096 o2k:2048:4096:4096 gu2k:2048:28672:4096 dn2k:2048:4096:14336 o:8192:8192:8192"}
export W="$BIN/gates"
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min 12 -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.limit \
    --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  for s in $SHAPES; do
    IFS=: read -r name m n k <<< "$s"
    mkdir -p "$W/$name"
    "$BIN/cwgemm" gate "$W/$name" "$m" "$n" "$k" 0x9ABC >> "$O/gate.jsonl" 2>> "$O/gate.err"; echo "gate $name rc $?" >> "$O/rc.txt"
  done'
echo "lease 1 rc $?" >> "$O/rc.txt"
dirs=""
for s in $SHAPES; do
  name=${s%%:*}
  dirs="$dirs $W/$name"
  cp "$W/$name/meta.json" "$O/gate_${name}_meta.json" 2>/dev/null
  cp "$W/$name/cw_meta.json" "$O/gate_${name}_cw_meta.json" 2>/dev/null
done
# shellcheck disable=SC2086
$REF $D/cwgemm_ref.py $dirs > "$O/ref.jsonl" 2> "$O/ref.err"
ref=$?
echo "ref rc $ref" | tee -a "$O/rc.txt"
if [ "$ref" != 0 ] || grep -q 'rc [1-9]' "$O/rc.txt"; then
  echo "a gate failed: nothing timed"; cat "$O/rc.txt"; cat "$O/gate.jsonl" "$O/ref.jsonl"; tail -20 "$O/ref.err"
  mkdir -p "$O/gates" && cp -r "$W"/* "$O/gates/" 2>/dev/null
  exit 5
fi
export REPS=${REPS:-15}
export CW_ARMS=${CW_ARMS:-"plain,plainq,c_l7qs,cw_l7qs,cquant,cwedit,cplane,cedited:c_l7qs,cwedited:cw_l7qs,cpar:c_l7qs,cwpar:cw_l7qs"}
export NCU_ONE=${NCU_ONE:-"plain c_l7qs cw_l7qs"}
export NCU_VEC=${NCU_VEC:-"cquant cwedit cplane"} NCU_VEC_SHAPES=${NCU_VEC_SHAPES:-"o2k dn2k"}
[ "$NCU_VEC" = - ] && NCU_VEC=""
export BENCH_SHAPES=${BENCH_SHAPES:-"q2k o2k gu2k dn2k o"}
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min 30 -- bash -c '
  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu --format=csv -lms 500 -i "$GPU_LEASE_UUID" > "$O/smi.csv" 2>&1 &
  smi=$!
  for s in $SHAPES; do
    IFS=: read -r name m n k <<< "$s"
    case " $BENCH_SHAPES " in *" $name "*) ;; *) continue ;; esac
    "$BIN/cwgemm" bench "$REPS" "$m" "$n" "$k" >> "$O/bench.jsonl" 2>> "$O/bench.err"; echo "bench $name rc $?" >> "$O/rc.txt"
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
    for a in $arms_; do
      ncu --clock-control none --profile-from-start off --import-source no "${S[@]}" -f -o "$O/ncu/$name-${a//:/_}" \
        "$BIN/cwgemm" one "$a" "$m" "$n" "$k" > "$O/ncu/$name-${a//:/_}.log" 2>&1 || echo "ncu one $name $a rc $?" >> "$O/rc.txt"
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
  --toml-arms "cw_l7qs,c_l7qs,plain" > "$O/profile.md" 2> "$O/profile.err"
echo "profile rc $?" | tee -a "$O/rc.txt"
