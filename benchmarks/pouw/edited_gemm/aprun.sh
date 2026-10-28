#!/usr/bin/env bash
# The adopted P.V (apv.cu) on one leased sm_120 GPU (cwd = the source tree's root). Builds apv.cu outside the lease and runs
# the SASS gate on it; then, gates before timing: lease 1 runs the device gates (poisoned buffers, every tile's f receipts
# against its O receipts, a no-write control) at APV_GATES (s tokens, H query heads, rep heads a KV head; small shapes,
# then Llama-3.1-8B's attention at 2,048 tokens) and dumps up to 24 tiles of each; apv_ref.py recomputes them on the CPU
# through checked_strip's pv_words and pv_served from P, V and the key (the gate's producers are vfold, checked on the
# device against r2's vedit and vsum); lease 1 also runs the V producers' own gate (vgate) at VP_SHAPES (s tokens, KV
# heads), past the GEMM's 4,096 tokens too. Only if every gate passes, lease 2 times APV_ARMS at APV_BENCH (interleaved
# hot items), the V producers alone (VP_ARMS) at VP_SHAPES, and runs ncu's sections on APV_NCU at APV_NCU_SHAPES
# (out/ncu/, read by eps_profile.py) and on VP_NCU at VP_SHAPES (out/ncu_vp/, read by vprod_profile.py); the two
# profiles and their TOML end the run. Clocks are read, never set.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --env GPU_LEASE_WHO=LANE \
#     -- bash benchmarks/pouw/edited_gemm/aprun.sh
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
export D=benchmarks/pouw/edited_gemm
export BIN=$(mktemp -d /tmp/apv.XXXXXX)
trap 'rm -rf "$BIN"' EXIT
PY=${CG_PYTHON:-python3}
REF=${CG_REF_PYTHON:-"uv run --no-project --with numpy --with python-sat --with-editable benchmarks/pouw/exact_kit python3"}
git rev-parse HEAD > "$O/commit.txt" 2>&1
nvcc -O3 -std=c++17 --no-compress -gencode arch=compute_120a,code=sm_120a -Xptxas -v -o "$BIN/apv" $D/apv.cu -lcuda -lcublasLt -lnvidia-ml \
  > "$O/build.log" 2>&1
rc=$?
echo "build rc $rc" | tee -a "$O/rc.txt"
[ "$rc" = 0 ] || { tail -40 "$O/build.log"; exit 4; }
$PY benchmarks/pouw/harness/sass_gate.py "$BIN/apv" --json "$O/sass_gate.json" > "$O/sass_gate.txt" 2>&1
echo "sass_gate (whole binary) rc $?" | tee -a "$O/sass_gate.txt"
# apv.cu includes cgemm.cu and so egemm.cu's kernels, some never launched; as in cpvrun.sh, the gate passes only if every
# flagged instruction is in one of those (an apv kernel takes ApArgs, so the pattern never excuses one)
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
cuobjdump -sass "$BIN/apv" > "$BIN/apv.sass" 2>/dev/null
awk '/Function :/{f=$3} /[[:space:]](LDL|STL)[.[:space:]]/{n[f]++} END{for (f in n) print n[f], f}' "$BIN/apv.sass" | sort -k2 \
  > "$O/sass_local_mem_ops.txt" 2>&1
gzip -c "$BIN/apv.sass" > "$O/apv.sass.gz"
# name:s:H:rep. d256: two tiles a head (qt 0 and 1); m512, m1k: KV heads shared; l8: Llama-3.1-8B (32 query heads, 8 KV
# heads) at 2,048 tokens; a16k: one head at 16,384 (Ycum's third digit Y2); l8x4k, l8x8k at 4,096 and 8,192 (bench only)
export APV_GATES=${APV_GATES:-"d256:256:2:1 m512:512:2:2 m1k:1024:4:2 l8:2048:32:4 a16k:16384:1:1"}
export APV_BENCH=${APV_BENCH:-"l8:2048:32:4 l8x4k:4096:32:4 l8x8k:8192:32:4"}
# s:KV heads, Llama-3.1-8B's 8 KV heads at 2K, 8K and 32K tokens
export VP_SHAPES=${VP_SHAPES:-"2048:8 8192:8 32768:8"}
export W="$BIN/gates"
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min 10 -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.limit \
    --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  for g in $APV_GATES; do
    IFS=: read -r name s h rep <<< "$g"
    mkdir -p "$W/$name"
    "$BIN/apv" gate "$W/$name" "$s" "$h" "$rep" 0x9ABC >> "$O/gate.jsonl" 2>> "$O/gate.err"; echo "gate $name rc $?" >> "$O/rc.txt"
  done
  for g in $VP_SHAPES; do
    IFS=: read -r s kvh <<< "$g"
    "$BIN/apv" vgate "$s" "$kvh" 0x9ABC >> "$O/vgate.jsonl" 2>> "$O/vgate.err"; echo "vgate $s rc $?" >> "$O/rc.txt"
  done'
echo "lease 1 rc $?" >> "$O/rc.txt"
dirs=""
for g in $APV_GATES; do
  name=${g%%:*}
  dirs="$dirs $W/$name"
  cp "$W/$name/meta.json" "$O/gate_${name}_meta.json" 2>/dev/null
done
# shellcheck disable=SC2086
$REF $D/apv_ref.py $dirs > "$O/ref.jsonl" 2> "$O/ref.err"
ref=$?
echo "ref rc $ref" | tee -a "$O/rc.txt"
if [ "$ref" != 0 ] || grep -q 'rc [1-9]' "$O/rc.txt"; then
  echo "a gate failed: nothing timed"; cat "$O/rc.txt"; cat "$O/gate.jsonl" "$O/vgate.jsonl"; cut -c1-3000 "$O/ref.jsonl"
  tail -20 "$O/ref.err" "$O/vgate.err"
  mkdir -p "$O/gates" && cp -r "$W"/d256 "$O/gates/" 2>/dev/null
  exit 5
fi
# eps_profile.py keys a shape by (m, n, k) = (H s, 128, s), from its gate's meta; a bench-only shape gets one here
for g in $APV_BENCH; do
  IFS=: read -r name s h rep <<< "$g"
  [ -e "$O/gate_${name}_meta.json" ] || echo "{\"m\": $((h * s)), \"n\": 128, \"k\": $s}" > "$O/gate_${name}_meta.json"
done
export REPS=${REPS:-15}
export APV_ARMS=${APV_ARMS:-"plain,apv,fu,uedit,pedit,vedit,vsum,vfold,vfold_x,apvedited,apvedited_r2,fuedited,fuedited_r2"}
export APV_NCU=${APV_NCU:-"plain apv fu"} APV_NCU_SHAPES=${APV_NCU_SHAPES:-"l8"}
export VP_ARMS=${VP_ARMS:-"vedit,vsum,vpair,vfold,vfold_x"} VP_NCU=${VP_NCU:-"vedit vsum vfold vfold_x"}
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min 30 -- bash -c '
  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu --format=csv -lms 500 -i "$GPU_LEASE_UUID" > "$O/smi.csv" 2>&1 &
  smi=$!
  for g in $APV_BENCH; do
    IFS=: read -r name s h rep <<< "$g"
    "$BIN/apv" bench "$REPS" "$s" "$h" "$rep" >> "$O/bench.jsonl" 2>> "$O/bench.err"; echo "bench $name rc $?" >> "$O/rc.txt"
  done
  for g in $VP_SHAPES; do
    IFS=: read -r s kvh <<< "$g"
    "$BIN/apv" vbench "$REPS" "$s" "$kvh" >> "$O/vbench.jsonl" 2>> "$O/vbench.err"; echo "vbench $s rc $?" >> "$O/rc.txt"
  done
  kill $smi
  mkdir -p "$O/ncu" "$O/ncu_vp"
  S=(--section SpeedOfLight --section ComputeWorkloadAnalysis --section InstructionStats --section WarpStateStats
     --section SchedulerStats --section SourceCounters --section LaunchStats --section Occupancy)
  for g in $APV_BENCH; do
    IFS=: read -r name s h rep <<< "$g"
    case " $APV_NCU_SHAPES " in *" $name "*) ;; *) continue ;; esac
    for a in $APV_NCU; do
      ncu --clock-control none --profile-from-start off --import-source no "${S[@]}" -f -o "$O/ncu/$name-$a" \
        "$BIN/apv" one "$a" "$s" "$h" "$rep" > "$O/ncu/$name-$a.log" 2>&1 || echo "ncu one $name $a rc $?" >> "$O/rc.txt"
    done
  done
  for g in $VP_SHAPES; do
    IFS=: read -r s kvh <<< "$g"
    for a in $VP_NCU; do
      ncu --clock-control none --profile-from-start off --import-source no "${S[@]}" --section MemoryWorkloadAnalysis \
        -f -o "$O/ncu_vp/vp-$s-$a" "$BIN/apv" vone "$a" "$s" "$kvh" > "$O/ncu_vp/vp-$s-$a.log" 2>&1 ||
        echo "ncu vone $s $a rc $?" >> "$O/rc.txt"
    done
  done'
echo "lease 2 rc $?" >> "$O/rc.txt"
for r in "$O"/ncu/*.ncu-rep "$O"/ncu_vp/*.ncu-rep; do
  [ -e "$r" ] || continue
  b=${r%.ncu-rep}
  ncu -i "$r" --page raw --csv --print-metric-instances values > "$b.raw.csv" 2> "$b.raw.err" ||
    ncu -i "$r" --page raw --csv > "$b.raw.csv" 2>> "$b.raw.err"
  ncu -i "$r" --page details --csv > "$b.details.csv" 2> "$b.details.err"
  ncu -i "$r" --page source --csv --print-source sass > "$b.sass.csv" 2> "$b.sass.err"
done
cat "$O/rc.txt"
$PY $D/eps_profile.py "$O" --mm benchmarks/pouw/machine_model --json "$O/profile.json" --toml "$O/profile.toml" \
  --toml-arms "apv,fu,plain" > "$O/profile.md" 2> "$O/profile.err"
echo "profile rc $?" | tee -a "$O/rc.txt"
$PY $D/vprod_profile.py "$O" --json "$O/vprod.json" --toml "$O/vprod.toml" > "$O/vprod.md" 2> "$O/vprod.err"
echo "vprod rc $?" | tee -a "$O/rc.txt"
cat "$O/vprod.md"
