#!/usr/bin/env bash
# Dither-C's P.V (cpv.cu, pv) on one leased sm_120 GPU (cwd = the source tree's root). Builds cpv.cu outside the lease and
# runs the SASS gate on it; then, gates before timing: lease 1 runs the device gates (poisoned buffers, every unit's f
# receipts against its O receipts, a no-write control) at PV_GATES (s tokens, H query heads, rep heads a KV head; the
# small ones all-diagonal and mixed, then Llama-3.1-8B's attention at 2,048 tokens) and dumps up to 32 unit tiles of
# each; cpv_ref.py recomputes them on the CPU through checked_strip's pv_words and pv_served from P, V and the key; only
# if every gate passes, lease 2 times PV_ARMS at PV_BENCH (interleaved hot items) and runs ncu's sections on PV_NCU at
# PV_NCU_SHAPES (out/ncu/, read by eps_profile.py, whose profile and TOML end the run). Clocks are read, never set.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --env GPU_LEASE_WHO=LANE \
#     -- bash benchmarks/pouw/edited_gemm/cpvrun.sh
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
export D=benchmarks/pouw/edited_gemm
export BIN=$(mktemp -d /tmp/cpv.XXXXXX)
trap 'rm -rf "$BIN"' EXIT
PY=${CG_PYTHON:-python3}
REF=${CG_REF_PYTHON:-"uv run --no-project --with numpy --with python-sat --with-editable benchmarks/pouw/exact_kit python3"}
git rev-parse HEAD > "$O/commit.txt" 2>&1
nvcc -O3 -std=c++17 --no-compress -gencode arch=compute_120a,code=sm_120a -Xptxas -v -o "$BIN/cpv" $D/cpv.cu -lcuda -lcublasLt -lnvidia-ml \
  > "$O/build.log" 2>&1
rc=$?
echo "build rc $rc" | tee -a "$O/rc.txt"
[ "$rc" = 0 ] || { tail -40 "$O/build.log"; exit 4; }
$PY benchmarks/pouw/harness/sass_gate.py "$BIN/cpv" --json "$O/sass_gate.json" > "$O/sass_gate.txt" 2>&1
echo "sass_gate (whole binary) rc $?" | tee -a "$O/sass_gate.txt"
# cpv.cu includes cgemm.cu and so egemm.cu's kernels, some never launched; as in crun.sh, the gate passes only if every
# flagged instruction is in one of those (a pv kernel takes PvArgs, so the pattern never excuses one)
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
cuobjdump -sass "$BIN/cpv" > "$BIN/cpv.sass" 2>/dev/null
awk '/Function :/{f=$3} /[[:space:]](LDL|STL)[.[:space:]]/{n[f]++} END{for (f in n) print n[f], f}' "$BIN/cpv.sass" | sort -k2 \
  > "$O/sass_local_mem_ops.txt" 2>&1
gzip -c "$BIN/cpv.sass" > "$O/cpv.sass.gz"
# name:s:H:rep. d256: every unit diagonal (qt 0 at 128 keys, qt 1 at 256); m512, m1k: full units beside them, KV heads
# shared; l8: Llama-3.1-8B (32 query heads, 8 KV heads) at 2,048 tokens; l8x4k at 4,096 (bench only)
export PV_GATES=${PV_GATES:-"d256:256:1:1 m512:512:2:2 m1k:1024:4:2 l8:2048:32:4"}
export PV_BENCH=${PV_BENCH:-"l8:2048:32:4 l8x4k:4096:32:4"}
export W="$BIN/gates"
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min 10 -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.limit \
    --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  for g in $PV_GATES; do
    IFS=: read -r name s h rep <<< "$g"
    mkdir -p "$W/$name"
    "$BIN/cpv" gate "$W/$name" "$s" "$h" "$rep" 0x9ABC >> "$O/gate.jsonl" 2>> "$O/gate.err"; echo "gate $name rc $?" >> "$O/rc.txt"
  done'
echo "lease 1 rc $?" >> "$O/rc.txt"
dirs=""
for g in $PV_GATES; do
  name=${g%%:*}
  dirs="$dirs $W/$name"
  cp "$W/$name/meta.json" "$O/gate_${name}_meta.json" 2>/dev/null
done
# shellcheck disable=SC2086
$REF $D/cpv_ref.py $dirs > "$O/ref.jsonl" 2> "$O/ref.err"
ref=$?
echo "ref rc $ref" | tee -a "$O/rc.txt"
if [ "$ref" != 0 ] || grep -q 'rc [1-9]' "$O/rc.txt"; then
  echo "a gate failed: nothing timed"; cat "$O/rc.txt"; cat "$O/gate.jsonl"; cut -c1-2000 "$O/ref.jsonl"; tail -20 "$O/ref.err"
  mkdir -p "$O/gates" && cp -r "$W"/* "$O/gates/" 2>/dev/null
  exit 5
fi
# eps_profile.py keys a shape by (m, n, k) = (H s, 128, s), from its gate's meta; a bench-only shape gets one here
for g in $PV_BENCH; do
  IFS=: read -r name s h rep <<< "$g"
  [ -e "$O/gate_${name}_meta.json" ] || echo "{\"m\": $((h * s)), \"n\": 128, \"k\": $s}" > "$O/gate_${name}_meta.json"
done
export REPS=${REPS:-15}
export PV_ARMS=${PV_ARMS:-"plain,pv,pv_i,pedit,vedit,vdig,pvedited"}
export PV_NCU=${PV_NCU:-"plain pv pedit vedit vdig"} PV_NCU_SHAPES=${PV_NCU_SHAPES:-"l8 l8x4k"}
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min 30 -- bash -c '
  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu --format=csv -lms 500 -i "$GPU_LEASE_UUID" > "$O/smi.csv" 2>&1 &
  smi=$!
  for g in $PV_BENCH; do
    IFS=: read -r name s h rep <<< "$g"
    "$BIN/cpv" bench "$REPS" "$s" "$h" "$rep" >> "$O/bench.jsonl" 2>> "$O/bench.err"; echo "bench $name rc $?" >> "$O/rc.txt"
  done
  kill $smi
  mkdir -p "$O/ncu"
  S=(--section SpeedOfLight --section ComputeWorkloadAnalysis --section InstructionStats --section WarpStateStats
     --section SchedulerStats --section SourceCounters --section LaunchStats --section Occupancy)
  for g in $PV_BENCH; do
    IFS=: read -r name s h rep <<< "$g"
    case " $PV_NCU_SHAPES " in *" $name "*) ;; *) continue ;; esac
    for a in $PV_NCU; do
      ncu --clock-control none --profile-from-start off --import-source no "${S[@]}" -f -o "$O/ncu/$name-$a" \
        "$BIN/cpv" one "$a" "$s" "$h" "$rep" > "$O/ncu/$name-$a.log" 2>&1 || echo "ncu one $name $a rc $?" >> "$O/rc.txt"
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
  --toml-arms "pv,plain" > "$O/profile.md" 2> "$O/profile.err"
echo "profile rc $?" | tee -a "$O/rc.txt"
