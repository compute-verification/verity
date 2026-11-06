#!/usr/bin/env bash
# H1's attention with S never stored (fattn.cu) on one leased sm_120 GPU (cwd = the source tree's root). Builds fattn.cu,
# apv.cu, cgemm.cu and cwgemm.cu outside the lease and runs the SASS gate on fattn and cwgemm (recorded; it doesn't stop
# the timing). Lease 1: fattn's device gates at FA_GATES (bitwise against the unfused h_l7qs, hsref and apv_l7qs; dumps for
# fattn_ref.py) and apv's at APV_GATES (s = 16,384: Ycum's third limb Y2) and cgemm's at FA_CG_GATES (the operand
# producers, c_fx1, and at k > 128 cwgemm's overlap orders), judged by apv_ref.py and cgemm_ref.py on the CPU beside
# lease 2. Only if every fattn gate and fattn_ref.py pass, lease 2 times FA_ARMS at FA_BENCH, the four linears (cwgemm.cu's
# LIN_ARMS at LIN_M tokens: cgemm.cu's arms and the cpar: orders), plain attention (flash_ref.py: torch SDPA, bf16, causal,
# GQA) at FLASH_SHAPES,
# and ncu's sections on FA_NCU at FA_NCU_SHAPES (out/ncu/, read by eps_profile.py, whose TOML ends the run); with
# FA_GATES_ONLY set, the run ends after the gates. Clocks are read, never set.
#   research run --on MACHINE --project verity --source . --cwd source --env GPU_LEASE_WHO=LANE \
#     -- bash benchmarks/pouw/edited_gemm/fattnrun.sh
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
export D=benchmarks/pouw/edited_gemm
export BIN=$(mktemp -d /tmp/fattn.XXXXXX)
trap 'rm -rf "$BIN"' EXIT
PY=${CG_PYTHON:-python3}
REF=${CG_REF_PYTHON:-"uv run --no-project --with numpy --with python-sat --with-editable benchmarks/pouw/exact_kit python3"}
git rev-parse HEAD > "$O/commit.txt" 2>&1
NV=(nvcc -O3 -std=c++17 --no-compress -gencode arch=compute_120a,code=sm_120a -Xptxas -v)
for b in fattn apv cgemm cwgemm; do
  "${NV[@]}" -o "$BIN/$b" $D/$b.cu -lcuda -lcublasLt -lnvidia-ml > "$O/build_$b.log" 2>&1 &
done
wait
for b in fattn apv cgemm cwgemm; do
  [ -x "$BIN/$b" ] || { echo "build $b failed"; tail -30 "$O/build_$b.log"; echo "build $b rc 1" >> "$O/rc.txt"; exit 4; }
done
echo "build rc 0" >> "$O/rc.txt"
$PY benchmarks/pouw/harness/sass_gate.py "$BIN/fattn" "$BIN/cwgemm" --json "$O/sass_gate.json" > "$O/sass_gate.txt" 2>&1
echo "sass_gate (whole binary) rc $?" >> "$O/sass_gate.txt"
# as aprun.sh: flagged instructions pass only in egemm.cu's never-launched kernels (an Args kernel of that name)
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
echo "sass_scope rc $?" > "$O/sass_rc.txt"
for b in fattn cwgemm; do
  cuobjdump -sass "$BIN/$b" > "$BIN/$b.sass" 2>/dev/null
  awk '/Function :/{f=$3} /[[:space:]](LDL|STL)[.[:space:]]/{n[f]++} END{for (f in n) print n[f], f}' "$BIN/$b.sass" | sort -k2 \
    > "$O/sass_local_mem_ops_$b.txt" 2>&1
  gzip -c "$BIN/$b.sass" > "$O/$b.sass.gz"
done
# name:s:H:rep. f1k: two KV heads at 1,024 tokens; l8: Llama-3.1-8B (32 query heads, 8 KV heads) at 2,048
export FA_GATES=${FA_GATES:-"f1k:1024:4:2 l8:2048:32:4"}
export APV_GATES=${APV_GATES:-"a16k:16384:1:1"}
# cgemm.cu's gates (its operand producers: cohalf's hquant and hwedit at k = 128, the linears' cquant and cwedit; c_fx1),
# through cwgemm's gate at k > 128 (cgemm's, then the overlap orders cpar:c_l7qs and cpar:c_fx1 against the serial
# kernels), judged by cgemm_ref.py on the CPU beside lease 2 (crun.sh's shapes, name:m:n:k)
export FA_CG_GATES=${FA_CG_GATES:-"hs:1024:256:128 h2k:2048:2048:128 small:1024:256:512 small3:1024:256:8192 q2k:2048:6144:4096 dn2k:2048:4096:14336"}
export W="$BIN/gates"
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min 15 -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.limit,memory.total \
    --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  for g in $FA_GATES; do
    IFS=: read -r name s h rep <<< "$g"
    mkdir -p "$W/$name"
    "$BIN/fattn" gate "$W/$name" "$s" "$h" "$rep" 0x9ABC >> "$O/gate.jsonl" 2>> "$O/gate.err"; echo "gate $name rc $?" >> "$O/rc.txt"
  done
  for g in $APV_GATES; do
    IFS=: read -r name s h rep <<< "$g"
    mkdir -p "$W/$name"
    "$BIN/apv" gate "$W/$name" "$s" "$h" "$rep" 0x9ABC >> "$O/apv_gate.jsonl" 2>> "$O/gate.err"; echo "apv gate $name rc $?" >> "$O/rc.txt"
  done
  for g in $FA_CG_GATES; do
    IFS=: read -r name m n k <<< "$g"
    mkdir -p "$W/cg_$name"
    b=cwgemm; [ "$k" = 128 ] && b=cgemm
    "$BIN/$b" gate "$W/cg_$name" "$m" "$n" "$k" 0x9ABC >> "$O/cg_gate.jsonl" 2>> "$O/gate.err"; echo "cg gate $name rc $?" >> "$O/rc.txt"
  done'
echo "lease 1 rc $?" >> "$O/rc.txt"
dirs=""
for g in $FA_GATES; do
  name=${g%%:*}
  dirs="$dirs $W/$name"
  cp "$W/$name/meta.json" "$O/gate_${name}_fattn_meta.json" 2>/dev/null
done
# shellcheck disable=SC2086
uv run --no-project --with numpy python3 $D/fattn_ref.py $dirs > "$O/ref.jsonl" 2> "$O/ref.err"
echo "ref rc $?" >> "$O/rc.txt"
if grep -q 'rc [1-9]' "$O/rc.txt"; then
  echo "a gate failed: nothing timed"; cat "$O/rc.txt"; cat "$O/gate.jsonl" "$O/apv_gate.jsonl"; cut -c1-3000 "$O/ref.jsonl"
  tail -20 "$O/ref.err" "$O/gate.err"
  mkdir -p "$O/gates" && cp -r "$W"/f1k "$O/gates/" 2>/dev/null
  exit 5
fi
apv_dirs=""
for g in $APV_GATES; do apv_dirs="$apv_dirs $W/${g%%:*}"; done
# shellcheck disable=SC2086
( $REF $D/apv_ref.py $apv_dirs > "$O/apv_ref.jsonl" 2> "$O/apv_ref.err"; echo "apv_ref rc $?" > "$O/apv_ref_rc.txt" ) &
apv_ref=$!
cg_dirs=""
for g in $FA_CG_GATES; do cg_dirs="$cg_dirs $W/cg_${g%%:*}"; done
# shellcheck disable=SC2086
( $REF $D/cgemm_ref.py $cg_dirs > "$O/cg_ref.jsonl" 2> "$O/cg_ref.err"; echo "cg_ref rc $?" > "$O/cg_ref_rc.txt" ) &
cg_ref=$!
if [ -n "${FA_GATES_ONLY:-}" ]; then
  wait $apv_ref $cg_ref
  cat "$O/apv_ref_rc.txt" "$O/cg_ref_rc.txt" >> "$O/rc.txt"
  echo "FA_GATES_ONLY: nothing timed"; cat "$O/rc.txt"
  exit 0
fi
# l8x32k: two KV heads (8 query heads) at 32,768 tokens; its whole-layer time is 4x
export FA_BENCH=${FA_BENCH:-"l8:2048:32:4 l8x8k:8192:32:4 l8x32k:32768:8:4"}
export FA_ARMS_UNF=${FA_ARMS_UNF:-"attn,attn_vf,qks,pv,pv_vf,hquant,hm,hm_full,hsm,lred,fu,hqk"}
export FA_ARMS_BIG=${FA_ARMS_BIG:-"attn,attn_vf,qks,pv,pv_vf,hquant,hm,hm_full,hsm,lred,fu"}
export LIN_ARMS=${LIN_ARMS:-"plain,plainq,c_l7qs,c_fx1,cedited:c_l7qs,cedited:c_fx1,cpar:c_l7qs,cpar:c_fx1,cquant,cwedit"}
export LIN_M=${LIN_M:-"2048 8192 32768"} LIN_SHAPES=${LIN_SHAPES:-"6144:4096 4096:4096 28672:4096 4096:14336"}
export FLASH_SHAPES=${FLASH_SHAPES:-"2048:32:8 8192:32:8 32768:32:8 32768:8:2"}
export FA_NCU=${FA_NCU:-"hm hsm fu lred hm_full"} FA_NCU_SHAPES=${FA_NCU_SHAPES:-"l8 l8x8k"}
export REPS=${REPS:-15}
# torch for flash_ref.py: the first of these that imports torch with CUDA
export TPY=""
for p in ${FA_TORCH_PY:-} /workspace/pouw/fill-out/pous/work/pous-bench/gpu312-torch2.14.0-cu130/bin/python \
  /workspace/pouw/mvp-e2e/venv312/bin/python; do
  [ -x "$p" ] && "$p" -c 'import torch; assert torch.cuda.is_available()' > /dev/null 2>&1 && { TPY=$p; break; }
done
[ -n "$TPY" ] || TPY="uv run --no-project --with torch --index https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match python3"
echo "torch python: $TPY" > "$O/torch.txt"
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min 40 -- bash -c '
  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,memory.used --format=csv -lms 500 -i "$GPU_LEASE_UUID" > "$O/smi.csv" 2>&1 &
  smi=$!
  for g in $FA_BENCH; do
    IFS=: read -r name s h rep <<< "$g"
    arms=$FA_ARMS_UNF; [ "$s" -gt 8192 ] && arms=$FA_ARMS_BIG
    FA_ARMS=$arms "$BIN/fattn" bench "$REPS" "$s" "$h" "$rep" >> "$O/fbench.jsonl" 2>> "$O/bench.err"; echo "bench $name rc $?" >> "$O/rc.txt"
  done
  for m in $LIN_M; do
    for sh in $LIN_SHAPES; do
      IFS=: read -r n k <<< "$sh"
      CW_ARMS=$LIN_ARMS "$BIN/cwgemm" bench "$REPS" "$m" "$n" "$k" >> "$O/lin.jsonl" 2>> "$O/lin.err"
      echo "lin $m $n $k rc $?" >> "$O/rc.txt"
    done
  done
  $TPY $D/flash_ref.py "$REPS" $FLASH_SHAPES >> "$O/flash.jsonl" 2>> "$O/flash.err"; echo "flash rc $?" >> "$O/rc.txt"
  kill $smi
  mkdir -p "$O/ncu"
  S=(--section SpeedOfLight --section ComputeWorkloadAnalysis --section InstructionStats --section WarpStateStats
     --section SchedulerStats --section SourceCounters --section LaunchStats --section Occupancy --section MemoryWorkloadAnalysis)
  for g in $FA_BENCH; do
    IFS=: read -r name s h rep <<< "$g"
    case " $FA_NCU_SHAPES " in *" $name "*) ;; *) continue ;; esac
    for a in $FA_NCU; do
      ncu --clock-control none --profile-from-start off --import-source no "${S[@]}" -f -o "$O/ncu/$name-$a" \
        "$BIN/fattn" one "$a" "$s" "$h" "$rep" > "$O/ncu/$name-$a.log" 2>&1 || echo "ncu one $name $a rc $?" >> "$O/rc.txt"
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
# eps_profile.py keys a shape by (m, n, k) from gate_NAME_meta.json and bench.jsonl: here (H s, s, 128)
$PY - "$O" <<'EOF'
import json, sys
from pathlib import Path
o = Path(sys.argv[1])
with open(o / "bench.jsonl", "w") as f:
    for line in (o / "fbench.jsonl").read_text().splitlines() if (o / "fbench.jsonl").exists() else []:
        b = json.loads(line)
        f.write(json.dumps({"m": b["H"] * b["s"], "n": b["s"], "k": 128, "arms": b["arms"]}) + "\n")
names = {"l8": (2048, 32), "l8x8k": (8192, 32), "l8x32k": (32768, 8)}
for name, (s, h) in names.items():
    (o / f"gate_{name}_meta.json").write_text(json.dumps({"m": h * s, "n": s, "k": 128}))
EOF
$PY $D/eps_profile.py "$O" --mm benchmarks/pouw/machine_model --json "$O/profile.json" --toml "$O/profile.toml" \
  --toml-arms "hm,hsm,fu" > "$O/profile.md" 2> "$O/profile.err"
echo "profile rc $?" >> "$O/rc.txt"
wait $apv_ref $cg_ref
cat "$O/apv_ref_rc.txt" "$O/cg_ref_rc.txt" >> "$O/rc.txt"
cat "$O/rc.txt"
