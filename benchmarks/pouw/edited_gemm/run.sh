#!/usr/bin/env bash
# The adopted design's edited GEMM against plain int8 at Llama-8B's linears (m = 8192 tokens; o_proj n = k = 4096, qkv
# n = 6144, gate_up n = 28672, down n = 4096 and k = 14336), on one leased sm_120 GPU (cwd = the source tree's root).
# Builds egemm.cu outside the lease and runs the SASS gate on it; then, gates before timing: lease 1 runs the device
# gates (a small shape and each linear, poisoned buffers, a no-write control) and dumps tiles; egemm_ref.py recomputes
# them on the CPU from x, W and the key, every kernel's receipts by its kind; only if both pass, lease 2 times o_proj's
# hot items (EG_ARMS), the slice's per-call interleaving (IL pairs ARM:BASE; IL_COLD the same with L2 flushed before each
# call), each other linear's hot items and interleaving (LAYER_ARMS, LAYER_IL), and ncu's issue and pipe counters at
# o_proj. EG_SET=best times each receipt kind's fastest kernels instead, also after wedit alone (edited:K); EG_SET=x13
# r9's XOR-in (13, 19) beside x_l2o and e_l4q; EG_SET=eps1 the fold-cost sweep at 8192^3 and Llama-3.1-8B's o_proj and
# gate_up at m = 2048, with ncu's sections on every arm at every shape (out/ncu/, read by eps_profile.py), EG_SET=eps1lag
# duo's absorb lag and tie spacing there, EG_SET=eps1sleep consumers that sleep on a stage's mbarrier, EG_SET=eps1wave
# the best of them at all four linears at m = 2048 and at a shape whose tiles fill whole waves, EG_SET=eps1pred the machine
# model's predictions against kernels with extra scalar work and duo's counts across k, EG_SET=eps1uni the uniform
# filler's stand-ins, EG_SET=eps1fill the four linears at 6016 tokens (whole waves) against 6144. The small gate's seed has
# a key that key_ok_r9 rejects. Clocks are read, never set.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --env GPU_LEASE_WHO=LANE \
#     -- bash benchmarks/pouw/edited_gemm/run.sh
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
export D=benchmarks/pouw/edited_gemm
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
export O_SHAPE=${O_SHAPE:-"8192 4096 4096"}
export LAYERS=${LAYERS:-"qkv:8192:6144:4096 gate_up:8192:28672:4096 down:8192:4096:14336"}
# EG_SET=eps1, the fold-cost sweep: 8192^3, and Llama-3.1-8B's o_proj and gate_up
# at m = 2048 (EG_SET=eps1lag too)
case "${EG_SET:-all}" in
  eps1 | eps1lag | eps1sleep) O_SHAPE="8192 8192 8192" LAYERS="o2k:2048:4096:4096 gu2k:2048:28672:4096" ;;
  eps1wave) O_SHAPE="2048 6016 4096" LAYERS="q2k:2048:6144:4096 o2k:2048:4096:4096 gu2k:2048:28672:4096 dn2k:2048:4096:14336" ;;
  eps1uni) O_SHAPE="8192 8192 8192" LAYERS="gu2k:2048:28672:4096" ;;
  eps1pred) O_SHAPE="8192 8192 8192" LAYERS="gu2k:2048:28672:4096 k1k:2048:4096:1024 k2k:2048:4096:2048 k8k:2048:4096:8192 q8k:2048:6144:8192" ;;
  eps1fill) O_SHAPE="4096 6016 4096"
    LAYERS="o48:4096:6144:4096 q47:6144:6016:4096 q48:6144:6144:4096 g47:28672:6016:4096 g48:28672:6144:4096 d47:4096:6016:14336 d48:4096:6144:14336" ;;
esac
gzip -c "$BIN/egemm.sass" > "$O/egemm.sass.gz"
export W="$BIN/gates"
mkdir -p "$W/small" "$W/o"
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min 10 -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.limit \
    --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  "$BIN/egemm" gate "$W/small" 1024 256 512 0x10483 >> "$O/gate.jsonl" 2>> "$O/gate.err"; echo "gate small rc $?" >> "$O/rc.txt"
  "$BIN/egemm" gate "$W/o" $O_SHAPE 0x5678 >> "$O/gate.jsonl" 2>> "$O/gate.err"; echo "gate o rc $?" >> "$O/rc.txt"
  for l in $LAYERS; do
    IFS=: read -r name m n k <<< "$l"
    mkdir -p "$W/$name"
    "$BIN/egemm" gate "$W/$name" "$m" "$n" "$k" 0x9ABC >> "$O/gate.jsonl" 2>> "$O/gate.err"; echo "gate $name rc $?" >> "$O/rc.txt"
  done'
echo "lease 1 rc $?" >> "$O/rc.txt"
dirs="$W/small $W/o"
for l in $LAYERS; do dirs="$dirs $W/${l%%:*}"; done
for g in $dirs; do cp "$g/meta.json" "$O/gate_$(basename "$g")_meta.json" 2>/dev/null; done
# shellcheck disable=SC2086
$PY $D/egemm_ref.py $dirs > "$O/ref.jsonl" 2> "$O/ref.err"
ref=$?
echo "ref rc $ref" | tee -a "$O/rc.txt"
if [ "$ref" != 0 ] || grep -q 'rc [1-9]' "$O/rc.txt"; then
  echo "a gate failed: nothing timed"; cat "$O/rc.txt"; cat "$O/gate.jsonl" "$O/ref.jsonl"; exit 5
fi
export REPS=${REPS:-15} IL_REPS=${IL_REPS:-41}
K32="e_l2q e_l3qm e_l2o e_l4q"
K64="d64_l2q d64_l7q d64_lsq d64_lsqm"
KX="x_l2 x_l2q x_l3qm x_l2o x64_l2q x64_l2o"
KTB="t4_l2 t4c_l2 t2w_l0 t2w_l2 t1w_l0 t1w_l2"
KALL="$K32 $K64 $KX $KTB"
arms="plain,plainq,qplain,qdither,wedit,qwe,e_plain,t_duo32"
for k in $KALL; do arms="$arms,$k,editedf:$k"; done
il=""
for k in e_plain $KALL; do il="$il $k:plain"; done
for k in $KALL; do il="$il editedf:$k:plainq"; done
cold=""
for k in e_l3qm e_l2q d64_lsq x_l2q t2w_l2 t1w_l2; do cold="$cold $k:plain editedf:$k:plainq"; done
lk="e_l2q e_l3qm d64_lsq x_l2q x_l3qm t2w_l2 t1w_l2"
larms="plain,plainq,qwe,e_plain"
for k in $lk; do larms="$larms,$k,editedf:$k"; done
lil=""
for k in $lk; do lil="$lil $k:plain editedf:$k:plainq"; done
ncu="e_plain e_l2q d64_lsq x_l2q t4_l2 t2w_l2 t1w_l2 qwe" stall="e_l2q x_l2q t1w_l2"
# EG_SET=best: each kind's fastest kernels at o_proj, also as edited:K (wedit, then K, against plain), at every linear
if [ "${EG_SET:-all}" = best ]; then
  kb="e_l2q e_l4q d64_l2q d64_lsqm x_l2o x_l2q x64_l2o x64_l2q t2w_l2 t1w_l2"
  arms="plain,plainq,wedit,qwe,e_plain" il="e_plain:plain"
  for k in $kb; do arms="$arms,$k,edited:$k,editedf:$k"; il="$il $k:plain edited:$k:plain editedf:$k:plainq"; done
  cold=""
  for k in e_l4q d64_l2q x_l2o x64_l2o; do cold="$cold $k:plain editedf:$k:plainq"; done
  lk="e_l2q e_l4q d64_l2q x_l2o x_l2q x64_l2o x64_l2q t1w_l2"
  larms="plain,plainq,wedit,qwe,e_plain" lil=""
  for k in $lk; do larms="$larms,$k,edited:$k,editedf:$k"; lil="$lil $k:plain edited:$k:plain editedf:$k:plainq"; done
  ncu="e_l4q d64_l2q x_l2o x64_l2o x64_l2q" stall="d64_l2q x_l2o x64_l2o"
fi
# EG_SET=x13: r9's served candidate x13_l2o beside x_l2o (rotl 7) and duo32's e_l4q, at every linear, each also as
# edited:K and editedf:K
if [ "${EG_SET:-all}" = x13 ]; then
  arms="plain,plainq,wedit,qwe,e_plain" il="e_plain:plain"
  for k in x13_l2o x_l2o e_l4q; do arms="$arms,$k,edited:$k,editedf:$k"; il="$il $k:plain edited:$k:plain editedf:$k:plainq"; done
  cold="x13_l2o:plain editedf:x13_l2o:plainq"
  larms=$arms lil=$il
  ncu="x13_l2o x_l2o" stall="x13_l2o"
fi
# EG_SET=eps1: the fold-cost sweep at r = 0 (e_plain), 1, 1.5, 2, 2.5, 3 (duo, e_l2q), 4 ops a word, all at e_l2q's
# schedule, beside cuBLASLt, duo's e_l4q and XIN's x_l2o; ncu on every arm at every shape (NCU_ONE: its sections;
# NCU_FULL: --set full; both only at the shapes NCU_SHAPES names when it is set, NCU_EVERY's arms at the others)
export NCU_ONE="" NCU_FULL="" NCU_SHAPES="" NCU_EVERY=""
if [ "${EG_SET:-all}" = eps1 ]; then
  sweep="e_plain f10_l2q f15_l2q f20_l2q f25_l2q e_l2q f40_l2q"
  arms="plain" il="e_plain:plain e_l2q:plain edited:e_l2q:plain"
  for k in $sweep e_l4q x_l2o; do arms="$arms,$k"; done
  cold="" larms=$arms lil=$il ncu="" stall=""
  NCU_ONE="f10_l2q f15_l2q f20_l2q f25_l2q f40_l2q e_l4q x_l2o" NCU_FULL="plain e_plain e_l2q"
fi
# EG_SET=eps1lag: duo's schedule against eps_sched, the absorb LAG 2, 4, 6, 7 MMAs behind with the tie every 4th (q) or
# 8th (o) MMA, at eps1's shapes, ncu sections on every arm
if [ "${EG_SET:-all}" = eps1lag ]; then
  arms="plain,e_plain,e_l2q,e_l4q,e_l4o,e_l6q,e_l7q,e_l7o" il="e_l4q:plain e_l7q:plain"
  cold="" larms=$arms lil=$il ncu="" stall=""
  NCU_ONE="plain e_plain e_l2q e_l4q e_l4o e_l6q e_l7q e_l7o"
fi
# EG_SET=eps1sleep: consumers that sleep on a stage's mbarrier (try_wait's suspend-time hint; the s kernels) against
# ptxas's retry loop
if [ "${EG_SET:-all}" = eps1sleep ]; then
  arms="plain,e_plain,e_plains,e_l2q,e_l2qs,e_l7q,e_l7qs" il="e_l7q:plain e_l7qs:plain"
  cold="" larms=$arms lil=$il ncu="" stall=""
  NCU_ONE="e_plain e_plains e_l2q e_l2qs e_l7q e_l7qs"
fi
# EG_SET=eps1wave: where the gap above the bound goes at m = 2048: o = 2048 x 6016 x 4096 is 752 tiles, 4 whole waves of
# 188 SMs, beside qkv's 768 (5 waves, the last 8% full), o_proj, gate_up and down
if [ "${EG_SET:-all}" = eps1wave ]; then
  arms="plain,e_plain,e_plains,e_l7q,e_l7qs" il="e_l7qs:plain"
  cold="" larms=$arms lil=$il ncu="" stall=""
  NCU_ONE="plain e_plains e_l7qs"
fi
# EG_SET=eps1pred: the machine model's predictions against kernels at e_l7qs's
# schedule: r = 2 (f20), duo on x-op filler chains (fx<x>, x = 4.5, 8, 12, 24), duo with 8 or 16 uniform-datapath
# instructions a MMA (u8, u16); ncu on every arm at 8192^3 and gate_up, and e_l7qs's counts at k = 1024 to 8192 for its
# closed-form instruction counts
if [ "${EG_SET:-all}" = eps1pred ]; then
  arms="plain,e_plains,e_l7qs,f20_l7qs,fx45_l7qs,fx8_l7qs,fx12_l7qs,fx24_l7qs,u8_l7qs,u16_l7qs" il=""
  cold="" larms=$arms lil="" ncu="" stall=""
  NCU_ONE=${arms//,/ } NCU_SHAPES="o gu2k" NCU_EVERY="e_plains e_l7qs"
fi
# EG_SET=eps1uni: the uniform filler's stand-ins with the chains broadcast from lane 0 (u8s, u16s before each atom's
# instructions, u16t before each tile's) beside u16 (ptxas's vector datapath) at 8192^3 and gate_up, ncu on every arm
if [ "${EG_SET:-all}" = eps1uni ]; then
  arms="plain,e_l7qs,u16_l7qs,u8s_l7qs,u16s_l7qs,u16t_l7qs" il=""
  cold="" larms=$arms lil="" ncu="" stall=""
  NCU_ONE=${arms//,/ }
fi
# EG_SET=eps1fill: the wave tail removed by the token count. 128 x 128 tiles over 188 = 4 x 47 SMs fill whole waves only
# when 47 divides the tiles, which for every Llama-3.1-8B linear means tokens a multiple of 47 x 128 = 6016; tokens are
# the n side (the kernel's m is a multiple of 1024), so o = o_proj at 6016 tokens (1504 tiles, 8 waves), q/g/d qkv,
# gate_up and down, each at 6016 (47) and 6144 (48: 8.17, 12.26, 57.19, 8.17 waves); ncu's sections on H at every shape
if [ "${EG_SET:-all}" = eps1fill ]; then
  arms="plain,e_plains,e_l7qs" il=""
  cold="" larms=$arms lil="" ncu="" stall=""
  NCU_ONE="e_l7qs"
fi
export EG_ARMS=${EG_ARMS:-$arms} IL=${IL:-$il} IL_COLD=${IL_COLD:-$cold}
export LAYER_ARMS=${LAYER_ARMS:-$larms} LAYER_IL=${LAYER_IL:-$lil}
export NCU=${NCU:-$ncu} NCU_STALL=${NCU_STALL:-$stall}
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min 30 -- bash -c '
  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu --format=csv -lms 500 -i "$GPU_LEASE_UUID" > "$O/smi.csv" 2>&1 &
  smi=$!
  "$BIN/egemm" bench "$REPS" $O_SHAPE > "$O/bench.jsonl" 2> "$O/bench.err"; echo "bench o rc $?" >> "$O/rc.txt"
  for p in $IL; do
    base=${p##*:}; arm=${p%:*}
    "$BIN/egemm" il "$IL_REPS" "$arm" "$base" $O_SHAPE >> "$O/il.jsonl" 2>> "$O/il.err"; echo "il $p rc $?" >> "$O/rc.txt"
  done
  for p in $IL_COLD; do
    base=${p##*:}; arm=${p%:*}
    EG_COLD=1 "$BIN/egemm" il "$IL_REPS" "$arm" "$base" $O_SHAPE >> "$O/il.jsonl" 2>> "$O/il.err"; echo "il cold $p rc $?" >> "$O/rc.txt"
  done
  for l in $LAYERS; do
    IFS=: read -r name m n k <<< "$l"
    EG_ARMS=$LAYER_ARMS "$BIN/egemm" bench "$REPS" "$m" "$n" "$k" >> "$O/bench.jsonl" 2>> "$O/bench.err"; echo "bench $name rc $?" >> "$O/rc.txt"
    for p in $LAYER_IL; do
      base=${p##*:}; arm=${p%:*}
      "$BIN/egemm" il "$IL_REPS" "$arm" "$base" "$m" "$n" "$k" >> "$O/il.jsonl" 2>> "$O/il.err"; echo "il $name $p rc $?" >> "$O/rc.txt"
    done
  done
  kill $smi
  M=gpu__time_duration.sum,sm__cycles_elapsed.avg,sm__inst_executed.sum,sm__inst_executed_pipe_tensor.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active,smsp__issue_active.avg.pct_of_peak_sustained_active,smsp__warps_eligible.avg.per_cycle_active
  for a in $NCU; do
    EG_ARMS=$a ncu --clock-control none --kernel-name "regex:^${a}\$" --launch-skip 4 --launch-count 1 --metrics $M --csv \
      "$BIN/egemm" bench 1 $O_SHAPE > "$O/ncu-$a.csv" 2>&1 || echo "ncu $a rc $?" >> "$O/rc.txt"
  done
  for a in $NCU_STALL; do
    EG_ARMS=$a ncu --clock-control none --kernel-name "regex:^${a}\$" --launch-skip 4 --launch-count 1 --csv \
      --metrics "regex:^smsp__average_warps_issue_stalled_.*_per_issue_active\.ratio$" \
      "$BIN/egemm" bench 1 $O_SHAPE > "$O/ncu-stall-$a.csv" 2>&1 || echo "ncu stall $a rc $?" >> "$O/rc.txt"
  done
  if [ -n "$NCU_ONE$NCU_FULL" ]; then
    mkdir -p "$O/ncu"
    S=(--section SpeedOfLight --section ComputeWorkloadAnalysis --section InstructionStats --section WarpStateStats
       --section SchedulerStats --section SourceCounters --section LaunchStats --section Occupancy)
    for sh in "o:$(echo $O_SHAPE | tr " " :)" $LAYERS; do
      IFS=: read -r name m n k <<< "$sh"
      arms_="$NCU_FULL $NCU_ONE"
      case " ${NCU_SHAPES:-$name} " in *" $name "*) ;; *) arms_=$NCU_EVERY ;; esac
      for a in $arms_; do
        case " $NCU_FULL " in *" $a "*) set_=(--set full) ;; *) set_=("${S[@]}") ;; esac
        ncu --clock-control none --profile-from-start off --import-source no "${set_[@]}" -f -o "$O/ncu/$name-${a//:/_}" \
          "$BIN/egemm" one "$a" "$m" "$n" "$k" > "$O/ncu/$name-${a//:/_}.log" 2>&1 || echo "ncu one $name $a rc $?" >> "$O/rc.txt"
      done
    done
  fi'
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
$PY $D/summary.py "$O" | tee "$O/summary.md"
