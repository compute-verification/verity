#!/usr/bin/env bash
# sha512x.cuh's compression variants (README "Compression variants") on commit_gpu's ruler, against commit_gpu's default
# build, on one leased GPU. commit_gpu (benchmarks/pouw/commit_gpu) is built from its own path, unedited: a0r0l as
# is (SHA_ADD 0, SHA_ROT 0, SHA_ROLL 1), and sx<variant> with as_commit_gpu.cuh force-included, so its sha::compress is
# sx::compress<V, ROLL>. Each build is checked bit for bit against reference.py before it is timed.
# cwd = the source tree's root. Outside the leases: the builds (sm_90a on an H200), reference.py's full cases, and each
# build's sha_rate SASS with its opcode and pipe counts (sass_count.py). Check leases (as many as it takes): selftest and the
# full check per build. Timing lease: ROUNDS rounds of `commit_gpu sha 20`, the builds interleaved, nvidia-smi every 200 ms,
# then ncu's issue and pipe utilization of one sha_rate launch per build (its clocks left alone). Clocks are read, never
# set; a lease waits its turn and holds the GPU at most LEASE_MIN minutes. mix.jsonl: per build, the rate and clock beside
# the SASS pipe mix's bound (ALU/64, IMAD slots/64 with IMAD.WIDE 2 and IMAD.HI 2.09, issue/128; SM-clocks a compression).
#
#   OUT=DIR bash benchmarks/torch_gpu/sha512_rows/ruler.sh
#
# Needs a CUDA GPU (an H200 for sm_90a), CUDA's nvcc and ncu, and `gpu-lease` on PATH.
# VARIANTS (default "0000 0011 0003 0030 0033 1033 0333"), ROUNDS (9), LEASE_MIN (14), LEASES (0: builds and
# counts only), JOBS (parallel nvcc, 6).
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda/bin:$PATH
export D=benchmarks/pouw/commit_gpu X=benchmarks/torch_gpu/sha512_rows PY=python3 PYTHONPATH=.:tools/config
W=$(mktemp -d /dev/shm/sha512imad.XXXXXX)
export W
trap 'rm -rf "$W"' EXIT
git rev-parse HEAD > "$O/commit.txt" 2>&1
nproc > "$O/nproc.txt"
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
case $CC in 90 | 100 | 120) SM=${CC}a ;; *) SM=$CC ;; esac
export SM LEASE_MIN=${LEASE_MIN:-14}
echo "compute_cap $CC sm_$SM" > "$O/arch.txt"

build() {
  local t=$1 flags="-DSHA_ADD=0 -DSHA_ROT=0 -DSHA_ROLL=1" sx
  if [ "$t" != a0r0l ]; then
    sx=$($PY -c "import sys; sys.path.insert(0, '$X'); import engine; print(*engine.variant_flags('${t#sx}'))") ||
      { echo "build $t rc flags" >> "$O/rc.txt"; return; }
    flags="$flags -include $X/as_commit_gpu.cuh $sx"
  fi
  # shellcheck disable=SC2086
  nvcc -O3 -std=c++17 -gencode arch=compute_$SM,code=sm_$SM -Xptxas -v $flags -o "$W/commit_gpu_$t" $D/commit_gpu.cu \
    -lcublasLt -lnvidia-ml > "$O/build-$t.log" 2>&1
  echo "build $t rc $?" >> "$O/rc.txt"
}
export -f build
BUILDS="a0r0l"
for v in ${VARIANTS:-0000 0011 0003 0030 0033 1033 0333}; do BUILDS="$BUILDS sx$v"; done
export BUILDS
# shellcheck disable=SC2086
printf '%s\n' $BUILDS | xargs -P "${JOBS:-6}" -I{} bash -c 'build {}' &
builds=$!
if [ "${LEASES:-1}" != 0 ]; then
  $PY $D/reference.py prepare "$W/cases" full > "$O/prepare.jsonl" 2> "$O/prepare.err"
  echo "prepare rc $?" >> "$O/rc.txt"
fi
wait $builds
BUILT=""
for t in $BUILDS; do
  [ -x "$W/commit_gpu_$t" ] || continue
  BUILT="$BUILT $t"
  cuobjdump -sass "$W/commit_gpu_$t" > "$W/sass-$t.txt" 2>&1
  $PY $X/sass_count.py "$W/commit_gpu_$t" sha_rate --trip 4 --sass "$W/sass-$t.txt" > "$O/sass-$t.json" 2>> "$O/sass.err"
  awk '/Function : /{p = /sha_rate/} p' "$W/sass-$t.txt" | gzip -c > "$O/sass-sha_rate-$t.txt.gz"
done
cat "$O/rc.txt"

mix() {   # the pipe-mix bound beside each build's rate and clock
  $PY - "$O" << 'EOF' | tee "$O/mix.jsonl"
import json, os, statistics, sys
out = sys.argv[1]
def lines(p):
    return [json.loads(x) for x in open(p) if x.startswith("{")] if os.path.exists(p) else []
sms = next((r["sms"] for r in lines(os.path.join(out, "sms.jsonl"))), 0)
base = None
for t in ["a0r0l"] + sorted(f[5:-5] for f in os.listdir(out) if f.startswith("sass-sx") and f.endswith(".json")):
    p = os.path.join(out, f"sass-{t}.json")
    if not os.path.exists(p):
        continue
    s = json.load(open(p))
    ops, pipes = s["per_block"], s["pipes_per_block"]
    slots = sum(c * {"IMAD.WIDE": 2, "IMAD.HI": 2.09}.get(op, 1) for op, c in ops.items() if op.startswith("IMAD"))
    other_fma = pipes.get("fma", 0) - sum(c for op, c in ops.items() if op.startswith("IMAD"))
    bound = {"alu": pipes.get("alu", 0) / 64, "imad": (slots + other_fma) / 64, "issue": s["total_per_block"] / 128}
    raw = [r["sha"] for r in lines(os.path.join(out, f"sha-{t}.jsonl"))]
    rec = {"build": t, "alu": pipes.get("alu", 0), "imad": ops.get("IMAD", 0), "imad_wide": ops.get("IMAD.WIDE", 0),
           "imad_hi": ops.get("IMAD.HI", 0), "imad_slots": round(slots, 2), "total": s["total_per_block"],
           "pipes": pipes, "sm_clocks_bound": {k: round(v, 2) for k, v in bound.items()},
           "binds": max(bound, key=bound.get), "T": round(max(bound.values()), 2)}
    if raw:
        g, mhz = statistics.median(r["gcomp_per_s"] for r in raw), statistics.median(r["mhz"] for r in raw)
        rec.update(rounds=len(raw), gcomp_per_s=g, gcomp_per_s_all=[r["gcomp_per_s"] for r in raw], mhz=mhz)
        if sms:
            rec["bound_gcomp_per_s"] = round(sms * mhz * 1e-3 / rec["T"], 3)
            rec["fraction_of_bound"] = round(g / rec["bound_gcomp_per_s"], 3)
        base = base or (g if t == "a0r0l" else None)
        if base:
            rec["vs_a0r0l"] = round(g / base, 4)
    print(json.dumps(rec))
EOF
}

if [ "${LEASES:-1}" = 0 ]; then
  echo "LEASES=0: builds and counts only" >> "$O/rc.txt"
  mix
  exit 0
fi

export TODO=$BUILT
while [ -n "${TODO// /}" ]; do
  rm -f "$W/left"
  # shellcheck disable=SC2016
  gpu-lease 1 --wait --max-min "$LEASE_MIN" -- bash -c '
    nvidia-smi --query-gpu=index,uuid,name,driver_version,compute_cap,clocks.sm,clocks.max.sm,power.limit --format=csv \
      -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
    start=$SECONDS left=""
    for t in $TODO; do
      if [ -n "$left" ] || [ $((SECONDS - start)) -gt $((LEASE_MIN * 60 - 300)) ]; then left="$left $t"; continue; fi
      "$W/commit_gpu_$t" selftest >> "$O/selftest.jsonl"; s=$?
      $PY $D/reference.py check "$W/commit_gpu_$t" "$W/cases" "$t" > "$O/check-$t.jsonl" 2> "$O/check-$t.err"; c=$?
      echo "selftest $t rc $s; check $t rc $c ($((SECONDS - start)) s into the lease)" >> "$O/rc.txt"
      [ $s = 0 ] && [ $c = 0 ] && echo "$t" >> "$W/timed"
    done
    echo "$left" > "$W/left"'
  echo "check lease rc $?" >> "$O/rc.txt"
  [ -f "$W/left" ] || break
  TODO=$(cat "$W/left")
done
TIMED=$(tr '\n' ' ' < "$W/timed" 2> /dev/null)
export TIMED ROUNDS=${ROUNDS:-9}
echo "timed: $TIMED" >> "$O/rc.txt"
[ -z "${TIMED// /}" ] && { cat "$O/rc.txt"; exit 1; }

NCU_METRICS="sm__cycles_elapsed.avg.per_second,smsp__issue_active.avg.pct_of_peak_sustained_active"
NCU_METRICS+=",smsp__inst_executed.avg.per_cycle_active,smsp__warps_active.avg.pct_of_peak_sustained_active"
NCU_METRICS+=",regex:^sm__inst_executed_pipe_[a-z0-9_]+\.avg\.pct_of_peak_sustained_active$"
NCU_METRICS+=",regex:^sm__pipe_[a-z0-9_]+_cycles_active\.avg\.pct_of_peak_sustained_active$"
NCU_METRICS+=",regex:^smsp__average_warps_issue_stalled_[a-z0-9_]+_per_issue_active\.ratio$"
export NCU_METRICS
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "$LEASE_MIN" -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,compute_cap,clocks.sm,clocks.max.sm,power.limit --format=csv \
    -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  $PY -c "
import ctypes, json
c = ctypes.CDLL(\"libcuda.so.1\"); c.cuInit(0); d, n = ctypes.c_int(), ctypes.c_int()
c.cuDeviceGet(ctypes.byref(d), 0); c.cuDeviceGetAttribute(ctypes.byref(n), 16, d); print(json.dumps({\"sms\": n.value}))" > "$O/sms.jsonl"
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active \
    --format=csv -lms 200 -i "$GPU_LEASE_UUID" > "$O/smi.csv" 2>&1 &
  smi=$!
  for r in $(seq "$ROUNDS"); do
    for t in $TIMED; do "$W/commit_gpu_$t" sha 20 >> "$O/sha-$t.jsonl" 2>> "$O/bench.err"; echo "round $r sha $t rc $?" >> "$O/rc.txt"; done
  done
  kill $smi
  for t in $TIMED; do
    ncu --clock-control none -k sha_rate -c 1 --csv --page raw --metrics "$NCU_METRICS" "$W/commit_gpu_$t" sha 1 \
      > "$O/ncu-$t.csv" 2> "$O/ncu-$t.err"
    echo "ncu $t rc $?" >> "$O/rc.txt"
  done'
echo "timing lease rc $?" >> "$O/rc.txt"
mix
cat "$O/rc.txt"
for f in "$O"/check-*.jsonl; do [ -e "$f" ] && tail -1 "$f"; done
