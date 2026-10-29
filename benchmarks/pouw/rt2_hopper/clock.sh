#!/usr/bin/env bash
# The H200 job (sm90_clock.cu), in two runs so no CPU phase runs on the shared H200 node. From a pushed commit:
#
#   research run --on auto --project verity --campaign eps1 --source <tree> --cwd source -- \
#     bash benchmarks/pouw/rt2_hopper/clock.sh build
#       builds sm90_clock.cu and msp90.cu for sm_90a, writes their SASS and sass_count.py's counts and gate.
#       Outputs: nvcc.txt, sm90_clock.sass, sass.json, gate.txt, msp90.sass, msp90-sass.json, bins.tgz.
#   research run --on vy-dc-h200 --project verity --campaign eps1 --source <tree> --cwd source --send bins.tgz \
#     --timeout 25m --env GPU_LEASE_WHO=LANE -- bash benchmarks/pouw/rt2_hopper/clock.sh time
#       under one gpu-lease: li's fold bit for bit against the host's reference (exact.jsonl), msp90's mma.sp
#       timings for (M′) (msp90.jsonl), then each load's SM clock
#       read in the kernel and its MACs a clock (clock.jsonl), with NVML sampled every 100 ms (smi.csv) and its clock,
#       performance and power reports before and after (smi-{before,after}.txt), and wg_spec with each role's own time
#       (roles.jsonl), the fold's cycles a unit with the lane alone in each checkpoint placement (lirate.jsonl) and
#       with NR of 8 units' checkpoints in memory (lirmix.jsonl). STEPS (default "exact,msp,joint,roles,lirate,lirmix,
#       clock") picks which of them run.
# Exit: 0 done, 1 build, gate or a probe failed, 2 no nvcc, no inputs or wrong device.
set -uo pipefail
MODE=${1:?usage: clock.sh build | time}
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONDONTWRITEBYTECODE=1

case "$MODE" in
build)
  NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
  [ -x "$NVCC" ] || NVCC=$(command -v nvcc) || { echo "clock.sh: no nvcc" >&2; exit 2; }
  WORK=$(mktemp -d)
  trap 'rm -rf "$WORK"' EXIT
  "$NVCC" --version >"$OUT/nvcc.txt" 2>&1
  "$NVCC" -std=c++20 -O3 -ldl -gencode arch=compute_90a,code=sm_90a -o "$WORK/sm90_clock" "$HERE/sm90_clock.cu" \
    >"$OUT/build.log" 2>&1 || { cat "$OUT/build.log" >&2; exit 1; }
  "$NVCC" -O3 -std=c++17 --expt-relaxed-constexpr -gencode arch=compute_90a,code=sm_90a -o "$WORK/msp90" \
    "$HERE/msp90.cu" -ldl >"$OUT/build-msp90.log" 2>&1 || { cat "$OUT/build-msp90.log" >&2; exit 1; }
  "$(dirname "$NVCC")/cuobjdump" -sass "$WORK/sm90_clock" >"$OUT/sm90_clock.sass" 2>&1
  "$(dirname "$NVCC")/cuobjdump" -sass "$WORK/msp90" | "$(dirname "$NVCC")/cu++filt" >"$OUT/msp90.sass" 2>&1
  python3 "$HERE/sass_count.py" "$OUT/msp90.sass" --json "$OUT/msp90-sass.json" >/dev/null 2>&1
  python3 "$HERE/sass_count.py" "$OUT/sm90_clock.sass" --json "$OUT/sass.json" --gate >"$OUT/gate.txt" 2>&1
  rc=$?
  cat "$OUT/gate.txt"
  [ $rc -eq 0 ] || exit 1
  tar -C "$WORK" -czf "$OUT/bins.tgz" sm90_clock msp90
  ;;
time)
  IN=${RESEARCH_RUN_DIR:?run under research run}/inputs
  if [ "${CLK_LEASED:-}" != 1 ]; then
    [ -f "$IN/bins.tgz" ] || { echo "clock.sh: no inputs/bins.tgz (pass the build's with --send)" >&2; exit 2; }
    cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
    [ "${cap%%.*}" = 9 ] || { echo "clock.sh: needs compute capability 9.x, the device reports '$cap'" >&2; exit 2; }
    WORK=$(mktemp -d)
    trap 'rm -rf "$WORK"' EXIT
    tar -C "$WORK" -xzf "$IN/bins.tgz"
    lease=()
    command -v gpu-lease >/dev/null && lease=(gpu-lease 1 --wait --max-min 15 --)
    [ ${#lease[@]} -eq 0 ] && echo "clock.sh: no gpu-lease on this node; running on its GPU directly" >&2
    "${lease[@]}" env CLK_LEASED=1 WORK="$WORK" bash "${BASH_SOURCE[0]}" time
    exit $?
  fi
  B="$WORK/sm90_clock"
  nvidia-smi -q -d CLOCK,PERFORMANCE,POWER,TEMPERATURE >"$OUT/smi-before.txt" 2>&1
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.draw,enforced.power.limit,temperature.gpu,clocks_event_reasons.active \
    --format=csv -lms 100 >"$OUT/smi.csv" 2>&1 &
  smi=$!
  rc=0
  STEPS=${STEPS:-exact msp joint roles lirate lirmix clock}
  STEPS=" ${STEPS//,/ } "
  [[ $STEPS == *" exact "* ]] && { "$B" exact 64 >"$OUT/exact.jsonl" 2>&1 || rc=1; }
  [[ $STEPS == *" msp "* ]] && [ -x "$WORK/msp90" ] && { "$WORK/msp90" >"$OUT/msp90.jsonl" 2>&1 || rc=1; }
  # the JP*90 rows' block-timed points beside their exact (until) twins and the tensor alone ('s test)
  [[ $STEPS == *" joint "* ]] && { "$B" probe --time wg_s8_n256_a1 wgspec_lop3_r256 wgU_lop3 wgspec_imad_r256 wgU_imad \
    wgspec_lop3_r64 wgspec_imad_r64 wgmix_lop3_r256 wgmix_imad_r256 --reps 5 >"$OUT/joint.jsonl" 2>&1 || rc=1; }
  [[ $STEPS == *" roles "* ]] && { "$B" roles 4096 3 >"$OUT/roles.jsonl" 2>&1 || rc=1; }
  [[ $STEPS == *" lirate "* ]] && { "$B" lirate 20000 5 >"$OUT/lirate.jsonl" 2>&1 || rc=1; }
  [[ $STEPS == *" lirmix "* ]] && { "$B" lirmix 10000 7 >"$OUT/lirmix.jsonl" 2>&1 || rc=1; }
  if [[ $STEPS == *" clock "* ]]; then
    : >"$OUT/clock.jsonl"
    # ~0.4 s a launch, 30 launches a load: the light loop first, then the wgmma alone, the design's mixed load, the
    # honest loop without filler, and the mixed load again (the clock at the end of a minute's saturation)
    for L in 3 0 2 1 2; do
      "$B" clock "$L" 600000 30 >>"$OUT/clock.jsonl" 2>&1 || rc=1
    done
  fi
  kill "$smi" 2>/dev/null
  wait "$smi" 2>/dev/null
  nvidia-smi -q -d CLOCK,PERFORMANCE,POWER,TEMPERATURE >"$OUT/smi-after.txt" 2>&1
  [ -f "$OUT/clock.jsonl" ] && python3 - "$OUT" <<'EOF'
import json, statistics, sys
out = sys.argv[1]
rows = [json.loads(l) for l in open(f"{out}/clock.jsonl") if l.startswith("{")]
summ, seen = {}, {}
for r in rows:
    if r["launch"] == 0:
        seen[r["load"]] = seen.get(r["load"], 0) + 1
    summ.setdefault(f"{r['load']}#{seen[r['load']]}", []).append(r)
res = {}
for k, rs in summ.items():
    rs = rs[5:] if len(rs) > 10 else rs   # past the first launches' ramp
    res[k] = {f: round(statistics.median(r[f] for r in rs), 3) for f in
              ("mhz_min", "mhz_med", "mhz_max", "event_mhz", "cycles_per_iter_med", "macs_per_clock_sm")}
    res[k]["launches"] = len(rs)
json.dump(res, open(f"{out}/clock-summary.json", "w"), indent=1)
print(json.dumps(res, indent=1))
EOF
  [ -f "$OUT/exact.jsonl" ] && cat "$OUT/exact.jsonl"
  exit $rc
  ;;
esac
