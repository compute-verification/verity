#!/usr/bin/env bash
# The certified stream's credit with the prover's tenant on the challenger's host (cwd = the source tree): ra-pcie-
# exhaust's configuration of record (`bwstream --mode foldtwo`, vendored verbatim from b208aa9d5: two processes under
# MPS, the challenger first, 10 generators with cached stores, the fold on 8 SMs, deadline 1,150 µs, 80 MiB of 256 KiB
# slots) on reserved physical cores, with and without `hog` (a memcpy tenant) on the other physical cores of verity's
# CPUs. Arms alternate quiet / tenant, three times each, after the gates pass; `--mode gen` too, quiet and tenant.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --campaign resource-accounting \
#     --declared-output 'out/*' --env GPU_LEASE_WHO=ra-rate-redteam -- taskset -c 48-95 bash benchmarks/rate_redteam/contend.sh
#
# Env: CHALLENGER (the challenger's CPUs, one thread a physical core; default 48,50,...,76), TENANT (the tenant's CPUs;
# default 78-95, both threads of the other physical cores), PHYS=1 (pick both at lease time instead: NEED idle physical
# cores, default 13, for the challenger and every other idle CPU of 48-95 for the tenant), GEN (foldtwo's generators,
# 10), STORES (`cached`, the configuration of record's `--temporal`, and/or `nt`; default cached), TENANT_MIB (each
# tenant thread's buffers, 256), REPS (3).
# Outputs under $RESEARCH_RUN_DIR/out: build.txt, host.txt, gpu.txt, env.txt, smi.csv, gates.json, gen_<arm>.json,
# fold_two_<arm><rep>.json, tenant_<arm><rep>.json (the tenant's bandwidth), busy_<arm><rep>.txt (each CPU's busy share).
set -euo pipefail
cd "$(dirname "$0")/../bandwidth"
export O=${RESEARCH_RUN_DIR:-$PWD}/out; mkdir -p "$O"
export PYTHONPATH=$(cd ../.. && pwd)${PYTHONPATH:+:$PYTHONPATH}
PY=${PY:-python3}
CHALLENGER=${CHALLENGER:-48,50,52,54,56,58,60,62,64,66,68,70,72,74,76}
TENANT=${TENANT:-78-95}

busy() {  # each CPU of verity's 48-95, its busy share over one second
  a=$(grep '^cpu[0-9]' /proc/stat); sleep 1; b=$(grep '^cpu[0-9]' /proc/stat)
  paste <(echo "$a") <(echo "$b") | awk '{n = NF / 2; c = substr($1, 4) + 0; if (c < 48 || c > 95) next;
    i0 = $5 + $6; i1 = $(n + 5) + $(n + 6); t0 = 0; t1 = 0; for (k = 2; k <= n; k++) { t0 += $k; t1 += $(n + k) }
    printf "%s:%.0f%% ", $1, 100 * (1 - (i1 - i0) / (t1 - t0 + 1e-9)) } END { print "" }'
}

if [ "${1:-}" = leased ]; then
  S=$2
  UUID=${GPU_LEASE_UUID:?run under gpu-lease}
  NAME=$(nvidia-smi -i "$UUID" --query-gpu=name --format=csv,noheader | head -1)
  nvidia-smi -i "$UUID" --query-gpu=index,uuid,name,pci.bus_id,pcie.link.gen.current,pcie.link.width.current,clocks.sm,clocks.max.sm,clocks.mem,power.draw \
    --format=csv > "$O/gpu.txt"
  case "$NAME" in *"RTX PRO 6000"*) ;; *) echo "REFUSED: GPU $UUID is '$NAME', not an RTX PRO 6000"; exit 3 ;; esac
  nvidia-smi -i "$UUID" --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,utilization.gpu,pcie.link.gen.current \
    --format=csv,nounits -lms 100 > "$O/smi.csv" &
  SMI=$!
  HOG=
  cleanup() { [ -n "$HOG" ] && kill "$HOG" 2> /dev/null; kill $SMI 2> /dev/null || true; }
  trap cleanup EXIT
  echo "at lease: $(busy)" >> "$O/host.txt"
  # PHYS=1: the challenger on the first NEED physical cores of 48-95 whose threads were both idle for a second (one
  # thread of each), the tenant on every other idle CPU of 48-95 outside those cores, as run_stream.sh's PHYS=1 picks.
  if [ -n "${PHYS:-}" ]; then
    read -r CHALLENGER TENANT < <("$PY" -c '
import time, sys
need = int(sys.argv[1])
def stat():
    return {int(l.split()[0][3:]): [int(x) for x in l.split()[1:]] for l in open("/proc/stat") if l[3].isdigit()}
def cpus(s):
    return [c for p in s.strip().split(",") for c in range(int(p.split("-")[0]), int(p.split("-")[-1]) + 1)]
a = stat(); time.sleep(1); b = stat()
busy = {c: 1 - (b[c][3] + b[c][4] - a[c][3] - a[c][4]) / max(1, sum(b[c]) - sum(a[c])) for c in a}
mine = range(48, 96)
sib = {c: cpus(open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read()) for c in mine}
idle = [c for c in mine if c == min(sib[c]) and all(s in mine and busy[s] < 0.1 for s in sib[c])]
chal = idle[:need]
held = {s for c in chal for s in sib[c]}
ten = [c for c in mine if c not in held and busy[c] < 0.1]
print(",".join(map(str, chal)), ",".join(map(str, ten)))' "${NEED:-13}")
    [ "$(echo "$CHALLENGER" | tr ',' '\n' | wc -l)" -ge "${NEED:-13}" ] ||
      { echo "REFUSED: PHYS=1 but fewer than ${NEED:-13} idle physical cores ($CHALLENGER)"; exit 3; }
    [ -n "$TENANT" ] || { echo "REFUSED: PHYS=1 left no idle CPU for the tenant"; exit 3; }
  fi
  echo "challenger $CHALLENGER; tenant $TENANT" >> "$O/host.txt"
  taskset -pc "$CHALLENGER" $$ > /dev/null
  MPS_CPUS=$(echo "$CHALLENGER" | tr ',' '\n' | tail -n 2 | paste -sd,)
  common=(--out "$O" --blocks 524288 --reps 7 --windows 20 --sizes-mib 80 --piece 1048576 --gen-threads 16)
  "$S/bwstream" --mode gates "${common[@]}" --vectors tests/stream_vectors.json --expect "$O/expect.json" ||
    { r=$?; echo "gates exited $r: nothing is timed"; exit $r; }
  tenant_on() {  # $1: the arm's name
    taskset -c "$TENANT" "$S/hog" "$TENANT" "${TENANT_MIB:-256}" 120 > "$O/tenant_$1.json" &
    HOG=$!
    sleep 3
  }
  tenant_off() { kill "$HOG" 2> /dev/null || true; wait "$HOG" || true; HOG=; }
  "$S/bwstream" --mode gen "${common[@]}" --suffix _quiet
  tenant_on gen; "$S/bwstream" --mode gen "${common[@]}" --suffix _tenant; tenant_off
  # an MPS control daemon of this run's own for the leased GPU, as run_stream.sh's MPS=1
  command -v nvidia-cuda-mps-control > /dev/null || { echo "REFUSED: no nvidia-cuda-mps-control"; exit 3; }
  export MPSDIR=$S/mps-pipe MPSLOG=$S/mps-log; mkdir -p "$MPSDIR" "$MPSLOG"
  CUDA_VISIBLE_DEVICES=$UUID CUDA_MPS_PIPE_DIRECTORY=$MPSDIR CUDA_MPS_LOG_DIRECTORY=$MPSLOG \
    taskset -c "$MPS_CPUS" nvidia-cuda-mps-control -d
  trap 'cleanup; echo quit | CUDA_MPS_PIPE_DIRECTORY=$MPSDIR nvidia-cuda-mps-control; sleep 1;
        nvidia-smi -i "$UUID" --query-compute-apps=pid,process_name,used_memory --format=csv > "$O/mps_after.txt" 2>&1' EXIT
  fold=(--mode foldtwo "${common[@]}" --windows 40 --live "${GEN:-10}" --fold-ctas 8 --deadline-us 1150
        --challenger-first --mps-dir "$MPSDIR")
  rc=0
  # STORES: the generators' stores, cached (`--temporal`, the configuration of record) and/or non-temporal
  for rep in $(seq "${REPS:-3}"); do
    for st in ${STORES:-cached}; do
      for arm in quiet tenant; do
        name=$arm$rep; [ "$st" = nt ] && name=${arm}_nt$rep
        [ $arm = tenant ] && tenant_on "$name"
        busy > "$O/busy_$name.txt"
        extra=(); [ "$st" = cached ] && extra=(--temporal)
        CUDA_VISIBLE_DEVICES=0 "$S/bwstream" "${fold[@]}" "${extra[@]}" --suffix "_$name" ||
          { r=$?; echo "foldtwo $name exited $r"; [ $rc = 0 ] && rc=$r; }
        [ $arm = tenant ] && tenant_off
      done
    done
  done
  exit $rc
fi

NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
FLAGS=(-O3 -std=c++17 -arch=sm_120 -Xptxas -v)
HOSTFLAGS=(-O3 -mavx512f -mavx512bw -mvaes -maes)
{ "$NVCC" --version; g++ --version | head -1; echo "flags: ${FLAGS[*]}; host: ${HOSTFLAGS[*]}"; } > "$O/build.txt"
g++ "${HOSTFLAGS[@]}" -c stream_host.cpp -o "$S/stream_host.o" >> "$O/build.txt" 2>&1 || { tail -30 "$O/build.txt"; exit 3; }
"$NVCC" "${FLAGS[@]}" stream.cu "$S/stream_host.o" -o "$S/bwstream" -ldl >> "$O/build.txt" 2>&1 || { tail -30 "$O/build.txt"; exit 3; }
if grep -E "[1-9][0-9]* bytes spill" "$O/build.txt"; then echo "REFUSED: a kernel spills"; exit 3; fi
gcc -O2 -pthread ../rate_redteam/hog.c -o "$S/hog" >> "$O/build.txt" 2>&1 || { tail -30 "$O/build.txt"; exit 3; }
sha256sum stream.cu tagger.cu stream_host.cpp stream.py tests/stream_vectors.json ../rate_redteam/hog.c "$S/bwstream" >> "$O/build.txt"
{ lscpu | grep -E 'Model name|L3'; echo "affinity: $(taskset -pc $$)"; echo "before build: $(busy)"; } > "$O/host.txt" 2>&1
"$PY" stream.py "$S/stream_vectors.json"
cmp "$S/stream_vectors.json" tests/stream_vectors.json || { echo "REFUSED: tests/stream_vectors.json is not stream.py's"; exit 3; }
"$PY" reference.py --blocks 524288 --seed 000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f \
  --workers 16 --out "$O/expect.json" > /dev/null

GPU_LEASE_WHO=${GPU_LEASE_WHO:-ra-rate-redteam} gpu-lease 1 --wait --on "${GPUS:-0,1,2,3}" --max-min 20 -- \
  bash "$PWD/../rate_redteam/contend.sh" leased "$S"
