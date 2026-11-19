#!/usr/bin/env bash
# `verity calibrate bandwidth chip|write_gpu|write_host`: this directory's one runner (cwd = the source tree), the
# device ceilings PoUB's timing cites, one probe a run, each the command of a `research run`. Each probe writes its
# records and the run's result.json through the harness's kit (`result.py`): `write_host` its fastest record as
# metrics.toml's number, `chip` and `write_gpu` their check's validation with no number.
#
#   chip        S_chip on one card, CARD (required: the `short` of its file under calibration/hardware/cards; the
#               records cited are an RTX PRO 6000's), `chip.cu`, `bwchip`. Outside the lease: build `bwchip` for the
#               card's plain architecture (refused on any
#               register spill), write the code-footprint kernels as PTX and assemble each with ptxas, and record each
#               cubin's SASS: its instructions, its add/xor steps (2 a chain a step, or the sweep's code bytes are not
#               what it says: refused) and its code bytes (sass.txt); count the uniform registers bwchip's kernels name
#               (uniform.txt); and rewrite urt.cubin (chip_ur.py) for each pair of uniform registers in UR_PAIRS.
#               Under `gpu-lease 1 --wait --on $GPUS --max-min 25`, refused where the leased card's file states
#               another `arch` than CARD's: `bwchip` (its sweeps' gates are the chains' and
#               chases' final states against the host's; a failed sweep reports nothing, and bwchip exits 3), then
#               `bwchip --ur` for each pair, one process each (a register that doesn't exist may fault the context):
#               ur.jsonl. Env: CUDA_HOME (the toolkit's directory, required: the probe names no version of its
#               own), GPUS (0,1,2,3: NUMA node 0's), TIMED, ONLY (one sweep), UR_PAIRS. Outputs under
#               $RESEARCH_RUN_DIR/out: build.txt, gpu.txt, sass.txt, uniform.txt, chip.json, ur.jsonl.
#   write_gpu   The write ceilings' GPU half: builds write.cu (bwwrite) for CARD (required: the `short` of the
#               card's file), its `arch` less the `a`, with toolkit 13.0 (NVCC, else $CUDA_HOME, else the machine's
#               record: `machine.py toolkit 13.0`) and runs every arm on one card, the queue's GPU (GPU_LEASE_UUID) or
#               one it leases, with the card's clocks, PCIe link and ECC mode before and after and nvidia-smi dmon
#               beside it. Outputs in $OUT: build.txt, sass.txt, gpu.txt, dmon.txt, write.jsonl, rc.txt. ARGS go to
#               bwwrite (default: 32 GiB on the card, 8 GiB pinned on the host, 7 reps).
#   write_host  The write ceilings' host half: builds write_host.c (bwwrite_host) and, on the CPUs the job was given,
#               writes host DRAM with KERNELS (default nt st stosb) at one thread a core, for 1, 2, 4, ... cores and
#               then every core given, by NUMA node and together, REPS times (default 2) for SECONDS_EACH each
#               (default 2), MIB a thread (default 512, above a socket's L3 at one thread). Outputs in $OUT: env.txt,
#               build.txt, write_host.jsonl, rc.txt; its number only where no record has a wrong byte (else exit 5).
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --campaign resource-accounting \
#     --declared-output 'out/*' -- env CUDA_HOME=DIR CARD=SHORT taskset -c 48-95 verity calibrate bandwidth chip
#   research run --queue --gpus 1 --max-min 30 --cpus 8 --mem-gb 32 --project verity --source . --cwd source \
#     -- env CARD=SHORT verity calibrate bandwidth write_gpu
#   research run --on auto --cpus 48 --mem-gb 64 --project verity --cwd source -- verity calibrate bandwidth write_host
USAGE="usage: verity calibrate bandwidth chip|write_gpu|write_host"

chip() {
set -euo pipefail
cd "$(dirname "$0")"
export O=${RESEARCH_RUN_DIR:-$PWD}/out; mkdir -p "$O"
PAIRS=${UR_PAIRS:-6:7 63:6 64:0 72:8 78:14 79:15 79:78 80:16 96:32 112:48 126:62 127:63 128:0 128:64 160:32 160:96 192:0 192:64 224:96 254:126 254:62}

if [ "${1:-}" = leased ]; then
  S=$2
  : "${CARD:?chip leased runs under run.sh chip}" "${ARCH:?chip leased runs under run.sh chip}"
  UUID=${GPU_LEASE_UUID:?run under gpu-lease}
  NAME=$(nvidia-smi -i "$UUID" --query-gpu=name --format=csv,noheader | head -1)
  nvidia-smi -i "$UUID" --query-gpu=index,uuid,name,pci.bus_id,clocks.sm,clocks.max.sm,clocks.mem,power.draw,temperature.gpu \
    --format=csv > "$O/gpu.txt"
  HAS=$(python3 ../../benchmarks/kernels/machine.py arch "$NAME" 2>&1) || { echo "REFUSED: GPU $UUID is '$NAME': $HAS"; exit 3; }
  [ "$HAS" = "$ARCH" ] || { echo "REFUSED: GPU $UUID is '$NAME', whose card file states $HAS, not the $ARCH of CARD=$CARD that bwchip is built for"; exit 3; }
  rc=0
  "$S/bwchip" --out "$O" --ptx "$S/ptx" ${ONLY:+--only "$ONLY"} > /dev/null || rc=$?
  : > "$O/ur.jsonl"
  for p in $PAIRS; do
    a=${p%:*} b=${p#*:} r=0
    line=$("$S/bwchip" --ur "$S/ur/urt_${a}_${b}.cubin" "$a" "$b" 2> "$S/ur.err") || r=$?
    [ -n "$line" ] || line="{\"a\":$a,\"b\":$b,\"error\":\"$(tr -d '"\n' < "$S/ur.err" | cut -c1-200)\"}"
    echo "${line%\}},\"rc\":$r}" >> "$O/ur.jsonl"
  done
  exit $rc
fi

CUDA=${CUDA:-${CUDA_HOME:-}}
[ -n "$CUDA" ] || { echo "REFUSED: chip: no CUDA toolkit given: set CUDA_HOME to its directory (the machine's own: python3 benchmarks/kernels/machine.py toolkit)" >&2; exit 3; }
[ -n "${CARD:-}" ] || { echo "REFUSED: chip: CARD is not given: the short name of the card's file under calibration/hardware/cards" >&2; exit 3; }
ARCH=$(python3 ../../benchmarks/kernels/machine.py card-arch "$CARD") || exit 3
export CARD ARCH
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
# Built for the plain architecture, the card's `arch` without its `a` suffix (sm_120 on an RTX PRO 6000, sm_90 on Hopper),
# on purpose: every record cited was built so, and the `a` target's build is other bytes (compile-only, both ways, toolkit
# 13.0: r20261118-083832-a37a).
FLAGS=(-O3 -std=c++17 -arch="${ARCH%a}" -Xptxas -v)
{ "$CUDA/bin/nvcc" --version; echo "flags: ${FLAGS[*]}"; } > "$O/build.txt"
"$CUDA/bin/nvcc" "${FLAGS[@]}" chip.cu -o "$S/bwchip" >> "$O/build.txt" 2>&1 || { tail -30 "$O/build.txt"; exit 3; }
if grep -E "[1-9][0-9]* bytes spill" "$O/build.txt"; then echo "REFUSED: a kernel spills"; exit 3; fi
mkdir "$S/ptx" "$S/ur"
"$S/bwchip" --emit "$S/ptx"
ls "$S"/ptx/*.ptx | xargs -P 16 -I{} sh -c '"$1" -arch="$3" -O3 "$2" -o "${2%.ptx}.cubin"' _ "$CUDA/bin/ptxas" {} "${ARCH%a}"
sha256sum chip.cu chip_ur.py "$S/bwchip" >> "$O/build.txt"

# Each kernel's steps: icache_N 4 chains of add/xor, icache8_N 8, unified_N, clevel_N and glevel_N 4, icachef_N 8 chains
# of FFMA with immediates (one instruction a chain a step), cpoll_0 none; unified, clevel and cpoll chase constants
# per thread (LDC, never the uniform path).
echo "kernel steps instructions alu_steps code_bytes sha256" > "$O/sass.txt"
for f in $(ls "$S"/ptx/*_*.cubin | sort -t_ -k1,1 -k2n); do
  k=$(basename "$f" .cubin); n=${k##*_}; kind=${k%_*}
  case $kind in icache8) need=$((16 * n)) op='IADD3|LOP3' ;; icachef) need=$((8 * n)) op=FFMA ;;
    *) need=$((8 * n)) op='IADD3|LOP3' ;; esac
  "$CUDA/bin/cuobjdump" -sass "$f" > "$S/sass.txt"
  ins=$(grep -cE '^\s+/\*[0-9a-f]+\*/' "$S/sass.txt")
  alu=$(grep -cE "^\s+/\*[0-9a-f]+\*/\s+($op)\b" "$S/sass.txt" || true)
  last=$(grep -E '^\s+/\*[0-9a-f]+\*/' "$S/sass.txt" | tail -1 | grep -oE '[0-9a-f]+' | head -1)
  echo "$kind $n $ins $alu $((16#$last + 16)) $(sha256sum "$f" | cut -c1-16)" >> "$O/sass.txt"
  [ "$alu" -ge "$need" ] || { echo "REFUSED: $k.cubin has $alu of $op, under $need"; exit 3; }
  if [ "$kind" = icachef ] && grep -qE 'FFMA [^;]*c\[' "$S/sass.txt"; then
    echo "REFUSED: $k.cubin reads its FFMA constants from a bank"; exit 3; fi
  case $kind in unified | clevel | cpoll) if grep -qE 'LDCU UR[0-9]+, c\[0x3\]' "$S/sass.txt"; then
    echo "REFUSED: $k.cubin chases through the uniform path"; exit 3; fi ;; esac
done
rm -f "$S"/ptx/*.ptx "$S/sass.txt"

# Uniform registers: how often each URn appears in bwchip's kernels, and up to three instructions naming URZ with their
# encodings (URZ is the register field's all-ones value), where a kernel has one.
"$CUDA/bin/cuobjdump" -sass "$S/bwchip" > "$S/all.sass"
{ grep -oE '\bUR[0-9]+\b' "$S/all.sass" | sort -t R -k2 -n | uniq -c
  grep -m3 -E 'UMOV UR[0-9]+, URZ' "$S/all.sass" || true; } > "$O/uniform.txt"
for p in $PAIRS; do
  python3 chip_ur.py "$CUDA/bin/cuobjdump" "$S/ptx/urt.cubin" "${p%:*}" "${p#*:}" "$S/ur/urt_${p%:*}_${p#*:}.cubin" >> "$O/uniform.txt"
done

rc=0; GPU_LEASE_WHO=${GPU_LEASE_WHO:-ra-pcie-exhaust} gpu-lease 1 --wait --on "${GPUS:-0,1,2,3}" ${TIMED:+--timed} --max-min 25 -- \
  bash "$PWD/run.sh" chip leased "$S" || rc=$?
uv run -q python result.py chip "$O" "$rc"
exit "$rc"
}

write_gpu() {
set -uxo pipefail
export O=${OUT:?run under research run}; mkdir -p "$O"
BW=$PWD/calibration/bandwidth
export B=$(mktemp -d /tmp/bwwrite.XXXXXX); trap 'rm -rf "$B"' EXIT
NVCC=${NVCC:-$(python3 benchmarks/kernels/machine.py toolkit 13.0)/bin/nvcc} || { echo 3 > "$O/rc.txt"; exit 3; }
[ -n "${CARD:-}" ] || { echo "write_gpu: CARD is not given: the short name of the card's file under calibration/hardware/cards" >&2; echo 3 > "$O/rc.txt"; exit 3; }
ARCH=$(python3 benchmarks/kernels/machine.py card-arch "$CARD") || { echo 3 > "$O/rc.txt"; exit 3; }
# Built for the plain architecture, the card's `arch` without its `a` suffix (sm_120 on an RTX PRO 6000, sm_90 on Hopper),
# on purpose: every record cited was built so, and the `a` target's build is other bytes (compile-only, both ways, toolkit
# 13.0: r20261118-083832-a37a).
if ! "$NVCC" -O3 -std=c++17 -arch="${ARCH%a}" -o "$B/bwwrite" "$BW/write.cu" > "$O/build.txt" 2>&1; then
  cat "$O/build.txt"; echo 4 > "$O/rc.txt"; exit 4
fi
{ "$NVCC" --version; sha256sum "$B/bwwrite" "$BW/write.cu"; } >> "$O/build.txt"
"$(dirname "$NVCC")/cuobjdump" -sass "$B/bwwrite" | grep -oE "\b(STG|UBLKCP|UTMASTG|LDG)[A-Z0-9._]*" | sort | uniq -c \
  > "$O/sass.txt"

measure() {
  local q=index,uuid,name,pci.bus_id,pcie.link.gen.current,pcie.link.width.current,pcie.link.gen.max,pcie.link.width.max
  q=$q,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,ecc.mode.current,power.draw,power.limit,memory.total
  { date -u +%FT%TZ; echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
    nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=$q --format=csv; } > "$O/gpu.txt" 2>&1
  nvidia-smi dmon -i "$GPU_LEASE_UUID" -s pucm -d 1 > "$O/dmon.txt" 2>&1 & local mon=$!
  # shellcheck disable=SC2086
  "$B/bwwrite" ${ARGS:-} > "$O/write.jsonl"; local rc=$?
  kill "$mon"
  { date -u +%FT%TZ; nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=$q --format=csv; } >> "$O/gpu.txt" 2>&1
  return $rc
}
export -f measure
if [ -n "${GPU_LEASE_UUID:-}" ]; then
  measure; rc=$?
else
  gpu-lease 1 --wait --max-min 15 -- bash -c measure; rc=$?
fi
echo "$rc" > "$O/rc.txt"
uv run -q python "$BW/result.py" write_gpu "$O" "$rc"
exit "$rc"
}

write_host() {
set -uxo pipefail
O=${OUT:?run under research run}; mkdir -p "$O"
BW=$PWD/calibration/bandwidth
B=$(mktemp -d /tmp/bwwrite-host.XXXXXX); trap 'rm -rf "$B"' EXIT
{ date -u +%FT%TZ; lscpu; numactl -H; taskset -pc $$; grep -H . /proc/self/cgroup; head -20 /proc/meminfo
  cat /sys/kernel/mm/transparent_hugepage/enabled; sudo -n dmidecode -t memory | grep -E "Speed|Type:|Size" | sort | uniq -c
  uptime; } > "$O/env.txt" 2>&1
if ! gcc -O3 -march=native -pthread -Wall -o "$B/bwwrite_host" "$BW/write_host.c" > "$O/build.txt" 2>&1; then
  cat "$O/build.txt"; echo 4 > "$O/rc.txt"; exit 4
fi
sha256sum "$B/bwwrite_host" "$BW/write_host.c" >> "$O/build.txt"

# One CPU a physical core among those allowed, as "node cpu" lines in core order.
python3 - > "$B/cores.txt" <<'EOF'
import os
seen = set()
for c in sorted(os.sched_getaffinity(0)):
    t = f"/sys/devices/system/cpu/cpu{c}/topology"
    key = (open(f"{t}/physical_package_id").read().strip(), open(f"{t}/core_id").read().strip())
    if key in seen:
        continue
    seen.add(key)
    node = next(int(d[4:]) for d in os.listdir(f"/sys/devices/system/cpu/cpu{c}") if d.startswith("node"))
    print(node, c)
EOF
cat "$B/cores.txt" >> "$O/env.txt"
first() { head -n "$1" | awk '{print $2}' | paste -sd,; }
ALL=$(first 100000 < "$B/cores.txt")
N=$(wc -l < "$B/cores.txt")
SETS=()
for k in 1 2 4 8 16 32 64; do [ "$k" -lt "$N" ] && SETS+=("cores$k $(first "$k" < "$B/cores.txt")"); done
for nd in $(awk '{print $1}' "$B/cores.txt" | sort -u); do
  SETS+=("node$nd-all $(awk -v n="$nd" '$1 == n {print $2}' "$B/cores.txt" | paste -sd,)")
done
[ "$(awk '{print $1}' "$B/cores.txt" | sort -u | wc -l)" -gt 1 ] && SETS+=("all $ALL")
rc=0
for r in $(seq "${REPS:-2}"); do
  for k in ${KERNELS:-nt st stosb}; do
    for s in "${SETS[@]}"; do
      "$B/bwwrite_host" --label "${s%% *}" --cpus "${s#* }" --kernel "$k" --mib "${MIB:-512}" --seconds "${SECONDS_EACH:-2}" \
        | tee -a "$O/write_host.jsonl"
      [ "${PIPESTATUS[0]}" = 0 ] || rc=6
    done
  done
done
uptime >> "$O/env.txt"
uv run -q python "$BW/result.py" write_host "$O"; wrote=$?
[ "$rc" = 0 ] && rc=$wrote
echo "$rc" > "$O/rc.txt"
exit "$rc"
}

case "${1:-}" in
  chip | write_gpu | write_host) name=$1; shift; "$name" "$@" ;;
  *) echo "$USAGE" >&2; exit 2 ;;
esac
