#!/usr/bin/env bash
# S_chip measured on one RTX PRO 6000 of vy-nebius-2 (`chip.cu`, `bwchip`), as one Attempt (cwd = the source tree):
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --campaign resource-accounting \
#     --declared-output 'out/*' -- taskset -c 48-95 bash benchmarks/bandwidth/run_chip.sh
#
# Outside the lease: build `bwchip` (refused on any register spill), write the code-footprint kernels as PTX and
# assemble each with ptxas, and record each cubin's SASS: its instructions, its add/xor steps (2 a chain a step, or the
# sweep's code bytes are not what it says: refused) and its code bytes (sass.txt); count the uniform registers bwchip's
# and bwstream's kernels name (uniform.txt); and rewrite urt.cubin (chip_ur.py) for each pair of uniform registers in
# UR_PAIRS.
# Under `gpu-lease 1 --wait --on $GPUS --max-min 25`: `bwchip` (its sweeps' gates are the chains' and chases' final
# states against the host's; a failed sweep reports nothing, and bwchip exits 3), then `bwchip --ur` for each pair, one
# process each (a register that doesn't exist may fault the context): ur.jsonl.
# Env: GPUS (0,1,2,3: NUMA node 0's), TIMED, ONLY (one sweep), UR_PAIRS.  Outputs under $RESEARCH_RUN_DIR/out: build.txt,
# gpu.txt, sass.txt, uniform.txt, chip.json, ur.jsonl.
set -euo pipefail
cd "$(dirname "$0")"
export O=${RESEARCH_RUN_DIR:-$PWD}/out; mkdir -p "$O"
PAIRS=${UR_PAIRS:-6:7 63:6 64:0 72:8 78:14 80:16 96:32 112:48 126:62 127:63 128:0 128:64 160:32 160:96 192:0 192:64 224:96 254:126 254:62}

if [ "${1:-}" = leased ]; then
  S=$2
  UUID=${GPU_LEASE_UUID:?run under gpu-lease}
  NAME=$(nvidia-smi -i "$UUID" --query-gpu=name --format=csv,noheader | head -1)
  nvidia-smi -i "$UUID" --query-gpu=index,uuid,name,pci.bus_id,clocks.sm,clocks.max.sm,clocks.mem,power.draw,temperature.gpu \
    --format=csv > "$O/gpu.txt"
  case "$NAME" in *"RTX PRO 6000"*) ;; *) echo "REFUSED: GPU $UUID is '$NAME', not an RTX PRO 6000"; exit 3 ;; esac
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

CUDA=${CUDA:-/usr/local/cuda}
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
FLAGS=(-O3 -std=c++17 -arch=sm_120 -Xptxas -v)
{ "$CUDA/bin/nvcc" --version; echo "flags: ${FLAGS[*]}"; } > "$O/build.txt"
"$CUDA/bin/nvcc" "${FLAGS[@]}" chip.cu -o "$S/bwchip" >> "$O/build.txt" 2>&1 || { tail -30 "$O/build.txt"; exit 3; }
if grep -E "[1-9][0-9]* bytes spill" "$O/build.txt"; then echo "REFUSED: a kernel spills"; exit 3; fi
mkdir "$S/ptx" "$S/ur"
"$S/bwchip" --emit "$S/ptx"
ls "$S"/ptx/*.ptx | xargs -P 16 -I{} sh -c '"$1" -arch=sm_120 -O3 "$2" -o "${2%.ptx}.cubin"' _ "$CUDA/bin/ptxas" {}
sha256sum chip.cu chip_ur.py "$S/bwchip" >> "$O/build.txt"

# Each kernel's steps: icache_N 4 chains, icache8_N 8, unified_N 4 (and its two chases per-thread LDCs).
echo "kernel steps instructions alu_steps code_bytes sha256" > "$O/sass.txt"
for f in $(ls "$S"/ptx/*_*.cubin | sort -t_ -k1,1 -k2n); do
  k=$(basename "$f" .cubin); n=${k##*_}; kind=${k%_*}
  case $kind in icache8) chains=8 ;; *) chains=4 ;; esac
  "$CUDA/bin/cuobjdump" -sass "$f" > "$S/sass.txt"
  ins=$(grep -cE '^\s+/\*[0-9a-f]+\*/' "$S/sass.txt")
  alu=$(grep -cE '^\s+/\*[0-9a-f]+\*/\s+(IADD3|LOP3)' "$S/sass.txt" || true)
  last=$(grep -E '^\s+/\*[0-9a-f]+\*/' "$S/sass.txt" | tail -1 | grep -oE '[0-9a-f]+' | head -1)
  echo "$kind $n $ins $alu $((16#$last + 16)) $(sha256sum "$f" | cut -c1-16)" >> "$O/sass.txt"
  [ "$alu" -ge $((2 * chains * n)) ] || { echo "REFUSED: $k.cubin has $alu add/xor, under $((2 * chains))·$n"; exit 3; }
  if [ "$kind" = unified ] && grep -qE 'LDCU UR[0-9]+, c\[0x3\]' "$S/sass.txt"; then
    echo "REFUSED: $k.cubin chases through the uniform path"; exit 3; fi
done
rm -f "$S"/ptx/*.ptx "$S/sass.txt"

# Uniform registers: how often each URn appears in bwchip's and bwstream's kernels, and three instructions naming URZ
# with their encodings (URZ is the register field's all-ones value).
"$CUDA/bin/nvcc" -O3 -std=c++17 -arch=sm_120 -cubin stream.cu -o "$S/stream.cubin" > /dev/null 2>&1
{ "$CUDA/bin/cuobjdump" -sass "$S/bwchip"; "$CUDA/bin/cuobjdump" -sass "$S/stream.cubin"; } > "$S/all.sass"
{ grep -oE '\bUR[0-9]+\b' "$S/all.sass" | sort -t R -k2 -n | uniq -c; grep -m3 -E 'UMOV UR[0-9]+, URZ' "$S/all.sass"; } \
  > "$O/uniform.txt"
for p in $PAIRS; do
  python3 chip_ur.py "$CUDA/bin/cuobjdump" "$S/ptx/urt.cubin" "${p%:*}" "${p#*:}" "$S/ur/urt_${p%:*}_${p#*:}.cubin" >> "$O/uniform.txt"
done

GPU_LEASE_WHO=${GPU_LEASE_WHO:-ra-pcie-exhaust} gpu-lease 1 --wait --on "${GPUS:-0,1,2,3}" ${TIMED:+--timed} --max-min 25 -- \
  bash "$PWD/run_chip.sh" leased "$S"
