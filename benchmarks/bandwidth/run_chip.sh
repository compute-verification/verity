#!/usr/bin/env bash
# S_chip measured on one RTX PRO 6000 of vy-nebius-2 (`chip.cu`, `bwchip`), as one Attempt (cwd = the source tree):
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --campaign resource-accounting \
#     --declared-output 'out/*' -- taskset -c 48-95 bash benchmarks/bandwidth/run_chip.sh
#
# Outside the lease: build `bwchip` (refused on any register spill), write the instruction-cache sweep's kernels as PTX
# and assemble each with ptxas, and record each cubin's SASS: its instructions, its add/xor steps (8 per step, or the
# sweep's code bytes are not what it says: refused) and its code bytes (icache_sass.txt), and the highest uniform
# register any kernel of bwchip, bwstream or the tagger names (uniform.txt).
# Under `gpu-lease 1 --wait --on $GPUS --max-min 25`: `bwchip` (its sweeps' gates are the chains' and chases' final
# states against the host's; a failed sweep reports nothing, and bwchip exits 3).
# Env: GPUS (0,1,2,3: NUMA node 0's), TIMED, ONLY (one sweep).  Outputs under $RESEARCH_RUN_DIR/out: build.txt, gpu.txt,
# icache_sass.txt, uniform.txt, chip.json.
set -euo pipefail
cd "$(dirname "$0")"
export O=${RESEARCH_RUN_DIR:-$PWD}/out; mkdir -p "$O"

if [ "${1:-}" = leased ]; then
  S=$2
  UUID=${GPU_LEASE_UUID:?run under gpu-lease}
  NAME=$(nvidia-smi -i "$UUID" --query-gpu=name --format=csv,noheader | head -1)
  nvidia-smi -i "$UUID" --query-gpu=index,uuid,name,pci.bus_id,clocks.sm,clocks.max.sm,clocks.mem,power.draw,temperature.gpu \
    --format=csv > "$O/gpu.txt"
  case "$NAME" in *"RTX PRO 6000"*) ;; *) echo "REFUSED: GPU $UUID is '$NAME', not an RTX PRO 6000"; exit 3 ;; esac
  exec "$S/bwchip" --out "$O" --icache "$S/icache" ${ONLY:+--only "$ONLY"}
fi

CUDA=${CUDA:-/usr/local/cuda}
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
FLAGS=(-O3 -std=c++17 -arch=sm_120 -Xptxas -v)
{ "$CUDA/bin/nvcc" --version; echo "flags: ${FLAGS[*]}"; } > "$O/build.txt"
"$CUDA/bin/nvcc" "${FLAGS[@]}" chip.cu -o "$S/bwchip" >> "$O/build.txt" 2>&1 || { tail -30 "$O/build.txt"; exit 3; }
if grep -E "[1-9][0-9]* bytes spill" "$O/build.txt"; then echo "REFUSED: a kernel spills"; exit 3; fi
mkdir "$S/icache"
"$S/bwchip" --emit-icache "$S/icache"
ls "$S"/icache/*.ptx | xargs -P 16 -I{} sh -c '"$1" -arch=sm_120 -O3 "$2" -o "${2%.ptx}.cubin"' _ "$CUDA/bin/ptxas" {}
sha256sum chip.cu "$S/bwchip" >> "$O/build.txt"

echo "steps instructions alu_steps code_bytes sha256" > "$O/icache_sass.txt"
for f in $(ls "$S"/icache/*.cubin | sort -t_ -k2 -n); do
  n=$(basename "$f" .cubin); n=${n#icache_}
  "$CUDA/bin/cuobjdump" -sass "$f" > "$S/sass.txt"
  ins=$(grep -cE '^\s+/\*[0-9a-f]+\*/' "$S/sass.txt")
  alu=$(grep -cE '^\s+/\*[0-9a-f]+\*/\s+(IADD3|LOP3)' "$S/sass.txt")
  last=$(grep -E '^\s+/\*[0-9a-f]+\*/' "$S/sass.txt" | tail -1 | grep -oE '[0-9a-f]+' | head -1)
  echo "$n $ins $alu $((16#$last + 16)) $(sha256sum "$f" | cut -c1-16)" >> "$O/icache_sass.txt"
  [ "$alu" -ge $((8 * n)) ] || { echo "REFUSED: icache_$n.cubin has $alu add/xor steps, under 8·$n"; exit 3; }
done
rm -f "$S"/icache/*.ptx "$S/sass.txt"

# Uniform registers: how often each URn appears in bwchip's and bwstream's kernels, and three instructions naming URZ
# with their encodings (URZ's index is the register field's all-ones value, so the field's width bounds the file).
"$CUDA/bin/nvcc" -O3 -std=c++17 -arch=sm_120 -cubin stream.cu -o "$S/stream.cubin" > /dev/null 2>&1
{ "$CUDA/bin/cuobjdump" -sass "$S/bwchip"; "$CUDA/bin/cuobjdump" -sass "$S/stream.cubin"; } > "$S/all.sass"
{ grep -oE '\bUR[0-9]+\b' "$S/all.sass" | sort -t R -k2 -n | uniq -c; grep -m3 -E 'UMOV UR[0-9]+, URZ' "$S/all.sass"; } \
  > "$O/uniform.txt"

GPU_LEASE_WHO=${GPU_LEASE_WHO:-ra-pcie-exhaust} gpu-lease 1 --wait --on "${GPUS:-0,1,2,3}" ${TIMED:+--timed} --max-min 25 -- \
  bash "$PWD/run_chip.sh" leased "$S"
