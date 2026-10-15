#!/usr/bin/env bash
# Instance census of the RTX PRO 6000 Blackwell Server Edition (sm_120) nodes: every constant an adversary or the
# honest prover can exploit, as microbenchmarks with a run id (CMP-12, worker W1; note lanes/compute-accounting
# *-draft-census-sm120.md). Run from a clean commit under `research run --cwd clone`; outputs go to $OUT.
#
#   research run --on vy-nebius-2 --project verity --campaign pouw --source <tree> --cwd clone --custody-ttl 12h \
#     --env GPU_LEASE_WHO=<id> -- bash -c 'bash benchmarks/pouw/census_sm120/census.sh tensor cuda memside memory'
#
# Sections (arguments; default: every GPU section). Each GPU section runs under `gpu-lease 1 --wait --max-min 20`
# (`CENSUS_GPU=<index|uuid>` pins the die), with nvidia-smi sampling the leased GPU's SM/memory clock, power and clock-event
# reasons every 200 ms, and `nvidia-smi -q -d CLOCK,POWER,PERFORMANCE` (read-only) before and after:
#   tensor     tensor.cu: every mma.sync kind sm_120a accepts (FP8 E4M3/E5M2 m16n8k32 with FP32/FP16 accumulate,
#              f8f6f4 FP4/FP6, block-scaled MXFP8 and NVFP4/MXFP4 k64, 2:4-sparse FP8/NVFP4/MXFP8/INT8/BF16/TF32, INT8,
#              INT4, binary, BF16/FP16 m16n8k16, TF32 m16n8k8, FP64 m8n8k4): MACs a SM a clock, the sub-partition
#              cycles a warp instruction holds the tensor pipe (slot time), dependent latency.
#   cuda       cuda_pipes.cu: FFMA, FADD, FMUL, HFMA2, BF16 FMA, IMAD, IADD3, LOP3, SHF, PRMT, POPC, DP4A, min/max,
#              BREV, FLO, F2FP packs (E4M3, E2M1, BF16, FP16), I2F, MUFU: warp instructions a SM a clock, alone and in
#              1:1 pairs.
#   coissue    coissue.cu: tensor and CUDA work in one kernel (k CUDA ops per MMA), in two concurrent kernels, and
#              warp-specialised (W MMA warps beside W' CUDA-op warps, per-warp clocks: the per-sub-partition frontier).
#   memside    memside.cu: L2 reductions and atomics by type, width and address pattern; TMA bulk loads and bulk
#              reductions; warp-specialised co-runs of RED, FFMA, QMMA and L2 reads.
#   memory     memory.cu: device properties; DRAM read/write by data pattern, plain vs COMP_GENERIC allocations; COMP_GENERIC
#              zeros by how they were written and what the allocation held before; reads by load path (cg, nc, default);
#              D2D copy; L2 footprint sweep; shared-memory bandwidth; random-row gather by row size.
# `CENSUS_ONLY=<label>` passes a filter to every GPU binary (e.g. `split` for coissue's warp-specialised cases, `fill` for
# memory's fill_methods).
#   links      links.cu under `gpu-lease 2` (`CENSUS_GPUS=i,j` pins the pair): PCIe H2D/D2H/bidirectional per GPU,
#              P2P, both GPUs' transfers at once; plus `nvidia-smi topo -m` and the PCIe tree.
#   host       host.c, no lease, nice 19, 16 threads: STREAM-like host DRAM bandwidth, AMX int8/bf16 tile throughput;
#              sequential read of a file written in the run's scratch (O_DIRECT); lscpu, lsblk.
# Then census.py reads every section's JSON lines and the SASS of each kernel's loop and writes $OUT/census.json and
# $OUT/census.md (constant, value, unit, run id, GPU UUID, method); `census.py --merge` joins runs' tables.
set -uo pipefail
OUT=${OUT:?run under research run}
SRC=${CENSUS_SRC:-$PWD/benchmarks/pouw/census_sm120}
[ -f "$SRC/tensor.cu" ] || SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
CUOBJDUMP=$(dirname "$NVCC")/cuobjdump
SECTIONS=${*:-tensor cuda coissue memside memory}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$OUT/sass"
echo "{\"run_id\":\"${RESEARCH_RUN_ID:-$(basename "${RESEARCH_RUN_DIR:-unknown}")}\",\"host\":\"$(hostname)\",\"sections\":\"$SECTIONS\",\"start_utc\":\"$(date -u +%FT%TZ)\",\"nvcc\":\"$($NVCC --version | tail -2 | head -1)\"}" >"$OUT/meta.json"

build() {  # build NAME [extra flags]: outside any lease
  local n=$1; shift
  if [ -x "$WORK/$n" ]; then return 0; fi
  if "$NVCC" -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v -I"$SRC" -o "$WORK/$n" "$SRC/$n.cu" "$@" \
      >"$OUT/build-$n.log" 2>&1; then
    "$CUOBJDUMP" -sass "$WORK/$n" >"$OUT/sass/$n.sass" 2>&1 || true
  else
    echo "{\"section\":\"$n\",\"build\":\"failed\"}" >>"$OUT/failures.jsonl"
    tail -20 "$OUT/build-$n.log" >&2
  fi
}

gpu_sections=()
for s in $SECTIONS; do
  case $s in
    tensor) build tensor; gpu_sections+=(tensor) ;;
    cuda) build cuda_pipes; gpu_sections+=(cuda_pipes) ;;
    coissue) build coissue; gpu_sections+=(coissue) ;;
    memside) build memside; gpu_sections+=(memside) ;;
    memory) build memory -lcuda; gpu_sections+=(memory) ;;
    links) build links ;;
    host) ;;
    *) echo "unknown section $s" >&2 ;;
  esac
done

if [ ${#gpu_sections[@]} -gt 0 ]; then
  cat >"$WORK/gpu.sh" <<EOF
set -uo pipefail
U=\$GPU_LEASE_UUID
echo "{\"gpu_uuid\":\"\$U\",\"cuda_visible_devices\":\"\${CUDA_VISIBLE_DEVICES:-}\",\"lease_start_utc\":\"\$(date -u +%FT%TZ)\"}" >"$OUT/lease.json"
nvidia-smi -q -d CLOCK,POWER,PERFORMANCE --id="\$U" >"$OUT/smi-before.txt" 2>&1
nvidia-smi --id="\$U" --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active,utilization.gpu --format=csv,noheader -lms 200 >"$OUT/smi-samples.csv" 2>&1 &
SMI=\$!
for b in ${gpu_sections[*]}; do
  [ -x "$WORK/\$b" ] || continue
  echo "\$(date -u +%FT%T.%NZ) start \$b" >>"$OUT/sections.log"
  timeout 600 "$WORK/\$b" \${CENSUS_ONLY:-} >"$OUT/\$b.jsonl" 2>"$OUT/\$b.err"
  echo "\$(date -u +%FT%T.%NZ) end \$b rc=\$?" >>"$OUT/sections.log"
done
kill \$SMI 2>/dev/null
nvidia-smi -q -d CLOCK,POWER,PERFORMANCE --id="\$U" >"$OUT/smi-after.txt" 2>&1
EOF
  pin=()
  [ -n "${CENSUS_GPU:-}" ] && pin=(--on "$CENSUS_GPU")
  gpu-lease 1 --wait --max-min 20 "${pin[@]}" -- bash "$WORK/gpu.sh"
  echo "gpu lease rc=$?" >>"$OUT/sections.log"
fi

for s in $SECTIONS; do
  case $s in
    links)
      nvidia-smi topo -m >"$OUT/topo.txt" 2>&1
      lspci -tv >"$OUT/lspci-tree.txt" 2>&1
      nvidia-smi --query-gpu=index,uuid,pci.bus_id,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv >"$OUT/pcie.csv" 2>&1
      if [ -x "$WORK/links" ]; then
        pin=()
        [ -n "${CENSUS_GPUS:-}" ] && pin=(--on "$CENSUS_GPUS")
        gpu-lease 2 --wait --max-min 15 "${pin[@]}" -- bash -c "echo \"{\\\"gpu_uuids\\\":\\\"\$GPU_LEASE_UUID\\\"}\" >'$OUT/links-lease.json'; timeout 600 '$WORK/links' >'$OUT/links.jsonl' 2>'$OUT/links.err'"
        echo "links lease rc=$?" >>"$OUT/sections.log"
      fi
      ;;
    host)
      bash "$SRC/host.sh" "$WORK" >>"$OUT/sections.log" 2>&1
      ;;
  esac
done

python3 "$SRC/census.py" "$OUT" >"$OUT/census-py.log" 2>&1 || echo "census.py failed rc=$?" >>"$OUT/sections.log"
ls -la "$OUT" >&2
cat "$OUT/sections.log" >&2
