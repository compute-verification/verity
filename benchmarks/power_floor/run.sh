#!/usr/bin/env bash
# POWER-MODEL's energy-per-operation run (energy.cu, CMP-172): build for this machine's card, then under one gpu-lease run
# every pipe alone, at partial SM loads, in pairs, and cuBLASLt's s8 GEMMs at the demo's shapes, each a steady window
# with NVML's energy counter. From a pushed commit:
#
#   research run --on vy-nebius-1 --project verity --campaign compute-research --source <tree> --cwd source -- \
#     env CHECK_SLOTS=/workspace/research/locks/gpu/slots python3 tools/verity/check/slot.py --by compute-research \
#     --wait-max 1800 --run-max 2400 -- bash benchmarks/power_floor/run.sh
#
# $OUT gets energy.jsonl (energy.cu's line a config, with its label), phases.jsonl (each config's start and end, UTC),
# smi_trace.csv (nvidia-smi every 50 ms on the leased GPU), volt/ (nvidia-smi -q -d VOLTAGE,CLOCK,POWER mid-window), the
# nvidia-smi -q snapshots before and after, energy.sass and build.log. PF_SECS (15) is a window, PF_WARM (6) the warm-up
# before it, PF_LEASE_ON the GPU to wait for (gpu-lease --on). PF_PLAN=full (the default): every config at the GPU's
# clocks, then a clock sweep (PF_CLOCKS, MHz); PF_PLAN=cap: the default clocks (nvidia-smi -rgc: the power cap holds the
# board), then PF_CAP_CLOCKS (2430 2250 2100). Clocks change only where `sudo -n nvidia-smi` is allowed, and every exit
# puts the GPU back: locked at PF_RELOCK_MHZ (2100 on the Nebius nodes, vm_setup.sh's vy-clocks), else nvidia-smi -rgc.
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
[ -x "$NVCC" ] || NVCC=$(command -v nvcc)
export PATH="$(dirname "$NVCC"):$PATH"

cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
case $cc in
  12.0) ARCH=(-gencode arch=compute_120a,code=sm_120a); MMA=imma ;;
  9.0) ARCH=(-gencode arch=compute_90a,code=sm_90a -DPF_SM90); MMA=wgmma ;;
  *) echo "no build for compute capability $cc" >&2; exit 1 ;;
esac
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
"$NVCC" -O3 -std=c++17 "${ARCH[@]}" -lineinfo -o "$WORK/energy" "$HERE/energy.cu" -lcublasLt -ldl -lpthread \
  >"$OUT/build.log" 2>&1 || { cat "$OUT/build.log" >&2; exit 1; }
cuobjdump -sass "$WORK/energy" >"$OUT/energy.sass" 2>&1
gpu-lease 1 --wait ${PF_LEASE_ON:+--on "$PF_LEASE_ON"} --max-min "${PF_LEASE_MIN:-30}" -- \
  bash "$HERE/run_lease.sh" "$WORK/energy" "$MMA"
