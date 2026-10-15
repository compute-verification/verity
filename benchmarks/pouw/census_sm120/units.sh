#!/usr/bin/env bash
# Round 3 of the census: the GPU's units outside the formal model's capacity table. Inside a research run, cwd the clone:
#   bash benchmarks/pouw/census_sm120/units.sh tex|rt...
# tex: tex.cu (texture filtering throughput and weight resolution) and its SASS. rt: rt_optix.cu (RT-core triangle tests,
# OptiX from the driver, headers fetched at OPTIX_TAG). Each builds outside the lease, then runs under one
# `gpu-lease 1 --wait --max-min 20`, with the GPU's clocks and throttle reasons read (never set) before and after.
# Writes $OUT/<part>.jsonl, <part>.sass, <part>-smi-*.csv, build-<part>.log.
set -uo pipefail
OUT=${OUT:-${RESEARCH_RUN_DIR:?}/out}
mkdir -p "$OUT"
SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NVCC=/usr/local/cuda-13.0/bin/nvcc
CUDA=/usr/local/cuda-13.0
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
Q=index,uuid,name,clocks.sm,clocks.max.sm,clocks.applications.graphics,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active

leased() { # PART CMD...: run CMD under a lease, with smi before, during (every 0.5 s) and after
  local part=$1
  shift
  cat >"$W/inner.sh" <<'EOF2'
part=$1
shift
nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu="$Q" --format=csv >"$OUT/$part-smi-before.csv" 2>&1
nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.active \
  --format=csv -lms 500 >"$OUT/$part-smi-during.csv" 2>&1 &
smi=$!
"$@" >>"$OUT/$part.jsonl" 2>>"$OUT/$part.err"
rc=$?
kill "$smi"
nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu="$Q" --format=csv >"$OUT/$part-smi-after.csv" 2>&1
exit $rc
EOF2
  OUT=$OUT Q=$Q gpu-lease 1 --wait --max-min 20 -- bash "$W/inner.sh" "$part" "$@" >>"$OUT/$part-lease.log" 2>&1
  echo "$(date -u +%H:%M:%SZ) $part: lease exit $?" | tee -a "$OUT/timeline.log"
}

for part in "$@"; do
  echo "$(date -u +%H:%M:%SZ) $part" | tee -a "$OUT/timeline.log"
  case $part in
    tex)
      $NVCC -O3 -arch=sm_120a -o "$W/tex" "$SRC/tex.cu" >"$OUT/build-tex.log" 2>&1 || { echo "tex build failed"; continue; }
      $CUDA/bin/cuobjdump -sass "$W/tex" >"$OUT/tex.sass" 2>&1
      leased tex "$W/tex" 20000
      ;;
    rt)
      tag=${OPTIX_TAG:-v9.0.0}
      git clone -q --depth 1 --branch "$tag" https://github.com/NVIDIA/optix-dev "$W/optix" >"$OUT/build-rt.log" 2>&1 \
        || { echo "optix headers ($tag) not fetched"; continue; }
      (cd "$W/optix" && git log -1 --format='optix-dev %H %s') >>"$OUT/build-rt.log"
      $NVCC -O3 -arch=sm_120a -optix-ir -I"$W/optix/include" -o "$W/rt_dev.optixir" "$SRC/rt_optix_dev.cu" \
        >>"$OUT/build-rt.log" 2>&1 || { echo "rt device build failed"; continue; }
      $NVCC -O3 -arch=sm_120a -I"$W/optix/include" -o "$W/rt" "$SRC/rt_optix.cu" -ldl >>"$OUT/build-rt.log" 2>&1 \
        || { echo "rt host build failed"; continue; }
      leased rt "$W/rt" "$W/rt_dev.optixir"
      ;;
  esac
done
echo "$(date -u +%H:%M:%SZ) done" | tee -a "$OUT/timeline.log"
