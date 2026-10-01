#!/usr/bin/env bash
# R1's deciding primitive on one RTX PRO 6000 (the store's docs/pouw/new-designs.md §2.1).  Builds pearl_c_sm120.cubin as
# build.sh does, with the node's CUDA 13.0.1 (CUDA, default /usr/local/cuda-13.0; the harness gate's other pinned toolkit,
# 12.9.1, spills gemm_plain), refuses it unless sass_gate.py passes, and with `bench` runs the harness's arms (ARMS, default
# R1's r1_arm.py; R1-H's is r1h_arm:R1HChainSm120) against the autotuned plain FP8 GEMMs at Llama-3.1-8B's linears, prefill
# (m 2,048) and decode (m 32).
#
# Research question: what does R1's binding cost on the card, over plain FP8, at a real model's shapes: the canonical
# unpromoted chain with no split-K at decode, and C's FP32 words hashed (fused into the epilogue at prefill, a side kernel at
# decode)?  For R1-H: the same chain from the hot start H = P_i Q_j, decode split-K merged exactly, y = bf16(fl(C - H')).
#
#   bash benchmarks/pouw/pearl_c_sm120/r1_job.sh build|bench   (from the source tree's root, under research run --cwd source)
#
# bench reads the harness build (libpouw_harness.so, build.json, libpouw_lt2.so) from INPUTS (default $RESEARCH_RUN_DIR/inputs,
# from --send), and needs a one-GPU lease (GPU_LEASE_UUID).  It records the job's CPU affinity and GPU (affinity.txt) and every
# core's busy share each second (cores.tsv), so a neighbour's load can be read against each timing.  Outputs:
# $RESEARCH_RUN_DIR/out (declare 'out/*'); scratch in $TMPDIR.
set -euo pipefail
mode="${1:?usage: r1_job.sh build|bench}"
OUT="${RESEARCH_RUN_DIR:?}/out"
SHIP="$OUT/ship"
SCRATCH="$(mktemp -d)"
LOGGER=
trap '[[ -n "$LOGGER" ]] && kill "$LOGGER" 2>/dev/null; rm -rf "$SCRATCH"' EXIT
mkdir -p "$SHIP"
export CUDA_HOME="${CUDA:-/usr/local/cuda-13.0}"
"$CUDA_HOME/bin/nvcc" --version | grep -q "V13.0.88" || { echo "$CUDA_HOME's nvcc is not 13.0.1 (V13.0.88)" >&2; exit 1; }
FLAGS=(-O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -fmad=false -Xptxas -v -cubin)
"$CUDA_HOME/bin/nvcc" "${FLAGS[@]}" -o "$SHIP/pearl_c_sm120.cubin" benchmarks/pouw/pearl_c_sm120/pearl_c_sm120.cu 2> "$SHIP/ptxas.txt" \
  || { tail -40 "$SHIP/ptxas.txt" >&2; exit 1; }
PATH="$CUDA_HOME/bin:$PATH" "$CUDA_HOME/bin/cuobjdump" -sass "$SHIP/pearl_c_sm120.cubin" > "$SCRATCH/sass.txt"
python3 benchmarks/pouw/pearl_c_sm120/sass_gate.py "$SHIP/ptxas.txt" "$SCRATCH/sass.txt" "$SHIP/sass.json"
{ "$CUDA_HOME/bin/nvcc" --version; echo "nvcc ${FLAGS[*]} -o pearl_c_sm120.cubin pearl_c_sm120.cu"; } > "$SHIP/toolchain.txt"
sha256sum "$SHIP/pearl_c_sm120.cubin" | tee "$SHIP/cubin.sha256"
[[ "$mode" == build ]] && exit 0

{ echo "affinity: $(taskset -pc $$)"; echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-}"; echo "GPU_LEASE_UUID=${GPU_LEASE_UUID-}"
  nvidia-smi --query-gpu=index,uuid,name,clocks.max.sm,power.limit --format=csv,noheader 2>&1; } > "$OUT/affinity.txt"
python3 - "$OUT/cores.tsv" <<'EOF' &
import sys, time
def read():
    with open("/proc/stat") as f:
        return {w[0]: list(map(int, w[1:])) for w in map(str.split, f) if w[0][:3] == "cpu" and w[0][3:].isdigit()}
prev = read()
names = sorted(prev, key=lambda c: int(c[3:]))
with open(sys.argv[1], "w", buffering=1) as out:
    out.write("unix_s\t" + "\t".join(names) + "\n")
    while True:
        time.sleep(1)
        cur = read()
        cells = []
        for c in names:
            d = [x - y for x, y in zip(cur[c], prev[c])]
            cells.append(f"{1 - (d[3] + d[4]) / sum(d):.2f}" if sum(d) else "nan")
        out.write(f"{time.time():.0f}\t" + "\t".join(cells) + "\n")
        prev = cur
EOF
LOGGER=$!

. benchmarks/pouw/harness/jobs/env.sh "${INPUTS:-$RESEARCH_RUN_DIR/inputs}"
# One harness run per arm and shape, each its own harness-<arm>-<shape>.json, so a preemption loses only the one it stopped.
ARMS="${ARMS:-r1_arm:R1ChainSm120}"
SHAPES="${SHAPES:-llama3.1-8b-o-decode llama3.1-8b-o-prefill llama3.1-8b-qkv-decode llama3.1-8b-qkv-prefill
  llama3.1-8b-down-decode llama3.1-8b-down-prefill llama3.1-8b-gate_up-decode llama3.1-8b-gate_up-prefill}"
rc=0
for arm in $ARMS; do
  for shape in $SHAPES; do
    tag="${arm%%:*}-$shape"
    PEARLC_SHIP="$SHIP" CUDA_DEVICE_ORDER=PCI_BUS_ID POUW_SASS_GATE_CACHE="$SCRATCH/sass-gate" \
      "$PY" benchmarks/pouw/harness/bench.py --gpu lease --expect-uuid "${GPU_LEASE_UUID:?no GPU lease}" \
      --arm-path benchmarks/pouw/pearl_c_sm120 --arm "$arm" --families fp8-e4m3 --shapes "$shape" \
      --out "$OUT/harness-$tag.json" > "$OUT/bench-$tag.log" 2>&1 && r=0 || { r=$?; rc=$r; tail -20 "$OUT/bench-$tag.log" >&2; }
    echo "$(date -u +%H:%M:%SZ) $tag: rc $r" | tee -a "$OUT/progress.txt" >&2
  done
done
exit "$rc"
