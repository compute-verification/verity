#!/usr/bin/env bash
# 1a's microbenchmark (zc1a.cu), beside the harness and never part of the prover's build: built against C-Flock's cuda-ghash
# headers, read from FLOCK_SRC without changing them (their sha256 in out/headers.sha256), and run on the job's GPU: the product
# check, both kernels' outputs byte for byte at M, their launch times at TIME_M (the GPU's clocks before and after in
# out/clocks.csv) and, with NCU_ROOT, Nsight Compute's full section set on one launch of each kernel at NCU_M
# (out/zc1a-ncu.ncu-rep and its CSV pages). Run from the verity tree (research run --cwd source) in a 1-GPU job; the build goes to
# a temporary directory, the results to RESEARCH_RUN_DIR/out (zc1a.log, zc1a.jsonl, build.log).
# env: FLOCK_SRC=/workspace/jobs/proofs-bf16-hill/flock-gpu-link/flock  CUDA_TK= (default: the first CUDA 13.3 of
#      FLOCK_WORK/cuda-root and /workspace/cache/flock/cuda-root)  CUDA_TOOLS= (cuobjdump and nvdisasm, for out/sass_counts.txt)
#      SM=120  M=13,20,28,32,35  TIME_M=35  ITERS=10  NCU_ROOT=  NCU_M=30
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
R=${RESEARCH_RUN_DIR:-/tmp/zc1a}; O=$R/out; mkdir -p $O
FLOCK_SRC=${FLOCK_SRC:-/workspace/jobs/proofs-bf16-hill/flock-gpu-link/flock}; G=$FLOCK_SRC/cuda-ghash
FLOCK_WORK=${FLOCK_WORK:-/workspace/jobs/proofs-bf16-hill}
for c in $FLOCK_WORK/cuda-root /workspace/cache/flock/cuda-root; do
  [ -z "${CUDA_TK:-}" ] && [ -x $c/usr/local/cuda-13.3/bin/nvcc ] && CUDA_TK=$c/usr/local/cuda-13.3
done
[ -x "${CUDA_TK:-}/bin/nvcc" ] || { echo "no CUDA 13.3 nvcc"; exit 1; }
B=$(mktemp -d); trap 'rm -rf $B' EXIT
(cd $G && sha256sum zerocheck_round1_cpustyle.cuh zerocheck_round1.cuh f128.cuh phi8_table.cuh) > $O/headers.sha256 || exit 1
sha256sum $HERE/zc1a.cu >> $O/headers.sha256
SM=${SM:-120}
$CUDA_TK/bin/nvcc -O3 -std=c++17 -lineinfo -gencode arch=compute_$SM,code=sm_$SM -Xptxas -v -I $G -o $B/zc1a $HERE/zc1a.cu \
  > $O/build.log 2>&1 || { cat $O/build.log; exit 1; }
PATH=$CUDA_TK/bin:${CUDA_TOOLS:-/workspace/research/lanes/proofs-bf16-hill/cuda-tools/usr/local/cuda-13.3}/bin:$PATH \
  cuobjdump -sass $B/zc1a > $B/zc1a.sass 2> $O/sass.err && python3 - $B/zc1a.sass > $O/sass_counts.txt <<'EOF'
import re, sys
# per kernel: its SASS instruction count and the counts of the opcodes that matter here (shared loads, byte permutes, integer ops)
name, counts = None, {}
for line in open(sys.argv[1]):
    m = re.match(r"\s*Function : (\S+)", line)
    if m:
        name = m.group(1); counts[name] = {}
        continue
    m = re.match(r"\s*/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_]*)(\.[A-Z0-9_.]+)?", line)
    if name and m:
        counts[name][m.group(1)] = counts[name].get(m.group(1), 0) + 1
for n, c in counts.items():
    if "zerocheck_first_round_cpu_structured" in n or "zc1a_first_round" in n:
        top = ", ".join(f"{k} {v}" for k, v in sorted(c.items(), key=lambda kv: -kv[1])[:14])
        print(f"{n}: {sum(c.values())} instructions; {top}")
EOF
Q=timestamp,clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw
nvidia-smi --query-gpu=$Q --format=csv > $O/clocks.csv 2>&1
$B/zc1a --m ${M:-13,20,28,32,35} --time-m ${TIME_M:-35} --iters ${ITERS:-10} > $O/zc1a.log 2>&1; rc=$?
nvidia-smi --query-gpu=$Q --format=csv,noheader >> $O/clocks.csv 2>&1
sed -n 's/^ZC1A //p' $O/zc1a.log > $O/zc1a.jsonl
if [ -n "${NCU_ROOT:-}" ]; then
  $NCU_ROOT/ncu --kernel-name regex:'zerocheck_first_round_cpu_structured|zc1a_first_round' --clock-control none --set full \
    -o $O/zc1a-ncu $B/zc1a --ncu ${NCU_M:-30} > $O/ncu.log 2>&1 || rc=1
  $NCU_ROOT/ncu --import $O/zc1a-ncu.ncu-rep --page details --csv > $O/zc1a-ncu.details.csv
  $NCU_ROOT/ncu --import $O/zc1a-ncu.ncu-rep --page raw --csv > $O/zc1a-ncu.raw.csv
fi
exit $rc
