#!/usr/bin/env bash
# d2h_bw.py on one GPU (CUDA_VISIBLE_DEVICES), then cpu_sha512.sh on the same host: the two sides of hashing the Commit's values on
# the host. Measurement only.
set -uo pipefail
. benchmarks/harness/env.sh
export OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}
mkdir -p "$OUT"
"$VENV/bin/python" benchmarks/served_commit/d2h_bw.py 2>&1 | tee "$OUT/d2h_bw.log"
echo "d2h_bw rc ${PIPESTATUS[0]}"
[ "${CPU_SHA:-1}" = 1 ] && bash benchmarks/served_commit/cpu_sha512.sh 2>&1 | tee "$OUT/cpu_sha512.log"
