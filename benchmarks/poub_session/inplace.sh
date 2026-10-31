#!/usr/bin/env bash
# The in-place pass's gate and its timing at 8B scale on one GPU (`inplace_gate.py`, cwd = the source tree): builds
# session.cu outside the lease and writes each kernel's SASS digest (sass.txt, so the kernels the in-place pass shares
# with the contiguous one compare with a build of the base), then the gate and the timing. Results go to
# $RESEARCH_RUN_DIR/out.
#
#   RESEARCH_RUN_DIR=DIR bash benchmarks/poub_session/inplace.sh
#
# Under a queue that leases the GPU around the whole job, QUEUED=1 skips gpu-lease. ARGS pass through to
# inplace_gate.py (--no-timing, --pages, --reps).
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
PY=${PY:-}
for p in /workspace/pouw/fill-out/pous/work/pous-bench/gpu312-torch2.14.0-cu130/bin/python \
         /workspace/pouw/mvp-e2e/venv312/bin/python; do
  [ -z "$PY" ] && [ -x "$p" ] && PY=$p
done
[ -n "$PY" ] || { echo "no torch venv on this node"; exit 3; }
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0} PATH=/usr/local/cuda-13.0/bin:$(dirname "$PY"):$PATH
RS=benchmarks/poub_session
B=$(mktemp -d /tmp/remote-inplace.XXXXXX); trap 'rm -rf "$B"' EXIT
{ nvcc --version; echo "python: $PY"; } > "$O/build.txt"
nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xcompiler -fPIC -shared -Xptxas -v \
  -o "$B/session.so" $RS/session.cu > "$O/session.ptxas" 2>&1 || { tail -40 "$O/session.ptxas"; exit 4; }
mkdir "$B/sass" && cuobjdump -sass "$B/session.so" | awk -v d="$B/sass" '/Function :/{f = d "/" $3} f {print > f}'
(cd "$B/sass" && md5sum -- *) > "$O/sass.txt"
LEASE=(gpu-lease 1 --wait --max-min "${MAXMIN:-20}" --)
[ "${QUEUED:-0}" = 1 ] && LEASE=()
# shellcheck disable=SC2086
"${LEASE[@]}" "$PY" $RS/inplace_gate.py --lib "$B/session.so" --out "$O" ${ARGS:-} 2>&1 | tee "$O/inplace.log"
exit "${PIPESTATUS[0]}"
