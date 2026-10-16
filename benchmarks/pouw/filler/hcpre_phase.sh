#!/usr/bin/env bash
# HC-PREFILL's phase probe runner: builds libhcpre_fill.so (dcap.cu's filler behind a C API) outside the lease, then under
# one lease runs hcpre_phase.py in node 2's vLLM venv (VENV, default /workspace/pouw/mvp-e2e/venv312) with ARGS, once per
# filler shape in FILLS (space-separated W,NC,S,LOG2,MAP; one process each, 16 points apart), or once when FILLS is
# unset. Run from the repository root (research run --cwd source); outputs under $OUT.
set -uo pipefail
O=${OUT:?}
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
VENV=${VENV:-/workspace/pouw/mvp-e2e/venv312}
B=$(mktemp -d /tmp/hcpre-phase.XXXXXX)
trap 'rm -rf "$B"' EXIT
"$NVCC" -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -shared -Xcompiler -fPIC -o "$B/libhcpre_fill.so" \
  benchmarks/pouw/filler/hcpre_fill.cu -lcublasLt -lnvidia-ml -ldl > "$O/build.log" 2>&1 || { echo "build failed"; exit 4; }
git rev-parse HEAD > "$O/commit.txt"
"$VENV/bin/python" -c "import torch, vllm; print(torch.__version__, vllm.__version__)" > "$O/venv.txt" 2>&1
q="clocks.sm,clocks.max.sm,power.draw,power.limit,clocks.applications.graphics,clocks.default_applications.graphics,clocks_event_reasons.active"
cat > "$B/inner.sh" <<EOF
set -uo pipefail
smi() { nvidia-smi --query-gpu=$q --format=csv,noheader,nounits \${GPU_LEASE_UUID:+-i \$GPU_LEASE_UUID} >> "$O/smi.csv"; }
smi
run() {
  LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:\${LD_LIBRARY_PATH:-} "$VENV/bin/python" benchmarks/pouw/filler/hcpre_phase.py \
    --lib "$B/libhcpre_fill.so" --out "$O" ${ARGS:-} "\$@" >> "$O/phase.log" 2>&1
  echo "phase \$* rc \$?" >> "$O/rc.txt"
}
if [ -z "${FILLS:-}" ]; then
  run
else
  p=0
  for f in ${FILLS:-}; do
    run --fill "\$f" --point0 "\$p"
    p=\$((p + 16))
  done
fi
smi
EOF
gpu-lease 1 --wait --max-min "${MAXMIN:-10}" -- bash "$B/inner.sh"
echo "lease rc $?" >> "$O/rc.txt"
