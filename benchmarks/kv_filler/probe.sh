#!/usr/bin/env bash
# What node 2 offers this benchmark (cwd = the source tree): the GPUs and their leases, the vLLM and torch environments,
# the CUDA tools (nvcc, nsys's GPU metrics, ncu, dcgmi) and the CPUs.  Writes $RESEARCH_RUN_DIR/out/probe.txt.
set -uo pipefail
O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
{
  date -u
  nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,clocks.sm,clocks.max.sm --format=csv
  gpu-lease status 2>&1 | head -30
  for d in /workspace/pouw/mvp-e2e /workspace/pouw/fill-out/pous/work/pous-bench; do echo "== $d"; ls "$d" 2>&1 | head; done
  cat /workspace/pouw/mvp-e2e/ready.json 2>&1 | head -c 1500; echo
  ls /workspace/pouw/mvp-e2e/models 2>&1
  ls /usr/local/ | rg -i cuda
  ls /usr/local/cuda-13.0/bin 2>&1 | tr '\n' ' '; echo
  for t in nsys ncu dcgmi nvcc; do echo "$t: $(command -v $t) $(ls /usr/local/cuda-13.0/bin/$t 2>/dev/null)"; done
  ls -d /opt/nvidia/nsight-systems* /usr/local/cuda-13.0/nsight-systems* 2>&1
  cat /proc/driver/nvidia/params 2>/dev/null | rg -i profil
  nproc; taskset -p $$
  /workspace/pouw/mvp-e2e/venv312/bin/python -c 'import vllm, torch; print(vllm.__version__, torch.__version__)' 2>&1 | tail -1
  df -h /workspace /tmp | tail -2
  free -g | head -2
} > "$O/probe.txt" 2>&1
cat "$O/probe.txt"
