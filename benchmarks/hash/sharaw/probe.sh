#!/usr/bin/env bash
# What a job on a node sees: git refs, the toolkit, profilers, CPUs. No GPU work.
set -u
O=${OUT:?}
{
  hostname; nproc; free -g | head -2
  git rev-parse HEAD; git remote -v | head -2
  git cat-file -t origin/cursor/commit-speed-21f3 2>&1 || true
  git ls-remote origin cursor/commit-speed-21f3 2>&1 | head -2 || true
  ls /usr/local/cuda/bin | grep -E "nvcc|nsys|ncu" ; ls -d /usr/local/cuda* /opt/nvidia/nsight* 2>/dev/null
  command -v nsys ncu gpu-lease python3
  /usr/local/cuda/bin/nvcc --version | tail -2
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,clocks.sm,clocks.max.sm --format=csv
} > "$O/probe.txt" 2>&1
cat "$O/probe.txt"
