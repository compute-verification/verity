#!/usr/bin/env bash
# CPERF 2026-09-17: worker environment probe (light job). Prints host/GPU provenance, venv facts,
# whether the blake3 wheel is present, and what B0 material already exists on the worker.
set -uo pipefail
echo "== date"; date -u +%FT%TZ
echo "== host"; hostname; lscpu | grep -E "Model name|^CPU\(s\)|Thread\(s\) per core|Socket|NUMA node\(s\)|Flags" | sed 's/Flags:.*sha_ni.*/Flags: ... sha_ni present/'
echo "== cgroup cpu.max"; cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo n/a
echo "== nproc"; nproc
echo "== mem"; free -g | head -2
echo "== gpu"; nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "== gpu procs"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo "== disk"; df -h /workspace | tail -1
echo "== venvs"
for V in /workspace/venv-cu129 /workspace/venv-py312; do
  echo "-- $V"
  $V/bin/python -c "import sys,ssl,hashlib; print(sys.version.split()[0], ssl.OPENSSL_VERSION)"
  $V/bin/python -c "import blake3; print('blake3', blake3.__version__)" 2>&1 | tail -1
  $V/bin/python -c "import torch; print('torch', torch.__version__, torch.version.cuda)" 2>&1 | tail -1
  $V/bin/python -c "import triton; print('triton', triton.__version__)" 2>&1 | tail -1
  $V/bin/python -c "import vllm; print('vllm', vllm.__version__)" 2>&1 | tail -1
  $V/bin/python -c "import numpy; print('numpy', numpy.__version__)" 2>&1 | tail -1
done
echo "== hf"; ls /workspace/hf/hub 2>/dev/null | head
echo "== cp"; ls /workspace/cp 2>/dev/null
echo "== cperf"; ls -la /workspace/cp/cperf 2>/dev/null || echo none
echo "== runs"; ls /workspace/cp/cov/runs 2>/dev/null | head -30
R=/workspace/cp/cov/runs/B0-cov_b0_b8-static_c256
if [ -d "$R" ]; then
  echo "== run dir $R"; ls "$R"
  echo "== du"; du -sh "$R"/* 2>/dev/null
  echo "== capture"; ls "$R/capture" 2>/dev/null | head -30
  for d in "$R"/capture/*; do [ -d "$d" ] && { echo "-- $d"; ls "$d" | head -20; }; done
  find "$R" -maxdepth 4 -name commitments.json 2>/dev/null | head
  find "$R" -maxdepth 4 -name "replay.json" 2>/dev/null | head
  find "$R" -maxdepth 5 -type d -name weights 2>/dev/null | head
  find "$R" -maxdepth 5 -type d -name ru 2>/dev/null | head
fi
echo "== snapshot"; pwd; ls; cat EXPORT.json 2>/dev/null | head -c 600; echo
ls e2e | head -5
echo "== done"
