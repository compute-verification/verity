#!/usr/bin/env bash
# CPU preflight of served_h1 on an RTX PRO 6000 node: the imports serve.py makes, the int8 pin's path, the research CLI,
# and one receipt
set -uo pipefail
ROOT=$PWD
cd integrations/vllm
PY=/workspace/jobs/venv312/bin/python3
export PYTHONPATH=.:$ROOT:$ROOT/tools/research/src:$ROOT/tools/config:$ROOT/benchmarks/pouw/served_h1 HF_HOME=/workspace/hf
echo "research: $(command -v research)"
ls -d /workspace/hf/hub/models--verity-int8--Llama-3.1-8B-INT8/snapshots/723d342ac443b040931242c768df342b0f6f15a7 || echo "PIN MISSING"
CUDA_VISIBLE_DEVICES= "$PY" - <<'PY'
import time, os, sys
t = time.perf_counter()
import serve, receipts as RC, commit as CM
from tests.engine import int8_exact_gpu as G
from verity.core.protocols.pouw.coins import Challenger
from verity_vllm import protocols as PO
from verity_vllm.engine import int8_exact as E, int8_ops
from verity_vllm.engine.build import build_engine, target_mismatches
from verity_vllm.program.frontend.target_profile import TargetProfile
from verity_vllm.protocols import pouw
from verity_vllm.protocols.pouw import h1 as PH
print("imports ok", round(time.perf_counter() - t, 1), "s; linear types", pouw.linear_types())
import torch
g = torch.Generator().manual_seed(1)
a = torch.randint(-127, 128, (40, 4096), generator=g, dtype=torch.int8)
b = torch.randint(-127, 128, (64, 4096), generator=g, dtype=torch.int8)
salt, ra, rb = os.urandom(64), os.urandom(64), os.urandom(64)
call = RC.Call(serve.padded(a, 64), b)
t = time.perf_counter()
rc = RC.call_receipts(call, salt, 11, ra, rb, only=[3])
print("cpu receipt", round(time.perf_counter() - t, 1), "s; equal reference", list(rc[0]) == RC.reference(call, salt, 11, ra, rb, 3)[1])
G.pin  # the pin lookup is exercised by the GPU run
PY
echo "preflight rc=$?"
