"""`test_fa3_launch_context`'s launch contexts on the stored Llama-3.2-1B B8 workload (`workloads/`, a fixture of the registry).
A sweep on a pod wrote a workload file of the same name under `out/gen/sweep/workloads`, which no tracked file holds; that the
stored one holds that file's bytes is not shown.
"""
from __future__ import annotations

import json

from verity_vllm.config import WORKLOADS, workload_file
from verity_vllm.pipeline import launch_context as LC

LLAMA_B8 = WORKLOADS / "llama32-1b__bf16__h100__tp1__b8__i1024__o128__mixed__greedy__bi-eager.json"


def test_stored_workload_file():
    with open(workload_file(LLAMA_B8)) as f:
        w = json.load(f)
    assert LC.spec_for(w, 8, 14) == json.dumps([1024] + [1] * 14, separators=(",", ":"))
    assert all(v[0] == 1024 for v in LC.shape_contexts(w).values())
