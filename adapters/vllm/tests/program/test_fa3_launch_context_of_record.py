"""`test_fa3_launch_context`'s launch contexts on the measured Llama-3.2-1B B8 workload file (`out/gen/sweep/workloads`, written by
a sweep on a pod, which no tracked file holds). Skipped where it is absent.
"""
from __future__ import annotations

import json

import pytest

from verity_vllm.pipeline import launch_context as LC


def test_measured_workload_file():
    import os
    p = os.path.join(os.path.dirname(__file__), "..", "..", "out", "gen", "sweep", "workloads",
                     "llama32-1b__bf16__h100__tp1__b8__i1024__o128__mixed__greedy__bi-eager.json")
    if not os.path.exists(p):
        pytest.skip("workload file not in this checkout")
    w = json.load(open(p))
    assert LC.spec_for(w, 8, 14) == json.dumps([1024] + [1] * 14, separators=(",", ":"))
    assert all(v[0] == 1024 for v in LC.shape_contexts(w).values())
