"""GP-01's chronology (`verity_vllm.program.workload`) of a request with a declared prefix."""
from __future__ import annotations

from verity.circuits.defs import REGISTRY

from tests.program.test_workload_program import _WP, _desc
from verity_vllm.program import workload as W


def test_a_declared_prefix_keeps_the_prefill_until_it_reaches_the_prompts_end():
    """A prefix shorter than the prompt is a prefix-cache hit: the prefill is programmed at the arrival and the decode steps
    after it, as with no prefix.  One that reaches the prompt's end is a request resumed mid-decode: no prefill, and its
    first step is a decode at its arrival."""
    c = W.request_component(_desc(3, 2), REGISTRY, _WP(3))

    def chronology(prefix):
        wl = W.WorkloadProgram([W.WorkloadRequest("r0", 0, 3, 3, 4, c, T=2, prefix=prefix)], manifest_digest="m" * 8, lag=0)
        ch = wl.chronology()
        return ch["requests"][0]["programmed_steps"], [m["phase"] for e in ch["engine_steps"] for m in e["members"]]

    cold = ({"prefill": 4, "decode": [5, 6], "count": 3}, ["prefill", "decode", "decode"])
    assert chronology(None) == cold
    assert chronology({"rows": 2, "producers": ["p"]}) == cold
    resumed = ({"prefill": None, "decode": [4, 6], "count": 3}, ["decode", "decode", "decode"])
    assert chronology({"rows": 3, "carried": True}) == resumed
    assert chronology({"rows": 5, "carried": True}) == resumed
