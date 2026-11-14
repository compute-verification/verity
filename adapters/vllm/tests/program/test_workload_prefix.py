"""GP-01's chronology (`verity_vllm.program.workload`) of a request with a declared prefix."""
from __future__ import annotations

from verity.circuits.defs import REGISTRY

from tests.program.test_workload_program import _WP, _desc
from verity_vllm.program import workload as W


def _chronology(prompt_len: int, T: int, prefix: dict | None) -> tuple[dict, list[str]]:
    """A lone request's programmed steps and the phase of each of its engine steps.  It arrives at step 4 with a prompt of
    `prompt_len` and a `max_tokens` of `T + 1`."""
    c = W.request_component(_desc(prompt_len, T), REGISTRY, _WP(prompt_len))
    wl = W.WorkloadProgram([W.WorkloadRequest("r0", 0, prompt_len, T + 1, 4, c, T=T, prefix=prefix)], manifest_digest="m" * 8, lag=0)
    ch = wl.chronology()
    return ch["requests"][0]["programmed_steps"], [m["phase"] for e in ch["engine_steps"] for m in e["members"]]


def test_a_declared_prefix_keeps_the_prefill_until_it_reaches_the_prompts_end():
    """A prefix shorter than the prompt is a prefix-cache hit: the prefill is programmed at the arrival and the decode steps
    after it, as with no prefix.  One that reaches the prompt's end is a request resumed mid-decode: no prefill, and its
    first step is a decode at its arrival."""
    cold = ({"prefill": 4, "decode": [5, 6], "count": 3}, ["prefill", "decode", "decode"])
    assert _chronology(3, 2, None) == cold
    assert _chronology(3, 2, {"rows": 2, "producers": ["p"]}) == cold
    resumed = ({"prefill": None, "decode": [4, 6], "count": 3}, ["decode", "decode", "decode"])
    assert _chronology(3, 2, {"rows": 3, "carried": True}) == resumed
    assert _chronology(3, 2, {"rows": 5, "carried": True}) == resumed


def test_the_prompts_length_and_not_max_tokens_says_whether_a_prefix_reaches_the_prompts_end():
    """The first test's request has a prompt of 3 and a `max_tokens` of 3, so it cannot tell which of the two a prefix's
    rows are held to.  Here the two differ, each way round, and a prefix's rows lie between them."""
    # a prompt of 5 and a `max_tokens` of 3: rows that reach `max_tokens` and not the prompt's end keep the prefill
    cold = ({"prefill": 4, "decode": [5, 6], "count": 3}, ["prefill", "decode", "decode"])
    assert _chronology(5, 2, {"rows": 3, "producers": ["p"]}) == cold
    assert _chronology(5, 2, {"rows": 4, "producers": ["p"]}) == cold
    assert _chronology(5, 2, {"rows": 5, "carried": True}) == ({"prefill": None, "decode": [4, 6], "count": 3}, ["decode"] * 3)
    # a prompt of 3 and a `max_tokens` of 5: rows that reach the prompt's end and not `max_tokens` are a resumed request's
    assert _chronology(3, 4, {"rows": 2, "producers": ["p"]}) == ({"prefill": 4, "decode": [5, 8], "count": 5}, ["prefill"] + ["decode"] * 4)
    resumed = ({"prefill": None, "decode": [4, 8], "count": 5}, ["decode"] * 5)
    assert _chronology(3, 4, {"rows": 3, "carried": True}) == resumed
    assert _chronology(3, 4, {"rows": 4, "carried": True}) == resumed
