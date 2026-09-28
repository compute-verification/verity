"""Timing through vLLM's engine, shared by `verity-vllm pous-bench` and `verity-vllm pous e2e`: fixed-token prompts, each engine
step timed, and the two configurations POUS slowdowns are reported for: prefill tokens/s over the time to every first token,
decode tokens/s over the steps after it."""
from __future__ import annotations

import time
from typing import Any


def prompt(i: int, length: int) -> list[int]:
    """Fixed token ids (no sampling): request `i`'s prompt of `length` tokens."""
    return [1000 + (i * 7919 + j * 104729) % 30000 for j in range(length)]


def run_config(engine, requests: list[list[int]], max_tokens: int, store_info) -> dict[str, Any]:
    """Step the engine over `requests` until done; each step's wall time and the requests with a first token after it."""
    from vllm import SamplingParams

    llm_engine = engine.llm.llm_engine
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    tag = f"r{time.monotonic_ns()}"
    for i, ids in enumerate(requests):
        llm_engine.add_request(f"{tag}-{i}", {"prompt_token_ids": ids}, params)
    before = store_info()
    steps, first, done = [], set(), {}
    t0 = time.perf_counter()
    while llm_engine.has_unfinished_requests():
        s = time.perf_counter()
        outs = llm_engine.step()
        e = time.perf_counter()
        for o in outs:
            if o.outputs and o.outputs[0].token_ids:
                first.add(o.request_id)
            if o.finished:
                done[o.request_id] = len(o.outputs[0].token_ids)
        steps.append({"s": e - s, "at": e - t0, "first": len(first)})
    after = store_info()
    row = {"requests": len(requests), "prompt": len(requests[0]), "max_tokens": max_tokens, "steps": steps,
           "generated": sum(done.values()), "wall_s": time.perf_counter() - t0}
    if before is not None:
        row["decode_s"] = after["decode_s"] - before["decode_s"]
        row["decodes"] = after["decodes"] - before["decodes"]
        row["decoded_bytes"] = after["decoded_bytes"] - before["decoded_bytes"]
    return row


def first_token_at(row: dict[str, Any]) -> float:
    return next(s["at"] for s in row["steps"] if s["first"] == row["requests"])


def prefill(engine, length: int, batch: int, store_info) -> dict[str, Any]:
    row = run_config(engine, [prompt(i, length) for i in range(batch)], 1, store_info)
    t = first_token_at(row)
    return {"kind": "prefill", "length": length, "batch": batch, "tokens": length * batch, "seconds": t,
            "tokens_per_s": length * batch / t, **row}


def decode(engine, batch: int, prompt_len: int, n: int, store_info) -> dict[str, Any]:
    row = run_config(engine, [prompt(i, prompt_len) for i in range(batch)], 1 + n, store_info)
    t = row["wall_s"] - first_token_at(row)
    return {"kind": "decode", "batch": batch, "tokens": batch * n, "seconds": t, "tokens_per_s": batch * n / t, **row}
